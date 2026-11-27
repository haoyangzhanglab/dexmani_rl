# MIT License
# Copyright (c) 2026 RoboFFT Authors
# Copyright (c) 2024 Intelligent Robot Motion Lab
# Copyright (c) 2022 Naoki Katsura
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""RoboFFT-F, following the official a9ca8c8 Can-state release recipe."""
import copy
import math
import numpy as np
import torch
from torch import nn
from .rfs import gae


class RoboFFTActor(nn.Module):
    def __init__(self, decoder, horizon, action_dim, inference_steps, n_samples=1, noise_std=.25):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (horizon, action_dim, inference_steps, n_samples)):
            raise ValueError('RoboFFT dimensions and sampling counts must be positive integers')
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError('RoboFFT noise_std must be finite and nonnegative')
        self.decoder = copy.deepcopy(decoder).eval().requires_grad_(True)
        self.horizon, self.action_dim = horizon, action_dim
        self.inference_steps, self.n_samples, self.noise_std = inference_steps, n_samples, noise_std

    @torch.no_grad()
    def sample(self, context, generator=None, eval_mode=False, *, return_clip_fraction=False):
        shape = (len(context), self.horizon, self.action_dim)
        sample = torch.randn(shape, device=context.device, dtype=context.dtype, generator=generator)
        dt = 1. / self.inference_steps
        context_kv = self.decoder.prepare_context_kv(context) if self.inference_steps > 1 else None
        if return_clip_fraction:
            clipped = sample.new_zeros(())
        for index in range(self.inference_steps):
            timestep = sample.new_full((len(context),), index * dt)
            sample = sample + dt * self.decoder(
                x=sample, timestep=timestep, context=context, context_kv=context_kv)
            if return_clip_fraction:
                clipped += (sample.abs() > 1).sum()
            sample = sample.clamp(-1, 1)
            if not eval_mode and self.noise_std > 0:
                sample = sample + self.noise_std * math.sqrt(dt) * torch.randn(
                    shape, device=sample.device, dtype=sample.dtype, generator=generator)
        if return_clip_fraction:
            clipped += (sample.abs() > 1).sum()
            total = (self.inference_steps + 1) * sample.numel()
        sample = sample.clamp(-1, 1)
        return (sample, clipped / total) if return_clip_fraction else sample

    def draw_matching(self, action, generator=None):
        shape = (len(action), self.n_samples)
        t = torch.randint(self.inference_steps, shape, device=action.device, generator=generator)
        t = t.to(action.dtype) / self.inference_steps
        eps = torch.randn((*shape, self.horizon, self.action_dim), device=action.device,
                          dtype=action.dtype, generator=generator)
        return eps, t

    def log_prob(self, context, action, eps, t):
        # This is a matching-loss surrogate, not a normalized action likelihood.
        target = action[:, None] - eps
        xt = t[..., None, None] * action[:, None] + (1 - t[..., None, None]) * eps
        condition = context[:, None].expand(-1, eps.shape[1], *context.shape[1:]).flatten(0, 1)
        velocity = self.decoder(x=xt.flatten(0, 1), timestep=t.flatten(), context=condition)
        return -(velocity.reshape_as(target) - target).square().mean(dim=(-2, -1))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, width=256):
        super().__init__()
        self.input = nn.Linear(state_dim, width)
        self.residual = nn.Sequential(nn.Mish(), nn.Linear(width, width), nn.Mish(), nn.Linear(width, width))
        self.output = nn.Linear(width, 1)

    def forward(self, state):
        hidden = self.input(state)
        return self.output(hidden + self.residual(hidden))


def ppo_loss(new_log_prob, old_log_prob, advantages, clip_coef=.02, ratio_alpha=8.,
             logprob_min=-1., logprob_max=1.):
    new = new_log_prob.clamp(logprob_min, logprob_max)
    old = old_log_prob.clamp(logprob_min, logprob_max)
    log_ratio = new - old
    ratio, scaled_ratio = log_ratio.exp(), (ratio_alpha * log_ratio).exp()
    advantages = advantages.reshape(-1, 1)
    loss = torch.maximum(-advantages * scaled_ratio,
                         -advantages * scaled_ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
    with torch.no_grad():
        metrics = dict(approx_kl=(ratio - 1 - log_ratio).mean(), ratio=ratio.mean(),
            clip_fraction=((ratio - 1).abs() > clip_coef).float().mean(),
            logprob_saturation=((new_log_prob <= logprob_min) | (new_log_prob >= logprob_max)).float().mean(),
            old_logprob_saturation=((old_log_prob <= logprob_min) | (old_log_prob >= logprob_max)).float().mean())
    return loss, metrics


class RewardScaler:
    def __init__(self, gamma):
        self.gamma = gamma
        self.mean, self.var, self.count = np.zeros(()), np.ones(()), 1e-4
        self.last_return = np.zeros(1)

    def __call__(self, reward, first):
        rewards = reward.detach().cpu().numpy().reshape(1, -1)
        starts = first.detach().cpu().numpy().reshape(1, -1)
        returns = np.zeros_like(rewards)
        previous = self.last_return
        for t in range(rewards.shape[1]):
            previous = returns[:, t] = rewards[:, t] + (1 - starts[:, t]) * self.gamma * previous
        self.last_return = returns[:, -1].copy()
        flat = returns.reshape(-1)
        count = len(flat)
        delta = flat.mean() - self.mean
        total = self.count + count
        moment = self.var * self.count + flat.var() * count + delta**2 * self.count * count / total
        self.mean = self.mean + delta * count / total
        # The release uses total-1, including after merging previous statistics.
        self.var, self.count = moment / (total - 1), total
        scaled = np.clip(rewards / np.sqrt(self.var + 1e-8), -10, 10)
        return torch.from_numpy(scaled.reshape(reward.shape)).to(dtype=reward.dtype)

    def state_dict(self):
        return dict(mean=self.mean.copy(), var=self.var.copy(), count=self.count,
                    last_return=self.last_return.copy())

    def load_state_dict(self, state):
        self.mean, self.var, self.count = (copy.deepcopy(state[key]) for key in ('mean', 'var', 'count'))
        self.last_return = state['last_return'].copy()


def warmup_cosine(step, minimum, maximum, warmup, cycle):
    step %= cycle
    if step < warmup:
        return minimum + (maximum - minimum) * step / warmup
    return minimum + (maximum - minimum) * (1 + math.cos(math.pi * (step - warmup) / (cycle - warmup))) / 2


class RoboFFT(nn.Module):
    def __init__(self, state_dim, actor, cfg, device):
        super().__init__()
        counts = ('value_width', 'batch_size', 'microbatch_size', 'update_epochs', 'repeat_samples',
                  'iterations')
        if any(type(cfg[key]) is not int or cfg[key] <= 0 for key in counts):
            raise ValueError('RoboFFT sizes and iteration counts must be positive integers')
        if (cfg['batch_size'] < 2 or type(cfg['lr_warmup_iterations']) is not int or
                not 0 <= cfg['lr_warmup_iterations'] < cfg['iterations'] or
                type(cfg['critic_warmup_iterations']) is not int or cfg['critic_warmup_iterations'] < 0):
            raise ValueError('Invalid RoboFFT minibatch or warmup settings')
        numeric = ('actor_lr', 'actor_min_lr', 'value_lr', 'gamma', 'gae_lambda', 'clip_coef',
                   'value_coef', 'ratio_alpha', 'target_kl', 'logprob_min', 'logprob_max')
        if (any(not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) for key in numeric) or
                not 0 < cfg['actor_min_lr'] <= cfg['actor_lr'] or cfg['value_lr'] <= 0 or
                not 0 < cfg['gamma'] <= 1 or not 0 <= cfg['gae_lambda'] <= 1 or
                not 0 < cfg['clip_coef'] < 1 or cfg['value_coef'] < 0 or cfg['ratio_alpha'] <= 0 or
                cfg['target_kl'] < 0 or cfg['logprob_min'] >= cfg['logprob_max']):
            raise ValueError('Invalid RoboFFT hyperparameters')
        self.cfg = cfg
        self.actor, self.value = actor, ValueNetwork(state_dim, cfg['value_width'])
        self.to(device)
        self.eval()
        self.optimizers = dict(actor=torch.optim.AdamW(self.actor.parameters(), lr=cfg['actor_lr'], weight_decay=0),
                               critic=torch.optim.AdamW(self.value.parameters(), lr=cfg['value_lr'], weight_decay=0))
        self.actor_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizers['actor'], lambda step: warmup_cosine(
                step, cfg['actor_min_lr'], cfg['actor_lr'],
                cfg['lr_warmup_iterations'], cfg['iterations']) / cfg['actor_lr'])
        self.reward_scaler = RewardScaler(cfg['gamma'])

    def finish_iteration(self, iteration):
        # Match the release's outer iteration clock, including evaluation iterations.
        if iteration >= self.cfg['critic_warmup_iterations']:
            self.actor_scheduler.step()

    def update(self, batch, iteration):
        cfg = self.cfg
        device = next(self.actor.parameters()).device
        repeat_count = len(batch['state']) * cfg['repeat_samples']
        if repeat_count < 2:
            raise ValueError('RoboFFT needs at least two samples for advantage standardization')
        reward = self.reward_scaler(batch['reward'], batch['first'])
        advantage, returns = gae(reward, batch['value'], batch['next_value'], batch['terminated'],
                                 batch['truncated'], cfg['gamma'], cfg['gae_lambda'])
        totals = dict(loss=0., actor_loss=0., value_loss=0., approx_kl=0., ratio=0., clip_fraction=0.,
                      logprob_saturation=0., old_logprob_saturation=0., actor_grad_norm=0., value_grad_norm=0.)
        metric_names = tuple(key for key in totals if not key.endswith('grad_norm'))
        samples, minibatches, actor_steps, last_kl = 0, 0, 0, 0.
        for _ in range(cfg['update_epochs']):
            # Repetition changes update exposure, not the cached forward-noising draws.
            order = torch.randperm(repeat_count) // cfg['repeat_samples']
            groups = list(order.split(cfg['batch_size']))
            if len(groups[-1]) == 1:
                tail = groups.pop()
                groups[-1] = torch.cat((groups[-1], tail))
            for indices in groups:
                # The minibatch that detects excess KL is still applied, as in release.
                if cfg['target_kl'] and last_kl > cfg['target_kl']:
                    break
                adv = advantage[indices].flatten()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                for optimizer in self.optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                metric_rows, weights = [], []
                if minibatches == 0:
                    matching_ok = torch.ones((), device=device, dtype=torch.bool)
                for start in range(0, len(indices), cfg['microbatch_size']):
                    selection = indices[start:start + cfg['microbatch_size']]
                    b = {key: batch[key][selection].to(device) for key in
                         ('context', 'state', 'action', 'eps', 't', 'old_log_prob')}
                    new = self.actor.log_prob(b['context'], b['action'], b['eps'], b['t'])
                    if minibatches == 0:
                        # Compare per draw before the PPO clamp can hide a changed velocity.
                        matching_ok.logical_and_(
                            torch.isclose(new.detach(), b['old_log_prob'], rtol=1e-5, atol=1e-7).all())
                    actor_loss, diagnostics = ppo_loss(new, b['old_log_prob'], adv[start:start + len(selection)].to(device),
                        cfg['clip_coef'], cfg['ratio_alpha'], cfg['logprob_min'], cfg['logprob_max'])
                    value_loss = .5 * (self.value(b['state']) - returns[selection].to(device)).square().mean()
                    loss = actor_loss + cfg['value_coef'] * value_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite RoboFFT loss')
                    weight = len(selection) / len(indices)
                    (loss * weight).backward()
                    values = dict(loss=loss, actor_loss=actor_loss, value_loss=value_loss, **diagnostics)
                    metric_rows.append(torch.stack([values[name].detach() for name in metric_names]))
                    weights.append(weight)
                metrics = dict.fromkeys(metric_names, 0.)
                for row, weight in zip(torch.stack(metric_rows).cpu().tolist(), weights):
                    for name, value in zip(metric_names, row):
                        metrics[name] += value * weight
                metrics['actor_grad_norm'] = nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'), error_if_nonfinite=True).item()
                metrics['value_grad_norm'] = nn.utils.clip_grad_norm_(self.value.parameters(), float('inf'), error_if_nonfinite=True).item()
                if minibatches == 0 and not matching_ok:
                    raise ValueError('Initial RoboFFT matching logp differs from stored logp; old policy or forward draws changed')
                self.optimizers['critic'].step()
                if iteration >= cfg['critic_warmup_iterations']:
                    self.optimizers['actor'].step()
                    actor_steps += 1
                last_kl = metrics['approx_kl']
                for name, value in metrics.items():
                    totals[name] += value * len(indices)
                samples += len(indices)
                minibatches += 1
        metrics = {name: total / samples for name, total in totals.items()}
        metrics.update(minibatches=minibatches, actor_steps=actor_steps,
                       actor_lr=self.optimizers['actor'].param_groups[0]['lr'],
                       value_lr=self.optimizers['critic'].param_groups[0]['lr'],
                       reward_variance=float(self.reward_scaler.var),
                       scaled_reward_mean=reward.mean().item(), early_stop=bool(cfg['target_kl'] and last_kl > cfg['target_kl']))
        return metrics
