import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def mlp(input_dim, output_dim, hidden_dims):
    layers = []
    for width in hidden_dims:
        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, state_dim, noise_dim, action_dim, hidden_dims, residual_scale):
        super().__init__()
        self.noise_dim = noise_dim
        self.residual_scale = residual_scale
        self.mean = mlp(state_dim, noise_dim + action_dim, hidden_dims)
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        self.log_std = nn.Parameter(torch.zeros(noise_dim + action_dim))

    def distribution(self, state):
        return Normal(self.mean(state), self.log_std.exp())

    def sample(self, state, generator=None):
        distribution = self.distribution(state)
        epsilon = torch.randn(distribution.loc.shape, device=state.device,
                              dtype=distribution.loc.dtype, generator=generator)
        joint = distribution.loc + distribution.scale * epsilon
        return joint, distribution.log_prob(joint).sum(-1, keepdim=True)

    def split(self, joint):
        noise = joint[:, :self.noise_dim]
        residual = self.residual_scale * joint[:, self.noise_dim:]
        return noise, residual


@torch.no_grad()
def gae(reward, value, next_value, terminated, truncated, gamma, gae_lambda):
    # 超时保留末观测的 bootstrap，但任何 episode 边界都切断优势递推。
    bootstrap = (~terminated.bool()).to(value.dtype)
    continuation = (~(terminated.bool() | truncated.bool())).to(value.dtype)
    delta = reward + gamma * bootstrap * next_value - value
    advantage = torch.empty_like(value)
    carry = torch.zeros_like(value[0])
    for step in reversed(range(len(value))):
        carry = delta[step] + gamma * gae_lambda * continuation[step] * carry
        advantage[step] = carry
    return advantage, advantage + value


class RFS(nn.Module):
    def __init__(self, state_dim, noise_dim, action_dim, cfg, device):
        super().__init__()
        hidden_dims = cfg['hidden_dims']
        if (not isinstance(hidden_dims, (list, tuple)) or not hidden_dims or
                any(type(width) is not int or width <= 0 for width in hidden_dims)):
            raise ValueError('RFS hidden_dims must contain positive integers')
        if any(type(cfg[key]) is not int or cfg[key] <= 0 for key in ('batch_size', 'update_epochs')):
            raise ValueError('RFS batch_size and update_epochs must be positive integers')
        numeric = ('actor_lr', 'value_lr', 'gamma', 'gae_lambda', 'clip_coef',
                   'value_coef', 'entropy_coef', 'max_grad_norm', 'residual_scale')
        if (any(type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) for key in numeric) or
                any(cfg[key] <= 0 for key in ('actor_lr', 'value_lr', 'max_grad_norm')) or
                any(cfg[key] < 0 for key in ('value_coef', 'entropy_coef', 'residual_scale')) or
                not 0 < cfg['gamma'] <= 1 or not 0 <= cfg['gae_lambda'] <= 1 or
                not 0 < cfg['clip_coef'] < 1):
            raise ValueError('Invalid RFS hyperparameters')
        self.cfg = cfg
        self.actor = Actor(state_dim, noise_dim, action_dim, hidden_dims, cfg['residual_scale'])
        self.value = mlp(state_dim, 1, hidden_dims)
        self.to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': cfg['actor_lr']},
            {'params': self.value.parameters(), 'lr': cfg['value_lr']},
        ])

    def update(self, batch):
        cfg = self.cfg
        device = self.actor.log_std.device
        b = {key: tensor.detach().to(device) for key, tensor in batch.items()}
        count = len(b['state'])
        if count == 0:
            raise ValueError('RFS requires a nonempty rollout')
        if any(not torch.isfinite(tensor).all() for tensor in b.values()):
            raise FloatingPointError('Nonfinite RFS rollout')
        advantage, returns = gae(b['reward'], b['value'], b['next_value'],
                                 b['terminated'], b['truncated'], cfg['gamma'], cfg['gae_lambda'])
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        totals = dict(loss=0., actor_loss=0., value_loss=0., entropy=0.,
                      approx_kl=0., clip_fraction=0., grad_norm=0.)
        parameters = list(self.parameters())
        for _ in range(cfg['update_epochs']):
            order = torch.randperm(count, device=device)
            for indices in order.split(cfg['batch_size']):
                state, adv = b['state'][indices], advantage[indices]
                distribution = self.actor.distribution(state)
                # PPO 的动作是原始联合采样，不是经过 residual 缩放或物理限位的动作。
                log_prob = distribution.log_prob(b['joint'][indices]).sum(-1, keepdim=True)
                log_ratio = log_prob - b['log_prob'][indices]
                ratio = log_ratio.exp()
                unclipped = -adv * ratio
                clipped = -adv * ratio.clamp(1 - cfg['clip_coef'], 1 + cfg['clip_coef'])
                actor_loss = torch.maximum(unclipped, clipped).mean()
                value_loss = F.mse_loss(self.value(state), returns[indices])
                entropy = distribution.entropy().sum(-1).mean()
                loss = actor_loss + cfg['value_coef'] * value_loss - cfg['entropy_coef'] * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite RFS loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'],
                                                    error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    values = dict(loss=loss, actor_loss=actor_loss, value_loss=value_loss,
                                  entropy=entropy, approx_kl=((ratio - 1) - log_ratio).mean(),
                                  clip_fraction=((ratio - 1).abs() > cfg['clip_coef']).float().mean(),
                                  grad_norm=grad_norm)
                    for name, metric in values.items():
                        totals[name] += metric.item() * len(indices)
        samples = count * cfg['update_epochs']
        metrics = {name: total / samples for name, total in totals.items()}
        std = self.actor.log_std.detach().exp()
        metrics['noise_std'] = std[:self.actor.noise_dim].mean().item()
        metrics['residual_std'] = (self.actor.residual_scale * std[self.actor.noise_dim:]).mean().item()
        return metrics
