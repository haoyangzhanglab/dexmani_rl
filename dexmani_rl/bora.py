"""BORA v2 on a FlowMatch base. The time axis below is in control steps."""
from copy import deepcopy
import math

import torch
from torch import nn

from .base_policy import executable_actions


def masked_mean(value, mask):
    """Give each sampled window equal weight (paper Eqs. 2–4)."""
    return (value.masked_fill(~mask, 0.).sum(-1) / mask.sum(-1)).mean()


def shifted_target(value, next_first, reward, done, mask, gamma):
    following = torch.cat((value[:, 1:], next_first[:, None]), dim=1)
    # A short window cut by a time limit bootstraps at the recorded final observation.
    last = mask.long().sum(-1, keepdim=True) - 1
    following = following.scatter(1, last, next_first[:, None])
    return reward + gamma * (~done) * following


def expectile_loss(delta, tau):
    return torch.where(delta < 0, 1 - tau, tau) * delta.square()


def chunk_advantage(delta, done, mask, discount):
    advantage = torch.zeros_like(delta)
    tail = torch.zeros_like(delta[:, 0])
    for i in reversed(range(delta.size(1))):
        tail = torch.where(mask[:, i], delta[:, i] + discount * (~done[:, i]) * tail, 0.)
        advantage[:, i] = tail
    return advantage


def gaussian_log_prob(mean, action, std):
    # Conditional on the same initial flow noise in both policies.
    return (-.5 * ((action - mean) / std).square() - math.log(std * math.sqrt(2 * math.pi))).sum(-1)


def clipped_surrogate(log_prob, old_log_prob, advantage, mask, clip):
    log_ratio = (log_prob - old_log_prob).masked_fill(~mask, 0.)
    if not log_ratio.isfinite().all():
        raise FloatingPointError('Non-finite BORA log probability ratio')
    # Choose the one-sided PPO bound before exp. Combining |A| in log space
    # also permits a representable weighted loss when the raw ratio overflows.
    effective = torch.where(advantage >= 0, log_ratio.clamp(max=math.log1p(clip)),
                            log_ratio.clamp(min=math.log1p(-clip)))
    surrogate = advantage.sign() * (effective + advantage.abs().log()).exp()
    return -masked_mean(surrogate, mask), log_ratio.detach()


def improvement_hinge(base_q, final_q, mask, margin):
    return masked_mean((base_q.detach() + margin - final_q).relu(), mask)


class StepValue(nn.Module):
    """Each relative position attends to the current observation's tokens."""
    def __init__(self, token_dim, positions, width, action_dim=0):
        super().__init__()
        self.token = nn.Linear(token_dim, width)
        self.position = nn.Parameter(torch.empty(positions, width))
        nn.init.normal_(self.position, std=.02)
        self.readout = nn.Sequential(nn.Linear(width + action_dim, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, context, action=None):
        tokens = self.token(context)
        weight = (self.position @ tokens.transpose(1, 2)) / math.sqrt(tokens.size(-1))
        state = self.position + weight.softmax(-1) @ tokens
        features = state if action is None else torch.cat((state, action), dim=-1)
        return self.readout(features).squeeze(-1)


class BORAActor(nn.Module):
    def __init__(self, policy, bc_cfg, cfg):
        super().__init__()
        self.policy = policy.eval()
        self.steps = bc_cfg['eval']['inference_steps']
        width = cfg['hidden_dim']
        output_dim = policy.n_action_steps * policy.action_dim
        input_dim = output_dim + policy.obs_encoder.out_dim + policy.action_dim
        self.residuals = nn.ModuleList()
        for _ in range(2):
            residual = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, output_dim))
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
            self.residuals.append(residual)
        self.register_buffer('blends', torch.full((2,), float(cfg['blend_start'])))
        self.register_buffer('round_number', torch.zeros((), dtype=torch.long))
        self.to(next(policy.parameters()).device)

    def flow_chunk(self, context, *, noise=None, generator=None):
        policy = self.policy
        future = policy.decode_condition(context, noise, self.steps, generator)
        return future[:, :policy.n_action_steps]

    def residual_input(self, context, obs):
        joint = obs['joint_state'][:, -1]
        if 'joint_state' in self.policy.normalizer.fields:
            joint = self.policy.normalizer['joint_state'].normalize(joint)
        return torch.cat((context.mean(1), joint), dim=-1)

    def correction(self, index, feature, base):
        return self.residuals[index](torch.cat((feature, base.flatten(1)), dim=-1)).reshape_as(base)

    def compose(self, base, feature, rounds):
        for i in range(rounds):
            base = base + self.blends[i] * self.correction(i, feature, base)
        return base

    @torch.no_grad()
    def normalized_actions(self, obs, *, generator=None, stage=None):
        rounds = int(self.round_number) if stage is None else stage
        if rounds not in (0, 1, 2) or rounds > int(self.round_number):
            raise ValueError('Requested BORA round has not been initialized')
        context = self.policy.encode_observations(obs)
        base = self.flow_chunk(context, generator=generator)
        return self.compose(base, self.residual_input(context, obs), rounds) if rounds else base


def optimize(loss, optimizer):
    if not loss.isfinite():
        raise FloatingPointError('Non-finite BORA loss')
    optimizer.zero_grad(set_to_none=True)
    parameters = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    loss.backward(inputs=parameters)
    nn.utils.clip_grad_norm_(parameters, math.inf, error_if_nonfinite=True)
    optimizer.step()


class BORA(nn.Module):
    def __init__(self, policy, bc_cfg, cfg):
        super().__init__()
        self.cfg = cfg
        if not (.5 < cfg['expectile'] < 1 and cfg['policy_std'] > 0 and 0 < cfg['ppo_clip'] < 1
                and type(cfg['ppo_epochs']) is int and cfg['ppo_epochs'] >= 1
                and cfg['blend_steps'] > 0 and cfg['bc_steps'] > 0):
            raise ValueError('Invalid BORA expectile, Gaussian, PPO or schedule configuration')
        self.actor = BORAActor(policy, bc_cfg, cfg)
        policy.obs_encoder.requires_grad_(True)
        tune_mode = (bc_cfg['agent'].get('rgb_backbone_config') or {}).get('tune_mode', 'freeze')
        for name, param in policy.obs_encoder.rgb_encoder.backbone.named_parameters():
            param.requires_grad_(tune_mode == 'full' or (tune_mode == 'lora' and 'lora_' in name))
        policy.action_decoder.requires_grad_(True)
        self.actor.residuals.requires_grad_(False)
        size = (policy.obs_encoder.out_dim, policy.n_action_steps, cfg['hidden_dim'])
        self.q = StepValue(*size, policy.action_dim)
        self.v = StepValue(*size)
        self.target_v = deepcopy(self.v).requires_grad_(False)
        self.to(next(policy.parameters()).device)
        adam = torch.optim.AdamW
        decay = cfg['weight_decay']
        encoder = [p for p in policy.obs_encoder.parameters() if p.requires_grad]
        self.optimizers = {
            'offline_actor': adam([{'params': encoder, 'lr': cfg['encoder_lr']},
                                   {'params': policy.action_decoder.parameters(), 'lr': cfg['decoder_lr']}],
                                  weight_decay=decay),
            'critic': adam([*self.q.parameters(), *self.v.parameters()], lr=cfg['critic_lr'], weight_decay=decay),
            'residual': adam(self.actor.residuals.parameters(), lr=cfg['residual_lr'], weight_decay=decay),
        }

    def update_critic(self, context, next_context, batch, *, online):
        context = context.detach()
        cfg, mask = self.cfg, batch['mask']
        q = self.q(context, batch['action'])
        value = self.v(context)
        with torch.no_grad():
            evaluator = self.target_v if online else self.v
            current = evaluator(context) if online else value.detach()
            target = shifted_target(current, evaluator(next_context)[:, 0], batch['reward'],
                                    batch['done'], mask, cfg['gamma'])
        delta = q.detach() - value
        q_loss = masked_mean((q - target).square(), mask)
        v_loss = masked_mean(expectile_loss(delta, cfg['expectile']), mask)
        optimize(q_loss + v_loss, self.optimizers['critic'])
        if online:
            with torch.no_grad():
                for target_param, param in zip(self.target_v.parameters(), self.v.parameters()):
                    target_param.lerp_(param, cfg['target_tau'])
        return delta.detach(), dict(q_loss=q_loss.item(), v_loss=v_loss.item())

    def offline_update(self, batch, *, ppo):
        if int(self.actor.round_number):
            raise ValueError('Offline update requested on an online BORA actor')
        cfg, flow, mask = self.cfg, self.actor.policy, batch['mask']
        context = flow.encode_observations(batch['obs'])
        with torch.no_grad():
            next_context = flow.encode_observations(batch['next_obs'])
        delta, metrics = self.update_critic(context, next_context, batch, online=False)
        advantage = chunk_advantage(delta, batch['done'], mask, cfg['gamma'] * cfg['gae_lambda'])
        gate_score = masked_mean(advantage, mask).item()
        metrics.update(advantage=gate_score, actor_updates=0, ppo_updates=0)
        if ppo and cfg['advantage_gate'] is not None and gate_score <= cfg['advantage_gate']:
            return metrics

        noise = torch.randn(len(mask), flow.horizon, flow.action_dim, device=context.device)
        if ppo:
            valid = advantage[mask]
            advantage = ((advantage - valid.mean()) / valid.std(unbiased=False).clamp_min(1e-8)).clamp(-2., 2.)
        for epoch in range(cfg['ppo_epochs'] if ppo else 1):
            if epoch:
                context = flow.encode_observations(batch['obs'])
            mean = self.actor.flow_chunk(context, noise=noise)
            bc = masked_mean((mean - batch['action']).square().mean(-1), mask)
            loss = bc
            if ppo:
                log_prob = gaussian_log_prob(mean, batch['action'], cfg['policy_std'])
                if epoch == 0:
                    old_log_prob = log_prob.detach()
                surrogate, log_ratio = clipped_surrogate(log_prob, old_log_prob, advantage, mask, cfg['ppo_clip'])
                loss = loss + cfg['ppo_weight'] * surrogate
                metrics.update(ppo_loss=surrogate.item(), mean_log_ratio=masked_mean(log_ratio.double(), mask).item(),
                               ppo_updates=epoch + 1)
            optimize(loss, self.optimizers['offline_actor'])
            metrics.update(bc_loss=bc.item(), actor_updates=epoch + 1)
        return metrics

    def begin_round(self, number):
        previous = int(self.actor.round_number)
        if number not in (1, 2) or number not in (previous, previous + 1):
            raise ValueError('BORA online rounds start at 1 and advance in order')
        if previous == 0:
            self.target_v.load_state_dict(self.v.state_dict())
        self.actor.round_number.fill_(number)
        self.actor.policy.eval().requires_grad_(False)
        self.actor.zero_grad(set_to_none=True)
        self.actor.residuals.requires_grad_(False)
        self.actor.residuals[number - 1].requires_grad_(True)

    def online_update(self, batch, *, step, joint_limits):
        cfg, actor, mask = self.cfg, self.actor, batch['mask']
        index = int(actor.round_number) - 1
        if index < 0:
            raise ValueError('Call begin_round before the first online update')
        blend = cfg['blend_start'] + min(step / cfg['blend_steps'], 1.) * (cfg['blend_end'] - cfg['blend_start'])
        bc_weight = cfg['bc_start'] + min(step / cfg['bc_steps'], 1.) * (cfg['bc_end'] - cfg['bc_start'])
        actor.blends[index] = blend
        with torch.no_grad():
            context = actor.policy.encode_observations(batch['obs'])
            next_context = actor.policy.encode_observations(batch['next_obs'])
            feature = actor.residual_input(context, batch['obs'])
            base = actor.compose(actor.flow_chunk(context), feature, index)
        _, metrics = self.update_critic(context, next_context, batch, online=True)
        normalizer = actor.policy.normalizer['action']
        with torch.no_grad():
            base_q = self.q(context, executable_actions(base, normalizer, joint_limits))
        residual = actor.correction(index, feature, base)
        final = base + blend * residual
        final_q = self.q(context, executable_actions(final, normalizer, joint_limits))
        hinge = improvement_hinge(base_q, final_q, mask, cfg['margin'])
        # Supervision on the proposal still points back to the recorded target at saturation.
        bc = masked_mean((final - batch['action']).square().mean(-1), mask)
        magnitude = masked_mean(residual.square().sum(-1), mask)
        loss = cfg['hinge_weight'] * hinge + bc_weight * bc + cfg['residual_l2'] * magnitude
        optimize(loss, self.optimizers['residual'])
        metrics.update(hinge=hinge.item(), bc_loss=bc.item(), residual_l2=magnitude.item(), blend=blend,
                       bc_weight=bc_weight, actor_loss=loss.item())
        return metrics
