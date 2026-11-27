import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, output_dim, width, depth):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(input_dim, width), nn.LayerNorm(width), nn.Tanh()]
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, state_dim, noise_dim, width, depth):
        super().__init__()
        self.net = mlp(state_dim, 2 * noise_dim, width, depth)

    def sample(self, state, generator=None):
        mean, log_std = self.net(state).chunk(2, -1)
        log_std = log_std.clamp(-20, 2)
        epsilon = torch.randn(mean.shape, device=mean.device, generator=generator)
        z = mean + log_std.exp() * epsilon
        log_prob = (-0.5 * (epsilon.square() + math.log(2 * math.pi)) - log_std
                    - 2 * (math.log(2) - z - F.softplus(-2 * z))).sum(-1, keepdim=True)
        return z.tanh(), log_prob


class Critics(nn.Module):
    def __init__(self, state_dim, action_dim, width, depth):
        super().__init__()
        self.q = nn.ModuleList([mlp(state_dim + action_dim, 1, width, depth) for _ in range(2)])

    def forward(self, state, action):
        features = torch.cat((state, action), -1)
        return torch.cat([q(features) for q in self.q], -1)


def td_target(reward, terminated, next_q, log_prob, alpha, gamma):
    return reward + gamma * (1 - terminated) * (next_q.min(-1, keepdim=True).values - alpha * log_prob)


class DSRL(nn.Module):
    def __init__(self, base, cfg):
        super().__init__()
        if (any(type(cfg[key]) is not int or cfg[key] <= 0 for key in ('width', 'batch_size', 'utd')) or
                any(type(cfg[key]) is not int or cfg[key] < 0 for key in ('depth', 'noise_updates'))):
            raise ValueError('Invalid DSRL network size or update counts')
        numeric = ('lr', 'noise_magnitude', 'gamma', 'tau')
        if (any(type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) for key in numeric) or
                cfg['lr'] <= 0 or cfg['noise_magnitude'] <= 0 or
                not 0 < cfg['gamma'] <= 1 or not 0 < cfg['tau'] <= 1):
            raise ValueError('Invalid DSRL hyperparameters')
        self.base, self.cfg = base, cfg
        args = (cfg['width'], cfg['depth'])
        self.actor = Actor(base.state_dim, base.noise_dim, *args)
        self.critic = Critics(base.state_dim, base.action_dim, *args)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.noise_critic = Critics(base.state_dim, base.noise_dim, *args)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.to(base.device)
        self.optimizers = {
            'actor': torch.optim.Adam(self.actor.parameters(), lr=cfg['lr']),
            'critic': torch.optim.Adam(self.critic.parameters(), lr=cfg['lr']),
            'noise_critic': torch.optim.Adam(self.noise_critic.parameters(), lr=cfg['lr']),
            'alpha': torch.optim.Adam([self.log_alpha], lr=cfg['lr']),
        }

    @staticmethod
    def optimize(optimizer, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite DSRL loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],
                                 float('inf'), error_if_nonfinite=True)
        optimizer.step()

    def update(self, replay):
        cfg = self.cfg
        for _ in range(cfg['utd']):
            b = replay.sample(cfg['batch_size'], self.base.device, fields=(
                'state', 'action', 'next_state', 'next_context', 'reward', 'terminated'))
            u, logp = self.actor.sample(b['state'])
            alpha = self.log_alpha.detach().exp()
            self.optimize(self.optimizers['alpha'], -(self.log_alpha * logp.detach()).mean())
            with torch.no_grad():
                next_u, next_logp = self.actor.sample(b['next_state'])
                next_a = self.base.actions(b['next_context'], cfg['noise_magnitude'] * next_u)
                y = td_target(b['reward'], b['terminated'], self.target(b['next_state'], next_a),
                              next_logp, alpha, cfg['gamma'])
            q_loss = 0.5 * (self.critic(b['state'], b['action']) - y).square().mean(0).sum()
            self.optimize(self.optimizers['critic'], q_loss)
            self.noise_critic.requires_grad_(False)
            actor_loss = (alpha * logp - self.noise_critic(b['state'], u).min(-1, keepdim=True).values).mean()
            self.optimize(self.optimizers['actor'], actor_loss)
            self.noise_critic.requires_grad_(True)
            with torch.no_grad():
                for target, current in zip(self.target.parameters(), self.critic.parameters()):
                    target.lerp_(current, cfg['tau'])
        metrics = dict(critic_loss=q_loss.item(), actor_loss=actor_loss.item(), alpha=alpha.item())
        for _ in range(cfg['noise_updates']):
            b = replay.sample(cfg['batch_size'], self.base.device, fields=('state', 'context'))
            with torch.no_grad():
                w = torch.randn(len(b['state']), self.base.noise_dim, device=self.base.device)
                a = self.base.actions(b['context'], w)
                teacher = self.critic(b['state'], a)
            prediction = self.noise_critic(b['state'], w / cfg['noise_magnitude'])
            loss = 0.5 * (prediction - teacher).square().mean(0).sum()
            self.optimize(self.optimizers['noise_critic'], loss)
        if cfg['noise_updates']:
            metrics['noise_loss'] = loss.item()
        return metrics
