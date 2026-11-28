"""Numerical reference copied from ff63d95, independent of the runtime under test."""

import unittest

import torch

from dexmani_rl.common.base_policy import FrozenPolicy, executable_actions as critic_actions
from test_architecture_contracts import FakeFlowMatchPolicy


def policy_dimensions(policy):
    return (policy.n_obs_steps * (policy.obs_encoder.out_dim + 19),
            policy.horizon * policy.action_dim,
            policy.n_action_steps * policy.action_dim)


def joint_targets(window, normalizer, limits):
    """Return requested and executable joint targets in radians."""
    requested = normalizer.unnormalize(window)
    return requested, requested.clamp(limits[:, 0], limits[:, 1])


def executable_actions(window, normalizer, limits):
    """Express the command the robot can execute in the critic's training coordinates."""
    _, applied = joint_targets(window, normalizer, limits)
    return normalizer.normalize(applied)


class OriginalFrozenPolicy:
    def __init__(self, policy, cfg, limits, decode_batch=64):
        self.policy, self.cfg, self.decode_batch = policy, cfg, decode_batch
        self.device = next(policy.parameters()).device
        self.state_dim, self.noise_dim, self.action_dim = policy_dimensions(policy)
        self.limits = torch.as_tensor(limits, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def encode(self, obs):
        context = self.policy.encode_observations(obs)
        frames = context.unflatten(1, (self.policy.n_obs_steps, -1))
        joint = obs['joint_state']
        if 'joint_state' in self.policy.normalizer.fields:
            joint = self.policy.normalizer['joint_state'].normalize(joint)
        state = torch.cat((frames[:, :, :-1].mean(2), joint), -1).flatten(1)
        return context, state

    @torch.no_grad()
    def actions(self, context, noise, residual=None, *, return_clip_fraction=False):
        chunks = []
        for c, w in zip(context.split(self.decode_batch), noise.split(self.decode_batch)):
            action = self.policy.decode_condition(c, w.reshape(-1, self.policy.horizon, self.policy.action_dim),
                                                  self.cfg['eval']['inference_steps'])
            chunks.append(action[:, :self.policy.n_action_steps])
        normalized = torch.cat(chunks)
        if residual is not None:
            normalized = normalized + residual.reshape_as(normalized)
        physical, clipped = joint_targets(normalized, self.policy.normalizer['action'], self.limits)
        actions = self.policy.normalizer['action'].normalize(clipped).flatten(1)
        if return_clip_fraction:
            return actions, (physical != clipped).float().mean()
        return actions

    @torch.no_grad()
    def robofft_actions(self, normalized):
        """Convert the execution window without changing the full matching sample."""
        start = self.policy.n_obs_steps - 1
        window = normalized[:, start:start + self.policy.n_action_steps]
        physical, clipped = joint_targets(window, self.policy.normalizer['action'], self.limits)
        return clipped, (physical != clipped).float().mean()

    def bounded_physical(self, window):
        """Map a normalized execution window to joint targets in radians."""
        return joint_targets(window, self.policy.normalizer['action'], self.limits)[1]

    def physical(self, actions):
        return self.policy.normalizer['action'].unnormalize(actions.reshape(-1, self.policy.n_action_steps, self.policy.action_dim))


class Affine:
    def normalize(self, value):
        return value * 2 + 0.5

    def unnormalize(self, value):
        return (value - 0.5) / 2


class NumericalPolicy(FakeFlowMatchPolicy):
    def encode_observations(self, obs):
        tokens = obs['joint_state'][..., :5]
        return torch.stack((tokens, 2 * tokens, -tokens), dim=2).flatten(1, 2)

    def decode_condition(self, context, noise, inference_steps):
        steps = 4 if inference_steps is None else inference_steps
        return (noise + context.mean((1, 2))[:, None, None] / steps)[:, self.n_obs_steps - 1:]


class LegacyCompatibilityTests(unittest.TestCase):
    def test_frozen_outputs(self):
        policy = NumericalPolicy()
        policy.normalization_dims['joint_state'] = 19
        policy.normalizer.values = {key: Affine() for key in ('joint_state', 'action')}
        rng = torch.Generator().manual_seed(17)
        obs = {'joint_state': torch.randn(3, 2, 19, generator=rng)}
        noise = torch.randn(3, 4, 3, generator=rng)
        residual = torch.randn(3, 6, generator=rng)
        limits = torch.tensor([[-0.5, 1], [-1, 0.75], [0, 0]])
        for steps in (None, 2):
            cfg = {'eval': {'inference_steps': steps}}
            old = OriginalFrozenPolicy(policy, cfg, limits, decode_batch=2)
            new = FrozenPolicy(policy, cfg, limits, decode_batch=2)
            self.assertEqual((new.state_dim, new.noise_dim, new.action_dim),
                             (old.state_dim, old.noise_dim, old.action_dim))
            context, state = old.encode(obs)
            torch.testing.assert_close(new.encode(obs), (context, state))
            for initial in (noise, noise.flatten(1)):
                for correction in (None, residual):
                    args = context, initial, correction
                    torch.testing.assert_close(new.actions(*args, return_clip_fraction=True),
                                               old.actions(*args, return_clip_fraction=True))
            torch.testing.assert_close(new.physical(residual), old.physical(residual))
            torch.testing.assert_close(new.bounded_physical(noise), old.bounded_physical(noise))
            torch.testing.assert_close(new.robofft_actions(noise), old.robofft_actions(noise))

    def test_critic_coordinates_and_boundary_gradient(self):
        field = Affine()
        limits = torch.tensor([[-1., 1.]])
        window = field.normalize(torch.arange(-2., 3.)[:, None]).requires_grad_()
        expected = executable_actions(window, field, limits)
        actual = critic_actions(window, field, limits)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), window)[0],
                                   torch.autograd.grad(expected.sum(), window)[0])
