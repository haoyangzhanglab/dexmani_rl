"""FlowMatch execution for RL training and evaluation."""

import torch

from ...envs.action_transform import JointActionTransform, joint_targets
from .runtime import FlowMatchActionDecoder, FlowMatchObservationEncoder


def policy_dimensions(policy):
    """Flattened widths of the RL state, full-horizon noise and execution window."""
    joint_dim = policy.normalization_dims.get("joint_state")
    if type(joint_dim) is not int or joint_dim <= 0:
        raise ValueError("FlowMatch policy requires a positive joint_state dimension")
    return (policy.n_obs_steps * (policy.obs_encoder.out_dim + joint_dim),
            policy.horizon * policy.action_dim,
            policy.n_action_steps * policy.action_dim)


class FlowMatchRLRuntime:
    """Encode observations and execute FlowMatch windows within robot joint limits."""

    def __init__(self, policy, cfg, limits, decode_batch=64):
        self.policy, self.cfg = policy, cfg
        self.device = next(policy.parameters()).device
        self.observation_encoder = FlowMatchObservationEncoder(policy)
        self.action_decoder = FlowMatchActionDecoder(
            policy, cfg['eval']['inference_steps'], decode_batch=decode_batch)
        self.action_transform = JointActionTransform(
            policy.normalizer['action'], torch.as_tensor(limits, device=self.device),
            action_dim=policy.action_dim,
            execute_steps=policy.n_action_steps,
        )
        self.state_dim, self.noise_dim, self.action_dim = policy_dimensions(policy)

    def encode(self, obs):
        return self.observation_encoder.encode(obs)

    @torch.no_grad()
    def actions(self, context, noise, residual=None, *, return_clip_fraction=False):
        normalized = self.action_decoder.decode(context, noise)
        if residual is not None:
            normalized = normalized + residual.reshape_as(normalized)
        return self.action_transform.executable_coordinates(
            normalized, return_clip_fraction=return_clip_fraction)

    @torch.no_grad()
    def physical_execution_window(self, normalized):
        """Select future joint targets and measure clipping on the execution window."""
        start = self.policy.n_obs_steps - 1
        window = normalized[:, start:start + self.policy.n_action_steps]
        physical, clipped = joint_targets(window, self.action_transform.normalizer, self.action_transform.limits)
        return clipped, (physical != clipped).float().mean()

    def bounded_physical(self, window):
        return self.action_transform.bounded_physical(window)

    def physical(self, actions):
        return self.action_transform.physical(actions)
