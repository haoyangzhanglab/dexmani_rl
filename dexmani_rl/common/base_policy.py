"""BC checkpoint loading and backwards-compatible FlowMatch RL runtime."""

import torch

from ..envs.action_transform import JointActionTransform, executable_actions, joint_targets
from ..policies.flowmatch.runtime import FlowMatchActionDecoder, FlowMatchObservationEncoder


def load_policy(path, device, weights=None):
    # Keep model-specific imports lazy: importing the common runtime must not
    # eagerly import vision backbones or optional VLA dependencies.
    from ..policy.flowmatch import FlowMatch
    from ..policy.normalizer import normalization_feature_shapes

    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or 'effective_config' not in checkpoint:
        raise ValueError('Expected a BC checkpoint with effective_config')
    cfg = checkpoint['effective_config']
    if cfg['agent']['_target_'].split('.')[-1] != 'FlowMatch' or cfg['action_key'] != 'action':
        raise ValueError('Requires a joint-action FlowMatch checkpoint')
    if any(cfg['dataset'].get(key) is not None for key in
           ('rgb_preprocess_size', 'rgb_random_crop_size', 'rgb_color_aug')):
        raise ValueError('Pinhole RGB must retain its input resolution; configure resize in rgb_backbone_config')
    weights = weights or ('ema' if cfg['eval']['use_ema'] else 'raw')
    if weights not in ('ema', 'raw'):
        raise ValueError('weights must be ema or raw')
    state = checkpoint.get('ema_model_state' if weights == 'ema' else 'model_state')
    if state is None:
        raise ValueError(f'Requested {weights} BC weights are absent')

    # These BC settings control training-time sampling, not frozen inference.
    args = {k: v for k, v in cfg['agent'].items() if k not in ('_target_', 'time_shift_alpha', 'fps_random_config')}
    policy = FlowMatch(**args)
    policy.configure_normalizer(cfg['normalization_dims'], normalization_feature_shapes(
        cfg['normalization_dims'], cfg['normalization'], required_identity_fields=policy.required_identity_fields))
    policy.load_state_dict(state, strict=True)
    policy.to(device=device, dtype=torch.float32)
    policy.normalizer.validate()
    policy.eval().requires_grad_(False)
    return policy, cfg, weights


def policy_dimensions(policy):
    observation = FlowMatchObservationEncoder(policy)
    return (observation.state_dim,
            observation.noise_dim,
            policy.n_action_steps * policy.action_dim)


class FrozenPolicy:
    """Compatibility facade retained for DSRL/RFS and existing training scripts.

    Observation encoding, FlowMatch decoding and physical action transforms are
    delegated to independent components so new policy backends do not extend
    this class or modify existing algorithms.
    """

    def __init__(self, policy, cfg, limits, decode_batch=64):
        self.policy, self.cfg, self.decode_batch = policy, cfg, decode_batch
        self.device = next(policy.parameters()).device
        self.observation_encoder = FlowMatchObservationEncoder(policy)
        self.action_decoder = FlowMatchActionDecoder(
            policy, cfg['eval']['inference_steps'], decode_batch=decode_batch)
        self.action_transform = JointActionTransform(
            policy.normalizer['action'], limits,
            action_dim=policy.action_dim,
            execute_steps=policy.n_action_steps,
        )
        self.state_dim = self.observation_encoder.state_dim
        self.noise_dim = self.observation_encoder.noise_dim
        self.action_dim = policy.n_action_steps * policy.action_dim
        self.limits = self.action_transform.limits.to(self.device)

    @torch.no_grad()
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
    def robofft_actions(self, normalized):
        """Convert RoboFFT's execution window without changing its matching sample."""
        start = self.policy.n_obs_steps - 1
        window = normalized[:, start:start + self.policy.n_action_steps]
        physical, clipped = joint_targets(window, self.policy.normalizer['action'], self.limits)
        return clipped, (physical != clipped).float().mean()

    def bounded_physical(self, window):
        return self.action_transform.bounded_physical(window)

    def physical(self, actions):
        return self.action_transform.physical(actions)
