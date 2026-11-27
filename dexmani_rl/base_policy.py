import torch
from .policy.flowmatch import FlowMatch
from .policy.normalizer import normalization_feature_shapes


def load_policy(path, device, weights=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    cfg = checkpoint['effective_config']
    if cfg['agent']['_target_'].split('.')[-1] != 'FlowMatch' or cfg['action_key'] != 'action':
        raise ValueError('Requires a joint-action FlowMatch checkpoint')
    if any(cfg['dataset'].get(key) is not None for key in
           ('rgb_preprocess_size', 'rgb_random_crop_size', 'rgb_color_aug')):
        raise ValueError('Pinhole RGB must retain its input resolution; configure resize in rgb_backbone_config')
    # These BC settings control training-time sampling, not frozen inference.
    args = {k: v for k, v in cfg['agent'].items() if k not in ('_target_', 'time_shift_alpha', 'fps_random_config')}
    policy = FlowMatch(**args)
    policy.configure_normalizer(cfg['normalization_dims'], normalization_feature_shapes(
        cfg['normalization_dims'], cfg['normalization'], required_identity_fields=policy.required_identity_fields))
    weights = weights or ('ema' if cfg['eval']['use_ema'] else 'raw')
    if weights not in ('ema', 'raw'):
        raise ValueError('weights must be ema or raw')
    state = checkpoint['ema_model_state'] if weights == 'ema' else checkpoint['model_state']
    if state is None:
        raise ValueError('Requested weights are absent')
    policy.load_state_dict(state, strict=True)
    policy.to(device=device, dtype=torch.float32)
    policy.normalizer.validate()
    policy.eval().requires_grad_(False)
    return policy, cfg, weights


class FrozenPolicy:
    def __init__(self, policy, cfg, limits, decode_batch=64):
        self.policy, self.cfg, self.decode_batch = policy, cfg, decode_batch
        self.device = next(policy.parameters()).device
        self.state_dim = policy.n_obs_steps * (policy.obs_encoder.out_dim + 19)
        self.limits = torch.as_tensor(limits, device=self.device, dtype=torch.float32)
        self.noise_dim = policy.horizon * policy.action_dim
        self.action_dim = policy.n_action_steps * policy.action_dim

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
    def actions(self, context, noise):
        chunks = []
        for c, w in zip(context.split(self.decode_batch), noise.split(self.decode_batch)):
            action = self.policy.decode_condition(c, w.reshape(-1, self.policy.horizon, self.policy.action_dim),
                                                  self.cfg['eval']['inference_steps'])
            chunks.append(action[:, :self.policy.n_action_steps].clamp(self.limits[:, 0], self.limits[:, 1]))
        physical = torch.cat(chunks)
        return self.policy.normalizer['action'].normalize(physical).flatten(1)

    def physical(self, actions):
        return self.policy.normalizer['action'].unnormalize(actions.reshape(-1, self.policy.n_action_steps, self.policy.action_dim))
