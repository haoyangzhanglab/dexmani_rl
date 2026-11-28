"""Load a pretrained joint-action FlowMatch checkpoint."""

import torch


def load_flowmatch_policy(path, device, weights=None):
    from .model.flowmatch import FlowMatch
    from .model.normalizer import normalization_feature_shapes

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
