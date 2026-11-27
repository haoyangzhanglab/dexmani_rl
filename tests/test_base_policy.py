from types import SimpleNamespace
import pytest
import torch
from dexmani_rl.base_policy import FrozenPolicy, load_policy
from dexmani_rl.policy.normalizer import LinearNormalizer, normalization_feature_shapes


@pytest.mark.parametrize('mode', ['identity', 'limits'])
def test_joint_history_uses_checkpoint_normalization(mode):
    shapes = normalization_feature_shapes({'joint_state': 19}, {'joint_state': mode})
    normalizer = LinearNormalizer(shapes)
    joint = torch.arange(38, dtype=torch.float32).reshape(1, 2, 19)
    expected = joint
    if mode == 'limits':
        normalizer['joint_state'].scale.fill_(2.)
        normalizer['joint_state'].offset.fill_(-1.)
        expected = 2 * joint - 1
    policy = SimpleNamespace(n_obs_steps=2, horizon=2, action_dim=19, n_action_steps=1,
        obs_encoder=SimpleNamespace(out_dim=3),
        normalizer=normalizer, parameters=lambda: iter([torch.zeros(())]),
        encode_observations=lambda obs: torch.zeros(1, 4, 3))
    base = FrozenPolicy(policy, {}, torch.tensor([[-1., 1.]]).repeat(19, 1))
    _, state = base.encode({'joint_state': joint})
    torch.testing.assert_close(state.reshape(1, 2, -1)[..., -19:], expected)


@pytest.mark.parametrize('key,value', [
    ('rgb_preprocess_size', [224, 224]),
    ('rgb_random_crop_size', [192, 192]),
    ('rgb_color_aug', {}),
])
def test_checkpoint_rejects_uncalibrated_dataset_rgb_processing(tmp_path, key, value):
    checkpoint = tmp_path / 'invalid.pt'
    cfg = dict(agent={'_target_': 'dexmani_policy.agents.policies.flowmatch.FlowMatch'},
               action_key='action', dataset={key: value})
    torch.save({'effective_config': cfg}, checkpoint)
    with pytest.raises(ValueError, match='Pinhole RGB'):
        load_policy(checkpoint, 'cpu')
