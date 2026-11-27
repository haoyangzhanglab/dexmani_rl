from types import SimpleNamespace
import pytest
import torch
from dexmani_rl.base_policy import FrozenPolicy, load_policy, policy_dimensions
from dexmani_rl.policy.flowmatch import FlowMatch, RectifiedFlow
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


def test_residual_is_added_before_physical_clipping():
    normalizer = LinearNormalizer({'action': (19,)})
    normalizer['action'].scale.fill_(2.)
    normalizer['action'].offset.fill_(1.)
    policy = SimpleNamespace(n_obs_steps=2, horizon=3, action_dim=19, n_action_steps=2,
        obs_encoder=SimpleNamespace(out_dim=3), normalizer=normalizer,
        parameters=lambda: iter([torch.zeros(())]),
        decode_condition=lambda context, noise, steps: normalizer['action'].normalize(
            context[:, None].expand(-1, 2, 19)))
    base = FrozenPolicy(policy, {'eval': {'inference_steps': 10}},
                        torch.tensor([[-1., 1.]]).repeat(19, 1), decode_batch=2)
    context = torch.tensor([[2.], [-2.], [0.]])
    noise = torch.zeros(3, 57)
    residual = torch.tensor([[-3.], [3.], [3.]]).expand(-1, 38).requires_grad_()
    actions, fraction = base.actions(context, noise, residual, return_clip_fraction=True)
    expected_physical = torch.tensor([.5, -.5, 1.])[:, None, None].expand(-1, 2, 19)
    torch.testing.assert_close(base.physical(actions), expected_physical)
    torch.testing.assert_close(fraction, torch.tensor(1 / 3))
    assert not actions.requires_grad
    original = base.actions(context, noise)
    torch.testing.assert_close(base.physical(original), context[:, None].expand(-1, 2, 19).clamp(-1, 1))
    torch.testing.assert_close(base.actions(context, noise, torch.zeros_like(residual)), original)


def test_flow_window_and_physical_prediction_with_nonidentity_normalizer():
    class ConstantVelocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def prepare_context_kv(self, context):
            return None

        def forward(self, x, timestep, context, context_kv=None):
            return context[:, :1, None].expand_as(x)

    # Exercise the actual Euler decoder and FlowMatch action API without a vision backbone.
    policy = FlowMatch.__new__(FlowMatch)
    torch.nn.Module.__init__(policy)
    policy.horizon, policy.n_obs_steps, policy.n_action_steps, policy.action_dim = 6, 3, 2, 19
    policy.obs_encoder = SimpleNamespace(out_dim=5)
    policy.action_decoder = RectifiedFlow(ConstantVelocity(), num_inference_steps=3)
    policy.normalizer = LinearNormalizer({'action': (19,)})
    scale = torch.linspace(.5, 2., 19)
    offset = torch.linspace(-1., 1., 19)
    policy.normalizer['action'].scale.copy_(scale)
    policy.normalizer['action'].offset.copy_(offset)
    policy.encode_observations = lambda obs: obs['context']
    context = torch.tensor([[.3], [-.2]])
    noise = torch.arange(2 * 6 * 19, dtype=torch.float32).reshape(2, 6, 19) / 100
    original_noise = noise.clone()
    normalized = (noise + context[:, :1, None])[:, 2:]
    decoded = policy.decode_condition(context, noise)
    torch.testing.assert_close(decoded, normalized)
    predicted = policy.predict_action({'context': context}, initial_noise=noise)['future_actions']
    torch.testing.assert_close(predicted, (normalized - offset) / scale)
    torch.testing.assert_close(noise, original_noise, atol=0, rtol=0)
    assert policy_dimensions(policy) == (3 * (5 + 19), 6 * 19, 2 * 19)
    base = FrozenPolicy(policy, {'eval': {'inference_steps': 3}},
                        torch.tensor([[-20., 20.]]).repeat(19, 1))
    actions = base.actions(context, noise.flatten(1))
    torch.testing.assert_close(actions, normalized[:, :2].flatten(1))
    torch.testing.assert_close(base.physical(actions), predicted[:, :2])
    with pytest.raises(ValueError, match='complete prediction horizon'):
        policy.decode_condition(context, noise[:, :-1])
