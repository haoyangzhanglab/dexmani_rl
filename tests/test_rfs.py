import copy
import math
import pytest
import torch
from dexmani_rl.rfs import Actor, RFS, gae


STATE_DIM, NOISE_DIM, ACTION_DIM = 3, 6, 2


def config():
    return dict(hidden_dims=[16, 8], residual_scale=.1, actor_lr=.001, value_lr=.001,
                batch_size=4, update_epochs=2, gamma=.99, gae_lambda=.95,
                clip_coef=.2, value_coef=.5, entropy_coef=0., max_grad_norm=1.)


def rollout(model, count=9):
    state = torch.randn(count, STATE_DIM)
    with torch.no_grad():
        joint, log_prob = model.actor.sample(state)
        value = model.value(state)
    return dict(state=state, joint=joint, log_prob=log_prob, value=value,
                reward=torch.linspace(-1, 1, count)[:, None], terminated=torch.zeros(count, 1),
                truncated=torch.zeros(count, 1), next_value=torch.ones(count, 1))


def test_joint_gaussian_and_residual_scaling():
    actor = Actor(3, 6, 2, [16, 8], residual_scale=.1)
    state = torch.randn(5, 3)
    joint, log_prob = actor.sample(state, torch.Generator().manual_seed(12))
    expected = torch.randn(5, 8, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(joint, expected)
    torch.testing.assert_close(log_prob, -.5 * (expected.square() + math.log(2 * math.pi)).sum(-1, keepdim=True))
    noise, residual = actor.split(joint)
    torch.testing.assert_close(noise, expected[:, :6])
    torch.testing.assert_close(residual, .1 * expected[:, 6:])
    # 原始 Gaussian 不做 tanh 或截断，缩放不能进入 PPO log probability。
    large = torch.full((1, 8), 20.)
    noise, residual = actor.split(large)
    torch.testing.assert_close(noise, torch.full((1, 6), 20.))
    torch.testing.assert_close(residual, torch.full((1, 2), 2.))
    torch.testing.assert_close(log_prob, actor.distribution(state).log_prob(joint).sum(-1, keepdim=True))


def test_gae_terminal_timeout_and_rollout_tail():
    reward = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
    value = torch.tensor([[.1], [.2], [.3], [.4], [.5]])
    next_value = torch.tensor([[.2], [100.], [.4], [2.], [3.]])
    terminated = torch.tensor([[False], [True], [False], [False], [False]])
    truncated = torch.tensor([[False], [False], [False], [True], [False]])
    advantage, returns = gae(reward, value, next_value, terminated, truncated, .9, .8)
    # 第 1 步 terminal 忽略 next_value；第 3 步 timeout 保留 bootstrap，隔离下一 episode。
    expected = torch.tensor([[1. + .9 * .2 - .1 + .9 * .8 * (2. - .2)],
                             [2. - .2],
                             [3. + .9 * .4 - .3 + .9 * .8 * (4. + .9 * 2. - .4)],
                             [4. + .9 * 2. - .4],
                             [5. + .9 * 3. - .5]])
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected + value)
    both, _ = gae(reward[:1], value[:1], next_value[:1],
                  torch.ones(1, 1), torch.ones(1, 1), .9, .8)
    torch.testing.assert_close(both, reward[:1] - value[:1])


def test_ppo_updates_both_heads_and_value_without_flow():
    torch.manual_seed(7)
    model = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, config(), 'cpu')
    batch = rollout(model)
    before = copy.deepcopy(model.state_dict())
    metrics = model.update(batch)
    assert all(math.isfinite(metric) for metric in metrics.values())
    for part in (slice(None, NOISE_DIM), slice(NOISE_DIM, None)):
        assert torch.count_nonzero(model.actor.mean[-1].weight[part]).item() > 0
        assert torch.count_nonzero(model.actor.log_std[part]).item() > 0
    assert any(not torch.equal(tensor, before[name])
               for name, tensor in model.state_dict().items() if name.startswith('value.'))
    assert metrics['noise_std'] > 0 and metrics['residual_std'] > 0
    assert len(model.optimizer.param_groups) == 2


def test_ppo_clipping_with_positive_and_negative_advantages():
    cfg = dict(config(), batch_size=4, update_epochs=1, value_coef=0., entropy_coef=0.)
    model = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, cfg, 'cpu')
    batch = rollout(model, count=4)
    batch['value'].zero_()
    batch['terminated'].fill_(1.)
    batch['reward'] = torch.tensor([[-3.], [-1.], [1.], [3.]])
    ratios = torch.tensor([[.5], [1.5], [.5], [1.5]])
    batch['log_prob'] -= ratios.log()
    metrics = model.update(batch)
    # 四项 clipped surrogate 分别为 [-2.4, -1.5, .5, 3.6] / sqrt(5)。
    assert metrics['actor_loss'] == pytest.approx(-.05 / math.sqrt(5), abs=1e-6)
    assert metrics['loss'] == pytest.approx(metrics['actor_loss'])
    assert metrics['clip_fraction'] == 1.


def test_checkpoint_restores_optimizer_and_next_update():
    torch.manual_seed(11)
    model = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, config(), 'cpu')
    model.update(rollout(model))
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(model.optimizer.state_dict())
    clone = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, config(), 'cpu')
    clone.load_state_dict(saved_model)
    clone.optimizer.load_state_dict(saved_optimizer)
    batch = rollout(model)
    rng = torch.get_rng_state()
    expected = model.update(batch)
    torch.set_rng_state(rng)
    actual = clone.update(batch)
    assert actual == pytest.approx(expected)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, clone.state_dict()[name])


def test_single_transition_has_finite_normalized_advantage():
    model = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, config(), 'cpu')
    metrics = model.update(rollout(model, count=1))
    assert all(math.isfinite(metric) for metric in metrics.values())


def test_nonfinite_rollout_stops_before_update():
    model = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, config(), 'cpu')
    before = copy.deepcopy(model.state_dict())
    batch = rollout(model)
    batch['reward'][0] = float('nan')
    with pytest.raises(FloatingPointError, match='Nonfinite RFS rollout'):
        model.update(batch)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def test_nonfinite_gradient_stops_before_optimizer_step():
    model = RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, config(), 'cpu')
    with torch.no_grad():
        model.value[-1].bias.zero_()
    model.value.forward = lambda state: model.value[-1].bias.sqrt().expand(len(state), 1)
    batch = rollout(model, count=4)
    batch['reward'].fill_(1.)
    batch['terminated'].fill_(1.)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match='non-finite'):
        model.update(batch)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


@pytest.mark.parametrize('key,value', [('hidden_dims', []), ('hidden_dims', [0]),
                                      ('actor_lr', float('nan')), ('value_lr', 0),
                                      ('batch_size', 0), ('update_epochs', 1.5),
                                      ('gamma', 0), ('gae_lambda', 1.1), ('clip_coef', 1),
                                      ('value_coef', -1), ('entropy_coef', -1),
                                      ('max_grad_norm', float('inf')), ('residual_scale', -1)])
def test_invalid_algorithm_config(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        RFS(STATE_DIM, NOISE_DIM, ACTION_DIM, cfg, 'cpu')
