from copy import deepcopy

import pytest
import torch
from torch import nn

from dexmani_rl.bora import (BORA, BORAActor, StepValue, chunk_advantage, clipped_surrogate,
                             expectile_loss, gaussian_log_prob, improvement_hinge, masked_mean, optimize,
                             shifted_target)
from dexmani_rl.policy.flowmatch import FlowMatch, RectifiedFlow
from dexmani_rl.policy.normalizer import LinearNormalizer


def test_bellman_targets_and_validity_masks():
    value = torch.tensor([[1., 2., 3.], [2., 4., 8.], [2., 4., 8.]])
    first = torch.tensor([5., 10., 30.])
    reward = torch.tensor([[0., 0., 1.], [0., 1., 0.], [0., 0., 0.]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    done = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.bool)
    actual = shifted_target(value, first, reward, done, mask, gamma=.5)
    torch.testing.assert_close(actual[mask], torch.tensor([1., 1.5, 3.5, 2., 1., 2., 15.]))
    torch.testing.assert_close(masked_mean(actual, mask), torch.tensor((2. + 1.5 + 8.5) / 3))

    delta = torch.tensor([[1., 2., 4.], [1., -2., 30.], [1., 2., 30.]])
    torch.testing.assert_close(chunk_advantage(delta, done, mask, .5),
                               torch.tensor([[3., 4., 4.], [0., -2., 0.], [2., 2., 0.]]))
    torch.testing.assert_close(expectile_loss(torch.tensor([-2., 0., 2.]), .75), torch.tensor([1., 0., 3.]))


def test_step_value_and_gaussian_ratio_are_local_to_each_position():
    torch.manual_seed(5)
    context, action = torch.randn(2, 4, 5), torch.randn(2, 3, 2)
    critic = StepValue(5, 3, 8, 2)
    q = critic(context, action)
    changed = action.clone()
    changed[:, 1] += 2.
    new = critic(context, changed)
    torch.testing.assert_close(q[:, [0, 2]], new[:, [0, 2]])
    assert not torch.equal(q[:, 1], new[:, 1])

    mean = torch.zeros_like(action)
    reference = torch.distributions.Independent(torch.distributions.Normal(mean, .6), 1)
    torch.testing.assert_close(gaussian_log_prob(mean, action, .6), reference.log_prob(action))
    log_ratio = torch.tensor([[1.4, .6]]).log()
    advantage = torch.tensor([[2., -1.]])
    mask = torch.ones_like(advantage, dtype=torch.bool)
    loss, _ = clipped_surrogate(log_ratio, 0., advantage, mask, .2)
    torch.testing.assert_close(loss, torch.tensor(-(2 * 1.2 - .8) / 2))
    final = torch.tensor([[.5, 1.5]], requires_grad=True)
    base = torch.ones_like(final, requires_grad=True)
    hinge = improvement_hinge(base, final, mask, 0.)
    hinge.backward()
    torch.testing.assert_close(hinge, torch.tensor(.25))
    torch.testing.assert_close(final.grad, torch.tensor([[-.5, 0.]]))
    assert base.grad is None


def test_log_space_ppo_keeps_only_the_active_gradient():
    log_prob = torch.tensor([[90., -90., 0., 0., 90., float('nan')]], requires_grad=True)
    advantage = torch.tensor([[1., -1., 2., -.5, 0., -1.]])
    mask = torch.tensor([[True, True, True, True, True, False]])
    loss, log_ratio = clipped_surrogate(log_prob, 0., advantage, mask, .2)
    torch.testing.assert_close(loss, torch.tensor(-(1.2 - .8 + 2. - .5) / 5))
    torch.testing.assert_close(log_ratio, torch.tensor([[90., -90., 0., 0., 90., 0.]]))
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.tensor([[0., 0., -.4, .1, 0., 0.]]))

    # exp(90) overflows FP32, but the negative-advantage objective and its gradient fit.
    log_prob = torch.tensor([[90.]], requires_grad=True)
    loss, _ = clipped_surrogate(log_prob, 0., torch.tensor([[-1e-30]]), mask[:, :1], .2)
    expected = (1e-30 * log_prob.detach().double().exp()).float()
    torch.testing.assert_close(loss, expected.squeeze())
    loss.backward()
    torch.testing.assert_close(log_prob.grad, expected)


def test_optimizer_rejects_unrepresentable_loss_and_nonfinite_gradient_before_step():
    p = nn.Parameter(torch.zeros(1, 1))
    optimizer = torch.optim.AdamW([p], lr=.01)
    loss, _ = clipped_surrogate(p + 90., 0., -torch.ones_like(p), torch.ones_like(p, dtype=torch.bool), .2)
    with pytest.raises(FloatingPointError, match='loss'):
        optimize(loss, optimizer)
    with pytest.raises(RuntimeError, match='non-finite'):
        optimize(p.sqrt().sum(), optimizer)
    torch.testing.assert_close(p, torch.zeros_like(p))
    assert not optimizer.state


class TinyEncoder(nn.Module):
    out_dim = 4

    def __init__(self):
        super().__init__()
        self.rgb_encoder = nn.Module()
        self.rgb_encoder.backbone = nn.Linear(19, 4)
        self.proj = nn.Linear(19, 4)

    def forward(self, obs):
        joint = obs['joint_state']
        return (self.proj(joint) + self.rgb_encoder.backbone(joint)).reshape(-1, 2, 4)


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond = nn.Linear(4, 19)
        self.scale = nn.Parameter(torch.tensor(.1))

    def prepare_context_kv(self, context):
        return self.cond(context.mean(1))[:, None]

    def forward(self, x, timestep, context, context_kv=None):
        return x * self.scale + context_kv


class TinyFlow(nn.Module):
    encode_observations = FlowMatch.encode_observations
    decode_condition = FlowMatch.decode_condition
    horizon, n_obs_steps, n_action_steps, action_dim = 4, 2, 2, 19

    def __init__(self):
        super().__init__()
        self.obs_encoder = TinyEncoder()
        self.action_decoder = RectifiedFlow(TinyVelocity(), num_inference_steps=2)
        self.normalizer = LinearNormalizer({'joint_state': (19,), 'action': (19,)})


def small_config():
    return dict(hidden_dim=8, encoder_lr=.01, decoder_lr=.01, critic_lr=.01, residual_lr=.01,
               weight_decay=0., gamma=.99, expectile=.7, gae_lambda=.95, policy_std=1.,
               ppo_clip=.2, ppo_weight=.01, ppo_epochs=2, advantage_gate=None, target_tau=.005,
               blend_start=.2, blend_end=.75, blend_steps=4, bc_start=2., bc_end=1., bc_steps=10,
               hinge_weight=.35, margin=0., residual_l2=.0002)


def test_online_q_scores_the_executed_command_and_bc_can_correct_saturation(monkeypatch):
    flow = TinyFlow().eval().requires_grad_(False)
    normalizer = flow.normalizer['action']
    normalizer.scale.copy_(torch.linspace(.75, 2., 19))
    normalizer.offset.fill_(.5)
    limits = torch.stack((torch.linspace(-.7, -.1, 19), torch.linspace(.2, .9, 19)), dim=-1)
    upper = normalizer.normalize(limits[:, 1]).expand(1, 2, -1)
    obs = {'joint_state': torch.zeros(1, 2, 19)}
    batch = dict(obs=obs, next_obs=obs, action=upper, mask=torch.ones(1, 2, dtype=torch.bool),
                 done=torch.zeros(1, 2, dtype=torch.bool), reward=torch.zeros(1, 2))
    cfg = small_config()
    cfg['residual_l2'] = 0.
    agent = BORA(flow, {'agent': {}, 'eval': {'inference_steps': 2}}, cfg)
    agent.begin_round(1)
    monkeypatch.setattr(agent.actor, 'flow_chunk', lambda context: upper + 1.)
    bias = agent.actor.residuals[0][-1].bias
    with torch.no_grad():
        bias.fill_(1.)
    # Both predictions overshoot the same limits. All Q queries must agree with replay.
    q_actions = []
    with agent.q.register_forward_pre_hook(lambda _, args: q_actions.append(args[1].detach().clone())):
        metrics = agent.online_update(batch, step=0, joint_limits=limits)
    assert len(q_actions) == 3
    for action in q_actions:
        torch.testing.assert_close(action, upper, rtol=0, atol=0)
    assert metrics['hinge'] == 0.
    assert metrics['bc_loss'] == pytest.approx((1 + cfg['blend_start']) ** 2)
    assert (bias.grad > 0).all() and (bias < 1.).all()


def test_offline_to_online_gradients_composition_and_restore(monkeypatch):
    torch.manual_seed(33)
    cfg = small_config()
    bc_cfg = {'agent': {}, 'eval': {'inference_steps': 2}}
    batch = dict(obs={'joint_state': torch.randn(3, 2, 19)},
                 next_obs={'joint_state': torch.randn(3, 2, 19)}, action=torch.randn(3, 2, 19),
                 reward=torch.randn(3, 2), done=torch.tensor([[False, False], [False, True], [True, False]]),
                 mask=torch.tensor([[True, True], [True, True], [True, False]]))
    flow = TinyFlow().eval().requires_grad_(False)
    agent = BORA(flow, bc_cfg, cfg)
    initial = deepcopy(flow.state_dict())
    agent.offline_update(batch, ppo=False)
    noises = []
    decode = flow.decode_condition

    def record_decode(context, noise, steps, generator):
        noises.append(noise.clone())
        return decode(context, noise, steps, generator)

    with monkeypatch.context() as patch:
        patch.setattr(flow, 'decode_condition', record_decode)
        metrics = agent.offline_update(batch, ppo=True)
    assert len(noises) == 2
    torch.testing.assert_close(noises[0], noises[1], rtol=0, atol=0)
    assert metrics['ppo_updates'] == 2
    assert metrics['mean_log_ratio'] != 0
    assert not torch.equal(flow.obs_encoder.proj.weight, initial['obs_encoder.proj.weight'])
    assert not torch.equal(flow.action_decoder.model.cond.weight, initial['action_decoder.model.cond.weight'])
    for key in ('normalizer.fields.action.scale', 'obs_encoder.rgb_encoder.backbone.weight'):
        assert torch.equal(initial[key], flow.state_dict()[key])

    cfg['advantage_gate'] = 1e6
    offline = deepcopy(flow.state_dict())
    prior_q = deepcopy(agent.q.state_dict())
    gated = agent.offline_update(batch, ppo=True)
    assert gated['actor_updates'] == gated['ppo_updates'] == 0
    assert any(not torch.equal(v, prior_q[k]) for k, v in agent.q.state_dict().items())
    agent.begin_round(1)
    assert all(torch.equal(v, agent.target_v.state_dict()[k]) for k, v in agent.v.state_dict().items())
    limits = torch.tensor([[-1., 1.]] * flow.action_dim)
    agent.online_update(batch, step=4, joint_limits=limits)
    stage_one = deepcopy(agent.actor.residuals[0].state_dict())
    agent.begin_round(2)
    agent.online_update(batch, step=2, joint_limits=limits)
    assert torch.equal(agent.actor.blends, torch.tensor([.75, .475]))
    assert all(torch.equal(v, stage_one[k]) for k, v in agent.actor.residuals[0].state_dict().items())
    assert all(torch.equal(v, offline[k]) for k, v in flow.state_dict().items())
    assert not any(p.requires_grad for p in flow.parameters())

    actor = BORAActor(TinyFlow(), bc_cfg, cfg)
    actor.load_state_dict(agent.actor.state_dict())
    obs = batch['obs']
    context = actor.policy.encode_observations(obs)
    feature = actor.residual_input(context, obs)
    base = actor.flow_chunk(context, generator=torch.Generator().manual_seed(4))
    round_one = base + actor.blends[0] * actor.correction(0, feature, base)
    full = round_one + actor.blends[1] * actor.correction(1, feature, round_one)
    for stage, expected in enumerate((base, round_one, full)):
        actual = actor.normalized_actions(obs, stage=stage, generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(actual, expected)
