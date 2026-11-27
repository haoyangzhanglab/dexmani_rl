import copy
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_rl.robofft import RoboFFT, RoboFFTActor, RewardScaler, ppo_loss


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2))
        self.bias = nn.Parameter(torch.tensor(.03))

    def forward(self, x, timestep, context, context_kv=None):
        return self.scale * x + self.bias + .1 * timestep[:, None, None] + context.mean((1, 2))[:, None, None]

    def prepare_context_kv(self, context):
        return None


def actor(n_samples=2):
    return RoboFFTActor(ToyDecoder(), horizon=3, action_dim=2, inference_steps=10, n_samples=n_samples)


def config(**overrides):
    cfg = OmegaConf.to_container(OmegaConf.load('configs/robofft.yaml'))
    cfg.update(value_width=16, batch_size=8, microbatch_size=3, update_epochs=2, n_samples=2)
    cfg.update(overrides)
    return cfg


def rollout(model, count=8):
    generator = torch.Generator().manual_seed(51)
    context = .1 * torch.randn(count, 2, 3, generator=generator)
    state = torch.randn(count, 4, generator=generator)
    action = .2 * torch.randn(count, 3, 2, generator=generator)
    eps, t = model.actor.draw_matching(action, generator)
    # Keep errors below the official log-probability floor to exercise actor gradients.
    eps = .25 * eps
    with torch.no_grad():
        log_prob = model.actor.log_prob(context, action, eps, t)
        value = model.value(state)
    return dict(context=context, state=state, action=action, eps=eps, t=t,
                old_log_prob=log_prob, value=value, next_value=.5 * value,
                reward=torch.linspace(-1, 1, count)[:, None],
                terminated=torch.zeros(count, 1, dtype=torch.bool),
                truncated=torch.zeros(count, 1, dtype=torch.bool),
                first=torch.zeros(count, 1, dtype=torch.bool))


@pytest.mark.parametrize('n_samples', [1, 2])
def test_forward_matching_full_horizon_and_shared_noise(n_samples):
    model = actor(n_samples)
    context = torch.tensor([[[.1]], [[-.2]]])
    action = torch.tensor([.3, -.4])[:, None, None].expand(2, 3, 2)
    eps = torch.tensor([.7, -.1])[:, None, None, None].expand(2, n_samples, 3, 2)
    t = torch.tensor([.2, .8])[:, None].expand(2, n_samples)
    # Analytic scalar velocity error, independent of the tensor reshaping in the implementation.
    predicted = torch.tensor([.274, -.158])
    target = torch.tensor([-.4, -.3])
    expected = -(predicted - target).square()[:, None].expand(2, n_samples)
    actual = model.log_prob(context, action, eps, t)
    torch.testing.assert_close(actual, expected)
    loss, metrics = ppo_loss(actual, actual.detach(), torch.tensor([-1., 1.]))
    torch.testing.assert_close(metrics['ratio'], torch.tensor(1.))
    torch.testing.assert_close(metrics['approx_kl'], torch.tensor(0.))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert not model.decoder.training


def test_matching_noise_grid_and_full_shape():
    model = actor(2)
    actions = torch.zeros(64, 3, 2)
    eps, t = model.draw_matching(actions, torch.Generator().manual_seed(91))
    assert eps.shape == (64, 2, 3, 2) and t.shape == (64, 2)
    torch.testing.assert_close(t * 10, (t * 10).round())
    assert t.min() == 0 and t.max() == pytest.approx(.9)
    assert eps.std().item() == pytest.approx(1., abs=.1)


@pytest.mark.parametrize('n_samples', [1, 2])
def test_per_mc_clipped_surrogate_and_gradient_sign(n_samples):
    ratios = torch.tensor([[.5, 1.5], [1.5, .5], [.5, .9], [1.5, 1.01]])[:, :n_samples]
    advantage = torch.tensor([-3., -1., 1., 3.]) / math.sqrt(20 / 3)
    new = (-.5 + ratios.log() / 8).requires_grad_()
    old = torch.full_like(new, -.5)
    loss, metrics = ppo_loss(new, old, advantage)
    expected = torch.tensor([[2.94, 4.5], [1.5, .98], [-.5, -.9], [-3.06, -3.03]])[:, :n_samples]
    torch.testing.assert_close(loss, expected.mean() / math.sqrt(20 / 3))
    loss.backward()
    gradient = torch.tensor([[0., 4.5], [1.5, 0.], [-.5, -.9], [0., -3.03]])[:, :n_samples]
    torch.testing.assert_close(new.grad, gradient * (2 / n_samples) / math.sqrt(20 / 3))
    # Diagnostic ratio deliberately has no alpha=8 temperature, unlike the actor objective.
    torch.testing.assert_close(metrics['ratio'], ratios.pow(1 / 8).mean())


def test_logprob_floor_saturates_without_silent_fallback():
    new = torch.tensor([[-2.], [-3.]], requires_grad=True)
    loss, metrics = ppo_loss(new, torch.full_like(new, -4), torch.tensor([-1., 1.]))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros_like(new))
    torch.testing.assert_close(metrics['ratio'], torch.tensor(1.))


@pytest.mark.parametrize('eval_mode', [False, True])
def test_sampling_matches_euler_maruyama_with_ordered_clipping(eval_mode):
    model = actor()
    context = torch.full((4, 2, 3), 2.)
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(4, 3, 2, generator=generator)
    clipped = 0
    with torch.no_grad():
        for index in range(10):
            velocity = .2 * x + .03 + .01 * index + 2
            x = x + .1 * velocity
            clipped += int((x.abs() > 1).sum())
            x = x.clamp(-1, 1)
            if not eval_mode:
                x = x + .25 * math.sqrt(.1) * torch.randn(x.shape, generator=generator)
        clipped += int((x.abs() > 1).sum())
        x = x.clamp(-1, 1)
    actual_rng = torch.Generator().manual_seed(13)
    actual, fraction = model.sample(context, actual_rng, eval_mode, return_clip_fraction=True)
    torch.testing.assert_close(actual, x)
    assert fraction.item() == pytest.approx(clipped / (11 * x.numel()))
    assert not actual.requires_grad
    assert torch.equal(actual_rng.get_state(), generator.get_state())
    no_clips = model.sample(context, actual_rng.manual_seed(13), eval_mode)
    torch.testing.assert_close(no_clips, actual, rtol=0, atol=0)
    assert torch.equal(actual_rng.get_state(), generator.get_state())


def test_sampling_reuses_context_projection_only_within_each_call(monkeypatch):
    model = actor()
    context = torch.zeros(2, 2, 3)
    caches, forwards = [], []
    original = model.decoder.forward

    def prepare(context):
        cache = context.mean((1, 2))
        caches.append(cache)
        return cache

    def forward(x, timestep, context, *, context_kv=None):
        forwards.append(context_kv)
        return original(x, timestep, context)

    monkeypatch.setattr(model.decoder, 'prepare_context_kv', prepare)
    monkeypatch.setattr(model.decoder, 'forward', forward)
    first = model.sample(context)
    model.sample(context + 1)
    model.log_prob(context, first, *model.draw_matching(first)).sum().backward()
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert all(value is caches[0] for value in forwards[:10])
    assert all(value is caches[1] for value in forwards[10:20])
    assert forwards[20:] == [None]
    assert model.decoder.scale.grad is not None


def test_reward_scaling_reset_and_cross_rollout_state():
    scaler = RewardScaler(.9)
    rewards = torch.tensor([[1.], [2.], [3.], [4.]])
    first = torch.tensor([[True], [False], [True], [False]])
    # Backward returns are [1, 2.9, 3, 6.7], not the future GAE returns.
    returns = np.array([1., 2.9, 3., 6.7], dtype=np.float32)
    count = 4.0001
    mean = returns.mean() * 4 / count
    variance = (1e-4 + returns.var() * 4 + returns.mean() ** 2 * 1e-4 * 4 / count) / (count - 1)
    actual = scaler(rewards, first)
    torch.testing.assert_close(actual, rewards / math.sqrt(variance + 1e-8))
    saved = copy.deepcopy(scaler.state_dict())
    assert saved['mean'] == pytest.approx(mean)
    assert saved['var'] == pytest.approx(variance)
    assert saved['last_return'] == pytest.approx(6.7)
    clone = RewardScaler(.9)
    clone.load_state_dict(saved)
    next_reward = torch.tensor([[2.], [1000.]])
    next_first = torch.tensor([[False], [True]])
    torch.testing.assert_close(scaler(next_reward, next_first), clone(next_reward, next_first))
    assert scaler.state_dict()['last_return'] == 1000.


def test_microbatch_accumulation_and_critic_only_warmup():
    torch.manual_seed(14)
    full = RoboFFT(4, actor(), config(microbatch_size=8), 'cpu')
    micro = RoboFFT(4, actor(), config(microbatch_size=3), 'cpu')
    micro.load_state_dict(copy.deepcopy(full.state_dict()))
    batch = rollout(full)
    before = copy.deepcopy(full.actor.state_dict())
    rng = torch.get_rng_state()
    full.update(batch, iteration=1)
    torch.set_rng_state(rng)
    micro.update(batch, iteration=1)
    for key, value in full.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for key, value in full.value.state_dict().items():
        torch.testing.assert_close(value, micro.value.state_dict()[key], atol=2e-6, rtol=2e-5)
    rng = torch.get_rng_state()
    first = full.update(batch, iteration=2)
    torch.set_rng_state(rng)
    second = micro.update(batch, iteration=2)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, micro.state_dict()[key], atol=2e-6, rtol=2e-5)
    assert any(not torch.equal(value, before[key]) for key, value in full.actor.state_dict().items())
    assert all(math.isfinite(value) for value in first.values())
    assert all(math.isfinite(value) for value in second.values())


@pytest.mark.parametrize('saturated', [False, True])
def test_initial_matching_guard_checks_raw_logp_before_optimizing(saturated):
    model = RoboFFT(4, actor(), config(repeat_samples=1, update_epochs=1, microbatch_size=1), 'cpu')
    batch = rollout(model, count=2)
    if saturated:
        batch['eps'] += 5.
    with torch.no_grad():
        actual = model.actor.log_prob(batch['context'], batch['action'], batch['eps'], batch['t'])
    batch['old_log_prob'] = actual - (.5 if saturated else torch.tensor([[.8], [1.2]]).log())
    _, diagnostics = ppo_loss(actual, batch['old_log_prob'], torch.ones(2))
    assert diagnostics['ratio'].item() == pytest.approx(1., abs=1e-6)
    with pytest.raises(ValueError, match='Initial RoboFFT matching logp differs'):
        model.update(batch, iteration=2)
    assert all(not optimizer.state for optimizer in model.optimizers.values())


def test_checkpoint_restores_next_update_with_scaler_and_scheduler():
    torch.manual_seed(29)
    model = RoboFFT(4, actor(), config(), 'cpu')
    model.update(rollout(model), iteration=2)
    model.finish_iteration(2)
    clone = RoboFFT(4, actor(), config(), 'cpu')
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    for name in model.optimizers:
        clone.optimizers[name].load_state_dict(copy.deepcopy(model.optimizers[name].state_dict()))
    clone.actor_scheduler.load_state_dict(copy.deepcopy(model.actor_scheduler.state_dict()))
    clone.reward_scaler.load_state_dict(copy.deepcopy(model.reward_scaler.state_dict()))
    batch = rollout(model)
    rng = torch.get_rng_state()
    expected = model.update(batch, iteration=3)
    torch.set_rng_state(rng)
    actual = clone.update(batch, iteration=3)
    assert actual == pytest.approx(expected)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, clone.state_dict()[key], atol=0, rtol=0)
    assert model.reward_scaler.state_dict() == clone.reward_scaler.state_dict()


def test_kl_detecting_minibatch_is_applied_before_early_stop():
    torch.manual_seed(39)
    cfg = config(actor_lr=.01, actor_min_lr=.01, batch_size=4, microbatch_size=2,
                 target_kl=1e-12, update_epochs=3)
    model = RoboFFT(4, actor(), cfg, 'cpu')
    metrics = model.update(rollout(model), iteration=2)
    assert metrics['early_stop']
    assert metrics['minibatches'] == metrics['actor_steps'] == 2
    for optimizer in model.optimizers.values():
        assert all(state['step'] == 2 for state in optimizer.state.values())


def test_release_repeat_exposure_without_new_matching_draws(monkeypatch):
    torch.manual_seed(43)
    model = RoboFFT(4, actor(), config(target_kl=0., batch_size=8, microbatch_size=8), 'cpu')
    batch = rollout(model, count=4)
    calls = []
    original = model.actor.log_prob

    def log_prob(context, action, eps, t):
        calls.extend(eps.detach().clone())
        return original(context, action, eps, t)

    monkeypatch.setattr(model.actor, 'log_prob', log_prob)
    metrics = model.update(batch, iteration=2)
    assert metrics['minibatches'] == metrics['actor_steps'] == 10
    observed = torch.stack(calls)
    for eps in batch['eps']:
        assert (observed == eps).flatten(1).all(1).sum() == 20


def test_scheduler_matches_fixed_upstream_iteration_clock():
    model = RoboFFT(4, actor(), config(), 'cpu')
    # Recorded from upstream CosineAnnealingWarmupRestarts at a9ca8c8.
    expected = {0: 1e-6, 1: 1e-6, 2: 1.1e-6, 10: 1.9e-6, 11: 2e-6,
                12: 1.9998758966336297e-6, 50: 1.8228139257794013e-6,
                100: 1.29968937726506e-6, 150: 1.0004963518588997e-6}
    for iteration in range(151):
        # No gradients are set; mark optimizer calls to avoid PyTorch's generic ordering warning.
        for optimizer in model.optimizers.values():
            optimizer.step()
        model.finish_iteration(iteration)
        if iteration in expected:
            assert model.optimizers['actor'].param_groups[0]['lr'] == pytest.approx(expected[iteration], abs=1e-15)
        assert model.optimizers['critic'].param_groups[0]['lr'] == .001


def test_trainable_decoder_is_independent_of_frozen_bc():
    baseline = ToyDecoder().eval().requires_grad_(False)
    initial = copy.deepcopy(baseline.state_dict())
    model = RoboFFTActor(baseline, 3, 2, 10)
    context = torch.zeros(2, 2, 3)
    action = torch.zeros(2, 3, 2)
    eps, t = model.draw_matching(action, torch.Generator().manual_seed(11))
    model.log_prob(context, action, eps, t).mean().backward()
    torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0).step()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in baseline.parameters())
    assert any(not torch.equal(value, initial[name]) for name, value in model.decoder.state_dict().items())
