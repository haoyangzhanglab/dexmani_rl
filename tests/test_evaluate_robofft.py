from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexmani_rl import evaluate as evaluation
from dexmani_rl.base_policy import FrozenPolicy
from dexmani_rl.policy.normalizer import LinearNormalizer
from dexmani_rl.robofft import RoboFFTActor


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.velocity = torch.nn.Parameter(torch.tensor(.4))

    def prepare_context_kv(self, context):
        return None

    def forward(self, x, timestep, context, *, context_kv=None):
        return self.velocity.expand_as(x)


def test_full_horizon_is_not_overwritten_by_execution_clipping():
    normalizer = LinearNormalizer({'action': (2,)})
    normalizer['action'].scale.copy_(torch.tensor([2., .5]))
    normalizer['action'].offset.copy_(torch.tensor([1., -.5]))
    policy = SimpleNamespace(n_obs_steps=2, n_action_steps=2, horizon=4, action_dim=2,
                             obs_encoder=SimpleNamespace(out_dim=3), normalizer=normalizer,
                             parameters=lambda: iter([torch.empty(0)]))
    base = FrozenPolicy(policy, {}, [[-.2, .2], [-.3, .3]])
    full = torch.tensor([[[99., 99.], [.6, -.4], [1., -.5], [-99., -99.]]])
    actual, fraction = base.robofft_actions(full)
    expected = torch.tensor([[[-.2, .2], [0., 0.]]])
    torch.testing.assert_close(actual, expected)
    assert fraction == 0
    full[:, 1] = torch.tensor([3., 1.])
    before = full.clone()
    actual, fraction = base.robofft_actions(full)
    torch.testing.assert_close(actual[:, 0], torch.tensor([[.2, .3]]))
    assert fraction == .5
    torch.testing.assert_close(full, before, rtol=0, atol=0)


@pytest.fixture
def simulation(monkeypatch):
    policy = SimpleNamespace(horizon=4, action_dim=2, n_obs_steps=2, n_action_steps=2,
                             action_decoder=SimpleNamespace(model=ConstantVelocity()),
                             obs_encoder=SimpleNamespace(out_dim=3), parameters=lambda: iter([torch.empty(0)]))
    policy.normalizer = LinearNormalizer({'action': (2,)})
    limits = torch.tensor([[-.8, .8], [-.8, .8]])
    clock, environments = [0.], []

    class Env:
        def __init__(self):
            self.env = SimpleNamespace(robot=SimpleNamespace(qlimits=limits))
            self.episodes = []
            self.closed = False

        def reset(self, seed):
            self.episodes.append(dict(seed=seed, actions=[]))
            return torch.zeros(1, 1)

        def step(self, action):
            self.episodes[-1]['actions'].append(action.copy())
            clock[0] += 1
            # A reused scene changes episode length, as observed in multi_grasp.
            done = len(self.episodes[-1]['actions']) == 1 + len(self.episodes)
            return torch.zeros(1, 1), float(done), done, False, dict(steps=2, success=done)

    @contextmanager
    def make_env(*args):
        env = Env()
        environments.append(env)
        try:
            yield env
        finally:
            env.closed = True

    def actions(context, noise, residual=None):
        # Native BC follows unclipped Euler before its physical execution limit.
        full = noise.reshape(-1, 4, 2) + policy.action_decoder.model.velocity
        return full[:, 1:3].clamp(-.8, .8).flatten(1)

    base = FrozenPolicy(policy, {}, limits)
    base.encode = lambda obs: (obs, obs)
    base.actions = actions
    monkeypatch.setattr(evaluation, 'make_env', make_env)
    monkeypatch.setattr(evaluation, 'FrozenPolicy', lambda *args: base)
    monkeypatch.setattr(evaluation, 'policy_dimensions', lambda policy: (1, 8, 4))
    monkeypatch.setattr(evaluation, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    return policy, environments


@pytest.mark.parametrize('deadline,completed', [(0., 0), (5., 0), (11., 1), (float('inf'), 3)])
def test_robofft_evaluation_requires_complete_triples(simulation, deadline, completed):
    policy, environments = simulation
    baseline = RoboFFTActor(policy.action_decoder.model, 4, 2, 2)
    actor = RoboFFTActor(policy.action_decoder.model, 4, 2, 2)
    results = evaluation.evaluate(policy, {}, [11, 12, 13],
        {'baseline': None, 'sampler_baseline': baseline, 'steered': actor}, deadline=deadline)
    for rows in results.values():
        assert [row['seed'] for row in rows] == [11, 12, 13][:completed]
    assert results['sampler_baseline'] == results['steered']
    assert all(env.closed for env in environments)
    if completed:
        assert 'normalized_clip_fraction' in results['steered'][0]
        assert 'physical_clip_fraction' in results['steered'][0]
        assert 'normalized_clip_fraction' not in results['baseline'][0]
        episodes = [env.episodes[0] for env in environments]
        for sampler, steered in zip(episodes[1::3][:completed], episodes[2::3][:completed]):
            np.testing.assert_array_equal(sampler['actions'], steered['actions'])


def test_robofft_cli_preserves_bc_and_loads_decoder_checkpoint(simulation, monkeypatch, tmp_path):
    policy, environments = simulation
    actor = RoboFFTActor(policy.action_decoder.model, 4, 2, 2)
    with torch.no_grad():
        actor.decoder.velocity.fill_(-.4)
    checkpoint = tmp_path / 'rl.pt'
    output = tmp_path / 'comparison.json'
    torch.save(dict(algorithm='robofft', base_checkpoint='base.pt', weights='raw',
                    actor=actor.state_dict(), config=dict(n_samples=1, noise_std=.25),
                    iteration=3, actor_updates=8, test_seeds=[11, 12, 13]), checkpoint)
    cfg = dict(env_runner=dict(task_name='test'), eval=dict(inference_steps=2))
    monkeypatch.setattr(evaluation, 'load_policy', lambda *args: (policy, cfg, 'raw'))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--rl-checkpoint', str(checkpoint),
                                    '--device', 'cpu', '--episodes', '2', '--output', str(output)])
    evaluation.main()
    result = json.loads(output.read_text())
    assert result['algorithm'] == 'robofft'
    assert result['selected_iteration'] == 3 and result['selected_actor_updates'] == 8
    for name in ('baseline', 'sampler_baseline', 'steered'):
        assert [row['seed'] for row in result[name]] == [11, 12]
    for name in ('comparison', 'comparison_vs_sampler'):
        assert result[name]['episodes'] == 2
        assert result[name]['steered_successes'] == 2
    assert policy.action_decoder.model.velocity.item() == pytest.approx(.4)
    assert len(environments) == 6 and all(env.closed and len(env.episodes) == 1 for env in environments)
    episodes = [env.episodes[0] for env in environments]
    for seed, native, sampler, steered in zip([11, 12], episodes[::3], episodes[1::3], episodes[2::3]):
        generator = torch.Generator().manual_seed(seed)
        for actual in native['actions']:
            expected = (torch.randn(1, 4, 2, generator=generator) + .4)[:, 1:3].clamp(-.8, .8)[0]
            torch.testing.assert_close(torch.from_numpy(actual), expected)
        for episode, velocity in ((sampler, .4), (steered, -.4)):
            generator.manual_seed(seed)
            for actual in episode['actions']:
                expected = torch.randn(1, 4, 2, generator=generator)
                for _ in range(2):
                    expected = (expected + velocity / 2).clamp(-1, 1)
                torch.testing.assert_close(torch.from_numpy(actual), expected[:, 1:3].clamp(-.8, .8)[0])
