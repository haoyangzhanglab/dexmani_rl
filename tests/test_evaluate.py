from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dexmani_rl import evaluate as evaluation
from dexmani_rl.dsrl import Actor
from dexmani_rl.evaluate import compare


@pytest.mark.parametrize('baseline,dsrl,expected', [
    (False, False, [-.9875, .9875]),
    (False, True, [-.975, 1.]),
    (True, False, [-1., .975]),
    (True, True, [-.9875, .9875]),
])
def test_single_pair_retains_uncertainty(baseline, dsrl, expected):
    result = compare([dict(seed=0, success=baseline)], [dict(seed=0, success=dsrl)])
    assert result['episodes'] == 1
    assert result['success_rate_delta'] == int(dsrl) - int(baseline)
    np.testing.assert_allclose(result['paired_conservative_95_ci'], expected)


def test_agreement_is_not_proof_of_equal_success_rates():
    results = [dict(seed=i, success=bool(i % 2)) for i in range(100)]
    comparison = compare(results, results)
    upper = 1 - .0125 ** (1 / 100)
    assert comparison['episodes'] == 100
    assert comparison['success_rate_delta'] == 0
    np.testing.assert_allclose(comparison['paired_conservative_95_ci'], [-upper, upper])


@pytest.mark.parametrize('seeds', [[1], [2, 1], [1, 3]])
def test_comparison_rejects_unpaired_results(seeds):
    baseline = [dict(seed=seed, success=False) for seed in (1, 2)]
    steered = [dict(seed=seed, success=False) for seed in seeds]
    with pytest.raises(ValueError, match='same seeds'):
        compare(baseline, steered)


@pytest.fixture
def simulation(monkeypatch):
    environments, clock = [], [0.]

    class Env:
        def __init__(self):
            self.env = SimpleNamespace(robot=SimpleNamespace(qlimits=[]))
            self.episodes, self.closed = [], False

        def reset(self, seed):
            self.episodes.append({'seed': seed, 'actions': []})
            return torch.zeros(1, 1)

        def step(self, action):
            self.episodes[-1]['actions'].append(action.copy())
            clock[0] += 1
            done = len(self.episodes[-1]['actions']) == 2
            return torch.zeros(1, 1), float(done), done, False, dict(steps=1, success=done)

    @contextmanager
    def make_env(*args):
        env = Env()
        environments.append(env)
        try:
            yield env
        finally:
            env.closed = True

    base = SimpleNamespace(device=torch.device('cpu'), noise_dim=2,
        encode=lambda obs: (None, obs), actions=lambda context, noise: noise,
        physical=lambda actions: actions)
    actor = Actor(1, 2, 4, 1)
    for parameter in actor.parameters():
        torch.nn.init.zeros_(parameter)
    monkeypatch.setattr(evaluation, 'make_env', make_env)
    monkeypatch.setattr(evaluation, 'FrozenPolicy', lambda *args: base)
    monkeypatch.setattr(evaluation, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    return environments, actor


@pytest.mark.parametrize('names,count,resets', [
    (('baseline',), 26, [25, 1]),
    (('dsrl',), 26, [25, 1]),
    (('baseline', 'dsrl'), 13, [24, 2]),
])
def test_evaluation_reuses_environments_and_seeds(simulation, names, count, resets):
    environments, actor = simulation
    policies = {name: None if name == 'baseline' else actor for name in names}
    result = evaluation.evaluate(None, {}, list(range(count)), policies, noise_magnitude=1.5)
    assert [len(env.episodes) for env in environments] == resets
    assert all(env.closed for env in environments)
    for name in names:
        assert result[name] == [dict(seed=i, success=True, steps=2, reward=1.) for i in range(count)]
    episodes = [episode for env in environments for episode in env.episodes]
    for index, episode in enumerate(episodes):
        generator = torch.Generator().manual_seed(episode['seed'])
        expected = torch.stack([torch.randn(1, 2, generator=generator)[0] for _ in range(2)])
        if names[index % len(names)] == 'dsrl':
            expected = expected.tanh() * 1.5
        torch.testing.assert_close(torch.from_numpy(np.stack(episode['actions'])), expected)


@pytest.mark.parametrize('deadline,completed', [(0., 0), (2., 0), (7., 1)])
def test_deadline_keeps_only_complete_pairs(simulation, deadline, completed):
    environments, actor = simulation
    result = evaluation.evaluate(None, {}, [11, 12, 13], {'baseline': None, 'dsrl': actor},
                                 noise_magnitude=1.5, deadline=deadline)
    assert result['baseline'] == result['dsrl']
    assert [episode['seed'] for episode in result['baseline']] == [11][:completed]
    assert all(env.closed for env in environments)
    if deadline == 0:
        assert not environments
