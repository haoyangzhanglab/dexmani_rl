from contextlib import contextmanager
from types import SimpleNamespace
import json
import sys
import numpy as np
import pytest
import torch
from dexmani_rl import evaluate as evaluation
from dexmani_rl.dsrl import Actor
from dexmani_rl.rfs import Actor as RFSActor
from dexmani_rl.evaluate import compare


@pytest.mark.parametrize('baseline,steered,expected', [
    (False, False, [-.9875, .9875]),
    (False, True, [-.975, 1.]),
    (True, False, [-1., .975]),
    (True, True, [-.9875, .9875]),
])
def test_single_pair_retains_uncertainty(baseline, steered, expected):
    result = compare([dict(seed=0, success=baseline)], [dict(seed=0, success=steered)])
    assert result['episodes'] == 1
    assert result['success_rate_delta'] == int(steered) - int(baseline)
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

    base = SimpleNamespace(device=torch.device('cpu'), state_dim=1, noise_dim=2, action_dim=2,
        encode=lambda obs: (None, obs), actions=lambda context, noise, residual=None: noise if residual is None else noise + residual,
        physical=lambda actions: actions)
    actor = Actor(1, 2, 4, 1)
    for parameter in actor.parameters():
        torch.nn.init.zeros_(parameter)
    monkeypatch.setattr(evaluation, 'make_env', make_env)
    monkeypatch.setattr(evaluation, 'FrozenPolicy', lambda *args: base)
    monkeypatch.setattr(evaluation, 'policy_dimensions', lambda policy: (1, 2, 2))
    monkeypatch.setattr(evaluation, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    return environments, actor


@pytest.mark.parametrize('names', [('baseline',), ('steered',), ('baseline', 'steered')])
def test_each_evaluation_starts_from_a_fresh_environment(simulation, names):
    environments, actor = simulation
    policies = {name: None if name == 'baseline' else actor for name in names}
    count = 3
    result = evaluation.evaluate(None, {}, list(range(count)), policies, noise_magnitude=1.5)
    assert len(environments) == count * len(names)
    assert all(env.closed and len(env.episodes) == 1 for env in environments)
    for name in names:
        assert result[name] == [dict(seed=i, success=True, steps=2, reward=1.) for i in range(count)]
    episodes = [episode for env in environments for episode in env.episodes]
    for index, episode in enumerate(episodes):
        generator = torch.Generator().manual_seed(episode['seed'])
        expected = torch.stack([torch.randn(1, 2, generator=generator)[0] for _ in range(2)])
        if names[index % len(names)] == 'steered':
            expected = expected.tanh() * 1.5
        torch.testing.assert_close(torch.from_numpy(np.stack(episode['actions'])), expected)


@pytest.mark.parametrize('deadline,completed', [(0., 0), (2., 0), (7., 1)])
def test_deadline_keeps_only_complete_pairs(simulation, deadline, completed):
    environments, actor = simulation
    result = evaluation.evaluate(None, {}, [11, 12, 13], {'baseline': None, 'steered': actor},
                                 noise_magnitude=1.5, deadline=deadline)
    assert result['baseline'] == result['steered']
    assert [episode['seed'] for episode in result['baseline']] == [11][:completed]
    assert all(env.closed for env in environments)
    if deadline == 0:
        assert not environments


def test_rfs_pairs_use_independent_seeded_joint_samples(simulation):
    environments, _ = simulation
    actor = RFSActor(1, 2, 2, [4], .1)
    result = evaluation.evaluate(None, {}, [11, 12], {'baseline': None, 'steered': actor})
    assert result['baseline'] == result['steered']
    episodes = [env.episodes[0] for env in environments]
    assert len(episodes) == 4
    for seed, baseline, rfs in zip([11, 12], episodes[::2], episodes[1::2]):
        generator = torch.Generator().manual_seed(seed)
        expected = torch.stack([torch.randn(1, 2, generator=generator)[0] for _ in range(2)])
        torch.testing.assert_close(torch.from_numpy(np.stack(baseline['actions'])), expected)
        generator.manual_seed(seed)
        expected = []
        for _ in range(2):
            noise, residual = actor.split(actor.sample(torch.zeros(1, 1), generator)[0])
            expected.append((noise + residual)[0].detach())
        torch.testing.assert_close(torch.from_numpy(np.stack(rfs['actions'])), torch.stack(expected))
    comparison = compare(result['baseline'], result['steered'])
    assert comparison['steered_successes'] == 2


@pytest.mark.parametrize('algorithm', ['dsrl', 'rfs'])
@pytest.mark.parametrize('episodes', [None, 2])
def test_rl_cli_loads_actor_without_extra_environment(simulation, monkeypatch, tmp_path, algorithm, episodes):
    environments, actor = simulation
    checkpoint = tmp_path / 'rl.pt'
    output = tmp_path / 'evaluation.json'
    config = dict(width=4, depth=1, noise_magnitude=1.5)
    if algorithm == 'rfs':
        actor = RFSActor(1, 2, 2, [4], .1)
        config = dict(hidden_dims=[4], residual_scale=.1)
    torch.save(dict(algorithm=algorithm, base_checkpoint='base.pt', weights='ema', actor=actor.state_dict(),
                    config=config, test_seeds=[11, 12, 13]), checkpoint)
    monkeypatch.setattr(evaluation, 'load_policy', lambda *args: (None, {'env_runner': {'task_name': 'task'}}, 'ema'))
    monkeypatch.setattr(evaluation, 'seed_split', lambda task: ([91, 92, 93], [], set()))
    argv = ['evaluate', '--rl-checkpoint', str(checkpoint), '--device', 'cpu', '--output', str(output)]
    if episodes is not None:
        argv += ['--episodes', str(episodes)]
    monkeypatch.setattr(sys, 'argv', argv)
    evaluation.main()
    result = json.loads(output.read_text())
    expected_seeds = [11, 12, 13][:episodes]
    assert [episode['seed'] for episode in result['steered']] == expected_seeds
    assert result['baseline'] == result['steered']
    assert result['comparison']['episodes'] == len(expected_seeds)
    assert result['comparison']['steered_successes'] == len(expected_seeds)
    assert result['algorithm'] == algorithm
    assert result['rl_checkpoint'] == str(checkpoint)
    assert result['checkpoint'] == 'base.pt'
    assert result['weights'] == 'ema'
    assert len(environments) == 2 * len(expected_seeds)
    assert all(env.closed and len(env.episodes) == 1 for env in environments)


def test_bc_cli_uses_shared_result_shape(simulation, monkeypatch, tmp_path):
    environments, _ = simulation
    output = tmp_path / 'evaluation.json'
    monkeypatch.setattr(evaluation, 'load_policy', lambda *args: (None, {'env_runner': {'task_name': 'task'}}, 'raw'))
    monkeypatch.setattr(evaluation, 'seed_split', lambda task: ([11, 12, 13], [], set()))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', 'base.pt', '--device', 'cpu',
                                    '--output', str(output)])
    evaluation.main()
    result = json.loads(output.read_text())
    assert result['algorithm'] == 'bc' and result['rl_checkpoint'] is None
    assert [episode['seed'] for episode in result['baseline']] == [11, 12, 13]
    assert result['steered'] == [] and result['comparison'] == {'episodes': 0}
    assert len(environments) == 3 and all(env.closed for env in environments)


def test_rl_cli_rejects_unknown_algorithm(simulation, monkeypatch, tmp_path):
    environments, _ = simulation
    checkpoint = tmp_path / 'unknown.pt'
    output = tmp_path / 'evaluation.json'
    torch.save({'algorithm': 'unknown', 'actor': {}}, checkpoint)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--rl-checkpoint', str(checkpoint),
                                    '--device', 'cpu', '--output', str(output)])
    with pytest.raises(ValueError, match='Unsupported RL algorithm'):
        evaluation.main()
    assert not environments and not output.exists()


@pytest.mark.parametrize('saved', [
    {'algorithm': 'rfs', 'model': {}},
    {'algorithm': 'robofft', 'model': {}, 'reward_scaler': {'var': np.ones(())}},
    {'effective_config': {}},
])
def test_rl_cli_rejects_resume_and_bc_checkpoints(monkeypatch, tmp_path, saved):
    checkpoint = tmp_path / 'wrong_checkpoint.pt'
    output = tmp_path / 'evaluation.json'
    torch.save(saved, checkpoint)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--rl-checkpoint', str(checkpoint), '--device', 'cpu',
                                    '--output', str(output)])
    with pytest.raises(ValueError, match='saved RL actor'):
        evaluation.main()
    assert not output.exists()


def test_cli_does_not_replace_output_created_during_evaluation(monkeypatch, tmp_path):
    output = tmp_path / 'evaluation.json'

    def evaluate(*args, **kwargs):
        output.write_text('first writer')
        return {'baseline': []}

    monkeypatch.setattr(evaluation, 'evaluate', evaluate)
    monkeypatch.setattr(evaluation, 'load_policy', lambda *args: (None, {'env_runner': {'task_name': 'task'}}, 'raw'))
    monkeypatch.setattr(evaluation, 'seed_split', lambda task: ([11], [], set()))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', 'base.pt', '--device', 'cpu',
                                    '--output', str(output)])
    with pytest.raises(FileExistsError):
        evaluation.main()
    assert output.read_text() == 'first writer'


@pytest.mark.parametrize('episodes', ['0', '-1', '4'])
def test_cli_rejects_invalid_episode_counts(monkeypatch, tmp_path, episodes):
    monkeypatch.setattr(evaluation, 'load_policy', lambda *args: (None, {'env_runner': {'task_name': 'task'}}, 'raw'))
    monkeypatch.setattr(evaluation, 'seed_split', lambda task: ([11, 12, 13], [], set()))
    output = tmp_path / 'evaluation.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--checkpoint', 'base.pt', '--device', 'cpu',
                                    '--output', str(output), '--episodes', episodes])
    with pytest.raises(SystemExit, match='2'):
        evaluation.main()
    assert not output.exists()
