from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_rl import evaluate as evaluation, train_rfs as train


def assert_same_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_same_state(value, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize('stop,expected_batches', [
    ('complete', [2, 2]), ('total_steps', [2, 1]),
    ('reset', []), ('partial_rollout', []),
])
def test_rfs_training_budget_resume_and_export(tmp_path, monkeypatch, stop, expected_batches):
    cfg = OmegaConf.load('configs/rfs.yaml')
    cfg.update(dict(hidden_dims=[8], rollout_steps=2, batch_size=2, update_epochs=2,
                    total_steps=3 if stop == 'total_steps' else 4, eval_interval_updates=1,
                    train_hours=1, max_hours=2))
    config = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, config)
    checkpoint = tmp_path / 'base.pt'
    checkpoint.write_bytes(b'test policy')
    policy = SimpleNamespace(n_obs_steps=1, n_action_steps=1, action_dim=2)
    policy_cfg = dict(env_runner=dict(task_name='test'))
    clock, environment_steps = [0.], []

    def actions(context, noise, residual=None, *, return_clip_fraction=False):
        action = noise[:, :2]
        if residual is not None:
            action = action + residual
        return (action, torch.tensor(0.)) if return_clip_fraction else action

    base = SimpleNamespace(device=torch.device('cpu'), state_dim=3, noise_dim=4, action_dim=2,
        encode=lambda obs: (obs, obs), actions=actions, physical=lambda action: action[:, None])

    def reset(seed):
        if stop == 'reset':
            clock[0] = 3601.
        return torch.zeros(1, 3)

    def step(action):
        assert stop != 'reset', 'Reset exhausted the budget before this action'
        environment_steps.append(action.copy())
        if stop == 'partial_rollout':
            clock[0] = 3601.
        return torch.zeros(1, 3), float(action.sum()) + 1., True, False, dict(steps=1, success=False)

    @contextmanager
    def make_env(*args):
        yield SimpleNamespace(env=SimpleNamespace(robot=SimpleNamespace(qlimits=[])),
                              reset=reset, step=step)

    updates, evaluated = [], []
    original_update = train.RFS.update
    original_save = train.save
    midway = tmp_path / 'midway.pt'

    def save(path, value):
        original_save(path, value)
        if stop == 'complete' and path.name == 'resume.pt' and value['updates'] == 1 and not midway.exists():
            original_save(midway, value)

    def update(agent, batch):
        previous = {key: value.clone() for key, value in agent.actor.state_dict().items()}
        metrics = original_update(agent, batch)
        gradients = [parameter.grad for parameter in agent.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert all(np.isfinite(value) for value in metrics.values())
        updates.append((len(batch['state']), previous))
        return metrics

    def evaluate(policy, policy_cfg, seeds, policies, **kwargs):
        for actor in policies.values():
            if actor is not None:
                assert updates, 'An untrained actor reached evaluation'
                evaluated.append({key: value.clone() for key, value in actor.state_dict().items()})
        return {name: [dict(seed=seed, success=False) for seed in seeds] for name in policies}

    seed_calls = []

    def split_seeds(task):
        seed_calls.append(task)
        assert len(seed_calls) == 1, 'Resume must preserve the saved seed partition'
        return [10], [11], {10, 11}

    monkeypatch.setattr(train, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(train.RFS, 'update', update)
    monkeypatch.setattr(train, 'save', save)
    for module in (train, evaluation):
        monkeypatch.setattr(module, 'load_policy', lambda *args: (policy, policy_cfg, 'raw'))
        monkeypatch.setattr(module, 'FrozenPolicy', lambda *args: base)
        monkeypatch.setattr(module, 'make_env', make_env)
        monkeypatch.setattr(module, 'seed_split', split_seeds)
        monkeypatch.setattr(module, 'evaluate', evaluate)
    monkeypatch.setattr(evaluation, 'policy_dimensions', lambda policy: (base.state_dim, base.noise_dim, base.action_dim))
    monkeypatch.setitem(sys.modules, 'dexmani_sim', SimpleNamespace(EpisodeRejected=RuntimeError))

    previous_output, previous_resume = None, None
    for name in ('first', 'resumed'):
        clock[0] = 0.
        output = tmp_path / name
        argv = ['train_rfs', '--output', str(output), '--device', 'cpu']
        if previous_output is None:
            argv += ['--config', str(config), '--checkpoint', str(checkpoint)]
        else:
            argv += ['--resume', str(previous_output / 'resume.pt')]
        monkeypatch.setattr(sys, 'argv', argv)
        train.main()
        result = json.loads((output / 'evaluation.json').read_text())
        saved = torch.load(output / 'resume.pt', weights_only=False)
        assert saved['updates'] == len(expected_batches)
        assert [size for size, _ in updates] == expected_batches
        if previous_resume is not None:
            assert_same_state(saved['model'], previous_resume['model'])
            assert_same_state(saved['optimizer'], previous_resume['optimizer'])
            assert saved['elapsed'] >= previous_resume['elapsed']
        if not expected_batches:
            assert not evaluated and not (output / 'best.pt').exists()
            assert result['rl_checkpoint'] is None and result['steered'] == []
            assert result['comparison']['episodes'] == 0
            assert saved['steps'] == (1 if stop == 'partial_rollout' else 0)
        else:
            assert saved['optimizer']['state']
            assert saved['last_eval_update'] == len(expected_batches)
            assert result['rl_checkpoint'] == 'best.pt'
            best = torch.load(output / 'best.pt', weights_only=False)['actor']
            last = torch.load(output / 'last.pt', weights_only=False)['actor']
            assert_same_state(best, evaluated[-1])
            assert_same_state(last, {key: saved['model']['actor.' + key] for key in last})
            assert any(not torch.equal(updates[0][1][key], last[key]) for key in last)
            if stop == 'complete':
                assert any(not torch.equal(best[key], last[key]) for key in best)
        previous_output, previous_resume = output, saved

    assert len(environment_steps) == (0 if stop == 'reset' else 1 if stop == 'partial_rollout' else cfg.total_steps)
    if stop == 'complete':
        continued_output = tmp_path / 'continued'
        monkeypatch.setattr(sys, 'argv', ['train_rfs', '--resume', str(midway),
                                         '--device', 'cpu', '--output', str(continued_output)])
        train.main()
        continued = torch.load(continued_output / 'resume.pt', weights_only=False)
        assert continued['updates'] == 2 and len(updates) == 3
        assert_same_state(continued['model'], saved['model'])
        assert_same_state(continued['optimizer'], saved['optimizer'])
        assert_same_state(continued['torch_rng'], saved['torch_rng'])
        assert continued['seed_rng'] == saved['seed_rng']
    if expected_batches:
        comparison = tmp_path / 'comparison.json'
        monkeypatch.setattr(sys, 'argv', ['evaluate', '--rl-checkpoint', str(output / 'best.pt'),
                                         '--device', 'cpu', '--episodes', '1', '--output', str(comparison)])
        evaluation.main()
        assert_same_state(best, evaluated[-1])
        assert json.loads(comparison.read_text())['checkpoint'] == str(checkpoint)
