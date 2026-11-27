from contextlib import contextmanager
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from dexmani_rl import train_dsrl as train, evaluate as evaluation
from dexmani_rl.training import training_arguments


@pytest.mark.parametrize('stop', [
    'complete', 'final_better', 'final_partial', 'warmup', 'reset', 'update_deadline',
])
def test_training_and_resume_export_the_evaluated_model(tmp_path, monkeypatch, stop):
    cfg = OmegaConf.load('configs/dsrl.yaml')
    cfg.update(dict(width=8, depth=1, batch_size=2, utd=1, noise_updates=1,
                    replay_capacity=4, warmup_steps=3, total_steps=4, eval_interval_chunks=2,
                    train_hours=1, max_hours=2))
    if stop == 'reset':
        cfg.warmup_steps = 0
    if stop == 'update_deadline':
        cfg.warmup_steps = 1
    config = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, config)
    checkpoint = tmp_path / 'base.pt'
    checkpoint.write_bytes(b'test policy')
    policy = SimpleNamespace(n_obs_steps=1, obs_encoder=SimpleNamespace(out_dim=1))
    policy_cfg = dict(env_runner=dict(task_name='test'))
    base = SimpleNamespace(device=torch.device('cpu'), state_dim=20, noise_dim=2, action_dim=2,
        encode=lambda obs: (obs, obs), actions=lambda context, noise: noise,
        physical=lambda action: action[:, None])
    clock = [0.]

    def reset(seed):
        if stop == 'reset':
            clock[0] = 3601.
        return torch.zeros(1, 20)

    def step(action):
        assert stop != 'reset', 'Training continued after reset exhausted its budget'
        if stop in ('warmup', 'update_deadline'):
            clock[0] = 3601.
        return torch.zeros(1, 20), 0., True, False, dict(steps=1)

    @contextmanager
    def make_env(*args):
        yield SimpleNamespace(env=SimpleNamespace(robot=SimpleNamespace(qlimits=[])),
                              reset=reset, step=step)

    evaluated = []
    updates = []
    original_update = train.DSRL.update

    def update(agent, replay):
        metrics = original_update(agent, replay)
        updates.append(metrics)
        return metrics

    def evaluate(policy, policy_cfg, seeds, policies, *, noise_magnitude=None, deadline=float('inf')):
        for actor in policies.values():
            if actor is not None:
                assert updates, 'An untrained actor reached evaluation'
                assert noise_magnitude == cfg.noise_magnitude
                evaluated.append({k: v.clone() for k, v in actor.state_dict().items()})
        final_candidate = list(policies) == ['steered'] and len(updates) > 1
        if stop == 'final_partial' and final_candidate:
            assert deadline < 7200.
            seeds = []
        return {name: [dict(seed=seed, success=stop == 'final_better' and final_candidate)
                       for seed in seeds] for name in policies}

    seed_calls = []

    def split_seeds(task):
        seed_calls.append(task)
        assert len(seed_calls) == 1, 'Resume must preserve the saved seed partition'
        return [10], [11], {10, 11}

    monkeypatch.setattr(train, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(train.DSRL, 'update', update)
    for module in (train, evaluation):
        monkeypatch.setattr(module, 'load_policy', lambda *args: (policy, policy_cfg, 'raw'))
        monkeypatch.setattr(module, 'FrozenPolicy', lambda *args: base)
        monkeypatch.setattr(module, 'make_env', make_env)
        monkeypatch.setattr(module, 'seed_split', split_seeds)
        monkeypatch.setattr(module, 'evaluate', evaluate)
    monkeypatch.setattr(evaluation, 'policy_dimensions', lambda policy: (base.state_dim, base.noise_dim, base.action_dim))
    monkeypatch.setitem(sys.modules, 'dexmani_sim', SimpleNamespace(EpisodeRejected=RuntimeError))
    previous = None
    for name in ('first', 'resumed'):
        clock[0] = 0.
        output = tmp_path / name
        argv = ['train_dsrl', '--output', str(output), '--device', 'cpu']
        if previous is None:
            argv += ['--config', str(config), '--checkpoint', str(checkpoint)]
        else:
            argv += ['--resume', str(previous / 'resume.pt')]
        monkeypatch.setattr(sys, 'argv', argv)
        train.main()
        result = json.loads((output / 'evaluation.json').read_text())
        saved = torch.load(output / 'resume.pt', weights_only=False)
        validations = [row for row in map(json.loads, (output / 'metrics.jsonl').read_text().splitlines())
                       if 'validation' in row]
        if stop in ('warmup', 'reset', 'update_deadline'):
            assert not updates and not evaluated and not (output / 'best.pt').exists()
            assert result['rl_checkpoint'] is None and result['steered'] == []
            assert result['comparison']['episodes'] == 0
            assert saved['actor_update_chunk'] == 0
            if stop == 'update_deadline':
                assert saved['steps'] == saved['replay']['size'] == 1
        else:
            assert result['rl_checkpoint'] == 'best.pt'
            best = torch.load(output / result['rl_checkpoint'], weights_only=False)['actor']
            last = torch.load(output / 'last.pt', weights_only=False)['actor']
            for key in best:
                torch.testing.assert_close(best[key], evaluated[-1][key], atol=0, rtol=0)
                torch.testing.assert_close(last[key], saved['model']['actor.' + key], atol=0, rtol=0)
            assert all(torch.equal(best[key], last[key]) for key in best) == (stop == 'final_better')
            assert saved['actor_update_chunk'] == 4
            assert saved['last_eval_chunk'] == (3 if stop == 'final_partial' else 4)
            phases = [row['validation_phase'] for row in validations]
            if name == 'first':
                assert phases == ['scheduled', 'final']
            elif stop == 'final_partial':
                assert phases == ['final']
            else:
                assert not phases
            if validations:
                assert validations[-1]['validation_complete'] == (stop != 'final_partial')
        previous = output
    if stop == 'complete':
        comparison = tmp_path / 'comparison.json'
        monkeypatch.setattr(sys, 'argv', ['evaluate', '--rl-checkpoint', str(output / 'best.pt'),
                                         '--device', 'cpu', '--output', str(comparison)])
        evaluation.main()
        for key in best:
            torch.testing.assert_close(best[key], evaluated[-1][key], atol=0, rtol=0)
        assert json.loads(comparison.read_text())['checkpoint'] == str(checkpoint)


@pytest.mark.parametrize('algorithm', ['dsrl', 'rfs', 'robofft'])
def test_default_config_from_other_working_directory(tmp_path, monkeypatch, algorithm):
    expected = OmegaConf.to_container(OmegaConf.load(Path('configs') / f'{algorithm}.yaml'), resolve=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['train', '--checkpoint', 'base.pt', '--output', 'run'])
    args, cfg, checkpoint, weights, saved = training_arguments(algorithm)
    assert args.output == Path('run')
    assert cfg == expected
    assert checkpoint == str(tmp_path / 'base.pt')
    assert weights is saved is None


def test_resume_rejects_actor_export_before_initialization(tmp_path, monkeypatch, capsys):
    export = tmp_path / 'best.pt'
    torch.save({'algorithm': 'dsrl', 'actor': {}}, export)
    output = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['train_dsrl', '--resume', str(export), '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert 'training snapshot (resume.pt)' in capsys.readouterr().err
    assert not output.exists()
