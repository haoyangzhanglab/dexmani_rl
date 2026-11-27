from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from dexmani_rl import train_dsrl as train, evaluate as evaluation


@pytest.mark.parametrize('stop', ['complete', 'warmup', 'reset'])
def test_training_and_resume_export_the_evaluated_model(tmp_path, monkeypatch, stop):
    cfg = OmegaConf.load('configs/dsrl.yaml')
    cfg.update(dict(width=8, depth=1, batch_size=2, utd=1, noise_updates=1,
                    replay_capacity=4, warmup_steps=3, total_steps=4, eval_interval_chunks=2,
                    train_hours=1, max_hours=2))
    if stop == 'reset':
        cfg.warmup_steps = 0
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
        if stop == 'warmup':
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
        return {name: [dict(seed=seed, success=False) for seed in seeds] for name in policies}

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
        if stop != 'complete':
            assert not evaluated and not (output / 'best.pt').exists()
            assert result['rl_checkpoint'] is None and result['steered'] == []
            assert result['comparison']['episodes'] == 0
        else:
            assert result['rl_checkpoint'] == 'best.pt'
            best = torch.load(output / result['rl_checkpoint'], weights_only=False)['actor']
            last = torch.load(output / 'last.pt', weights_only=False)['actor']
            for key in best:
                torch.testing.assert_close(best[key], evaluated[-1][key], atol=0, rtol=0)
            assert any(not torch.equal(best[k], last[k]) for k in best)
            resumed = torch.load(output / 'resume.pt', weights_only=False)
            assert resumed['last_eval_chunk'] == 3
            for key in last:
                torch.testing.assert_close(last[key], resumed['model']['actor.' + key], atol=0, rtol=0)
        previous = output
    if stop == 'complete':
        comparison = tmp_path / 'comparison.json'
        monkeypatch.setattr(sys, 'argv', ['evaluate', '--rl-checkpoint', str(output / 'best.pt'),
                                         '--device', 'cpu', '--output', str(comparison)])
        evaluation.main()
        for key in best:
            torch.testing.assert_close(best[key], evaluated[-1][key], atol=0, rtol=0)
        assert json.loads(comparison.read_text())['checkpoint'] == str(checkpoint)
