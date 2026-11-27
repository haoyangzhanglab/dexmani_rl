from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dexmani_rl import train_robofft as train
from dexmani_rl.base_policy import FrozenPolicy
from dexmani_rl.policy.normalizer import LinearNormalizer
from test_train_rfs import assert_same_state


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x, timestep, context, context_kv=None):
        return self.linear(x) + context[:, :1, :2] * .01

    def prepare_context_kv(self, context):
        return None


@pytest.mark.parametrize('scenario', ['tie', 'better', 'partial', 'budget_before_update', 'budget_after_update'])
def test_training_resume_and_budget(tmp_path, monkeypatch, scenario):
    cfg = OmegaConf.load('configs/robofft.yaml')
    cfg.update(dict(iterations=4, rollout_steps=2, batch_size=2, microbatch_size=1,
                    repeat_samples=1, update_epochs=1, value_width=8, eval_interval=10, lr_warmup_iterations=1,
                    train_hours=1, max_hours=2, logprob_min=-10.))
    # Random fixtures saturate the release's -1 floor; only this chain test widens it.
    config = tmp_path / 'config.yaml'
    OmegaConf.save(cfg, config)
    checkpoint = tmp_path / 'fixture.pt'
    checkpoint.write_bytes(b'fixture')
    torch.manual_seed(19)
    decoder = Decoder().eval().requires_grad_(False)
    policy = SimpleNamespace(action_decoder=SimpleNamespace(model=decoder), horizon=3,
                             n_obs_steps=2, n_action_steps=2, action_dim=2,
                             normalizer=LinearNormalizer({'action': (2,)}))
    policy_cfg = dict(env_runner=dict(task_name='test'), eval=dict(inference_steps=2))
    limits = torch.tensor([[-1., 1.], [-1., 1.]])
    base = FrozenPolicy.__new__(FrozenPolicy)
    base.policy, base.limits = policy, limits
    base.device, base.state_dim = torch.device('cpu'), 3
    base.encode = lambda obs: (obs[:, None], obs)
    clock, actions, seed_calls = [0.], [], []

    @contextmanager
    def make_env(*args):
        def step(action):
            actions.append(action.copy())
            if scenario == 'budget_before_update' or (scenario == 'budget_after_update' and len(actions) == 5):
                clock[0] = 3601.
            return torch.ones(1, 3), float(action.sum()) + 2., True, False, dict(steps=2, success=False)
        yield SimpleNamespace(env=SimpleNamespace(robot=SimpleNamespace(qlimits=limits)),
                              reset=lambda seed: torch.zeros(1, 3), step=step)

    def split(task):
        seed_calls.append(task)
        assert len(seed_calls) == 1
        return [20, 21], [22, 23], {20, 21, 22, 23}

    def evaluate(policy, cfg, seeds, policies, **kwargs):
        success = False
        if list(policies) == ['steered']:
            assert seeds == [22, 23]
            trained = any(not torch.equal(v, before[k]) for k, v in policies['steered'].decoder.state_dict().items())
            success = trained and scenario in ('better', 'partial', 'budget_after_update')
            if trained and scenario == 'partial':
                seeds = seeds[:1]
            if trained and scenario == 'budget_after_update':
                assert clock[0] < kwargs['deadline'] < 7200.
        else:
            assert seeds == [20, 21]
        return {name: [dict(seed=seed, success=success) for seed in seeds] for name in policies}

    original_save = train.save
    midway = tmp_path / 'midway.pt'

    def save(path, value):
        original_save(path, value)
        if path.name == 'resume.pt' and value['iteration'] == 3 and not midway.exists():
            original_save(midway, value)

    monkeypatch.setattr(train, 'load_policy', lambda *args: (policy, policy_cfg, 'raw'))
    monkeypatch.setattr(train, 'FrozenPolicy', lambda *args: base)
    monkeypatch.setattr(train, 'make_env', make_env)
    monkeypatch.setattr(train, 'seed_split', split)
    monkeypatch.setattr(train, 'evaluate', evaluate)
    monkeypatch.setattr(train, 'save', save)
    monkeypatch.setattr(train, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setitem(sys.modules, 'dexmani_sim', SimpleNamespace(EpisodeRejected=RuntimeError))
    before = {k: v.clone() for k, v in decoder.state_dict().items()}
    output = tmp_path / 'first'
    monkeypatch.setattr(sys, 'argv', ['train_robofft', '--config', str(config), '--checkpoint', str(checkpoint),
                                     '--output', str(output), '--device', 'cpu'])
    train.main()
    saved = torch.load(output / 'resume.pt', weights_only=False)
    result = json.loads((output / 'evaluation.json').read_text())
    assert result['comparison_vs_sampler']['episodes'] == 2
    assert_same_state(decoder.state_dict(), before)
    assert all(p.grad is None for p in decoder.parameters())
    validations = [row for row in map(json.loads, (output / 'metrics.jsonl').read_text().splitlines())
                   if 'validation' in row]
    if scenario == 'budget_before_update':
        assert saved['updates'] == saved['actor_updates'] == 0
        assert saved['discarded_chunks'] == 1
        assert saved['steps'] == 2
        assert len(validations) == 1
        return
    if scenario == 'budget_after_update':
        assert result['selected_actor_updates'] == 1 and result['selected_iteration'] == 2
        assert [row['actor_updates'] for row in validations] == [0, 1]
        return

    assert saved['updates'] == 3 and saved['actor_updates'] == 2
    assert [(row['iteration'], row['actor_updates']) for row in validations] == [(0, 0), (3, 2)]
    assert validations[-1]['validation_phase'] == 'final'
    assert validations[-1]['validation_complete'] == (scenario != 'partial')
    assert result['selected_iteration'] == (3 if scenario == 'better' else 0)
    assert result['selected_actor_updates'] == (2 if scenario == 'better' else 0)
    assert saved['iteration'] == 4 and saved['steps'] == 12
    assert saved['optimizers']['actor']['state'] and saved['optimizers']['critic']['state']
    assert any(not torch.equal(before[k], saved['model']['actor.decoder.' + k]) for k in before)
    assert len(actions) == 6
    if scenario == 'partial':
        assert saved['last_validated_actor_updates'] == 0
        return
    resumed = tmp_path / 'resumed'
    monkeypatch.setattr(sys, 'argv', ['train_robofft', '--resume', str(midway),
                                     '--output', str(resumed), '--device', 'cpu'])
    train.main()
    continued = torch.load(resumed / 'resume.pt', weights_only=False)
    for key in ('model', 'optimizers', 'schedulers', 'reward_scaler', 'torch_rng', 'seed_rng'):
        assert_same_state(continued[key], saved[key])
    assert continued['actor_updates'] == saved['actor_updates']
    metrics = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
    phase = [row['validation_phase'] for row in metrics if 'validation' in row]
    assert phase == ['final']
    assert continued['best_actor_updates'] == saved['best_actor_updates']
    assert continued['last_validated_actor_updates'] == saved['actor_updates']
    assert all(np.isfinite(row['normalized_clip_fraction']) for row in metrics if 'updates' in row)
    if scenario == 'better':
        done = tmp_path / 'done'
        monkeypatch.setattr(sys, 'argv', ['train_robofft', '--resume', str(resumed / 'resume.pt'),
                                         '--output', str(done), '--device', 'cpu'])
        train.main()
        assert (done / 'metrics.jsonl').read_text() == ''


def test_eval_only_schedule_rejected(tmp_path, monkeypatch):
    config = tmp_path / 'config.yaml'
    cfg = OmegaConf.load('configs/robofft.yaml')
    cfg.eval_interval = 1
    OmegaConf.save(cfg, config)
    monkeypatch.setattr(sys, 'argv', ['train_robofft', '--config', str(config), '--checkpoint', 'unused.pt',
                                     '--output', str(tmp_path / 'run')])
    with pytest.raises(ValueError, match='eval_interval'):
        train.main()
