"""Short CUDA run on real multi_grasp demonstrations using a synthetic FlowMatch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from omegaconf import OmegaConf

from dexmani_rl.policy.flowmatch import FlowMatch


def make_fixture(path):
    torch.manual_seed(12)
    model_args = dict(horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19,
                      hidden_dim=64, n_layers=1, n_head=4, num_points=128,
                      pc_encoder_config=dict(num_patches=16, group_size=8, depth=1),
                      rgb_backbone_config=dict(image_size=56, architecture=dict(model_type='dinov2',
                          hidden_size=48, num_attention_heads=4, num_hidden_layers=1, image_size=56)))
    policy = FlowMatch(**model_args)
    dims = dict(action=19, joint_state=19, point_cloud=3, rgb=3, camera_intrinsic=9, camera_extrinsic=12)
    policy.configure_normalizer(dims, {'action': (19,), 'joint_state': (19,)})
    decoder = policy.action_decoder.model
    torch.nn.init.normal_(decoder.adaln_modulation.projection.weight, std=.01)
    torch.nn.init.normal_(decoder.final_layer.ffn_final.fc2.weight, std=.01)
    cfg = dict(agent=dict(_target_='FlowMatch', **model_args), action_key='action', normalization_dims=dims,
               normalization={k: 'limits' if k in ('action', 'joint_state') else 'identity' for k in dims},
               eval=dict(use_ema=False, inference_steps=10), control_dt_s=1/16, dataset={},
               env_runner=dict(task_name='multi_grasp', env_kwargs=dict(action_cnt_limit=16,
                               randomize_model_id=False, randomize_table_height=True)))
    torch.save(dict(effective_config=cfg, model_state=policy.state_dict(), test_fixture=True), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / 'fixture.pt'
    make_fixture(fixture)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/bora.yaml')
    cfg.update(dict(offline_updates=3, bc_warmup_updates=1, offline_batch_size=2, online_batch_size=2,
                    episodes_per_round=1, updates_per_episode=2, hidden_dim=32))
    config_path = output / 'smoke.yaml'
    OmegaConf.save(cfg, config_path)
    first, second = output / 'round1', output / 'round2'
    train = [sys.executable, '-m', 'dexmani_rl.train_bora']
    subprocess.run([*train, '--checkpoint', fixture, '--config', config_path,
                    '--through', 'round1', '--output', first], check=True)
    subprocess.run([*train, '--resume', first / 'resume.pt', '--output', second], check=True)
    report = output / 'evaluation.json'
    subprocess.run([sys.executable, '-m', 'dexmani_rl.evaluate', '--rl-checkpoint', second / 'last.pt',
                    '--episodes', '2', '--output', report], check=True)

    def actor(path):
        return torch.load(path, map_location='cpu', weights_only=True)['actor']

    base = torch.load(fixture, map_location='cpu', weights_only=True)['model_state']
    offline, r1, r2 = (actor(path) for path in (first / 'offline.pt', first / 'last.pt', second / 'last.pt'))
    for prefix in ('obs_encoder.', 'action_decoder.'):
        assert any(not torch.equal(t, offline['policy.' + k]) for k, t in base.items() if k.startswith(prefix))
    for k in offline:
        if k.startswith('policy.'):
            assert torch.equal(offline[k], r1[k]) and torch.equal(r1[k], r2[k]), k
        if k.startswith('residuals.0.'):
            assert torch.equal(r1[k], r2[k]), k
    assert torch.equal(r1['blends'][0], r2['blends'][0])
    assert any(not torch.equal(offline[k], r1[k]) for k in r1 if k.startswith('residuals.0.'))
    assert any(not torch.equal(r2[k], r1[k]) for k in r2 if k.startswith('residuals.1.'))
    rows = [json.loads(line) for run in (first, second) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    ppo = [r for r in rows if r.get('phase') == 'offline_ppo']
    assert len(ppo) == 2 and all(r['ppo_updates'] == 2 and r['mean_log_ratio'] != 0 for r in ppo)
    rollouts = [r for r in rows if r.get('phase') == 'collect']
    assert [r['round'] for r in rollouts] == [1, 2]
    assert all(r['takeovers'] == r['recoveries'] == 0 for r in rollouts)
    evaluation = json.loads(report.read_text())
    seeds = [e['seed'] for e in evaluation['baseline']]
    assert len(seeds) == 2 and all([e['seed'] for e in evaluation[k]] == seeds for k in ('offline', 'steered'))
    print(json.dumps(dict(test_fixture=True, collected_steps=sum(r['steps'] for r in rollouts),
                          ppo_updates=sum(r['ppo_updates'] for r in ppo), paired_seeds=seeds)))


if __name__ == '__main__':
    main()
