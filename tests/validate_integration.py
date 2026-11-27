"""Opt-in GPU integration check; generated weights are test fixtures, never a BC baseline."""
import argparse
import json
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf
from dexmani_rl.base_policy import load_policy, FrozenPolicy
from dexmani_rl.env import make_env
from dexmani_rl.dsrl import DSRL
from dexmani_rl.replay import Replay
from dexmani_rl.policy.flowmatch import FlowMatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-root')
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(12)
    kwargs = dict(horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19,
                  hidden_dim=64, n_layers=1, n_head=4)
    reference_class = FlowMatch
    if args.reference_root:
        sys.path.insert(0, str(Path(args.reference_root).resolve()))
        from dexmani_policy.agents.policies.flowmatch import FlowMatch as Reference
        reference_class = Reference
    reference = reference_class(**kwargs).eval().requires_grad_(False)
    dims = dict(action=19, joint_state=19, point_cloud=3, rgb=3, camera_intrinsic=9, camera_extrinsic=12)
    reference.configure_normalizer(dims, {'action': (19,), 'joint_state': (19,)})
    # Nonzero velocity ensures the comparison exercises time integration, not just identity noise.
    torch.nn.init.normal_(reference.action_decoder.model.final_layer.ffn_final.fc2.weight, std=.01)
    cfg = dict(agent={'_target_': 'dexmani_policy.agents.policies.flowmatch.FlowMatch', **kwargs,
                      'time_shift_alpha': 3., 'fps_random_config': {'use_random': True}},
        action_key='action', normalization_dims=dims,
        normalization={k: 'limits' if k in ('action', 'joint_state') else 'identity' for k in dims},
        eval=dict(use_ema=False, inference_steps=10), control_dt_s=1/16, dataset={},
        env_runner=dict(task_name='multi_grasp', env_kwargs=dict(randomize_table_height=True, randomize_model_id=False)))
    checkpoint = directory / 'fixture.pt'
    torch.save(dict(effective_config=cfg, model_state=reference.state_dict(), ema_model_state=None,
                    test_fixture=True), checkpoint)
    policy, cfg, _ = load_policy(checkpoint, 'cuda:0')
    reference = reference.to('cuda')
    replay = Replay(8)
    rl_cfg = OmegaConf.to_container(OmegaConf.load('configs/dsrl.yaml'))
    rl_cfg.update(width=32, depth=2, batch_size=2, utd=1, noise_updates=1, replay_capacity=8,
                  warmup_steps=0, total_steps=8, max_hours=.012, train_hours=.008)
    OmegaConf.save(rl_cfg, directory / 'smoke.yaml')
    with make_env(policy, cfg) as env:
        obs = env.reset(48556)
        base = FrozenPolicy(policy, cfg, env.env.robot.qlimits, decode_batch=2)
        with torch.no_grad():
            expected_context = reference._encode_obs(obs) if args.reference_root else reference.encode_observations(obs)
            context, state = base.encode(obs)
            torch.testing.assert_close(context, expected_context, atol=1e-5, rtol=1e-5)
            seed = 51
            w = torch.randn(1, 16, 19, device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed))
            expected = reference.predict_action(obs, generator=torch.Generator(device='cuda').manual_seed(seed))['future_actions']
            actual = policy.predict_action(obs, initial_noise=w)['future_actions']
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            assert not torch.allclose(actual, policy.predict_action(obs, initial_noise=w + .5)['future_actions'])
        agent = DSRL(base, rl_cfg)
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        for _ in range(2):
            w = torch.randn(1, base.noise_dim, device='cuda')
            action = base.actions(context, w)
            obs, reward, done, truncated, info = env.step(base.physical(action)[0].cpu().numpy())
            next_context, next_state = base.encode(obs)
            replay.add(context=context[0], state=state[0], action=action[0], next_context=next_context[0],
                next_state=next_state[0], reward=[reward], terminated=[float(done)])
            metrics = agent.update(replay)
            context, state = next_context, next_state
            if done or truncated:
                break
        for k, v in policy.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)
        assert all(p.grad is None for p in policy.parameters())
        torch.save(agent.state_dict(), directory / 'agent.pt')
        clone = DSRL(base, rl_cfg)
        clone.load_state_dict(torch.load(directory / 'agent.pt', weights_only=True))
        g1 = torch.Generator(device='cuda').manual_seed(42)
        g2 = torch.Generator(device='cuda').manual_seed(42)
        torch.testing.assert_close(agent.actor.sample(state, g1)[0], clone.actor.sample(state, g2)[0])
    result = dict(reference_comparison=bool(args.reference_root), fixture_only=True,
                  checkpoint_loading=True, frozen_policy=True, real_simulation=True, metrics=metrics)
    (directory / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
