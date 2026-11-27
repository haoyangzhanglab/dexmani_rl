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
from dexmani_rl.rfs import RFS
from dexmani_rl.policy.flowmatch import FlowMatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-root')
    parser.add_argument('--algorithm', choices=['dsrl', 'rfs'], default='dsrl')
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
    reference.normalizer['action'].scale.copy_(torch.linspace(.5, 2., 19))
    reference.normalizer['action'].offset.copy_(torch.linspace(-.5, .5, 19))
    # Exercise both time/context conditioning and integration; zero gates hide embedding differences.
    torch.nn.init.normal_(reference.action_decoder.model.adaln_modulation.projection.weight, std=.01)
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
    rl_cfg = OmegaConf.to_container(OmegaConf.load(f'configs/{args.algorithm}.yaml'))
    if args.algorithm == 'rfs':
        rl_cfg.update(hidden_dims=[32, 16], rollout_steps=4, batch_size=2, update_epochs=2,
                      total_steps=64, eval_interval_updates=100, max_hours=.025, train_hours=.015)
    else:
        rl_cfg.update(width=32, depth=2, batch_size=2, utd=1, noise_updates=1, replay_capacity=8,
                      warmup_steps=0, total_steps=8, max_hours=.012, train_hours=.008)
        replay = Replay(rl_cfg['replay_capacity'])
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
        agent = (RFS(base.state_dim, base.noise_dim, base.action_dim, rl_cfg, base.device)
                 if args.algorithm == 'rfs' else DSRL(base, rl_cfg))
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        actor_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
        updates = 0
        for rollout_index in range(2):
            if rollout_index == 1:
                snapshot = dict(model=agent.state_dict(), torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(base.device))
                if args.algorithm == 'rfs':
                    snapshot['optimizer'] = agent.optimizer.state_dict()
                else:
                    snapshot['optimizers'] = {name: optimizer.state_dict() for name, optimizer in agent.optimizers.items()}
                    snapshot['replay'] = replay.state_dict()
                torch.save(snapshot, directory / 'update_state.pt')
                saved = torch.load(directory / 'update_state.pt', map_location='cpu', weights_only=True)
                clone = (RFS(base.state_dim, base.noise_dim, base.action_dim, rl_cfg, base.device)
                         if args.algorithm == 'rfs' else DSRL(base, rl_cfg))
                clone.load_state_dict(saved['model'])
                if args.algorithm == 'rfs':
                    clone.optimizer.load_state_dict(saved['optimizer'])
                else:
                    for name, optimizer in clone.optimizers.items():
                        optimizer.load_state_dict(saved['optimizers'][name])
                    replay = Replay(rl_cfg['replay_capacity'])
                    replay.load_state_dict(saved['replay'])
                g1 = torch.Generator(device='cuda').manual_seed(42)
                g2 = torch.Generator(device='cuda').manual_seed(42)
                torch.testing.assert_close(agent.actor.sample(state, g1)[0], clone.actor.sample(state, g2)[0])
                agent = clone
                torch.set_rng_state(saved['torch_rng'])
                torch.cuda.set_rng_state(saved['cuda_rng'], base.device)
                resumed_actor = {k: v.clone() for k, v in agent.actor.state_dict().items()}

            transitions = []
            for _ in range(4 if args.algorithm == 'rfs' else 2):
                with torch.no_grad():
                    if args.algorithm == 'rfs':
                        joint, log_prob = agent.actor.sample(state)
                        w, residual = agent.actor.split(joint)
                        value = agent.value(state)
                        action = base.actions(context, w, residual)
                    else:
                        w = torch.randn(1, base.noise_dim, device='cuda')
                        action = base.actions(context, w)
                obs, reward, done, truncated, info = env.step(base.physical(action)[0].cpu().numpy())
                next_context, next_state = base.encode(obs)
                if args.algorithm == 'rfs':
                    with torch.no_grad():
                        next_value = agent.value(next_state)
                    transitions.append(dict(state=state[0], joint=joint[0], log_prob=log_prob[0],
                        value=value[0], next_value=next_value[0],
                        reward=torch.tensor([reward], device='cuda'),
                        terminated=torch.tensor([bool(done)], device='cuda'),
                        truncated=torch.tensor([bool(truncated)], device='cuda')))
                else:
                    replay.add(context=context[0], state=state[0], action=action[0], next_context=next_context[0],
                        next_state=next_state[0], reward=[reward], terminated=[float(done)])
                    metrics = agent.update(replay)
                    updates += 1
                context, state = next_context, next_state
                if done or truncated:
                    context, state = base.encode(env.reset(48556))
            if args.algorithm == 'rfs':
                metrics = agent.update({key: torch.stack([row[key] for row in transitions]) for key in transitions[0]})
                updates += 1
        assert any(not torch.equal(actor_before[k], v) for k, v in agent.actor.state_dict().items())
        assert any(not torch.equal(resumed_actor[k], v) for k, v in agent.actor.state_dict().items())
        for k, v in policy.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)
        assert all(p.grad is None for p in policy.parameters())
    result = dict(algorithm=args.algorithm, updates=updates, resumed_updates=True,
                  reference_comparison=bool(args.reference_root), fixture_only=True,
                  checkpoint_loading=True, frozen_policy=True, real_simulation=True, metrics=metrics)
    (directory / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
