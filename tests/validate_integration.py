"""Exercise the three post-training algorithms on CUDA with synthetic FlowMatch weights."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
from dexmani_rl.base_policy import load_policy, FrozenPolicy
from dexmani_rl.env import make_env
from dexmani_rl.dsrl import DSRL
from dexmani_rl.replay import Replay
from dexmani_rl.rfs import RFS


PROJECT = Path(__file__).resolve().parents[1]
ROLLOUT_SEED = 48556


def assert_tree_equal(actual, expected, *, rtol=0., atol=0.):
    if isinstance(expected, torch.Tensor):
        floating = expected.is_floating_point() or expected.is_complex()
        torch.testing.assert_close(actual.cpu(), expected.cpu(),
                                   rtol=rtol if floating else 0., atol=atol if floating else 0.)
    elif isinstance(expected, np.ndarray):
        if expected.dtype.kind in 'fc':
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            assert_tree_equal(value, reference, rtol=rtol, atol=atol)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    else:
        assert actual == expected


def validate_identical_rollouts(policy, cfg):
    from unittest.mock import patch
    from dexmani_rl.env import ChunkEnv
    from dexmani_rl.evaluate import evaluate
    from dexmani_rl.robofft import RoboFFTActor

    seed = 42039
    starts, first_chunks = [], []
    reset, step = ChunkEnv.reset, ChunkEnv.step

    def record_reset(env, seed):
        obs = reset(env, seed)
        starts.append({name: value.cpu().clone() for name, value in obs.items()})
        return obs

    def record_step(env, actions):
        if len(first_chunks) < len(starts):
            first_chunks.append(actions.copy())
        return step(env, actions)

    actor = RoboFFTActor(policy.action_decoder.model, policy.horizon, policy.action_dim, cfg['eval']['inference_steps'])
    with patch.object(ChunkEnv, 'reset', record_reset), patch.object(ChunkEnv, 'step', record_step):
        results = evaluate(policy, cfg, [seed], {'first': actor, 'second': actor})
    assert_tree_equal(starts[0], starts[1])
    np.testing.assert_array_equal(first_chunks[0], first_chunks[1])
    assert results['first'] == results['second']
    return dict(results['first'][0], identical_initial_observations=True, identical_first_chunk=True)


def validate_robofft(env, base, context, state, cfg, directory):
    from dexmani_rl.checkpoint import capture_rng, restore_rng
    from dexmani_rl.robofft import RoboFFT, RoboFFTActor

    def make_agent():
        policy = base.policy
        actor = RoboFFTActor(policy.action_decoder.model, policy.horizon, policy.action_dim,
                            base.cfg['eval']['inference_steps'], cfg['n_samples'], cfg['noise_std'])
        return RoboFFT(base.state_dim, actor, cfg, base.device)

    def snapshot(agent):
        return dict(model=agent.state_dict(),
                    optimizers={name: value.state_dict() for name, value in agent.optimizers.items()},
                    schedulers={'actor': agent.actor_scheduler.state_dict()},
                    reward_scaler=agent.reward_scaler.state_dict(), **capture_rng(seed_rng, base.device))

    agent = make_agent()
    seed_rng = np.random.default_rng(12)
    before = {key: value.cpu().clone() for key, value in base.policy.state_dict().items()}
    first, metrics_history = True, []
    update_tolerance = dict(rtol=1e-5, atol=1e-7)
    for iteration in (1, 2, 3):
        if iteration == 3:
            torch.save(snapshot(agent), directory / 'update_state.pt')
            saved = torch.load(directory / 'update_state.pt', map_location='cpu', weights_only=False)
            original = agent
            agent = make_agent()
            agent.load_state_dict(saved['model'])
            for name in agent.optimizers:
                agent.optimizers[name].load_state_dict(saved['optimizers'][name])
            agent.actor_scheduler.load_state_dict(saved['schedulers']['actor'])
            agent.reward_scaler.load_state_dict(saved['reward_scaler'])
            restore_rng(saved, seed_rng, base.device)
            assert_tree_equal(snapshot(agent), saved)

        actor_before = {key: value.cpu().clone() for key, value in agent.actor.state_dict().items()}
        transitions, normalized_clips, physical_clips = [], [], []
        for _ in range(4):
            with torch.no_grad():
                action, normalized_clip = agent.actor.sample(context, return_clip_fraction=True)
                eps, t = agent.actor.draw_matching(action)
                log_prob = agent.actor.log_prob(context, action, eps, t)
                physical, physical_clip = base.robofft_actions(action)
                value = agent.value(state)
            obs, reward, done, truncated, _ = env.step(physical[0].cpu().numpy())
            next_context, next_state = base.encode(obs)
            with torch.no_grad():
                next_value = agent.value(next_state)
            row = dict(context=context[0], state=state[0], action=action[0], eps=eps[0], t=t[0],
                       old_log_prob=log_prob[0], value=value[0], next_value=next_value[0],
                       reward=torch.tensor([reward]), terminated=torch.tensor([bool(done)]),
                       truncated=torch.tensor([bool(truncated)]), first=torch.tensor([first]))
            transitions.append({key: value.detach().cpu() for key, value in row.items()})
            normalized_clips.append(normalized_clip.item())
            physical_clips.append(physical_clip.item())
            context, state, first = next_context, next_state, False
            if done or truncated:
                context, state = base.encode(env.reset(ROLLOUT_SEED))
                first = True
        batch = {key: torch.stack([row[key] for row in transitions]) for key in transitions[0]}
        if iteration == 3:
            update_rng = capture_rng(seed_rng, base.device)
            expected_metrics = original.update(batch, iteration)
            original.finish_iteration(iteration)
            expected_rng = capture_rng(seed_rng, base.device)
            restore_rng(update_rng, seed_rng, base.device)
        metrics = agent.update(batch, iteration)
        agent.finish_iteration(iteration)
        changed = any(not torch.equal(actor_before[key], value.cpu()) for key, value in agent.actor.state_dict().items())
        assert changed == (iteration >= 2), f'Unexpected actor update at iteration {iteration}: {metrics}'
        if iteration == 3:
            # Separate CUDA backward passes can differ at FP32 reduction precision.
            # Loading, frozen weights and RNG comparisons remain exact.
            actual_state, expected_state = agent.state_dict(), original.state_dict()
            update_max_abs_error = max((actual_state[key] - expected_state[key]).abs().max().item()
                                       for key in actual_state)
            assert_tree_equal(actual_state, expected_state, **update_tolerance)
            for name in agent.optimizers:
                assert_tree_equal(agent.optimizers[name].state_dict(), original.optimizers[name].state_dict(),
                                  **update_tolerance)
            assert_tree_equal(agent.actor_scheduler.state_dict(), original.actor_scheduler.state_dict())
            assert_tree_equal(agent.reward_scaler.state_dict(), original.reward_scaler.state_dict())
            assert_tree_equal(metrics, expected_metrics, **update_tolerance)
            assert_tree_equal(capture_rng(seed_rng, base.device), expected_rng)
        assert all(np.isfinite(value) for value in metrics.values())
        metrics_history.append(dict(iteration=iteration, normalized_clip_fraction=float(np.mean(normalized_clips)),
                                    physical_clip_fraction=float(np.mean(physical_clips)), **metrics))
    assert_tree_equal(base.policy.state_dict(), before)
    assert all(parameter.grad is None for parameter in base.policy.parameters())
    return dict(updates=len(metrics_history), actor_update_rounds=sum(m['actor_steps'] > 0 for m in metrics_history),
                actor_optimizer_steps=sum(m['actor_steps'] for m in metrics_history),
                resumed_update_max_abs_error=update_max_abs_error, resumed_update_tolerance=update_tolerance,
                logprob_min=cfg['logprob_min'],
                fixture_decoder='synthetic_contraction', metrics=metrics_history)


def validate_steering(env, base, context, state, cfg, directory, algorithm):
    is_rfs = algorithm == 'rfs'

    def make_agent():
        if is_rfs:
            return RFS(base.state_dim, base.noise_dim, base.action_dim, cfg, base.device)
        return DSRL(base, cfg)

    agent = make_agent()
    replay = None if is_rfs else Replay(cfg['replay_capacity'])
    policy_before = copy.deepcopy(base.policy.state_dict())
    actor_before = copy.deepcopy(agent.actor.state_dict())
    updates = 0
    for rollout_index in range(2):
        if rollout_index == 1:
            snapshot = dict(model=agent.state_dict(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(base.device))
            if is_rfs:
                snapshot['optimizer'] = agent.optimizer.state_dict()
            else:
                snapshot['optimizers'] = {name: opt.state_dict() for name, opt in agent.optimizers.items()}
                snapshot['replay'] = replay.state_dict()
            checkpoint = directory / 'update_state.pt'
            torch.save(snapshot, checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            agent = make_agent()
            agent.load_state_dict(saved['model'])
            assert_tree_equal(agent.state_dict(), saved['model'])
            if is_rfs:
                agent.optimizer.load_state_dict(saved['optimizer'])
                assert_tree_equal(agent.optimizer.state_dict(), saved['optimizer'])
            else:
                for name, optimizer in agent.optimizers.items():
                    optimizer.load_state_dict(saved['optimizers'][name])
                    assert_tree_equal(optimizer.state_dict(), saved['optimizers'][name])
                replay = Replay(cfg['replay_capacity'])
                replay.load_state_dict(saved['replay'])
                assert_tree_equal(replay.state_dict(), saved['replay'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'], base.device)
            resumed_actor = copy.deepcopy(agent.actor.state_dict())

        transitions = []
        for _ in range(4 if is_rfs else 2):
            with torch.no_grad():
                if is_rfs:
                    joint, log_prob = agent.actor.sample(state)
                    noise, residual = agent.actor.split(joint)
                    value = agent.value(state)
                    action = base.actions(context, noise, residual)
                else:
                    noise = torch.randn(1, base.noise_dim, device=base.device)
                    action = base.actions(context, noise)
            obs, reward, done, truncated, _ = env.step(base.physical(action)[0].cpu().numpy())
            next_context, next_state = base.encode(obs)
            if is_rfs:
                with torch.no_grad():
                    next_value = agent.value(next_state)
                transitions.append(dict(state=state[0], joint=joint[0], log_prob=log_prob[0],
                    value=value[0], next_value=next_value[0],
                    reward=torch.tensor([reward], device=base.device),
                    terminated=torch.tensor([bool(done)], device=base.device),
                    truncated=torch.tensor([bool(truncated)], device=base.device)))
            else:
                replay.add(context=context[0], state=state[0], action=action[0], next_context=next_context[0],
                    next_state=next_state[0], reward=[reward], terminated=[float(done)])
                metrics = agent.update(replay)
                updates += 1
            context, state = next_context, next_state
            if done or truncated:
                context, state = base.encode(env.reset(ROLLOUT_SEED))
        if is_rfs:
            metrics = agent.update({key: torch.stack([row[key] for row in transitions]) for key in transitions[0]})
            updates += 1
    assert any(not torch.equal(actor_before[k], v) for k, v in agent.actor.state_dict().items())
    assert any(not torch.equal(resumed_actor[k], v) for k, v in agent.actor.state_dict().items())
    assert_tree_equal(base.policy.state_dict(), policy_before)
    assert all(p.grad is None for p in base.policy.parameters())
    return dict(updates=updates, metrics=metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-root', required=True, type=Path, help='dexmani_policy source directory')
    parser.add_argument('--algorithm', choices=['dsrl', 'rfs', 'robofft'], default='dsrl')
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    if not (reference_root / 'dexmani_policy/__init__.py').is_file():
        parser.error(f'Cannot find dexmani_policy in {reference_root}')
    sys.path.insert(0, str(reference_root))
    from dexmani_policy.agents.policies.flowmatch import FlowMatch

    reference_source = Path(inspect.getfile(FlowMatch)).resolve()
    if not reference_source.is_relative_to(reference_root):
        parser.error(f'FlowMatch was loaded from {reference_source}, expected {reference_root}')
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(12)
    kwargs = dict(horizon=16, n_obs_steps=2, n_action_steps=8, action_dim=19,
                  hidden_dim=64, n_layers=1, n_head=4)
    reference = FlowMatch(**kwargs).eval().requires_grad_(False)
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
    if args.algorithm == 'robofft':
        cfg['env_runner']['env_kwargs']['action_cnt_limit'] = 16
        # Give the fixture a contracting velocity field. A random, nearly zero
        # velocity saturates the release logp floor on these 19D action sequences.
        # Only initialize the readout, using synthetic Gaussian vectors independent of RL rollouts.
        decoder = reference.action_decoder.model
        x = torch.randn(64, reference.horizon, reference.action_dim, generator=torch.Generator().manual_seed(381))
        head = decoder.final_layer.ffn_final
        hidden = decoder.final_layer.norm_final(decoder.input_embedder(x) + decoder.input_pos_embed)
        features = head.act(head.fc1(hidden)).flatten(0, 1)
        head.fc2.weight.copy_(torch.linalg.lstsq(features.double(), -x.flatten(0, 1).double(), driver='gels').solution.T)
    checkpoint = directory / 'fixture.pt'
    torch.save(dict(effective_config=cfg, model_state=reference.state_dict(), ema_model_state=None,
                    test_fixture=True), checkpoint)
    policy, cfg, _ = load_policy(checkpoint, 'cuda:0')
    reference = reference.to('cuda')
    rl_cfg = OmegaConf.to_container(OmegaConf.load(PROJECT / 'configs' / f'{args.algorithm}.yaml'))
    if args.algorithm == 'robofft':
        rl_cfg.update(value_width=32, rollout_steps=4, batch_size=2, microbatch_size=1,
                      repeat_samples=1, update_epochs=2, lr_warmup_iterations=1,
                      iterations=4, eval_interval=10, max_hours=.35, train_hours=.2)
    elif args.algorithm == 'rfs':
        rl_cfg.update(hidden_dims=[32, 16], rollout_steps=4, batch_size=2, update_epochs=2,
                      total_steps=64, eval_interval_updates=100, max_hours=.025, train_hours=.015)
    else:
        rl_cfg.update(width=32, depth=2, batch_size=2, utd=1, noise_updates=1, replay_capacity=8,
                      warmup_steps=0, total_steps=8, max_hours=.012, train_hours=.008)
    OmegaConf.save(rl_cfg, directory / 'smoke.yaml')
    result = dict(algorithm=args.algorithm, reference_source=str(reference_source))
    if args.algorithm == 'robofft':
        result['identical_policy_comparison'] = validate_identical_rollouts(policy, cfg)
    with make_env(policy, cfg) as env:
        obs = env.reset(ROLLOUT_SEED)
        base = FrozenPolicy(policy, cfg, env.env.robot.qlimits, decode_batch=2)
        with torch.no_grad():
            expected_context = reference._encode_obs(obs)
            context, state = base.encode(obs)
            torch.testing.assert_close(context, expected_context, atol=1e-5, rtol=1e-5)
            seed = 51
            w = torch.randn(1, 16, 19, device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed))
            expected = reference.predict_action(obs, generator=torch.Generator(device='cuda').manual_seed(seed))['future_actions']
            actual = policy.predict_action(obs, initial_noise=w)['future_actions']
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            assert not torch.allclose(actual, policy.predict_action(obs, initial_noise=w + .5)['future_actions'])
            result.update(flowmatch_context_max_abs_error=(context - expected_context).abs().max().item(),
                          flowmatch_action_max_abs_error=(actual - expected).abs().max().item())
        if args.algorithm == 'robofft':
            result.update(validate_robofft(env, base, context, state, rl_cfg, directory))
        else:
            result.update(validate_steering(env, base, context, state, rl_cfg, directory, args.algorithm))
    (directory / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
