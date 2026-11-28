import json
import time
from pathlib import Path

import torch

from dexmani_rl.envs.dexmani_sim import make_flowmatch_env, seed_split
from dexmani_rl.evaluation.metrics import evaluation_report
from dexmani_rl.policies.flowmatch.loader import load_flowmatch_policy
from dexmani_rl.policies.flowmatch.rl_runtime import (
    FlowMatchRLRuntime,
    policy_dimensions,
)

from .._shared.cli import evaluation_inputs
from .algorithm import Actor


@torch.no_grad()
def rollout(env, base, seed, actor, deadline):
    if time.monotonic() >= deadline:
        return None
    obs = env.reset(seed)
    generator = torch.Generator(device=base.device).manual_seed(seed)
    steps, reward_sum = 0, 0.
    while time.monotonic() < deadline:
        context, state = base.encode(obs)
        residual = None
        if actor is None:
            w = torch.randn(1, base.noise_dim, device=base.device, generator=generator)
        else:
            w, residual = actor.split(actor.sample(state, generator)[0])
        normalized = base.actions(context, w, residual)
        actions = base.physical(normalized)[0].cpu().numpy()
        obs, reward, done, truncated, info = env.step(actions)
        steps += info['steps']
        reward_sum += reward
        if done or truncated:
            result = {'seed': seed, 'success': bool(info['success']), 'steps': steps, 'reward': reward_sum}
            return result
    return None


def evaluate(policy, cfg, seeds, policies, *, deadline=float('inf')):
    results: dict[str, list[dict]] = {name: [] for name in policies}
    for seed in seeds:
        group = {}
        for name, actor in policies.items():
            if time.monotonic() >= deadline:
                return results
            # PhysX retains scene history across resets, even with the same seed.
            with make_flowmatch_env(policy, cfg) as env:
                base = FlowMatchRLRuntime(policy, cfg, env.env.robot.qlimits)
                episode = rollout(env, base, seed, actor, deadline)
            if episode is None:
                return results
            group[name] = episode
        # Commit a seed only after every requested policy has completed it.
        for name in policies:
            results[name].append(group[name])
    return results


def evaluate_noise_policy(run, actor, has_updates):
    policies = {'baseline': None}
    checkpoint = None
    if has_updates:
        checkpoint, policies['steered'] = run.selected_actor(actor)
    results = evaluate(run.policy, run.policy_cfg, run.test, policies,
                       deadline=run.deadline)
    return dict(algorithm=run.metadata['algorithm'], rl_checkpoint=checkpoint, **evaluation_report(results, run.test))


def main():
    args, checkpoint, weights, saved = evaluation_inputs('rfs')
    policy, cfg, weights = load_flowmatch_policy(checkpoint, args.device, weights)
    if args.rl_checkpoint:
        seeds = saved['test_seeds']
    else:
        seeds, _, _ = seed_split(cfg['env_runner']['task_name'])
    if args.episodes is not None:
        if args.episodes > len(seeds):
            raise ValueError(f'--episodes must not exceed the test seed count ({len(seeds)})')
        seeds = seeds[:args.episodes]
    policies = {'baseline': None}
    if saved is not None:
        actor_cfg = saved['config']
        state_dim, noise_dim, action_dim = policy_dimensions(policy)
        actor = Actor(state_dim, noise_dim, action_dim,
                      actor_cfg['hidden_dims'], actor_cfg['residual_scale'])
        actor.to(args.device)
        actor.load_state_dict(saved['actor'])
        actor.eval()
        policies['steered'] = actor
    results = evaluate(policy, cfg, seeds, policies)
    result = dict(algorithm='rfs' if saved is not None else 'bc', checkpoint=checkpoint, weights=weights,
                  rl_checkpoint=str(Path(args.rl_checkpoint).resolve()) if args.rl_checkpoint else None,
                  **evaluation_report(results, seeds))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result, indent=2)
    with output.open('x') as file:
        file.write(report)
    print(json.dumps(result['comparison'] if args.rl_checkpoint else {'episodes': len(results['baseline'])}))


if __name__ == '__main__':
    main()
