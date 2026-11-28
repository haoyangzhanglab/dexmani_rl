import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from dexmani_rl.envs.dexmani_sim import make_flowmatch_env, seed_split
from dexmani_rl.evaluation.metrics import compare
from dexmani_rl.evaluation.metrics import evaluation_report as paired_report
from dexmani_rl.policies.flowmatch.loader import load_flowmatch_policy
from dexmani_rl.policies.flowmatch.rl_runtime import FlowMatchRLRuntime

from .._shared.cli import evaluation_inputs
from .algorithm import BORAActor


@dataclass(frozen=True)
class BORAStage:
    actor: BORAActor
    round_number: int


def evaluation_report(results, seeds):
    result = paired_report(results, seeds)
    if result['evaluation_complete'] and 'offline' in results:
        result.update(comparison_offline_vs_bc=compare(results['baseline'], results['offline']),
                      comparison_vs_offline=compare(results['offline'], results['steered']))
    return result


@torch.no_grad()
def rollout(env, base, seed, actor, deadline):
    if time.monotonic() >= deadline:
        return None
    obs = env.reset(seed)
    generator = torch.Generator(device=base.device).manual_seed(seed)
    steps, reward_sum = 0, 0.
    while time.monotonic() < deadline:
        if actor is not None:
            window = actor.actor.normalized_actions(obs, generator=generator, stage=actor.round_number)
            actions = base.bounded_physical(window)[0].cpu().numpy()
        else:
            context, _ = base.encode(obs)
            noise = torch.randn(1, base.noise_dim, device=base.device, generator=generator)
            actions = base.physical(base.actions(context, noise))[0].cpu().numpy()
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


def main():
    args, checkpoint, weights, saved = evaluation_inputs('bora')
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
        bora = BORAActor(deepcopy(policy), cfg, saved['config'])
        bora.load_state_dict(saved['actor'])
        bora.eval()
        round_number = int(bora.round_number)
        if round_number:
            policies['offline'] = BORAStage(bora, 0)
        policies['steered'] = BORAStage(bora, round_number)
    results = evaluate(policy, cfg, seeds, policies)
    result = dict(algorithm='bora' if saved is not None else 'bc', checkpoint=checkpoint, weights=weights,
                  rl_checkpoint=str(Path(args.rl_checkpoint).resolve()) if args.rl_checkpoint else None,
                  **evaluation_report(results, seeds))
    if saved is not None:
        result['round'] = round_number
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result, indent=2)
    with output.open('x') as file:
        file.write(report)
    print(json.dumps(result['comparison'] if args.rl_checkpoint else {'episodes': len(results['baseline'])}))


if __name__ == '__main__':
    main()
