import time
from copy import deepcopy

import numpy as np
import torch

from dexmani_rl.envs.dexmani_sim import make_flowmatch_env
from dexmani_rl.evaluation.metrics import compare
from dexmani_rl.evaluation.metrics import evaluation_report as paired_report
from dexmani_rl.policies.flowmatch.loader import load_flowmatch_policy
from dexmani_rl.policies.flowmatch.rl_runtime import FlowMatchRLRuntime

from .._shared.cli import evaluation_inputs, evaluation_seeds, write_evaluation
from .algorithm import RoboFFTActor


def evaluation_report(results, seeds):
    result = paired_report(results, seeds)
    if result['evaluation_complete'] and 'sampler_baseline' in results:
        result['comparison_vs_sampler'] = compare(results['sampler_baseline'], results['steered'])
    return result


@torch.no_grad()
def rollout(env, base, seed, actor, deadline):
    if time.monotonic() >= deadline:
        return None
    obs = env.reset(seed)
    generator = torch.Generator(device=base.device).manual_seed(seed)
    steps, reward_sum = 0, 0.
    normalized_clips, physical_clips = [], []
    while time.monotonic() < deadline:
        if actor is not None:
            context, _ = base.encode(obs)
            normalized, fraction = actor.sample(context, generator, eval_mode=True, return_clip_fraction=True)
            physical, physical_fraction = base.physical_execution_window(normalized)
            actions = physical[0].cpu().numpy()
            normalized_clips.append(fraction.item())
            physical_clips.append(physical_fraction.item())
        else:
            context, _ = base.encode(obs)
            noise = torch.randn(1, base.noise_dim, device=base.device, generator=generator)
            actions = base.physical(base.actions(context, noise))[0].cpu().numpy()
        obs, reward, done, truncated, info = env.step(actions)
        steps += info['steps']
        reward_sum += reward
        if done or truncated:
            result = {'seed': seed, 'success': bool(info['success']), 'steps': steps, 'reward': reward_sum}
            if actor is not None:
                result.update(normalized_clip_fraction=float(np.mean(normalized_clips)),
                              physical_clip_fraction=float(np.mean(physical_clips)))
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
    args, checkpoint, weights, saved = evaluation_inputs('robofft')
    policy, cfg, weights = load_flowmatch_policy(checkpoint, args.device, weights)
    seeds = evaluation_seeds(saved, cfg, args.episodes)
    policies: dict[str, RoboFFTActor | None] = {'baseline': None}
    if saved is not None:
        actor_cfg = saved['config']
        actor = RoboFFTActor.from_policy(policy, cfg, actor_cfg)
        policies['sampler_baseline'] = deepcopy(actor)
        actor.load_state_dict(saved['actor'])
        actor.eval()
        policies['steered'] = actor
    results = evaluate(policy, cfg, seeds, policies)
    report = evaluation_report(results, seeds)
    if saved is not None:
        report.update(selected_iteration=saved['iteration'], selected_actor_updates=saved['actor_updates'])
    write_evaluation(args, checkpoint, weights, saved, report)


if __name__ == '__main__':
    main()
