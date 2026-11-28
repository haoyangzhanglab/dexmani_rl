import time

import torch

from dexmani_rl.envs.dexmani_sim import make_flowmatch_env
from dexmani_rl.evaluation.metrics import evaluation_report
from dexmani_rl.policies.flowmatch.loader import load_flowmatch_policy
from dexmani_rl.policies.flowmatch.rl_runtime import (
    FlowMatchRLRuntime,
    policy_dimensions,
)

from .._shared.cli import evaluation_inputs, evaluation_seeds, write_evaluation
from .algorithm import Actor


@torch.no_grad()
def rollout(env, base, seed, actor, noise_magnitude, deadline):
    if time.monotonic() >= deadline:
        return None
    obs = env.reset(seed)
    generator = torch.Generator(device=base.device).manual_seed(seed)
    steps, reward_sum = 0, 0.
    while time.monotonic() < deadline:
        context, state = base.encode(obs)
        if actor is None:
            w = torch.randn(1, base.noise_dim, device=base.device, generator=generator)
        else:
            w = actor.sample(state, generator)[0] * noise_magnitude
        normalized = base.actions(context, w)
        actions = base.physical(normalized)[0].cpu().numpy()
        obs, reward, done, truncated, info = env.step(actions)
        steps += info['steps']
        reward_sum += reward
        if done or truncated:
            return {'seed': seed, 'success': bool(info['success']), 'steps': steps, 'reward': reward_sum}
    return None


def evaluate(policy, cfg, seeds, policies, *, noise_magnitude=None, deadline=float('inf')):
    results: dict[str, list[dict]] = {name: [] for name in policies}
    for seed in seeds:
        group = {}
        for name, actor in policies.items():
            if time.monotonic() >= deadline:
                return results
            # PhysX retains scene history across resets, even with the same seed.
            with make_flowmatch_env(policy, cfg) as env:
                base = FlowMatchRLRuntime(policy, cfg, env.env.robot.qlimits)
                episode = rollout(env, base, seed, actor, noise_magnitude, deadline)
            if episode is None:
                return results
            group[name] = episode
        # Commit a seed only after every requested policy has completed it.
        for name in policies:
            results[name].append(group[name])
    return results


def evaluate_noise_policy(run, actor, has_updates, *, noise_magnitude=None):
    policies: dict[str, Actor | None] = {'baseline': None}
    checkpoint = None
    if has_updates:
        checkpoint, policies['steered'] = run.selected_actor(actor)
    results = evaluate(run.policy, run.policy_cfg, run.test, policies,
                       noise_magnitude=noise_magnitude, deadline=run.deadline)
    return dict(algorithm=run.metadata['algorithm'], rl_checkpoint=checkpoint, **evaluation_report(results, run.test))


def main():
    args, checkpoint, weights, saved = evaluation_inputs('dsrl')
    policy, cfg, weights = load_flowmatch_policy(checkpoint, args.device, weights)
    seeds = evaluation_seeds(saved, cfg, args.episodes)
    policies: dict[str, Actor | None] = {'baseline': None}
    noise_magnitude = None
    if saved is not None:
        actor_cfg = saved['config']
        state_dim, noise_dim, _ = policy_dimensions(policy)
        actor = Actor(state_dim, noise_dim, actor_cfg['width'], actor_cfg['depth'])
        noise_magnitude = actor_cfg['noise_magnitude']
        actor.to(args.device)
        actor.load_state_dict(saved['actor'])
        actor.eval()
        policies['steered'] = actor
    results = evaluate(policy, cfg, seeds, policies, noise_magnitude=noise_magnitude)
    report = evaluation_report(results, seeds)
    write_evaluation(args, checkpoint, weights, saved, report)


if __name__ == '__main__':
    main()
