import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from .base_policy import load_policy, FrozenPolicy
from .env import make_env, seed_split
from .dsrl import Actor


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
        actions = base.physical(base.actions(context, w))[0].cpu().numpy()
        obs, reward, done, truncated, info = env.step(actions)
        steps += info['steps']
        reward_sum += reward
        if done or truncated:
            return dict(seed=seed, success=bool(info['success']), steps=steps, reward=reward_sum)
    return None


def evaluate(policy, cfg, seeds, policies, *, noise_magnitude=None, deadline=float('inf')):
    results = {name: [] for name in policies}
    seeds_per_env = 25 // len(policies)
    for start in range(0, len(seeds), seeds_per_env):
        if time.monotonic() >= deadline:
            break
        with make_env(policy, cfg) as env:
            base = FrozenPolicy(policy, cfg, env.env.robot.qlimits)
            for seed in seeds[start:start + seeds_per_env]:
                group = {}
                for name, actor in policies.items():
                    episode = rollout(env, base, seed, actor, noise_magnitude, deadline)
                    if episode is None:
                        return results
                    group[name] = episode
                # Commit a seed only after every requested policy has completed it.
                for name in policies:
                    results[name].append(group[name])
    return results


def compare(baseline, steered):
    if [x['seed'] for x in baseline] != [x['seed'] for x in steered]:
        raise ValueError('Paired evaluation requires the same seeds in the same order')
    count = len(baseline)
    if not count:
        return {'episodes': 0}
    a = np.array([x['success'] for x in baseline], dtype=int)
    b = np.array([x['success'] for x in steered], dtype=int)
    diff = b - a
    # Delta = P(DSRL-only success) - P(BC-only success). Bonferroni combines
    # two 97.5% Clopper-Pearson intervals into a conservative 95% interval.
    wins = binomtest(int((diff == 1).sum()), count).proportion_ci(.975, method='exact')
    losses = binomtest(int((diff == -1).sum()), count).proportion_ci(.975, method='exact')
    ci = [wins.low - losses.high, wins.high - losses.low]
    return dict(episodes=count, baseline_successes=int(a.sum()),
                dsrl_successes=int(b.sum()), success_rate_delta=float(diff.mean()),
                paired_conservative_95_ci=ci, training_seeds=1)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--dsrl-checkpoint')
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.dsrl_checkpoint:
        if args.weights:
            parser.error('--dsrl-checkpoint uses the saved base policy and weight selection')
        saved = torch.load(args.dsrl_checkpoint, map_location='cpu', weights_only=False)
        checkpoint, weights = saved['base_checkpoint'], saved['weights']
    else:
        checkpoint, weights = str(Path(args.checkpoint).resolve()), args.weights
    policy, cfg, weights = load_policy(checkpoint, args.device, weights)
    seeds, _, _ = seed_split(cfg['env_runner']['task_name'])
    policies = {'baseline': None}
    noise_magnitude = None
    if args.dsrl_checkpoint:
        with make_env(policy, cfg) as env:
            base = FrozenPolicy(policy, cfg, env.env.robot.qlimits)
            actor = Actor(base.state_dim, base.noise_dim, saved['config']['width'], saved['config']['depth']).to(base.device)
            actor.load_state_dict(saved['actor'])
            actor.eval()
        policies['dsrl'] = actor
        noise_magnitude = saved['config']['noise_magnitude']
    results = evaluate(policy, cfg, seeds, policies, noise_magnitude=noise_magnitude)
    result = dict(checkpoint=checkpoint, weights=weights, **results)
    if args.dsrl_checkpoint:
        result['dsrl_checkpoint'] = str(Path(args.dsrl_checkpoint).resolve())
        result['comparison'] = compare(result['baseline'], result['dsrl'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result.get('comparison', {'episodes': len(result['baseline'])})))


if __name__ == '__main__':
    main()
