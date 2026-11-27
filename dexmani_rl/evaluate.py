import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from .base_policy import load_policy, FrozenPolicy, policy_dimensions
from .env import make_env, seed_split
from .dsrl import Actor as DSRLActor
from .rfs import Actor as RFSActor


@torch.no_grad()
def rollout(env, base, seed, actor, noise_magnitude, deadline):
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
        elif isinstance(actor, RFSActor):
            w, residual = actor.split(actor.sample(state, generator)[0])
        else:
            w = actor.sample(state, generator)[0] * noise_magnitude
        normalized = base.actions(context, w, residual)
        actions = base.physical(normalized)[0].cpu().numpy()
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
    # Delta = P(steered-only success) - P(BC-only success). Bonferroni combines
    # two 97.5% Clopper-Pearson intervals into a conservative 95% interval.
    wins = binomtest(int((diff == 1).sum()), count).proportion_ci(.975, method='exact')
    losses = binomtest(int((diff == -1).sum()), count).proportion_ci(.975, method='exact')
    ci = [wins.low - losses.high, wins.high - losses.low]
    return dict(episodes=count, baseline_successes=int(a.sum()),
                steered_successes=int(b.sum()), success_rate_delta=float(diff.mean()),
                paired_conservative_95_ci=ci, training_seeds=1)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--rl-checkpoint')
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, help='default: full test pool (100)')
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error('--episodes must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    algorithm = 'bc'
    if args.rl_checkpoint:
        if args.weights:
            parser.error('--rl-checkpoint uses the saved base policy and weight selection')
        saved = torch.load(args.rl_checkpoint, map_location='cpu', weights_only=False)
        algorithm = saved['algorithm']
        if algorithm not in ('dsrl', 'rfs'):
            raise ValueError(f'Unsupported RL algorithm: {algorithm!r}')
        checkpoint, weights = saved['base_checkpoint'], saved['weights']
    else:
        checkpoint, weights = str(Path(args.checkpoint).resolve()), args.weights
    policy, cfg, weights = load_policy(checkpoint, args.device, weights)
    if args.rl_checkpoint:
        seeds = saved['test_seeds']
    else:
        seeds, _, _ = seed_split(cfg['env_runner']['task_name'])
    if args.episodes is not None:
        if args.episodes > len(seeds):
            parser.error(f'--episodes must not exceed the test seed count ({len(seeds)})')
        seeds = seeds[:args.episodes]
    policies = {'baseline': None}
    noise_magnitude = None
    if args.rl_checkpoint:
        state_dim, noise_dim, action_dim = policy_dimensions(policy)
        if algorithm == 'rfs':
            actor = RFSActor(state_dim, noise_dim, action_dim,
                             saved['config']['hidden_dims'], saved['config']['residual_scale']).to(args.device)
        else:
            actor = DSRLActor(state_dim, noise_dim, saved['config']['width'], saved['config']['depth']).to(args.device)
            noise_magnitude = saved['config']['noise_magnitude']
        actor.load_state_dict(saved['actor'])
        actor.eval()
        policies['steered'] = actor
    results = evaluate(policy, cfg, seeds, policies, noise_magnitude=noise_magnitude)
    baseline, steered = results['baseline'], results['steered'] if args.rl_checkpoint else []
    result = dict(algorithm=algorithm, checkpoint=checkpoint, weights=weights,
                  rl_checkpoint=str(Path(args.rl_checkpoint).resolve()) if args.rl_checkpoint else None,
                  baseline=baseline, steered=steered,
                  comparison=compare(baseline, steered) if args.rl_checkpoint else {'episodes': 0})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['comparison'] if args.rl_checkpoint else {'episodes': len(baseline)}))


if __name__ == '__main__':
    main()
