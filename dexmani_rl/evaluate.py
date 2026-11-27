import argparse
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError
import numpy as np
import torch
from scipy.stats import binomtest
from .base_policy import load_policy, FrozenPolicy, policy_dimensions
from .env import make_env, seed_split
from .dsrl import Actor as DSRLActor
from .rfs import Actor as RFSActor
from .robofft import RoboFFTActor
from .bora import BORAActor


@dataclass(frozen=True)
class BORAStage:
    actor: BORAActor
    round_number: int


@torch.no_grad()
def rollout(env, base, seed, actor, noise_magnitude, deadline):
    if time.monotonic() >= deadline:
        return None
    obs = env.reset(seed)
    generator = torch.Generator(device=base.device).manual_seed(seed)
    steps, reward_sum = 0, 0.
    robofft = isinstance(actor, RoboFFTActor)
    normalized_clips, physical_clips = [], []
    while time.monotonic() < deadline:
        if isinstance(actor, BORAStage):
            window = actor.actor.normalized_actions(obs, generator=generator, stage=actor.round_number)
            actions = base.bounded_physical(window)[0].cpu().numpy()
        elif robofft:
            context, _ = base.encode(obs)
            normalized, fraction = actor.sample(context, generator, eval_mode=True, return_clip_fraction=True)
            physical, physical_fraction = base.robofft_actions(normalized)
            actions = physical[0].cpu().numpy()
            normalized_clips.append(fraction.item())
            physical_clips.append(physical_fraction.item())
        else:
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
            result = dict(seed=seed, success=bool(info['success']), steps=steps, reward=reward_sum)
            if robofft:
                result.update(normalized_clip_fraction=float(np.mean(normalized_clips)),
                              physical_clip_fraction=float(np.mean(physical_clips)))
            return result
    return None


def evaluate(policy, cfg, seeds, policies, *, noise_magnitude=None, deadline=float('inf')):
    results = {name: [] for name in policies}
    for seed in seeds:
        group = {}
        for name, actor in policies.items():
            if time.monotonic() >= deadline:
                return results
            # PhysX retains scene history across resets, even with the same seed.
            with make_env(policy, cfg) as env:
                base = FrozenPolicy(policy, cfg, env.env.robot.qlimits)
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
        actor_error = '--rl-checkpoint requires a saved RL actor'
        try:
            saved = torch.load(args.rl_checkpoint, map_location='cpu', weights_only=True)
        except UnpicklingError as error:
            raise ValueError(actor_error) from error
        if not isinstance(saved, dict) or 'algorithm' not in saved or 'actor' not in saved:
            raise ValueError(actor_error)
        algorithm = saved['algorithm']
        if algorithm not in ('dsrl', 'rfs', 'robofft', 'bora'):
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
    if algorithm == 'bora':
        bora = BORAActor(deepcopy(policy), cfg, saved['config'])
        bora.load_state_dict(saved['actor'])
        bora.eval()
        round_number = int(bora.round_number)
        if round_number:
            policies['offline'] = BORAStage(bora, 0)
        policies['steered'] = BORAStage(bora, round_number)
    elif args.rl_checkpoint:
        actor_cfg = saved['config']
        if algorithm == 'robofft':
            actor = RoboFFTActor(policy.action_decoder.model, policy.horizon, policy.action_dim,
                                 cfg['eval']['inference_steps'], actor_cfg['n_samples'], actor_cfg['noise_std'])
            policies['sampler_baseline'] = deepcopy(actor)
        else:
            state_dim, noise_dim, action_dim = policy_dimensions(policy)
            if algorithm == 'rfs':
                actor = RFSActor(state_dim, noise_dim, action_dim,
                                 actor_cfg['hidden_dims'], actor_cfg['residual_scale'])
            else:
                actor = DSRLActor(state_dim, noise_dim, actor_cfg['width'], actor_cfg['depth'])
                noise_magnitude = actor_cfg['noise_magnitude']
            actor.to(args.device)
        actor.load_state_dict(saved['actor'])
        actor.eval()
        policies['steered'] = actor
    results = evaluate(policy, cfg, seeds, policies, noise_magnitude=noise_magnitude)
    baseline, steered = results['baseline'], results['steered'] if args.rl_checkpoint else []
    result = dict(algorithm=algorithm, checkpoint=checkpoint, weights=weights,
                  rl_checkpoint=str(Path(args.rl_checkpoint).resolve()) if args.rl_checkpoint else None,
                  baseline=baseline, steered=steered,
                  comparison=compare(baseline, steered) if args.rl_checkpoint else {'episodes': 0})
    if algorithm == 'robofft':
        result.update(sampler_baseline=results['sampler_baseline'],
                      selected_iteration=saved['iteration'], selected_actor_updates=saved['actor_updates'],
                      comparison_vs_sampler=compare(results['sampler_baseline'], steered))
    if algorithm == 'bora':
        result['round'] = round_number
        if 'offline' in results:
            result.update(offline=results['offline'], comparison_offline_vs_bc=compare(baseline, results['offline']),
                          comparison_vs_offline=compare(results['offline'], steered))
    output.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result, indent=2)
    with output.open('x') as file:
        file.write(report)
    print(json.dumps(result['comparison'] if args.rl_checkpoint else {'episodes': len(baseline)}))


if __name__ == '__main__':
    main()
