import argparse
import copy
import json
import time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from .base_policy import load_policy, FrozenPolicy
from .env import make_env, seed_split
from .evaluate import evaluate, compare
from .rfs import RFS
from .checkpoint import save, capture_rng, restore_rng


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--resume')
    parser.add_argument('--config')
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    saved = None
    if args.resume:
        if args.config or args.weights:
            parser.error('--resume uses the saved config and weight selection')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['algorithm'] != 'rfs':
            raise ValueError('Requires an RFS-PPO resume checkpoint')
        cfg, checkpoint, weights = (saved[k] for k in ('config', 'base_checkpoint', 'weights'))
    else:
        cfg = OmegaConf.to_container(OmegaConf.load(args.config or 'configs/rfs.yaml'), resolve=True)
        checkpoint, weights = str(Path(args.checkpoint).resolve()), args.weights
    if (any(type(cfg[k]) is not int or cfg[k] <= 0 for k in
            ('total_steps', 'rollout_steps', 'eval_interval_updates', 'decode_batch')) or
            cfg['rollout_steps'] < 2 or not 0 < cfg['train_hours'] <= cfg['max_hours']):
        raise ValueError('Invalid RFS collection or time budget')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(cfg['seed'])
    policy, policy_cfg, weights = load_policy(checkpoint, args.device, weights)
    if saved is None:
        test_seeds, validation_seeds, excluded = seed_split(policy_cfg['env_runner']['task_name'])
    else:
        test_seeds, validation_seeds = saved['test_seeds'], saved['validation_seeds']
        excluded = set(saved['excluded_seeds'])
    with make_env(policy, policy_cfg) as env:
        base = FrozenPolicy(policy, policy_cfg, env.env.robot.qlimits, cfg['decode_batch'])
        agent = RFS(base.state_dim, base.noise_dim, base.action_dim, cfg, base.device)
    steps, chunks, updates, last_eval_update, discarded_chunks = 0, 0, 0, 0, 0
    elapsed, best_score, best_actor = 0., -1., None
    seed_rng = np.random.default_rng(cfg['seed'])
    if saved is not None:
        agent.load_state_dict(saved['model'])
        agent.optimizer.load_state_dict(saved['optimizer'])
        steps, chunks, updates, last_eval_update, discarded_chunks = (
            saved[k] for k in ('steps', 'chunks', 'updates', 'last_eval_update', 'discarded_chunks'))
        elapsed, best_score, best_actor = (saved[k] for k in ('elapsed', 'best_score', 'best_actor'))
        restore_rng(saved, seed_rng, base.device)
        del saved
    deadline = started + max(0, cfg['max_hours'] * 3600 - elapsed)
    training_deadline = started + max(0, cfg['train_hours'] * 3600 - elapsed)
    metadata = dict(algorithm='rfs', config=cfg, base_checkpoint=checkpoint, weights=weights,
                    torch=str(torch.__version__),
                    test_seeds=test_seeds, validation_seeds=validation_seeds,
                    excluded_seeds=sorted(excluded))
    (output / 'config.json').write_text(json.dumps(dict(metadata, policy_config=policy_cfg), indent=2))

    def snapshot():
        return dict(metadata, model=agent.state_dict(), optimizer=agent.optimizer.state_dict(),
            steps=steps, chunks=chunks, updates=updates, last_eval_update=last_eval_update,
            discarded_chunks=discarded_chunks, elapsed=elapsed + time.monotonic() - started,
            best_score=best_score, best_actor=best_actor,
            **capture_rng(seed_rng, base.device))

    from dexmani_sim import EpisodeRejected
    with (output / 'metrics.jsonl').open('a', buffering=1) as log, ExitStack() as environments:
        needs_reset, reset_attempts = True, 25
        while steps < cfg['total_steps'] and time.monotonic() < training_deadline:
            transitions, clip_fractions = [], []
            while (len(transitions) < cfg['rollout_steps'] and steps < cfg['total_steps'] and
                   time.monotonic() < training_deadline):
                if needs_reset:
                    if reset_attempts == 25:
                        environments.close()
                        env = environments.enter_context(make_env(policy, policy_cfg))
                        reset_attempts = 0
                    seed = int(seed_rng.integers(2**32))
                    if seed in excluded:
                        continue
                    reset_attempts += 1
                    try:
                        obs = env.reset(seed)
                    except EpisodeRejected as error:
                        log.write(json.dumps(dict(rejected_seed=seed, error=str(error))) + '\n')
                        continue
                    context, state = base.encode(obs)
                    episode_steps, episode_reward = 0, 0.
                    needs_reset = False
                if time.monotonic() >= training_deadline:
                    break
                with torch.no_grad():
                    joint, log_prob = agent.actor.sample(state)
                    noise, residual = agent.actor.split(joint)
                    value = agent.value(state)
                    action, clip_fraction = base.actions(context, noise, residual, return_clip_fraction=True)
                obs, reward, terminated, truncated, info = env.step(base.physical(action)[0].cpu().numpy())
                next_context, next_state = base.encode(obs)
                with torch.no_grad():
                    next_value = agent.value(next_state)
                transitions.append(dict(state=state[0], joint=joint[0], log_prob=log_prob[0],
                    value=value[0], next_value=next_value[0],
                    reward=torch.tensor([reward], device=base.device),
                    terminated=torch.tensor([bool(terminated)], device=base.device),
                    truncated=torch.tensor([bool(truncated)], device=base.device)))
                clip_fractions.append(clip_fraction)
                steps += info['steps']
                chunks += 1
                episode_steps += info['steps']
                episode_reward += reward
                context, state = next_context, next_state
                needs_reset = terminated or truncated
                if needs_reset:
                    log.write(json.dumps(dict(episode_seed=seed, success=bool(info['success']),
                        episode_steps=episode_steps, episode_reward=episode_reward, steps=steps)) + '\n')
            # Pending on-policy samples are never carried across a resume or time limit.
            if not transitions or time.monotonic() >= training_deadline:
                discarded_chunks += len(transitions)
                break
            batch = {key: torch.stack([row[key] for row in transitions]) for key in transitions[0]}
            metrics = agent.update(batch)
            updates += 1
            log.write(json.dumps(dict(steps=steps, chunks=chunks, updates=updates,
                rollout_chunks=len(transitions), action_clip_fraction=torch.stack(clip_fractions).mean().item(),
                **metrics)) + '\n')
            if updates - last_eval_update >= cfg['eval_interval_updates']:
                results = evaluate(policy, policy_cfg, validation_seeds, {'steered': agent.actor},
                                   deadline=training_deadline)['steered']
                if len(results) == len(validation_seeds):
                    last_eval_update = updates
                    score = float(np.mean([row['success'] for row in results]))
                    log.write(json.dumps(dict(validation=results, success_rate=score, steps=steps)) + '\n')
                    if score > best_score:
                        best_score = score
                        best_actor = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
                        save(output / 'best.pt', {**metadata, 'actor': best_actor})
            save(output / 'resume.pt', snapshot())
        log.write(json.dumps(dict(steps=steps, chunks=chunks, updates=updates,
                                  discarded_chunks=discarded_chunks)) + '\n')
    save(output / 'last.pt', {**metadata, 'actor': agent.actor.state_dict()})
    save(output / 'resume.pt', snapshot())
    actor = agent.actor
    if best_actor is not None:
        save(output / 'best.pt', {**metadata, 'actor': best_actor})
        actor = copy.deepcopy(agent.actor)
        actor.load_state_dict(best_actor)
    policies = {'baseline': None}
    if updates:
        policies['steered'] = actor
    results = evaluate(policy, policy_cfg, test_seeds, policies, deadline=deadline)
    baseline, steered = results['baseline'], results['steered'] if updates else []
    evaluated_checkpoint = ('best.pt' if best_actor is not None else 'last.pt') if updates else None
    result = dict(algorithm='rfs', rl_checkpoint=evaluated_checkpoint, baseline=baseline, steered=steered,
                  comparison=compare(baseline, steered) if updates else {'episodes': 0})
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    save(output / 'resume.pt', snapshot())
    print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
