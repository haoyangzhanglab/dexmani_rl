"""RoboFFT-F online finetuning; iteration numbering follows the pinned release."""
import copy
import json
import time
from contextlib import ExitStack

import numpy as np
import torch

from .base_policy import FrozenPolicy, load_policy
from .checkpoint import capture_rng, restore_rng, save
from .env import make_env, seed_split
from .evaluate import compare, evaluate
from .robofft import RoboFFT, RoboFFTActor
from .training import actor_snapshot, record_validation, selection_deadline, training_arguments


def main():
    args, cfg, checkpoint, weights, saved = training_arguments('robofft')
    if (type(cfg['rollout_steps']) is not int or cfg['rollout_steps'] < 2
            or not 0 < cfg['train_hours'] <= cfg['max_hours']):
        raise ValueError('Invalid RoboFFT collection or time budget')
    if type(cfg['eval_interval']) is not int or cfg['eval_interval'] < 2:
        raise ValueError('eval_interval must be at least 2 to leave iterations for training')

    output = args.output
    best_path = output / 'best.pt'
    started = time.monotonic()
    torch.manual_seed(cfg['seed'])
    policy, policy_cfg, weights = load_policy(checkpoint, args.device, weights)
    if saved is None:
        test_seeds, validation_seeds, excluded = seed_split(policy_cfg['env_runner']['task_name'])
    else:
        test_seeds, validation_seeds = saved['test_seeds'], saved['validation_seeds']
        excluded = set(saved['excluded_seeds'])
    with make_env(policy, policy_cfg) as env:
        base = FrozenPolicy(policy, policy_cfg, env.env.robot.qlimits)
    actor = RoboFFTActor(policy.action_decoder.model, policy.horizon, policy.action_dim,
                        policy_cfg['eval']['inference_steps'], cfg['n_samples'], cfg['noise_std'])
    agent = RoboFFT(base.state_dim, actor, cfg, base.device)
    iteration, steps, chunks, updates, actor_updates, discarded_chunks = 0, 0, 0, 0, 0, 0
    elapsed, best_score, best_actor = 0., -1., None
    best_iteration, best_actor_updates = None, 0
    last_validated_actor_updates = None
    seed_rng = np.random.default_rng(cfg['seed'])
    if saved is not None:
        agent.load_state_dict(saved['model'])
        for name, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(saved['optimizers'][name])
        agent.actor_scheduler.load_state_dict(saved['schedulers']['actor'])
        agent.reward_scaler.load_state_dict(saved['reward_scaler'])
        iteration, steps, chunks, updates, actor_updates, discarded_chunks = (
            saved[k] for k in ('iteration', 'steps', 'chunks', 'updates', 'actor_updates', 'discarded_chunks'))
        elapsed, best_score, best_actor = (saved[k] for k in ('elapsed', 'best_score', 'best_actor'))
        best_iteration, best_actor_updates = saved['best_iteration'], saved['best_actor_updates']
        last_validated_actor_updates = saved['last_validated_actor_updates']
        restore_rng(saved, seed_rng, base.device)
        del saved
    deadline = started + max(0, cfg['max_hours'] * 3600 - elapsed)
    training_deadline = started + max(0, cfg['train_hours'] * 3600 - elapsed)
    metadata = dict(algorithm='robofft', config=cfg,
                    base_checkpoint=checkpoint, weights=weights, torch=str(torch.__version__),
                    test_seeds=test_seeds, validation_seeds=validation_seeds,
                    excluded_seeds=sorted(excluded))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(dict(metadata, policy_config=policy_cfg), indent=2))

    def snapshot():
        return dict(metadata, model=agent.state_dict(),
                    optimizers={k: v.state_dict() for k, v in agent.optimizers.items()},
                    schedulers={'actor': agent.actor_scheduler.state_dict()},
                    reward_scaler=agent.reward_scaler.state_dict(), iteration=iteration,
                    steps=steps, chunks=chunks, updates=updates, actor_updates=actor_updates,
                    discarded_chunks=discarded_chunks, elapsed=elapsed + time.monotonic() - started,
                    best_score=best_score, best_actor=best_actor,
                    best_iteration=best_iteration, best_actor_updates=best_actor_updates,
                    last_validated_actor_updates=last_validated_actor_updates,
                    **capture_rng(seed_rng, base.device))

    def validate(candidate_iteration, end, phase, log):
        nonlocal best_score, best_actor, best_iteration, best_actor_updates, last_validated_actor_updates
        results = evaluate(policy, policy_cfg, validation_seeds, {'steered': agent.actor}, deadline=end)['steered']
        score = record_validation(results, len(validation_seeds), phase, log, iteration=candidate_iteration,
                                  steps=steps, actor_updates=actor_updates)
        if score is None:
            return False
        last_validated_actor_updates = actor_updates
        if score > best_score:
            best_score = score
            best_iteration, best_actor_updates = candidate_iteration, actor_updates
            best_actor = actor_snapshot(agent.actor)
            save(best_path, {**metadata, 'actor': best_actor,
                             'iteration': best_iteration, 'actor_updates': best_actor_updates})
        return True

    from dexmani_sim import EpisodeRejected
    with (output / 'metrics.jsonl').open('a', buffering=1) as log, ExitStack() as environments:
        needs_reset, reset_attempts = True, 25
        while iteration < cfg['iterations'] and time.monotonic() < training_deadline:
            # The release evaluates at iteration zero and then every val_freq iterations.
            if iteration % cfg['eval_interval'] == 0:
                environments.close()
                needs_reset, reset_attempts = True, 25
                if (actor_updates != last_validated_actor_updates and
                        not validate(iteration, training_deadline, 'scheduled', log)):
                    break
            else:
                transitions, normalized_clips, physical_clips = [], [], []
                while len(transitions) < cfg['rollout_steps'] and time.monotonic() < training_deadline:
                    first = needs_reset
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
                        action, normalized_clip = agent.actor.sample(context, return_clip_fraction=True)
                        eps, t = agent.actor.draw_matching(action)
                        old_log_prob = agent.actor.log_prob(context, action, eps, t)
                        value = agent.value(state)
                        physical, physical_clip = base.robofft_actions(action)
                    obs, reward, terminated, truncated, info = env.step(physical[0].cpu().numpy())
                    next_context, next_state = base.encode(obs)
                    with torch.no_grad():
                        next_value = agent.value(next_state)
                    row = dict(context=context[0], state=state[0], action=action[0], eps=eps[0],
                               t=t[0], old_log_prob=old_log_prob[0], value=value[0], next_value=next_value[0])
                    row = {key: value.detach().cpu() for key, value in row.items()}
                    row.update(reward=torch.tensor([reward]), terminated=torch.tensor([bool(terminated)]),
                               truncated=torch.tensor([bool(truncated)]), first=torch.tensor([first]))
                    transitions.append(row)
                    normalized_clips.append(float(normalized_clip))
                    physical_clips.append(float(physical_clip))
                    steps += info['steps']
                    chunks += 1
                    episode_steps += info['steps']
                    episode_reward += reward
                    context, state = next_context, next_state
                    needs_reset = terminated or truncated
                    if needs_reset:
                        log.write(json.dumps(dict(episode_seed=seed, success=bool(info['success']),
                            episode_steps=episode_steps, episode_reward=episode_reward, steps=steps)) + '\n')
                if not transitions or time.monotonic() >= training_deadline:
                    discarded_chunks += len(transitions)
                    break
                batch = {key: torch.stack([row[key] for row in transitions]) for key in transitions[0]}
                metrics = agent.update(batch, iteration)
                updates += 1
                actor_updates += int(metrics['actor_steps'])
                log.write(json.dumps(dict(iteration=iteration, steps=steps, chunks=chunks, updates=updates,
                    rollout_chunks=len(transitions), normalized_clip_fraction=float(np.mean(normalized_clips)),
                    physical_clip_fraction=float(np.mean(physical_clips)), **metrics)) + '\n')
                del batch, transitions
            agent.finish_iteration(iteration)
            iteration += 1
            save(output / 'resume.pt', snapshot())

        environments.close()
        if actor_updates > 0 and actor_updates != last_validated_actor_updates and time.monotonic() < deadline:
            end = selection_deadline(training_deadline, deadline, time.monotonic(),
                                     len(validation_seeds), 3 * len(test_seeds))
            validate(iteration - 1, end, 'final', log)

    save(output / 'last.pt', {**metadata, 'actor': agent.actor.state_dict(),
                             'iteration': iteration - 1, 'actor_updates': actor_updates})
    save(output / 'resume.pt', snapshot())
    selected_actor = agent.actor
    if best_actor is not None:
        if not best_path.exists():
            save(best_path, {**metadata, 'actor': best_actor,
                             'iteration': best_iteration, 'actor_updates': best_actor_updates})
        selected_actor = copy.deepcopy(agent.actor)
        selected_actor.load_state_dict(best_actor)
    sampler_baseline = RoboFFTActor(policy.action_decoder.model, policy.horizon, policy.action_dim,
                                   policy_cfg['eval']['inference_steps'], cfg['n_samples'], cfg['noise_std'])
    results = evaluate(policy, policy_cfg, test_seeds,
                       {'baseline': None, 'sampler_baseline': sampler_baseline, 'steered': selected_actor},
                       deadline=deadline)
    result = dict(algorithm='robofft', rl_checkpoint='best.pt' if best_actor is not None else 'last.pt',
                  actor_updates=actor_updates,
                  selected_iteration=best_iteration if best_actor is not None else iteration - 1,
                  selected_actor_updates=best_actor_updates if best_actor is not None else actor_updates, **results,
                  comparison=compare(results['baseline'], results['steered']),
                  comparison_vs_sampler=compare(results['sampler_baseline'], results['steered']))
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    save(output / 'resume.pt', snapshot())
    print(json.dumps(result['comparison_vs_sampler']))


if __name__ == '__main__':
    main()
