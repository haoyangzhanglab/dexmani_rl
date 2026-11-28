"""RoboFFT-F online finetuning; iteration numbering follows the pinned release."""
import json
import time
from contextlib import ExitStack, closing

import numpy as np
import torch

from .common.base_policy import FrozenPolicy
from .common.checkpoint import restore_rng, save
from .common.env import make_env, training_episodes
from .common.training import OnlineRun, training_arguments
from .evaluate import evaluate, evaluation_report
from .robofft import RoboFFT, RoboFFTActor


def main():
    args, cfg, checkpoint, weights, saved = training_arguments('robofft')
    if (type(cfg['rollout_steps']) is not int or cfg['rollout_steps'] < 2
            or not 0 < cfg['train_hours'] <= cfg['max_hours']):
        raise ValueError('Invalid RoboFFT collection or time budget')
    if type(cfg['eval_interval']) is not int or cfg['eval_interval'] < 2:
        raise ValueError('eval_interval must be at least 2 to leave iterations for training')

    output = args.output
    run = OnlineRun('robofft', args, cfg, checkpoint, weights, saved)
    policy, policy_cfg = run.policy, run.policy_cfg
    with make_env(policy, policy_cfg) as env:
        base = FrozenPolicy(policy, policy_cfg, env.env.robot.qlimits)
    actor = RoboFFTActor.from_policy(policy, policy_cfg, cfg)
    agent = RoboFFT(base.state_dim, actor, cfg, base.device)
    iteration, steps, chunks, updates, actor_updates, discarded_chunks = 0, 0, 0, 0, 0, 0
    best_iteration, best_actor_updates = None, 0
    last_validated_actor_updates = None
    if saved is not None:
        agent.load_state_dict(saved['model'])
        for name, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(saved['optimizers'][name])
        agent.actor_scheduler.load_state_dict(saved['schedulers']['actor'])
        agent.reward_scaler.load_state_dict(saved['reward_scaler'])
        iteration, steps, chunks, updates, actor_updates, discarded_chunks = (
            saved[k] for k in ('iteration', 'steps', 'chunks', 'updates', 'actor_updates', 'discarded_chunks'))
        best_iteration, best_actor_updates = saved['best_iteration'], saved['best_actor_updates']
        last_validated_actor_updates = saved['last_validated_actor_updates']
        restore_rng(saved, run.rng, base.device)
        del saved
    deadline, training_deadline = run.deadline, run.training_deadline
    run.open_output()

    def snapshot():
        return dict(run.snapshot_fields(base.device), model=agent.state_dict(),
                    optimizers={k: v.state_dict() for k, v in agent.optimizers.items()},
                    schedulers={'actor': agent.actor_scheduler.state_dict()},
                    reward_scaler=agent.reward_scaler.state_dict(), iteration=iteration,
                    steps=steps, chunks=chunks, updates=updates, actor_updates=actor_updates,
                    discarded_chunks=discarded_chunks, best_iteration=best_iteration, best_actor_updates=best_actor_updates,
                    last_validated_actor_updates=last_validated_actor_updates)

    def validate(candidate_iteration, end, phase, log):
        nonlocal best_iteration, best_actor_updates, last_validated_actor_updates
        results = evaluate(policy, policy_cfg, run.validation, {'steered': agent.actor}, deadline=end)['steered']
        previous = run.best_score
        progress = {'iteration': candidate_iteration, 'steps': steps, 'actor_updates': actor_updates}
        score = run.record_validation(results, agent.actor, phase, log, progress,
                                      iteration=candidate_iteration, actor_updates=actor_updates)
        if score is not None:
            last_validated_actor_updates = actor_updates
            if score > previous:
                best_iteration, best_actor_updates = candidate_iteration, actor_updates
        return score is not None

    with (output / 'metrics.jsonl').open('a', buffering=1) as log, ExitStack() as environments:
        episodes = None
        while iteration < cfg['iterations'] and time.monotonic() < training_deadline:
            # The release evaluates at iteration zero and then every val_freq iterations.
            if iteration % cfg['eval_interval'] == 0:
                environments.close()
                episodes = None
                if (actor_updates != last_validated_actor_updates and
                        not validate(iteration, training_deadline, 'scheduled', log)):
                    break
            else:
                if episodes is None:
                    episodes = environments.enter_context(closing(
                        training_episodes(policy, policy_cfg, run.rng, run.excluded, training_deadline, log)))
                    needs_reset = True
                transitions: list[dict[str, torch.Tensor]] = []
                normalized_clips, physical_clips = [], []
                while len(transitions) < cfg['rollout_steps'] and time.monotonic() < training_deadline:
                    first = needs_reset
                    if needs_reset:
                        episode = next(episodes, None)
                        if episode is None:
                            break
                        env, seed, obs = episode
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
                    row = {'context': context[0], 'state': state[0], 'action': action[0], 'eps': eps[0],
                           't': t[0], 'old_log_prob': old_log_prob[0], 'value': value[0], 'next_value': next_value[0]}
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
                        log.write(json.dumps({'episode_seed': seed, 'success': bool(info['success']),
                                              'episode_steps': episode_steps, 'episode_reward': episode_reward, 'steps': steps}) + '\n')
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
            end = run.selection_deadline(3)
            validate(iteration - 1, end, 'final', log)

    save(output / 'last.pt', {**run.metadata, 'actor': agent.actor.state_dict(),
                             'iteration': iteration - 1, 'actor_updates': actor_updates})
    save(output / 'resume.pt', snapshot())
    checkpoint_name, selected_actor = run.selected_actor(
        agent.actor, iteration=best_iteration, actor_updates=best_actor_updates)
    sampler_baseline = RoboFFTActor.from_policy(policy, policy_cfg, cfg)
    results = evaluate(policy, policy_cfg, run.test,
                       {'baseline': None, 'sampler_baseline': sampler_baseline, 'steered': selected_actor},
                       deadline=deadline)
    result = dict(algorithm='robofft', rl_checkpoint=checkpoint_name,
                  actor_updates=actor_updates,
                  selected_iteration=best_iteration if best_iteration is not None else iteration - 1,
                  selected_actor_updates=best_actor_updates if best_iteration is not None else actor_updates,
                  **evaluation_report(results, run.test))
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    save(output / 'resume.pt', snapshot())
    print(json.dumps(result['comparison_vs_sampler'] if result['evaluation_complete'] else result['comparison']))


if __name__ == '__main__':
    main()
