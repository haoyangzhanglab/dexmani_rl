import json
import time
from contextlib import closing
from pathlib import Path

import torch

from dexmani_rl.envs.dexmani_sim import make_flowmatch_env, training_episodes
from dexmani_rl.policies.flowmatch.rl_runtime import FlowMatchRLRuntime

from .._shared.cli import training_inputs, training_parser
from .._shared.flowmatch_online import FlowMatchOnlineRun
from .._shared.torch_checkpoint import restore_rng, save
from .algorithm import RFS
from .evaluate import evaluate, evaluate_noise_policy

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / 'configs' / 'methods' / 'rfs.yaml'


def main():
    parser = training_parser()
    args = parser.parse_args()
    cfg, checkpoint, weights, saved = training_inputs(parser, args, method='rfs', default_config=DEFAULT_CONFIG)
    if (any(type(cfg[k]) is not int or cfg[k] <= 0 for k in
            ('total_steps', 'rollout_steps', 'eval_interval_updates')) or
            cfg['rollout_steps'] < 2 or not 0 < cfg['train_hours'] <= cfg['max_hours']):
        raise ValueError('Invalid RFS collection or time budget')
    output = args.output
    run = FlowMatchOnlineRun('rfs', args, cfg, checkpoint, weights, saved)
    policy, policy_cfg = run.policy, run.policy_cfg
    with make_flowmatch_env(policy, policy_cfg) as env:
        base = FlowMatchRLRuntime(policy, policy_cfg, env.env.robot.qlimits)
        agent = RFS(base.state_dim, base.noise_dim, base.action_dim, cfg, base.device)
    steps, chunks, updates, last_eval_update, discarded_chunks = 0, 0, 0, 0, 0
    if saved is not None:
        agent.load_state_dict(saved['model'])
        agent.optimizer.load_state_dict(saved['optimizer'])
        steps, chunks, updates, last_eval_update, discarded_chunks = (
            saved[k] for k in ('steps', 'chunks', 'updates', 'last_eval_update', 'discarded_chunks'))
        restore_rng(saved, run.rng, base.device)
        del saved
    deadline, training_deadline = run.deadline, run.training_deadline
    run.open_output()

    def snapshot():
        return dict(run.snapshot_fields(base.device), model=agent.state_dict(), optimizer=agent.optimizer.state_dict(),
            steps=steps, chunks=chunks, updates=updates, last_eval_update=last_eval_update,
            discarded_chunks=discarded_chunks)

    def validate(end, phase, log):
        nonlocal last_eval_update
        results = evaluate(policy, policy_cfg, run.validation, {'steered': agent.actor}, deadline=end)['steered']
        if run.record_validation(results, agent.actor, phase, log, {'steps': steps, 'updates': updates}) is not None:
            last_eval_update = updates

    with (output / 'metrics.jsonl').open('a', buffering=1) as log, closing(
            training_episodes(policy, policy_cfg, run.rng, run.excluded, training_deadline, log)) as episodes:
        needs_reset = True
        while steps < cfg['total_steps'] and time.monotonic() < training_deadline:
            transitions: list[dict[str, torch.Tensor]] = []
            clip_fractions = []
            while (len(transitions) < cfg['rollout_steps'] and steps < cfg['total_steps'] and
                   time.monotonic() < training_deadline):
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
                    joint, log_prob = agent.actor.sample(state)
                    noise, residual = agent.actor.split(joint)
                    value = agent.value(state)
                    action, clip_fraction = base.actions(context, noise, residual, return_clip_fraction=True)
                obs, reward, terminated, truncated, info = env.step(base.physical(action)[0].cpu().numpy())
                next_context, next_state = base.encode(obs)
                with torch.no_grad():
                    next_value = agent.value(next_state)
                transitions.append({'state': state[0], 'joint': joint[0], 'log_prob': log_prob[0],
                                    'value': value[0], 'next_value': next_value[0],
                                    'reward': torch.tensor([reward], device=base.device),
                                    'terminated': torch.tensor([bool(terminated)], device=base.device),
                                    'truncated': torch.tensor([bool(truncated)], device=base.device)})
                clip_fractions.append(clip_fraction)
                steps += info['steps']
                chunks += 1
                episode_steps += info['steps']
                episode_reward += reward
                context, state = next_context, next_state
                needs_reset = terminated or truncated
                if needs_reset:
                    log.write(json.dumps({'episode_seed': seed, 'success': bool(info['success']),
                                          'episode_steps': episode_steps, 'episode_reward': episode_reward, 'steps': steps}) + '\n')
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
            if updates - last_eval_update >= cfg['eval_interval_updates'] and time.monotonic() < training_deadline:
                validate(training_deadline, 'scheduled', log)
            save(output / 'resume.pt', snapshot())
        episodes.close()
        if updates != last_eval_update and time.monotonic() < deadline:
            end = run.selection_deadline(2)
            validate(end, 'final', log)
        log.write(json.dumps({'steps': steps, 'chunks': chunks, 'updates': updates,
                              'discarded_chunks': discarded_chunks}) + '\n')
    save(output / 'last.pt', {**run.metadata, 'actor': agent.actor.state_dict()})
    save(output / 'resume.pt', snapshot())
    result = evaluate_noise_policy(run, agent.actor, updates > 0)
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    save(output / 'resume.pt', snapshot())
    print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
