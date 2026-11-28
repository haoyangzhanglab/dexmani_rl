import json
import time

import torch

from .common.base_policy import FrozenPolicy
from .common.checkpoint import restore_rng, save
from .common.env import make_env
from .common.training import OnlineRun, training_arguments
from .dsrl import DSRL
from .evaluate import evaluate, evaluate_noise_policy
from .replay import Replay


def main():
    args, cfg, checkpoint, weights, saved = training_arguments('dsrl')
    counts = ('total_steps', 'replay_capacity', 'decode_batch', 'eval_interval_chunks')
    if (any(type(cfg[key]) is not int or cfg[key] <= 0 for key in counts)
            or type(cfg['warmup_steps']) is not int
            or not 0 <= cfg['warmup_steps'] <= cfg['total_steps']
            or not 0 < cfg['train_hours'] <= cfg['max_hours']):
        raise ValueError('Invalid DSRL collection or time budget')
    output = args.output
    run = OnlineRun('dsrl', args, cfg, checkpoint, weights, saved)
    policy, policy_cfg = run.policy, run.policy_cfg
    replay = Replay(cfg['replay_capacity'])
    steps, chunks, actor_update_chunk, last_eval_chunk = 0, 0, 0, 0
    with make_env(policy, policy_cfg) as env:
        base = FrozenPolicy(policy, policy_cfg, env.env.robot.qlimits, cfg['decode_batch'])
        agent = DSRL(base, cfg)
    if saved is not None:
        agent.load_state_dict(saved['model'])
        for name, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(saved['optimizers'][name])
        replay.load_state_dict(saved['replay'])
        steps, chunks, actor_update_chunk, last_eval_chunk = (
            saved[k] for k in ('steps', 'chunks', 'actor_update_chunk', 'last_eval_chunk'))
        restore_rng(saved, run.rng, base.device)
        del saved
    deadline, training_deadline = run.deadline, run.training_deadline
    run.open_output()

    def snapshot():
        return dict(run.snapshot_fields(base.device), model=agent.state_dict(),
            optimizers={name: optimizer.state_dict() for name, optimizer in agent.optimizers.items()}, replay=replay.state_dict(),
            steps=steps, chunks=chunks, actor_update_chunk=actor_update_chunk, last_eval_chunk=last_eval_chunk)

    def validate(end, phase, log):
        nonlocal last_eval_chunk
        results = evaluate(policy, policy_cfg, run.validation, {'steered': agent.actor},
                           noise_magnitude=cfg['noise_magnitude'], deadline=end)['steered']
        if run.record_validation(results, agent.actor, phase, log, {'steps': steps, 'chunks': chunks}) is not None:
            last_eval_chunk = actor_update_chunk

    from dexmani_sim import EpisodeRejected
    with (output / 'metrics.jsonl').open('a', buffering=1) as log:
        while steps < cfg['total_steps'] and time.monotonic() < training_deadline:
            with make_env(policy, policy_cfg) as env:
                for _ in range(25):
                    if steps >= cfg['total_steps'] or time.monotonic() >= training_deadline:
                        break
                    seed = int(run.rng.integers(2**32))
                    if seed in run.excluded:
                        continue
                    try:
                        obs = env.reset(seed)
                    except EpisodeRejected as error:
                        log.write(json.dumps({'rejected_seed': seed, 'error': str(error)}) + '\n')
                        continue
                    context, state = base.encode(obs)
                    while steps < cfg['total_steps'] and time.monotonic() < training_deadline:
                        with torch.no_grad():
                            w = (torch.randn(1, base.noise_dim, device=base.device) if steps < cfg['warmup_steps']
                                 else agent.actor.sample(state)[0] * cfg['noise_magnitude'])
                            action = base.actions(context, w)
                        obs, reward, done, truncated, info = env.step(base.physical(action)[0].cpu().numpy())
                        next_context, next_state = base.encode(obs)
                        replay.add(context=context[0], state=state[0], action=action[0],
                            next_context=next_context[0], next_state=next_state[0], reward=[reward],
                            terminated=[float(done)])
                        steps += info['steps']
                        chunks += 1
                        metrics = {}
                        if steps >= cfg['warmup_steps'] and time.monotonic() < training_deadline:
                            metrics = agent.update(replay)
                            actor_update_chunk = chunks
                        log.write(json.dumps(dict(steps=steps, chunks=chunks, seed=seed, reward=reward, **metrics)) + '\n')
                        context, state = next_context, next_state
                        if done or truncated:
                            break
                    if (actor_update_chunk - last_eval_chunk >= cfg['eval_interval_chunks']
                            and time.monotonic() < training_deadline):
                        validate(training_deadline, 'scheduled', log)
            if steps < cfg['total_steps'] and time.monotonic() < training_deadline:
                save(output / 'resume.pt', snapshot())
        if actor_update_chunk != last_eval_chunk and time.monotonic() < deadline:
            end = run.selection_deadline(2)
            validate(end, 'final', log)
    save(output / 'last.pt', {**run.metadata, 'actor': agent.actor.state_dict()})
    save(output / 'resume.pt', snapshot())
    result = evaluate_noise_policy(run, agent.actor, actor_update_chunk > 0,
                                   noise_magnitude=cfg['noise_magnitude'])
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
