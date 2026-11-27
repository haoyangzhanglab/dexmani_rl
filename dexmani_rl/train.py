import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from .base_policy import load_policy, FrozenPolicy
from .dsrl import DSRL
from .replay import Replay
from .env import make_env, seed_split
from .evaluate import evaluate, compare


def save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


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
        cfg, checkpoint, weights = (saved[k] for k in ('config', 'base_checkpoint', 'weights'))
    else:
        cfg = OmegaConf.to_container(OmegaConf.load(args.config or 'configs/dsrl.yaml'), resolve=True)
        checkpoint, weights = str(Path(args.checkpoint).resolve()), args.weights
    if not (0 <= cfg['warmup_steps'] <= cfg['total_steps'] and
            0 < cfg['train_hours'] <= cfg['max_hours'] and
            0 < cfg['gamma'] <= 1 and 0 < cfg['tau'] <= 1 and cfg['noise_magnitude'] > 0):
        raise ValueError('Invalid training budget or algorithm hyperparameters')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(cfg['seed'])
    policy, policy_cfg, weights = load_policy(checkpoint, args.device, weights)
    test_seeds, validation_seeds, excluded = seed_split(policy_cfg['env_runner']['task_name'])
    replay = Replay(cfg['replay_capacity'])
    steps, chunks, last_eval_chunk, elapsed, best_score = 0, 0, 0, 0., -1.
    best_actor = None
    seed_rng = np.random.default_rng(cfg['seed'])
    with make_env(policy, policy_cfg) as env:
        base = FrozenPolicy(policy, policy_cfg, env.env.robot.qlimits, cfg['decode_batch'])
        agent = DSRL(base, cfg)
    if saved is not None:
        agent.load_state_dict(saved['model'])
        for optimizer, state in zip(agent.optimizers, saved['optimizers']):
            optimizer.load_state_dict(state)
        replay.load_state_dict(saved['replay'])
        steps, chunks, last_eval_chunk, elapsed = (saved[k] for k in ('steps', 'chunks', 'last_eval_chunk', 'elapsed'))
        best_score = saved['best_score']
        best_actor = saved['best_actor']
        torch.set_rng_state(saved['torch_rng'])
        if base.device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'], base.device)
        seed_rng.bit_generator.state = saved['seed_rng']
        del saved
    deadline = started + max(0, cfg['max_hours'] * 3600 - elapsed)
    training_deadline = started + max(0, cfg['train_hours'] * 3600 - elapsed)
    metadata = dict(config=cfg, base_checkpoint=checkpoint, weights=weights,
                    policy_config=policy_cfg, torch=torch.__version__,
                    test_seeds=test_seeds, validation_seeds=validation_seeds)
    (output / 'config.json').write_text(json.dumps(metadata, indent=2))

    def snapshot():
        return dict(metadata, model=agent.state_dict(),
            optimizers=[o.state_dict() for o in agent.optimizers], replay=replay.state_dict(),
            steps=steps, chunks=chunks, last_eval_chunk=last_eval_chunk,
            elapsed=elapsed + time.monotonic() - started, best_score=best_score, best_actor=best_actor,
            seed_rng=seed_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(base.device) if base.device.type == 'cuda' else None)

    from dexmani_sim import EpisodeRejected
    with (output / 'metrics.jsonl').open('a', buffering=1) as log:
        while steps < cfg['total_steps'] and time.monotonic() < training_deadline:
            with make_env(policy, policy_cfg) as env:
                for _ in range(25):
                    if steps >= cfg['total_steps'] or time.monotonic() >= training_deadline:
                        break
                    seed = int(seed_rng.integers(2**32))
                    if seed in excluded:
                        continue
                    try:
                        obs = env.reset(seed)
                    except EpisodeRejected as error:
                        log.write(json.dumps(dict(rejected_seed=seed, error=str(error))) + '\n')
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
                        metrics = agent.update(replay) if steps >= cfg['warmup_steps'] else {}
                        log.write(json.dumps(dict(steps=steps, chunks=chunks, seed=seed, reward=reward, **metrics)) + '\n')
                        context, state = next_context, next_state
                        if done or truncated:
                            break
                    if steps >= cfg['warmup_steps'] and chunks - last_eval_chunk >= cfg['eval_interval']:
                        results = evaluate(policy, policy_cfg, validation_seeds, {'dsrl': agent.actor},
                            noise_magnitude=cfg['noise_magnitude'], deadline=training_deadline)['dsrl']
                        if len(results) == len(validation_seeds):
                            last_eval_chunk = chunks
                            score = np.mean([r['success'] for r in results])
                            log.write(json.dumps(dict(validation=results, success_rate=float(score), steps=steps)) + '\n')
                            if score > best_score:
                                best_score = float(score)
                                best_actor = {k: v.detach().cpu().clone() for k, v in agent.actor.state_dict().items()}
                                save(output / 'best.pt', {**metadata, 'actor': best_actor})
            save(output / 'resume.pt', snapshot())
    save(output / 'last.pt', {**metadata, 'actor': agent.actor.state_dict()})
    actor = agent.actor
    if best_actor is not None:
        save(output / 'best.pt', {**metadata, 'actor': best_actor})
        actor = copy.deepcopy(agent.actor)
        actor.load_state_dict(best_actor)
    has_updates = steps > 0 and steps >= cfg['warmup_steps']
    evaluated_checkpoint = ('best.pt' if best_actor is not None else 'last.pt') if has_updates else None
    policies = {'baseline': None}
    if has_updates:
        policies['dsrl'] = actor
    results = evaluate(policy, policy_cfg, test_seeds, policies,
                       noise_magnitude=cfg['noise_magnitude'], deadline=deadline)
    baseline, steered = results['baseline'], results['dsrl'] if has_updates else []
    result = dict(dsrl_checkpoint=evaluated_checkpoint, baseline=baseline, dsrl=steered,
                  comparison=compare(baseline, steered) if has_updates else {'episodes': 0})
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2))
    save(output / 'resume.pt', snapshot())
    print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
