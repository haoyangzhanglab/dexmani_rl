"""BORA v2 offline updates followed by two rounds of residual adaptation."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from dexmani_rl.envs.action_transform import JointActionTransform
from dexmani_rl.envs.dexmani_sim import make_flowmatch_env, seed_split
from dexmani_rl.policies.flowmatch.loader import load_flowmatch_policy

from .._shared.cli import RECIPE_DIR, training_inputs, training_parser
from .._shared.torch_checkpoint import capture_rng, restore_rng, save
from .algorithm import BORA
from .data import FrameReplay, collect_episode

DEFAULT_CONFIG = RECIPE_DIR / 'bora.yaml'


def main(correction=None, confirm_recovery=None):
    parser = training_parser()
    parser.add_argument('--through', choices=['offline', 'round1', 'round2'], default='round2',
                        help='Last stage to run before saving the training snapshot')
    parser.add_argument('--demonstrations', type=Path, help='Directory containing episode_*.h5')
    args = parser.parse_args()
    if args.resume and args.demonstrations:
        parser.error('--resume uses the saved demonstration data')
    cfg, checkpoint, weights, saved = training_inputs(parser, args, method='bora', default_config=DEFAULT_CONFIG)
    from dexmani_sim import DATA_DIR, EpisodeRejected

    through = ('offline', 'round1', 'round2').index(args.through)
    for key in ('offline_updates', 'offline_batch_size', 'online_batch_size', 'episodes_per_round',
                'updates_per_episode', 'save_interval'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (type(cfg['bc_warmup_updates']) is not int or not 0 <= cfg['bc_warmup_updates'] < cfg['offline_updates']
            or cfg['online_batch_size'] % 2):
        raise ValueError('BORA requires a BC+PPO phase and an even mixed batch size')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    policy, bc_cfg, weights = load_flowmatch_policy(checkpoint, args.device, weights)
    device = next(policy.parameters()).device
    task = bc_cfg['env_runner']['task_name']
    if saved is None:
        test_seeds, _, excluded = seed_split(task)
        demo_dir = args.demonstrations or DATA_DIR / 'episodes' / task / 'succeeded_episode'
        demo_files = sorted(demo_dir.glob('episode_*.h5'))
        online_files = []
    else:
        test_seeds, excluded = saved['test_seeds'], set(saved['excluded_seeds'])
        demo_files, online_files = saved['demo_files'], saved['online_files']
    if not demo_files:
        raise ValueError('No episode_*.h5 demonstrations. Pass --demonstrations with the episode directory.')
    rewards = cfg['rewards']
    offline = FrameReplay(demo_files, policy, bc_cfg, rewards, excluded_seeds=excluded)
    online = FrameReplay(online_files, policy, bc_cfg, rewards, excluded_seeds=excluded)
    agent = BORA(policy, bc_cfg, cfg)
    offline_steps = episodes = 0
    if saved is not None:
        agent.load_state_dict(saved['model'])
        offline_steps, episodes = saved['offline_steps'], saved['episodes']
        round_number = int(agent.actor.round_number)
        if through < round_number:
            raise ValueError('--through precedes the saved training stage')
        if round_number:
            agent.begin_round(round_number)
        for name, optimizer in agent.optimizers.items():
            optimizer.load_state_dict(saved['optimizers'][name])
        restore_rng(saved, rng, device)
        del saved

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    version = None
    if shutil.which('git'):
        version = subprocess.run(['git', 'describe', '--always', '--dirty', '--abbrev=40'],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=False).stdout.strip()
    metadata = {'algorithm': 'bora', 'config': cfg, 'base_checkpoint': checkpoint, 'weights': weights,
                'git_revision': version or None, 'torch': str(torch.__version__), 'test_seeds': test_seeds,
                'excluded_seeds': sorted(excluded), 'demo_files': [str(p) for p in offline.paths]}
    (output / 'config.json').write_text(json.dumps(dict(metadata, policy_config=bc_cfg,
        resumed_from=args.resume, correction_enabled=correction is not None,
        recovery_confirmation_enabled=confirm_recovery is not None), indent=2))

    def snapshot():
        save(output / 'resume.pt', dict(metadata, model=agent.state_dict(),
             offline_steps=offline_steps, episodes=episodes, online_files=[str(p) for p in online.paths],
             optimizers={k: v.state_dict() for k, v in agent.optimizers.items()}, **capture_rng(rng, device)))

    def export(name):
        save(output / name, dict(metadata, actor=agent.actor.state_dict(),
                                 offline_steps=offline_steps, episodes=episodes))

    with (output / 'metrics.jsonl').open('x', buffering=1) as log:
        def record(**values):
            log.write(json.dumps(values, allow_nan=False) + '\n')

        while offline_steps < cfg['offline_updates']:
            ppo = offline_steps >= cfg['bc_warmup_updates']
            metrics = agent.offline_update(offline.sample(cfg['offline_batch_size'], rng), ppo=ppo)
            offline_steps += 1
            record(phase='offline_ppo' if ppo else 'offline_bc', offline_steps=offline_steps, **metrics)
            if offline_steps % cfg['save_interval'] == 0 and offline_steps < cfg['offline_updates']:
                snapshot()
        if int(agent.actor.round_number) == 0:
            export('offline.pt')
            if through:
                snapshot()
        # Each complete episode becomes available to the 1:1 sampler before the next rollout.
        while episodes < through * cfg['episodes_per_round']:
            round_number = 1 + episodes // cfg['episodes_per_round']
            if round_number != int(agent.actor.round_number):
                agent.begin_round(round_number)
            seed = int(rng.integers(2**32))
            if seed in excluded:
                continue
            path = output / 'online' / f'episode_{episodes:03}_{seed}.h5'
            with make_flowmatch_env(policy, bc_cfg) as env:
                transform = JointActionTransform(
                    policy.normalizer['action'], torch.as_tensor(env.env.robot.qlimits, device=device),
                    action_dim=policy.action_dim, execute_steps=policy.n_action_steps,
                )

                def act(obs, transform=transform):
                    return transform.bounded_physical(agent.actor.normalized_actions(obs))[0]

                try:
                    rollout = collect_episode(env, seed, act, path, rewards, correction=correction,
                                              confirm_recovery=confirm_recovery, blend=cfg['intervention_blend'])
                except EpisodeRejected as error:
                    record(rejected_seed=seed, error=str(error))
                    continue
            online.add(path)
            record(phase='collect', round=round_number, episode=episodes + 1, **rollout)
            first_update = (episodes % cfg['episodes_per_round']) * cfg['updates_per_episode']
            for step in range(first_update, first_update + cfg['updates_per_episode']):
                batch = offline.sample(cfg['online_batch_size'], rng, online=online)
                metrics = agent.online_update(batch, step=step, joint_limits=transform.limits)
                record(phase='online', round=round_number, round_update=step + 1, **metrics)
            episodes += 1
            if episodes < through * cfg['episodes_per_round']:
                snapshot()
        snapshot()
        export('last.pt')
    print(json.dumps({'round': int(agent.actor.round_number), 'offline_steps': offline_steps, 'episodes': episodes,
                      'checkpoint': str(output / 'last.pt')}))


if __name__ == '__main__':
    main()
