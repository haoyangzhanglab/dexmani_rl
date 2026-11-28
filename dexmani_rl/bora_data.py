"""Control-frame windows for BORA demonstrations and online replay."""
import os
import tempfile
from bisect import bisect_right
from pathlib import Path

import h5py
import numpy as np
import torch

from .policy.preprocess import prepare_observations


class FrameReplay:
    """Index HDF5 episodes by valid control frame and read visual windows on demand."""

    def __init__(self, paths, policy, bc_cfg, rewards, excluded_seeds=()) -> None:
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.task = bc_cfg['env_runner']['task_name']
        self.dt = bc_cfg['control_dt_s']
        self.rewards = rewards
        self.excluded_seeds = set(excluded_seeds)
        self.paths: list[Path] = []
        self.ends: list[int] = []
        for path in paths:
            self.add(path)

    def add(self, path):
        path = Path(path).resolve()
        with h5py.File(path, 'r') as episode:
            if episode.attrs['task_name'] != self.task or episode.attrs['action_space'] != 'joint':
                raise ValueError(f'Task or action space differs from checkpoint: {path}')
            if not np.isclose(episode.attrs['dt'], self.dt, rtol=1e-6):
                raise ValueError(f'Control period differs from checkpoint: {path}')
            if int(episode.attrs['seed']) in self.excluded_seeds:
                raise ValueError(f'Evaluation seed in training replay: {path}')
            done = episode['done'][:]
            if (done.ndim != 1 or not len(done) or
                    episode['action'].shape != (len(done), self.policy.action_dim)):
                raise ValueError(f'Invalid action or termination shape: {path}')
            terminal = np.flatnonzero(done)
            length = int(terminal[0]) + 1 if len(terminal) else len(done)
            if not len(terminal) and not episode.attrs['truncated']:
                raise ValueError(f'Missing episode end: {path}')
            # Only timeouts bootstrap from a post-step observation.
            needed = length + int(not len(terminal))
            for key in self.policy.normalization_dims:
                if key != 'action' and len(episode[key]) < needed:
                    raise ValueError(f'Missing observation for {key} at episode boundary: {path}')
        self.paths.append(path)
        self.ends.append(len(self) + length)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _window(self, index):
        episode_idx = bisect_right(self.ends, index)
        origin = self.ends[episode_idx - 1] if episode_idx else 0
        start = int(index - origin)
        k = self.policy.n_action_steps
        remaining = self.ends[episode_idx] - int(index)
        count = min(k, remaining)
        end = start + count
        with h5py.File(self.paths[episode_idx], 'r') as episode:
            action = np.zeros((k, self.policy.action_dim), dtype=np.float32)
            action[:count] = episode['action'][start:end]
            if not np.isfinite(action).all():
                raise ValueError(f'Nonfinite action in {self.paths[episode_idx]}')
            mask = np.arange(k) < count
            done = np.zeros(k, dtype=bool)
            done[:count] = episode['done'][start:end]
            reward = mask.astype(np.float32) * self.rewards['time_penalty']
            if episode.attrs['success']:
                reward += done * self.rewards['success_reward']
            for key, coefficient in [('takeover', 'takeover_penalty'), ('recovery', 'recovery_reward')]:
                if key in episode:
                    reward[:count] += episode[key][start:end] * self.rewards[coefficient]

            # A terminal tail is masked and needs no post-step observation.
            next_frame = end - 1 if done[count - 1] else end
            offsets = np.arange(1 - self.policy.n_obs_steps, 1)
            current = np.maximum(start + offsets, 0)
            following = np.maximum(next_frame + offsets, 0)
            obs, next_obs = {}, {}
            for key in self.policy.normalization_dims:
                if key == 'action':
                    continue
                source = episode[key][current[0]:following[-1] + 1]
                obs[key] = source[current - current[0]]
                next_obs[key] = source[following - current[0]]
        dims = self.policy.normalization_dims
        return {'obs': prepare_observations(obs, normalization_dims=dims, device='cpu'),
                'next_obs': prepare_observations(next_obs, normalization_dims=dims, device='cpu'),
                'action': torch.from_numpy(action), 'reward': torch.from_numpy(reward),
                'done': torch.from_numpy(done), 'mask': torch.from_numpy(mask)}

    def _draw(self, count, rng):
        if not len(self):
            raise ValueError('Cannot sample an empty frame replay')
        return [self._window(index) for index in rng.integers(len(self), size=count)]

    def sample(self, batch_size, rng, *, online=None):
        if online is None:
            rows = self._draw(batch_size, rng)
        else:
            if batch_size < 2 or batch_size % 2:
                raise ValueError('A 1:1 mixed batch requires a positive even batch size')
            half = batch_size // 2
            rows = self._draw(half, rng) + online._draw(half, rng)
        observations = {
            key: {name: torch.cat([row[key][name] for row in rows]).to(self.device) for name in rows[0][key]}
            for key in ('obs', 'next_obs')
        }
        batch = {key: torch.stack([row[key] for row in rows]).to(self.device)
                 for key in ('action', 'reward', 'done', 'mask')}
        action = self.policy.normalizer['action'].normalize(batch['action'])
        batch['action'] = action.masked_fill(~batch['mask'][..., None], 0.)
        return {**observations, **batch}


@torch.no_grad()
def collect_episode(env, seed, act, path, rewards, correction=None, blend=1., *, confirm_recovery=None):
    """Choose an optional human target before each step and confirm its recovery on the resulting observation."""
    if not 0 < blend <= 1:
        raise ValueError('Intervention blend must be in (0, 1]')
    if confirm_recovery is not None and correction is None:
        raise ValueError('Recovery confirmation requires an action correction callback')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = env.reset(seed)
    frames = [env.history[-1]]
    actions, dones, active_frames, takeovers, recoveries = [], [], [], [], []
    limits = np.asarray(env.env.robot.qlimits)
    was_active = recovery_confirmed = False
    terminated = truncated = False
    while not (terminated or truncated):
        proposed = act(obs)
        if torch.is_tensor(proposed):
            proposed = proposed.detach().cpu().numpy()
        proposed = np.asarray(proposed)
        if proposed.shape != (env.policy.n_action_steps, env.policy.action_dim) or not np.isfinite(proposed).all():
            raise ValueError('Expected one finite physical action chunk [K,D]')
        for target in proposed:
            human = correction(env.history[-1], target.copy()) if correction is not None else None
            active = human is not None
            takeover = active and not was_active
            if active:
                human = np.asarray(human, dtype=np.float32)
                if human.shape != target.shape or not np.isfinite(human).all():
                    raise ValueError('An active correction needs a finite physical joint target')
                target = (1. - blend) * target + blend * human
            target = np.clip(target, limits[:, 0], limits[:, 1])
            raw, _, terminated, truncated, info = env.env.step(target)
            env.observe(raw)
            frames.append(env.history[-1])
            actions.append(np.asarray(info['executed_joint_action'], dtype=np.float32).copy())
            recovered = (active and not recovery_confirmed and confirm_recovery is not None
                         and bool(confirm_recovery(raw, terminated, truncated, info)))
            recovery_confirmed = active and (recovery_confirmed or recovered)
            dones.append(terminated)
            active_frames.append(active)
            takeovers.append(takeover)
            recoveries.append(recovered)
            was_active = active
            if terminated or truncated:
                break
        if not (terminated or truncated):
            obs = env.observations()

    # Keep only one trajectory's raw frames in RAM. Publish a completed HDF5
    # exclusively, so a failed or concurrent rollout cannot replace another one.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.h5') as temporary:
        with h5py.File(temporary.name, 'w') as episode:
            for key in env.policy.normalization_dims:
                if key != 'action':
                    episode.create_dataset(key, data=np.stack([frame[key] for frame in frames]),
                                           compression='lzf')
            for key, values in (('action', actions), ('done', dones), ('intervention', active_frames),
                                ('takeover', takeovers), ('recovery', recoveries)):
                episode.create_dataset(key, data=np.asarray(values))
            episode.attrs.update(task_name=env.env.name, dt=env.env.dt, action_space='joint', seed=seed,
                                 success=bool(info['success']), truncated=bool(truncated))
        os.link(temporary.name, path)
    return {'seed': seed, 'steps': len(actions), 'success': bool(info['success']), 'terminated': bool(terminated),
            'truncated': bool(truncated), 'takeovers': sum(takeovers), 'recoveries': sum(recoveries),
            'time_reward': len(actions) * rewards['time_penalty'],
            'success_reward': int(bool(info['success']) and terminated) * rewards['success_reward'],
            'takeover_reward': sum(takeovers) * rewards['takeover_penalty'],
            'recovery_reward': sum(recoveries) * rewards['recovery_reward']}
