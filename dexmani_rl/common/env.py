import json
import random
import time
from collections import deque
from contextlib import contextmanager

import numpy as np

from ..policy.preprocess import prepare_observations


class ChunkEnv:
    def __init__(self, env, policy, cfg):
        self.env, self.policy = env, policy
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=policy.n_obs_steps)
        if not np.isclose(env.dt, cfg['control_dt_s'], rtol=1e-6):
            raise ValueError('Control period differs from checkpoint')

    def observe(self, obs):
        self.history.append({k: v.copy() for k, v in obs.items()})

    def observations(self):
        frames = list(self.history)
        frames = [frames[0]] * (self.policy.n_obs_steps - len(frames)) + frames
        return prepare_observations({k: np.stack([f[k] for f in frames]) for k in frames[0]},
            normalization_dims=self.policy.normalization_dims,
            device=next(self.policy.parameters()).device)

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.history.clear()
        self.observe(obs)
        return self.observations()

    def step(self, actions):
        reward = 0.
        for k, action in enumerate(actions):
            obs, _, terminated, truncated, info = self.env.step(action)
            self.observe(obs)
            reward += float(info['success'])
            if terminated or truncated:
                break
        return self.observations(), reward, terminated, truncated, {**info, 'steps': k + 1}


@contextmanager
def make_env(policy, cfg):
    from dexmani_sim import resolve_env_class
    from dexmani_sim.envs.lifecycle import open_env
    kwargs = dict(cfg['env_runner'].get('env_kwargs') or {})
    kwargs.update(render_mode='rgb_array', record_video=False,
                  observation_keys=tuple(k for k in policy.normalization_dims if k != 'action'))
    with open_env(resolve_env_class(cfg['env_runner']['task_name']), **kwargs) as env:
        yield ChunkEnv(env, policy, cfg)


def seed_split(task):
    from dexmani_sim import DATA_DIR
    seeds = [int(s) for s in (DATA_DIR / 'eval_seeds' / f'{task}.txt').read_text().split()]
    if len(seeds) < 120 or len(seeds) != len(set(seeds)):
        raise ValueError('Need at least 120 distinct evaluation seeds')
    seeds.sort()
    random.Random(1024).shuffle(seeds)
    return seeds[:100], seeds[-20:], set(seeds)


def training_episodes(policy, cfg, rng, excluded, deadline, log):
    from dexmani_sim import EpisodeRejected

    while time.monotonic() < deadline:
        with make_env(policy, cfg) as env:
            attempts = 0
            while attempts < 25 and time.monotonic() < deadline:
                seed = int(rng.integers(2**32))
                if seed in excluded:
                    continue
                attempts += 1
                try:
                    obs = env.reset(seed)
                except EpisodeRejected as error:
                    log.write(json.dumps({'rejected_seed': seed, 'error': str(error)}) + '\n')
                    continue
                yield env, seed, obs
