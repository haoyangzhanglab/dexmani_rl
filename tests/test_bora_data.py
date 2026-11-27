from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from dexmani_rl.bora_data import FrameReplay, collect_episode
from dexmani_rl.env import ChunkEnv
from dexmani_rl.policy.normalizer import LinearNormalizer


CONFIG = dict(control_dt_s=1/16, env_runner={'task_name': 'multi_grasp'})
REWARDS = dict(time_penalty=-.01, success_reward=1., takeover_penalty=-.5, recovery_reward=.25)


def policy():
    normalizer = LinearNormalizer({'action': (19,)})
    normalizer['action'].scale.fill_(2)
    normalizer['action'].offset.fill_(1)
    return SimpleNamespace(n_obs_steps=2, n_action_steps=3, action_dim=19,
                           normalization_dims={'joint_state': 19, 'action': 19},
                           normalizer=normalizer, parameters=lambda: iter([torch.zeros(())]))


def demo(path, *, done=(False, False, True, True, True), truncated=False, post=False):
    steps = len(done)
    with h5py.File(path, 'w') as f:
        f['action'] = np.arange(steps, dtype='f4')[:, None].repeat(19, axis=1)
        f['joint_state'] = np.arange(steps + post, dtype='f4')[:, None].repeat(19, axis=1)
        f['done'] = np.array(done)
        f.attrs.update(task_name='multi_grasp', action_space='joint', dt=1/16, seed=17,
                       success=any(done), truncated=truncated)
    return path


class Indices:
    def __init__(self, *indices):
        self.indices = indices

    def integers(self, high, size):
        assert len(self.indices) == size
        assert max(self.indices) < high
        return self.indices


def test_demo_uses_first_done_and_aligned_sliding_windows(tmp_path):
    path = demo(tmp_path/'demo.h5')
    replay = FrameReplay([path], policy(), CONFIG, REWARDS)
    assert len(replay) == 3
    b = replay.sample(3, Indices(0, 1, 2))
    torch.testing.assert_close(b['obs']['joint_state'][..., 0], torch.tensor([[0., 0.], [0., 1.], [1., 2.]]))
    torch.testing.assert_close(b['next_obs']['joint_state'][..., 0], torch.tensor([[0., 1.], [1., 2.], [1., 2.]]))
    torch.testing.assert_close(b['action'][..., 0], torch.tensor([[1., 3., 5.], [3., 5., 0.], [5., 0., 0.]]))
    assert b['done'].tolist() == [[False, False, True], [False, True, False], [True, False, False]]
    assert b['mask'].tolist() == [[True, True, True], [True, True, False], [True, False, False]]
    torch.testing.assert_close(b['reward'], torch.tensor([[-.01, -.01, .99], [-.01, .99, 0.], [.99, 0., 0.]]))
    with pytest.raises(ValueError, match='Evaluation seed'):
        FrameReplay([path], policy(), CONFIG, REWARDS, excluded_seeds={17})


def test_timeout_bootstraps_actual_final_observation(tmp_path):
    flags = (False, False, False, False, False)
    path = demo(tmp_path/'timeout.h5', done=flags, truncated=True, post=True)
    replay = FrameReplay([path], policy(), CONFIG, REWARDS)
    b = replay.sample(3, Indices(0, 2, 4))
    torch.testing.assert_close(b['next_obs']['joint_state'][..., 0],
                               torch.tensor([[0., 1.], [4., 5.], [4., 5.]]))
    assert not b['done'].any()
    assert b['mask'].sum(-1).tolist() == [3, 3, 1]
    missing = demo(tmp_path/'missing.h5', done=flags, truncated=True)
    with pytest.raises(ValueError, match='Missing observation'):
        FrameReplay([missing], policy(), CONFIG, REWARDS)


def test_replay_indices_cross_episode_boundary(tmp_path):
    replay = FrameReplay([demo(tmp_path/'one.h5')], policy(), CONFIG, REWARDS)
    replay.add(demo(tmp_path/'two.h5', done=(False, True)))
    batch = replay.sample(3, Indices(2, 3, 4))
    assert len(replay) == 5
    assert batch['mask'].sum(-1).tolist() == [1, 2, 1]
    torch.testing.assert_close(batch['action'][:, 0, 0], torch.tensor([5., 1., 3.]))


class Sim:
    name = 'multi_grasp'
    dt = 1/16
    robot = SimpleNamespace(qlimits=np.array([[-1., 1.]] * 19))

    def reset(self, seed):
        self.i = 0
        return self.observe(), {}

    def observe(self):
        return {'joint_state': np.full(19, self.i, dtype='f4')}

    def step(self, action):
        self.i += 1
        end = self.i == 5
        # Test that replay follows the command actually executed after sim processing.
        return self.observe(), 42., end, False, dict(success=end, executed_joint_action=action * .5)


def test_record_executed_action_and_reward_edges(tmp_path):
    small = policy()
    env = ChunkEnv(Sim(), small, CONFIG)
    visited = []

    def actor(obs):
        visited.append(obs['joint_state'][0, :, 0].tolist())
        return torch.zeros(3, 19)

    def correct(obs, target):
        frame = int(obs['joint_state'][0])
        assert np.array_equal(target, np.zeros(19))
        return np.full(19, 3.) if frame in (1, 2, 4) else None

    confirmations = []

    def confirm(next_obs, terminated, truncated, info):
        frame = int(next_obs['joint_state'][0])
        confirmations.append((frame, terminated, truncated))
        np.testing.assert_array_equal(info['executed_joint_action'], np.full(19, .5))
        return frame >= 2

    path = tmp_path/'episode.h5'
    report = collect_episode(env, 60, actor, path, REWARDS, correct, blend=.5, confirm_recovery=confirm)
    assert visited == [[0., 0.], [2., 3.]]
    # The first takeover lasts two steps. The second takeover ends the episode.
    assert confirmations == [(2, False, False), (5, True, False)]
    assert report == dict(seed=60, steps=5, success=True, terminated=True, truncated=False,
                          takeovers=2, recoveries=2, time_reward=-.05, success_reward=1.,
                          takeover_reward=-1., recovery_reward=.5)
    with h5py.File(path) as episode:
        np.testing.assert_array_equal(episode['joint_state'][:, 0], np.arange(6))
        np.testing.assert_array_equal(episode['action'][:, 0], [0., .5, .5, 0., .5])
        np.testing.assert_array_equal(episode['takeover'][:], [0, 1, 0, 0, 1])
        np.testing.assert_array_equal(episode['recovery'][:], [0, 1, 0, 0, 1])
    online = FrameReplay([path], small, CONFIG, REWARDS)
    b = online.sample(1, Indices(2))
    torch.testing.assert_close(b['reward'][0], torch.tensor([-.01, -.01, .74]))
    offline = FrameReplay([demo(tmp_path/'demo.h5')], small, CONFIG, REWARDS)
    mixed = offline.sample(2, Indices(0), online=online)
    torch.testing.assert_close(mixed['reward'][:, 0], torch.tensor([-.01, -.01]))
    torch.testing.assert_close(mixed['reward'][:, 1], torch.tensor([-.01, -.26]))
    assert mixed['obs']['joint_state'].shape == (2, 2, 19)
    with pytest.raises(FileExistsError):
        collect_episode(env, 60, actor, path, REWARDS)


def test_recovery_uses_the_result_of_a_corrected_step_including_timeout(tmp_path):
    class TimeoutSim(Sim):
        def step(self, action):
            obs, reward, end, _, info = super().step(action)
            return obs, reward, False, end, {**info, 'success': False}

    small = policy()
    env = ChunkEnv(TimeoutSim(), small, CONFIG)

    def correct(obs, target):
        frame = int(obs['joint_state'][0])
        return target if frame in (0, 3, 4) else None

    results = []

    def confirm(next_obs, terminated, truncated, info):
        frame = int(next_obs['joint_state'][0])
        results.append((frame, terminated, truncated))
        # Recovery becomes visible on autonomous frame 2 and on the last correction.
        return frame == 2 or truncated

    path = tmp_path/'timeout.h5'
    report = collect_episode(env, 83, lambda obs: np.zeros((3, 19)), path, REWARDS,
                             correction=correct, confirm_recovery=confirm)
    assert results == [(1, False, False), (4, False, False), (5, False, True)]
    assert report['takeovers'] == 2 and report['recoveries'] == 1
    with h5py.File(path) as episode:
        np.testing.assert_array_equal(episode['recovery'][:], [0, 0, 0, 0, 1])
    b = FrameReplay([path], small, CONFIG, REWARDS).sample(1, Indices(4))
    assert not b['done'].any()
    torch.testing.assert_close(b['reward'][0], torch.tensor([.24, 0., 0.]))


def test_autonomous_collection_has_no_intervention_events(tmp_path):
    env = ChunkEnv(Sim(), policy(), CONFIG)
    report = collect_episode(env, 7, lambda obs: np.zeros((3, 19)), tmp_path/'episode.h5', REWARDS)
    assert report['takeovers'] == report['recoveries'] == 0
    assert report['takeover_reward'] == report['recovery_reward'] == 0.
