import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from dexmani_rl.methods.bora import train
from dexmani_rl.methods.bora.algorithm import shifted_target
from dexmani_rl.methods.bora.data import FrameReplay
from tests.test_method_updates import TrainablePolicy, recipe


class BoraReplayTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.episode = self.root / 'demo.h5'
        self.policy = TrainablePolicy(state_dim=19)
        self.config = {'env_runner': {'task_name': 'test'}, 'control_dt_s': 0.1,
                       'eval': {'inference_steps': 1}, 'agent': {}}
        self.rewards = {'time_penalty': -1., 'success_reward': 5.}

    def write_episode(self, done, frames, *, truncated=False):
        with h5py.File(self.episode, 'w') as f:
            f.attrs.update(task_name='test', action_space='joint', dt=0.1, seed=1,
                           truncated=truncated, success=any(done))
            f['done'] = done
            f['action'] = np.zeros((len(done), self.policy.action_dim), dtype=np.float32)
            f['joint_state'] = np.broadcast_to(np.arange(1, frames + 1)[:, None], (frames, 19))

    def test_final_transition(self):
        # Some demonstrations continue recording after the first terminal action.
        for done, frames, truncated, expected in (([True, True], 2, False, 4.), ([False], 2, True, 0.)):
            with self.subTest(truncated=truncated):
                self.write_episode(done, frames, truncated=truncated)
                replay = FrameReplay([self.episode], self.policy, self.config, self.rewards)
                self.assertEqual(len(replay), 1)
                batch = replay.sample(1, np.random.default_rng(0))
                self.assertEqual(batch['mask'].tolist(), [[True, False]])
                next_value = batch['next_obs']['joint_state'][:, -1, 0]
                target = shifted_target(torch.zeros_like(batch['reward']), next_value,
                                        batch['reward'], batch['done'], batch['mask'], gamma=0.5)
                self.assertEqual(target[batch['mask']].tolist(), [expected])

    def test_timeout_requires_final_observation(self):
        self.write_episode([False], 1, truncated=True)
        with self.assertRaisesRegex(ValueError, 'Missing observation for joint_state'):
            FrameReplay([self.episode], self.policy, self.config, self.rewards)

    def test_resume_after_removing_selection_symlink(self):
        self.write_episode([True], 1)
        selection = self.root / 'selection'
        selection.mkdir()
        link = selection / 'episode_0001.h5'
        link.symlink_to(self.episode)
        config = self.root / 'config.yaml'
        OmegaConf.save(recipe('bora', offline_updates=2, bc_warmup_updates=1, offline_batch_size=2,
                              hidden_dim=8, ppo_epochs=1), config)
        first = self.root / 'first'
        resumed = self.root / 'resumed'

        def run(source, output):
            args = ['train', *source, '--device', 'cpu', '--through', 'offline', '--output', str(output)]
            with (
                patch.object(sys, 'argv', args),
                patch.object(train, 'load_flowmatch_policy',
                             return_value=(TrainablePolicy(state_dim=19), self.config, 'raw')),
                patch.object(train, 'seed_split', return_value=([2], [3], {2, 3})),
                patch.dict(sys.modules, dexmani_sim=SimpleNamespace(DATA_DIR=self.root, EpisodeRejected=RuntimeError)),
            ):
                train.main()

        run(['--checkpoint', 'test.pt', '--config', str(config), '--demonstrations', str(selection)], first)
        link.unlink()
        run(['--resume', str(first / 'resume.pt')], resumed)
        saved = torch.load(resumed / 'resume.pt', map_location='cpu', weights_only=False)
        self.assertEqual(saved['demo_files'], [str(self.episode)])
        self.assertEqual(saved['offline_steps'], 2)
