"""Paired rollouts exercise the same paths used by validation and the CLI."""
import unittest
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.test_method_updates import TrainablePolicy, Velocity


class OneChunkEnv:
    def __init__(self, actions):
        self.env = SimpleNamespace(robot=SimpleNamespace(qlimits=[[-.5, .5]] * 3))
        self.actions = actions

    def reset(self, seed):
        return {'joint_state': torch.full((1, 2, 3), float(seed))}

    def step(self, action):
        self.actions.append(action.copy())
        return {}, float(action.sum()), False, True, {'steps': len(action), 'success': True}


class MethodEvaluationTests(unittest.TestCase):
    def test_paired_candidates_and_deadline(self):
        for name in ('dsrl', 'rfs', 'robofft', 'bora'):
            with self.subTest(method=name):
                torch.manual_seed(37)
                evaluator = import_module(f'dexmani_rl.methods.{name}.evaluate')
                policy = TrainablePolicy()
                cfg = {'eval': {'inference_steps': 1}}
                candidates = {'baseline': None}
                kwargs = {}
                if name == 'dsrl':
                    actor = evaluator.Actor(16, 12, 8, 1)
                    kwargs['noise_magnitude'] = 1.5
                elif name == 'rfs':
                    actor = evaluator.Actor(16, 12, 6, [8], .1)
                elif name == 'robofft':
                    actor = evaluator.RoboFFTActor(Velocity(3, 3), 4, 3, 1)
                    candidates['sampler_baseline'] = deepcopy(actor)
                else:
                    model = evaluator.BORAActor(policy, cfg, {'hidden_dim': 8, 'blend_start': .2})
                    model.round_number.fill_(2)
                    candidates['offline'] = evaluator.BORAStage(model, 0)
                    actor = evaluator.BORAStage(model, 2)
                candidates['steered'] = actor
                actions = []

                @contextmanager
                def open_env(policy, cfg):
                    yield OneChunkEnv(actions)

                with patch.object(evaluator, 'make_flowmatch_env', open_env):
                    rows = evaluator.evaluate(policy, cfg, [2, 3], candidates, **kwargs)
                    report = evaluator.evaluation_report(rows, [2, 3])
                    self.assertTrue(report['evaluation_complete'])
                    self.assertEqual(report['comparison']['episodes'], 2)
                    for episodes in rows.values():
                        self.assertEqual([e['seed'] for e in episodes], [2, 3])
                        self.assertEqual([e['steps'] for e in episodes], [2, 2])
                    self.assertEqual(len(actions), 2 * len(candidates))
                    self.assertTrue(all(abs(action).max() <= .5 for action in actions))
                    # The deadline expires after the first candidate of the second seed.
                    cutoff = len(actions) + len(candidates) + 1
                    with patch.object(evaluator.time, 'monotonic', side_effect=lambda: int(len(actions) >= cutoff)):
                        partial = evaluator.evaluate(policy, cfg, [2, 3], candidates, deadline=1, **kwargs)
                    self.assertTrue(all(len(episodes) == 1 for episodes in partial.values()))
                    self.assertFalse(evaluator.evaluation_report(partial, [2, 3])['evaluation_complete'])
