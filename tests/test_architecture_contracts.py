import subprocess
import sys
import unittest

import torch

from dexmani_rl.common.base_policy import FrozenPolicy, policy_dimensions
from dexmani_rl.core.contracts import StepResult
from dexmani_rl.core.specs import ActionSegment, ActionSpec
from dexmani_rl.envs.adapters import ChunkEnvAdapter
from dexmani_rl.policies.flowmatch.adapter import FlowMatchPolicyAdapter
from tests.flowmatch_fixtures import FakeFlowMatchPolicy


class FakeLegacyEnv:
    def __init__(self, terminated=False):
        self.terminated = terminated

    def close(self):
        raise AssertionError("Borrowed environment closed twice")

    def reset(self, seed):
        return {"seed": seed}

    def step(self, action):
        return {"obs": action}, 1.0, self.terminated, not self.terminated, {"steps": 3, "source": "fake"}


class ArchitectureContractTests(unittest.TestCase):
    def test_action_spec_splits_dynamic_segments(self):
        spec = ActionSpec(
            segments=(ActionSegment("arm", 7, "joint_position", "rad"),
                      ActionSegment("hand_latent", 9, "latent", "normalized")),
            horizon=32,
            execute_steps=25,
            control_dt=0.02,
        )
        action = torch.zeros(2, 32, 16)
        parts = spec.split(action)
        self.assertEqual(spec.action_dim, 16)
        self.assertEqual(parts["arm"].shape, (2, 32, 7))
        self.assertEqual(parts["hand_latent"].shape, (2, 32, 9))

    def test_action_spec_rejects_invalid_or_duplicate_segments(self):
        with self.assertRaises(ValueError):
            ActionSpec((ActionSegment("arm", 7), ActionSegment("arm", 3)))
        with self.assertRaises(ValueError):
            ActionSpec((ActionSegment("arm", 7),), horizon=4, execute_steps=5)

    def test_core_import_does_not_load_policy_backends(self):
        code = """
import sys
import dexmani_rl.core
import dexmani_rl.envs.adapters
import dexmani_rl.policies

assert 'dexmani_rl.policy' not in sys.modules
assert not any(name.startswith('dexmani_rl.policies.') for name in sys.modules)
assert not {'torch', 'torchvision', 'transformers', 'timm', 'dexmani_sim'}.intersection(sys.modules)
"""
        subprocess.run([sys.executable, "-E", "-S", "-c", code], check=True)

    def test_bora_critic_actions_without_policy_modules(self):
        code = """
import sys
sys.modules['dexmani_rl.policies.flowmatch'] = None
sys.modules['dexmani_rl.policy'] = None
sys.modules['dexmani_rl.common.base_policy'] = None
import torch
from dexmani_rl.bora import executable_actions
from tests.flowmatch_fixtures import IdentityField
field = IdentityField()
window = torch.tensor([[[2., -0.5]]])
limits = torch.tensor([[-1., 1.], [-1., 1.]])
expected = torch.tensor([[[1., -0.5]]])
torch.testing.assert_close(executable_actions(window, field, limits), expected)
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_method_imports(self):
        code = """
import sys
import dexmani_rl.policies.flowmatch.adapter
import dexmani_rl.dsrl, dexmani_rl.rfs, dexmani_rl.robofft, dexmani_rl.bora
import dexmani_rl.train_dsrl, dexmani_rl.train_rfs, dexmani_rl.train_robofft, dexmani_rl.train_bora
assert not {'dexmani_rl.policy.flowmatch', 'torchvision', 'transformers', 'timm', 'dexmani_sim'}.intersection(sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_action_dimensions_and_period(self):
        for width in (19, 32):
            spec = ActionSpec((ActionSegment('code', width, 'latent'),), horizon=8, execute_steps=3)
            values = torch.arange(width)
            torch.testing.assert_close(spec.split(values)['code'], values)
        for dim in (0, -1, True):
            with self.assertRaises(ValueError):
                ActionSegment('code', dim)
        for dt in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ActionSpec((ActionSegment('joint', 2),), control_dt=dt)

    def test_step_result_reports_episode_boundary(self):
        result = StepResult({}, 1.0, False, True, 2, {})
        self.assertTrue(result.done)
        with self.assertRaises(ValueError):
            StepResult({}, float("nan"), False, False)

    def test_chunk_env_adapter_bridges_legacy_tuple_api(self):
        adapter = ChunkEnvAdapter(FakeLegacyEnv())
        self.assertEqual(adapter.reset(12), {"seed": 12})
        result = adapter.step(torch.zeros(1))
        self.assertEqual(result.observation, {"obs": torch.zeros(1)})
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.info, {"steps": 3, "source": "fake"})
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.executed_steps, 3)
        adapter.close()
        terminal = ChunkEnvAdapter(FakeLegacyEnv(terminated=True)).step([1, 2, 3])
        self.assertTrue(terminal.terminated)
        self.assertFalse(terminal.truncated)

    def test_flowmatch_dimensions_are_derived_from_checkpoint_shape(self):
        policy = FakeFlowMatchPolicy()
        self.assertEqual(policy_dimensions(policy), (16, 12, 6))

    def test_adapter_returns_clamped_physical_window(self):
        from dexmani_rl.policy.normalizer import FieldNormalizer

        policy = FakeFlowMatchPolicy()
        policy.normalizer.values["action"] = FieldNormalizer([2.] * 3, [100.] * 3)
        limits = torch.tensor([[-0.1, 0.1]] * 3)
        base = FrozenPolicy(policy, {"eval": {"inference_steps": 2}}, limits)
        adapter = FlowMatchPolicyAdapter(base)
        obs = {"joint_state": torch.zeros(1, 2, 3)}
        noise = torch.tensor([[101., 99., 100.] * 4])
        expected = torch.tensor([[[0.1, -0.1, 0.]] * 2])
        torch.testing.assert_close(adapter.predict(obs, noise=noise).action, expected, rtol=0, atol=0)

    def test_frozen_policy_facade_and_generic_adapter(self):
        policy = FakeFlowMatchPolicy()
        limits = torch.tensor([[-0.5, 0.5]] * policy.action_dim)
        base = FrozenPolicy(policy, {"eval": {"inference_steps": 2}, "control_dt_s": 0.05}, limits)
        obs = {"joint_state": torch.zeros(2, policy.n_obs_steps, policy.action_dim)}
        context, state = base.encode(obs)
        self.assertEqual(state.shape, (2, 16))
        noise = torch.ones(2, base.noise_dim)
        normalized = base.actions(context, noise)
        self.assertEqual(normalized.shape, (2, 6))
        self.assertTrue(torch.all(normalized <= 0.5))
        adapter = FlowMatchPolicyAdapter(base)
        for kind, unit, dt in [('latent', 'normalized', .05), ('joint_position', 'rad', .02)]:
            with self.assertRaises(ValueError):
                FlowMatchPolicyAdapter(base, ActionSpec((ActionSegment('joint', 3, kind, unit),), 4, 2, dt))
        output = adapter.predict(obs, noise=noise)
        self.assertEqual(output.action.shape, (2, 2, 3))
        self.assertEqual(output.features.shape, (2, 16))
        self.assertEqual(adapter.action_spec.action_dim, 3)


if __name__ == "__main__":
    unittest.main()
