import subprocess
import sys
import unittest

import torch

from dexmani_rl.core.contracts import StepResult
from dexmani_rl.core.specs import ActionSegment, ActionSpec
from dexmani_rl.envs.adapters import ChunkEnvAdapter
from dexmani_rl.common.base_policy import FrozenPolicy, policy_dimensions
from dexmani_rl.policies.flowmatch.adapter import FlowMatchPolicyAdapter


class IdentityField:
    def normalize(self, value):
        return value

    def unnormalize(self, value):
        return value


class FakeNormalizer:
    fields = {"joint_state", "action"}

    def __init__(self):
        self.values = {name: IdentityField() for name in self.fields}

    def __getitem__(self, name):
        return self.values[name]


class FakeObservationEncoder:
    out_dim = 5


class FakeFlowMatchPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.n_obs_steps = 2
        self.horizon = 4
        self.n_action_steps = 2
        self.action_dim = 3
        self.normalization_dims = {"joint_state": 3}
        self.obs_encoder = FakeObservationEncoder()
        self.normalizer = FakeNormalizer()

    def encode_observations(self, obs):
        batch = obs["joint_state"].shape[0]
        return torch.zeros(batch, self.n_obs_steps * 3, self.obs_encoder.out_dim)

    def decode_condition(self, context, noise, inference_steps):
        return noise


class FakeLegacyEnv:
    def reset(self, seed):
        return {"seed": seed}

    def step(self, action):
        return {"obs": action}, 1.0, False, True, {"steps": 3, "source": "fake"}


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
        code = (
            "import sys; import dexmani_rl.core; "
            "assert 'torch' not in sys.modules; "
            "assert 'dexmani_rl.policy.flowmatch' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_common_runtime_keeps_flowmatch_model_import_lazy(self):
        code = (
            "import sys; import dexmani_rl.common.base_policy; "
            "assert 'dexmani_rl.policy.flowmatch' not in sys.modules; "
            "assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

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
        self.assertTrue(result.truncated)
        self.assertEqual(result.executed_steps, 3)
        adapter.close()

    def test_flowmatch_dimensions_are_derived_from_checkpoint_shape(self):
        policy = FakeFlowMatchPolicy()
        self.assertEqual(policy_dimensions(policy), (16, 12, 6))

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
        output = adapter.predict(obs, noise=noise)
        self.assertEqual(output.action.shape, (2, 2, 3))
        self.assertEqual(output.features.shape, (2, 16))
        self.assertEqual(adapter.action_spec.action_dim, 3)


if __name__ == "__main__":
    unittest.main()
