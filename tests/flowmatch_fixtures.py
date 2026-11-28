"""FlowMatch doubles shared by the CPU regression tests."""

import torch


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
