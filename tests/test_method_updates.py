"""Seeded CPU updates on tiny batches using the production update rules."""
import math
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from dexmani_rl.policies.flowmatch.rl_runtime import FlowMatchRLRuntime
from tests.flowmatch_fixtures import FakeFlowMatchPolicy


def recipe(method, **overrides):
    path = Path(__file__).resolve().parents[1] / 'configs' / 'methods' / f'{method}.yaml'
    return {**OmegaConf.to_container(OmegaConf.load(path)), **overrides}


class Velocity(nn.Linear):
    def forward(self, x, timestep, context, context_kv=None):
        return super().forward(x)


class TrainablePolicy(FakeFlowMatchPolicy):
    def __init__(self, state_dim=3):
        super().__init__()
        self.normalization_dims = {'joint_state': state_dim}
        self.obs_encoder = nn.Linear(state_dim, 5)
        self.obs_encoder.out_dim = 5
        self.obs_encoder.rgb_encoder = nn.Module()
        self.obs_encoder.rgb_encoder.backbone = nn.Identity()
        self.action_decoder = nn.Linear(3, 3)

    def encode_observations(self, obs):
        return self.obs_encoder(obs['joint_state']).repeat_interleave(2, dim=1)

    def decode_condition(self, context, noise, inference_steps, generator=None):
        if noise is None:
            noise = torch.randn(len(context), self.horizon, self.action_dim, generator=generator)
        return self.action_decoder(noise) + context.mean((1, 2))[:, None, None]


class MethodUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.state = torch.randn(2, 16)
        self.context = torch.randn(2, 6, 5)
        self.transitions = dict(state=self.state, value=torch.zeros(2, 1), next_value=torch.ones(2, 1),
                                reward=torch.tensor([[.1], [1.]]), terminated=torch.tensor([[False], [True]]),
                                truncated=torch.zeros(2, 1, dtype=torch.bool))

    def check_update(self, parameter, before, metrics):
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertFalse(torch.equal(parameter, before))

    def test_dsrl(self):
        from dexmani_rl.methods.dsrl.algorithm import DSRL
        from dexmani_rl.methods.dsrl.replay import Replay

        base = FlowMatchRLRuntime(FakeFlowMatchPolicy(), {'eval': {'inference_steps': 1}}, [[-1, 1]] * 3)
        agent = DSRL(base, recipe('dsrl', width=8, depth=1, utd=1, noise_updates=1, batch_size=2))
        replay = Replay(2)
        for i in range(2):
            replay.add(state=self.state[i], action=torch.randn(6), next_state=self.state[1-i],
                       context=self.context[i], next_context=self.context[1-i],
                       reward=self.transitions['reward'][i], terminated=self.transitions['terminated'][i])
        p = next(agent.target.parameters())
        before = p.detach().clone()
        self.check_update(p, before, agent.update(replay))

    def test_rfs(self):
        from dexmani_rl.methods.rfs.algorithm import RFS

        agent = RFS(16, 12, 6, recipe('rfs', hidden_dims=[8], batch_size=2, update_epochs=1), 'cpu')
        joint, log_prob = agent.actor.sample(self.state)
        p = agent.actor.log_std
        before = p.detach().clone()
        self.check_update(p, before, agent.update(dict(self.transitions, joint=joint, log_prob=log_prob)))

    def test_robofft(self):
        from dexmani_rl.methods.robofft.algorithm import RoboFFT, RoboFFTActor

        actor = RoboFFTActor(Velocity(3, 3), 4, 3, 1)
        cfg = recipe('robofft', batch_size=2, microbatch_size=1, critic_warmup_iterations=0,
                     repeat_samples=1, update_epochs=1, value_width=8, logprob_min=-20.)
        agent = RoboFFT(16, actor, cfg, 'cpu')
        action = actor.sample(self.context)
        eps, t = actor.draw_matching(action)
        batch = dict(self.transitions, context=self.context, action=action, eps=eps, t=t,
                     first=torch.tensor([1, 0]), old_log_prob=actor.log_prob(self.context, action, eps, t).detach())
        p = actor.decoder.weight
        before = p.detach().clone()
        self.check_update(p, before, agent.update(batch, 0))
        agent.finish_iteration(0)

    def test_bora(self):
        from dexmani_rl.methods.bora.algorithm import BORA

        policy = TrainablePolicy()
        agent = BORA(policy, {'eval': {'inference_steps': 1}, 'agent': {}},
                     recipe('bora', hidden_dim=8, advantage_gate=None))
        batch = dict(obs={'joint_state': torch.randn(2, 2, 3)}, next_obs={'joint_state': torch.randn(2, 2, 3)},
                     action=torch.randn(2, 2, 3), mask=torch.ones(2, 2, dtype=torch.bool),
                     done=torch.tensor([[False, False], [False, True]]), reward=torch.randn(2, 2))
        for ppo in (False, True):
            p = policy.action_decoder.weight
            before = p.detach().clone()
            self.check_update(p, before, agent.offline_update(batch, ppo=ppo))
        for stage in (1, 2):
            agent.begin_round(stage)
            p = agent.actor.residuals[stage - 1][-1].weight
            before = p.detach().clone()
            metrics = agent.online_update(batch, step=0, joint_limits=torch.tensor([[-1., 1.]] * 3))
            self.check_update(p, before, metrics)
