"""Compare RoboFFT-F on CPU against a supplied copy of the upstream source."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from dexmani_rl.robofft import RoboFFT, RewardScaler, ValueNetwork, ppo_loss
from test_robofft import actor


class ReferenceActor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = copy.deepcopy(decoder)

    def forward(self, x, t, obs):
        return self.decoder(x, t, obs['state'])


class ReferenceValue(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = copy.deepcopy(value)

    def forward(self, obs):
        return self.value(obs['critic_state'])


def reference_flow(flow_class, local, value):
    # Upstream __init__ reads a Robomimic checkpoint. Here both sides use the same toy weights.
    model = flow_class.__new__(flow_class)
    nn.Module.__init__(model)
    options = dict(device='cpu', inference_steps=10, horizon_steps=3, action_dim=2,
                   n_samples_per_action=local.n_samples, matching_style='u',
                   logprob_min=-1., logprob_max=1., logratio_alpha=8.,
                   use_per_sample_ratio=True, clip_ploss_coef=.02, clip_ploss_style='ppo',
                   clip_vloss_coef=None, min_sampling_denoising_std=.25,
                   max_logprob_denoising_std=.25, zero_start_eval=False, zero_start_train=False,
                   denoised_clip_value=1., act_min=-1., act_max=1., sample_t_type='linspace')
    for name, val in options.items():
        setattr(model, name, val)
    model.actor_ft = ReferenceActor(local.decoder)
    model.critic = ReferenceValue(value)
    return model


def check_objective(flow_class, n_samples):
    torch.manual_seed(31)
    local, value = actor(n_samples), ValueNetwork(4, 16)
    reference = reference_flow(flow_class, local, value)
    context = torch.randn(4, 2, 3) * .1
    state = torch.randn(4, 4)
    action = torch.randn(4, 3, 2) * .2
    eps, t = local.draw_matching(action)
    # Include active gradients on each side of PPO's ratio clip for both advantage signs.
    eps = eps * .3
    obs = {'state': context, 'critic_state': state}
    norm = dict(normalize_act_space_dimension=True, normalize_act_horizon_dimension=True)
    logp = local.log_prob(context, action, eps, t)
    expected, _, _ = reference.get_logprobs_eps_t(obs, action, eps=eps, t=t, **norm)
    torch.testing.assert_close(logp, expected, rtol=1e-6, atol=1e-7)
    old = logp.detach() + torch.tensor([[-.04, .02], [.04, -.01], [-.1, .01], [.2, -.2]])[:, :n_samples]
    advantage = torch.tensor([-3., -1., 1., 3.])
    returns = torch.tensor([-.2, -.1, .3, .7])
    original = reference.loss(obs, action, returns, torch.zeros(4), advantage, old, eps, t, verbose=False, **norm)
    policy_loss, diagnostics = ppo_loss(logp, old, advantage / advantage.std())
    value_loss = .5 * (value(state).flatten() - returns).square().mean()
    torch.testing.assert_close(policy_loss, original[0])
    torch.testing.assert_close(value_loss, original[2])
    for key, target in zip(('clip_fraction', 'approx_kl', 'ratio'), original[4:7]):
        torch.testing.assert_close(diagnostics[key], torch.tensor(target), atol=1e-7, rtol=0)
    (policy_loss + .5 * value_loss).backward()
    (original[0] + .5 * original[2]).backward()
    gradient_error = 0.
    for model, ref, lr in ((local, reference.actor_ft, 1e-6), (value, reference.critic, 1e-3)):
        for a, b in zip(model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-7)
            gradient_error = max(gradient_error, (a.grad - b.grad).abs().max().item())
        torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0).step()
        torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=0).step()
        for a, b in zip(model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    return dict(logprob_max_error=(logp - expected).abs().max().item(), policy_loss=policy_loss.item(),
                gradient_max_error=gradient_error, adamw_update=True)


def check_release(source):
    for package in ('model', 'util'):
        if not (source / package / '__init__.py').is_file():
            raise ValueError(f'Cannot find the upstream RoboFFT-F package {package!r} in {source}')
        # The upstream files use these top-level imports, which bypass sys.path when already loaded.
        if any(name == package or name.startswith(f'{package}.') for name in sys.modules):
            raise RuntimeError(f'Package {package!r} is already loaded. Run validation in a fresh process.')
    sys.path.insert(0, str(source))
    from model.common.mlp import ResidualMLP
    from model.flow.ft_fpo.fpoflow import FPOFlow
    from util.reward_scaling import RunningRewardScaler
    from util.scheduler import CosineAnnealingWarmupRestarts

    results = {f'mc{n}': check_objective(FPOFlow, n) for n in (1, 2)}
    for eval_mode in (False, True):
        local = actor()
        reference = reference_flow(FPOFlow, local, ValueNetwork(4, 16))
        context = torch.full((4, 2, 3), 2.)
        torch.manual_seed(73)
        expected = reference.get_actions({'state': context}, eval_mode=eval_mode)
        actual = local.sample(context, torch.Generator().manual_seed(73), eval_mode=eval_mode)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        results[f'sampling_eval_{eval_mode}'] = (actual - expected).abs().max().item()

    reference_scaler = RunningRewardScaler(1, gamma=.999)
    local_scaler = RewardScaler(.999)
    for reward, first in (([1., 0., 1., 0.], [1., 0., 0., 0.]), ([0., 2., 3., 4.], [0., 0., 1., 0.])):
        reward, first = np.array([reward], dtype=np.float32), np.array([first], dtype=np.float32)
        expected = reference_scaler(reward, first)
        actual = local_scaler(torch.from_numpy(reward.T), torch.from_numpy(first.T)).numpy().T
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(local_scaler.var, reference_scaler.ret_rms.var)
        np.testing.assert_array_equal(local_scaler.mean, reference_scaler.ret_rms.mean)
    results['reward_scaling'] = 'exact'

    official_mlp = ResidualMLP([4, 16, 16, 16, 1], activation_type='Mish')
    local_mlp = ValueNetwork(4, 16)
    with torch.no_grad():
        for actual, expected in zip(local_mlp.parameters(), official_mlp.parameters(), strict=True):
            actual.copy_(expected)
    state = torch.randn(10, 4)
    torch.testing.assert_close(local_mlp(state), official_mlp(state), atol=0, rtol=0)
    results['critic'] = 'exact'

    cfg = OmegaConf.to_container(OmegaConf.load(PROJECT / 'configs/robofft.yaml'))
    model = RoboFFT(4, actor(), cfg, 'cpu')
    ref_optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(()))], lr=2e-6, weight_decay=0)
    scheduler = CosineAnnealingWarmupRestarts(ref_optimizer,
        first_cycle_steps=151, max_lr=2e-6, min_lr=1e-6, warmup_steps=10, cycle_mult=1., gamma=1.)
    lr_error = 0.
    for iteration in range(151):
        actual = model.optimizers['actor'].param_groups[0]['lr']
        expected = ref_optimizer.param_groups[0]['lr']
        error = abs(actual - expected)
        if not error <= 1e-15:
            raise AssertionError(f'Iteration {iteration}: local lr {actual}, upstream lr {expected}')
        lr_error = max(lr_error, error)
        if iteration >= 2:
            model.optimizers['actor'].step()
            ref_optimizer.step()
            scheduler.step()
        model.finish_iteration(iteration)
    return dict(results, actor_lr_max_abs_error=lr_error, scheduler_iterations=151, reference_source=str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True, help='Upstream robofft-f source directory')
    parser.add_argument('--output', type=Path, required=True, help='New JSON result path')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = check_release(args.reference_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
