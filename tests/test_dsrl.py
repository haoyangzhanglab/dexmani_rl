import copy
import pytest
import torch
from dexmani_rl.dsrl import Actor, DSRL, td_target
from dexmani_rl.replay import Replay


class Base:
    device = torch.device('cpu')
    state_dim = 3
    noise_dim = 6
    action_dim = 2
    def actions(self, context, noise):
        return noise[:, :2].tanh()


def test_log_probability():
    torch.manual_seed(0)
    actor = Actor(3, 6, 16, 2)
    state = torch.randn(8, 3)
    generator = torch.Generator().manual_seed(12)
    u, logp = actor.sample(state, generator)
    mean, log_std = actor.net(state).chunk(2, -1)
    std = log_std.clamp(-20, 2).exp()
    z = mean + std * torch.randn(mean.shape, generator=torch.Generator().manual_seed(12))
    transform = torch.distributions.TanhTransform()
    expected = (torch.distributions.Normal(mean, std).log_prob(z)
                - transform.log_abs_det_jacobian(z, z.tanh())).sum(-1, keepdim=True)
    torch.testing.assert_close(u, z.tanh())
    torch.testing.assert_close(logp, expected, atol=2e-5, rtol=2e-5)
    actual_grads = torch.autograd.grad(logp.sum(), actor.parameters())
    expected_grads = torch.autograd.grad(expected.sum(), actor.parameters())
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


def test_terminal_and_timeout_targets():
    y = td_target(torch.tensor([[1.], [0.]]), torch.tensor([[1.], [0.]]),
                  torch.tensor([[2., 3.], [2., 3.]]), torch.zeros(2, 1), 1., .99)
    torch.testing.assert_close(y, torch.tensor([[1.], [1.98]]))


def test_nonfinite_gradient_stops_before_optimizer_step():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    with pytest.raises(RuntimeError, match='non-finite'):
        DSRL.optimize(optimizer, parameter.sqrt())
    assert parameter.item() == 0.


def test_updates_and_restore():
    cfg = dict(width=16, depth=2, lr=.001, batch_size=4, utd=2, noise_updates=2,
               noise_magnitude=1.5, gamma=.99, tau=.005)
    model = DSRL(Base(), cfg)
    replay = Replay(5)
    for _ in range(8):
        replay.add(context=torch.randn(2, 3), next_context=torch.randn(2, 3), state=torch.randn(3),
                   next_state=torch.randn(3), action=torch.randn(2), reward=[1.], terminated=[0.])
    before = copy.deepcopy(model.state_dict())
    metrics = model.update(replay)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    for prefix in ('actor.', 'critic.', 'noise_critic.', 'target.'):
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if k.startswith(prefix))
    assert all(p.grad is None for p in model.target.parameters())
    other = Replay(1)
    other.load_state_dict(replay.state_dict())
    assert other.position == replay.position and other.size == replay.size
    for key in replay.data:
        torch.testing.assert_close(other.data[key], replay.data[key])
    clone = DSRL(Base(), cfg)
    clone.load_state_dict(model.state_dict())
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b)
