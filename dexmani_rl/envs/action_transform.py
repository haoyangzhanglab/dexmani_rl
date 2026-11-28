"""Conversions between normalized policy actions and executable joint targets."""

from typing import Any

import torch


def _limits_tensor(limits: Any, *, device: torch.device, dtype: torch.dtype, action_dim: int) -> torch.Tensor:
    value = torch.as_tensor(limits, device=device, dtype=dtype)
    if value.shape != (action_dim, 2):
        raise ValueError(f"Expected joint limits [{action_dim}, 2], got {tuple(value.shape)}")
    if not torch.isfinite(value).all() or not (value[:, 0] <= value[:, 1]).all():
        raise ValueError("Joint limits must be finite with lower <= upper")
    return value


def joint_targets(window: torch.Tensor, normalizer: Any, limits: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return requested and limit-clamped physical joint targets."""
    requested = normalizer.unnormalize(window)
    bounds = torch.as_tensor(limits, device=requested.device)
    executable = requested.clamp(bounds[:, 0], bounds[:, 1])
    return requested, executable


def executable_actions(window: torch.Tensor, normalizer: Any, limits: Any) -> torch.Tensor:
    """Express the actually executable action in the critic's normalized coordinates."""
    _, executable = joint_targets(window, normalizer, limits)
    return normalizer.normalize(executable)


class JointActionTransform:
    """Own joint-limit and normalization conversions, independent of policy models."""

    def __init__(self, normalizer: Any, limits: Any, *, action_dim: int, execute_steps: int) -> None:
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        if type(execute_steps) is not int or execute_steps <= 0:
            raise ValueError("execute_steps must be a positive integer")
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.execute_steps = execute_steps
        self.limits = _limits_tensor(
            limits, device=torch.as_tensor(limits).device, dtype=torch.float32,
            action_dim=action_dim,
        )

    def executable_coordinates(self, window: torch.Tensor, *, return_clip_fraction: bool = False):
        requested, executable = joint_targets(window, self.normalizer, self.limits)
        normalized = self.normalizer.normalize(executable).flatten(1)
        if return_clip_fraction:
            return normalized, (requested != executable).float().mean()
        return normalized

    def physical(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        window = normalized_actions.reshape(-1, self.execute_steps, self.action_dim)
        return self.normalizer.unnormalize(window)

    def bounded_physical(self, normalized_window: torch.Tensor) -> torch.Tensor:
        return joint_targets(normalized_window, self.normalizer, self.limits)[1]
