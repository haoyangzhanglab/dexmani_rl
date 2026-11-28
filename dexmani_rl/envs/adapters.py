"""Environment adapters and compatibility bridges."""

from typing import Any

from ..core.contracts import StepResult


class ChunkEnvAdapter:
    """Adapt the existing five-tuple ChunkEnv API to the EnvAdapter contract.

    The wrapped environment remains owned by its existing context manager. This
    adapter deliberately does not close borrowed environments on close().
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self, seed: int | None = None):
        return self.env.reset(seed)

    def step(self, action) -> StepResult:
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        return StepResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            executed_steps=info["steps"],
            info=info,
        )

    def close(self) -> None:
        """No-op: the caller that created the wrapped context owns its lifetime."""
        return None
