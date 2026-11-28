"""Minimal structural contracts; implementations remain owned by each backend."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, Protocol

from .specs import ActionSpec


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    action: Any
    features: Any | None = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    executed_steps: int = 1
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reward)):
            raise ValueError("StepResult.reward must be finite")
        if type(self.executed_steps) is not int or self.executed_steps < 1:
            raise ValueError("StepResult.executed_steps must be a positive integer")

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class PolicyAdapter(Protocol):
    @property
    def action_spec(self) -> ActionSpec: ...

    def predict(self, observation: Any, **kwargs: Any) -> PolicyOutput: ...


class EnvAdapter(Protocol):
    def reset(self, seed: int | None = None) -> Any: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...
