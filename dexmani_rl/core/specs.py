"""Small, model-agnostic specifications shared across policies and environments."""

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionSegment:
    """One semantically homogeneous part of an action vector."""

    name: str
    dim: int
    kind: str = "continuous"
    unit: str = "normalized"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ValueError("ActionSegment.name must be a non-empty trimmed string")
        if type(self.dim) is not int or self.dim <= 0:
            raise ValueError("ActionSegment.dim must be a positive integer")
        if (not isinstance(self.kind, str) or not self.kind or
                not isinstance(self.unit, str) or not self.unit):
            raise ValueError("ActionSegment.kind and unit must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Describes action meaning and temporal shape without owning action logic.

    horizon is the predicted number of control frames; execute_steps is the
    prefix sent to the environment. segments describe the final axis.
    """

    segments: tuple[ActionSegment, ...]
    horizon: int = 1
    execute_steps: int = 1
    control_dt: float | None = None

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        object.__setattr__(self, "segments", segments)
        if not segments:
            raise ValueError("ActionSpec requires at least one action segment")
        if any(not isinstance(segment, ActionSegment) for segment in segments):
            raise TypeError("ActionSpec.segments must contain ActionSegment values")
        names = [segment.name for segment in segments]
        if len(names) != len(set(names)):
            raise ValueError("ActionSpec segment names must be unique")
        if type(self.horizon) is not int or self.horizon <= 0:
            raise ValueError("ActionSpec.horizon must be a positive integer")
        if type(self.execute_steps) is not int or not 1 <= self.execute_steps <= self.horizon:
            raise ValueError("ActionSpec.execute_steps must be in [1, horizon]")
        if self.control_dt is not None and (
            isinstance(self.control_dt, bool)
            or not isinstance(self.control_dt, (int, float))
            or not math.isfinite(self.control_dt)
            or self.control_dt <= 0
        ):
            raise ValueError("ActionSpec.control_dt must be finite and positive")

    @property
    def action_dim(self) -> int:
        return sum(segment.dim for segment in self.segments)

    @property
    def slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        start = 0
        for segment in self.segments:
            result[segment.name] = slice(start, start + segment.dim)
            start += segment.dim
        return result

    def split(self, action: Any) -> dict[str, Any]:
        """Split a NumPy array or tensor along its final dimension."""
        if not hasattr(action, "shape") or not action.shape or action.shape[-1] != self.action_dim:
            shape = getattr(action, "shape", None)
            raise ValueError(f"Expected final action dimension {self.action_dim}, got {shape}")
        return {name: action[..., part] for name, part in self.slices.items()}
