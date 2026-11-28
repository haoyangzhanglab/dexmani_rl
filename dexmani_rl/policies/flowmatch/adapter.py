"""Generic inference adapter for the repository's FlowMatch policy runtime."""

from dataclasses import replace
from math import isclose

import torch

from ...core.contracts import PolicyOutput
from ...core.specs import ActionSegment, ActionSpec


class FlowMatchPolicyAdapter:
    """Expose a FlowMatch policy through the model-agnostic PolicyAdapter contract.

    base is the backwards-compatible FrozenPolicy runtime used by the current
    DSRL/RFS implementations. Supplying an explicit ActionSpec allows callers to
    describe arm/hand segments without hard-coding those dimensions here.
    """

    def __init__(self, base, action_spec: ActionSpec | None = None) -> None:
        self.base = base
        policy = base.policy
        self._action_spec = action_spec or ActionSpec(
            segments=(ActionSegment("joint", policy.action_dim, "joint_position", "rad"),),
            horizon=policy.horizon,
            execute_steps=policy.n_action_steps,
            control_dt=base.cfg.get("control_dt_s"),
        )
        if any((part.kind, part.unit) != ("joint_position", "rad") for part in self._action_spec.segments):
            raise ValueError("FlowMatch outputs joint positions in radians")
        dt = base.cfg.get("control_dt_s")
        if dt is not None:
            if self._action_spec.control_dt is None:
                self._action_spec = replace(self._action_spec, control_dt=dt)
            elif not isclose(self._action_spec.control_dt, dt, rel_tol=1e-6):
                raise ValueError("ActionSpec control period differs from checkpoint")
        if self._action_spec.action_dim != policy.action_dim:
            raise ValueError("ActionSpec dimension must match the FlowMatch policy action dimension")
        if self._action_spec.horizon != policy.horizon:
            raise ValueError("ActionSpec horizon must match the FlowMatch policy horizon")
        if self._action_spec.execute_steps != policy.n_action_steps:
            raise ValueError("ActionSpec execute_steps must match the FlowMatch execution window")

    @property
    def action_spec(self) -> ActionSpec:
        return self._action_spec

    @torch.no_grad()
    def predict(self, observation, *, noise=None, generator=None, include_context: bool = False) -> PolicyOutput:
        context, state = self.base.encode(observation)
        if noise is None:
            noise = torch.randn(
                (len(context), self.base.noise_dim), device=self.base.device,
                dtype=context.dtype, generator=generator,
            )
        normalized = self.base.actions(context, noise)
        auxiliary = {"context": context} if include_context else {}
        return PolicyOutput(action=self.base.physical(normalized), features=state, auxiliary=auxiliary)
