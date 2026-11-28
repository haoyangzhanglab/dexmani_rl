# Architecture and Extension Rules

## Design goals

- Keep the shared kernel small and independent of any paper or robot.
- Keep each policy backend and each paper method in its own module.
- Describe action dimensions and semantics with data, not constants in trainers.
- Preserve the current training entry points while migrating behind compatibility facades.
- Make optional model dependencies lazy so removing one backend does not break imports of another.

## Dependency direction

~~~text
core
 ↑       ↑       ↑
policies algorithms envs data
  \       |       /
       methods
~~~

The core package must not import a policy, algorithm, simulator, or dataset implementation. A method may compose the components it needs. A policy must not import a trainer. An algorithm must not import a concrete policy implementation. Environment adapters must not know which RL algorithm is using them.

## Stable contracts

- ActionSpec describes action segments, prediction horizon, execution prefix, and control period. It does not encode or decode actions.
- PolicyAdapter exposes action_spec and predict. PolicyOutput.action follows the declared ActionSpec and may be a physical command or a latent action representation. If it is latent, the method must decode it before calling EnvAdapter.step. Model-specific features can be returned in PolicyOutput.auxiliary without expanding the common interface for every paper.
- EnvAdapter exposes reset, step returning StepResult, and close.
- ActionCodec and MotionPrior should be introduced as separate representation contracts when their first concrete implementation lands. A temporal codec and a history-conditioned prior are not the same abstraction and must not be forced into one class.

## Adding a policy

1. Add the model and its adapter under policies/<backend>/.
2. Implement PolicyAdapter and provide an ActionSpec matching the actual output.
3. Keep backend-specific imports inside the backend module or loader.
4. Add tests for output shape, action semantics, and checkpoint loading.
5. Add a method only if the policy needs a distinct training procedure.

Do not modify existing policy adapters, RL algorithms, or the shared evaluator merely to register a new model. Do not add a global registry until there is a demonstrated need for third-party runtime discovery.

## Adding a method

Give the paper its own methods/<method>/ package, configuration, and entry point. It may compose a policy adapter, one or more algorithms, representations, data sources, and an environment adapter. Keep paper-specific losses, schedules, and training stages inside that method.

Do not create a universal BaseTrainer or BaseRLAgent. PPO, SAC, TD3, DAgger, and generative-policy matching have materially different update lifecycles. The current evaluate.py remains the compatibility evaluator for the four existing methods; a new method should own its evaluation entry point or use a genuinely generic adapter evaluator rather than adding method-specific branches to the legacy file.

## Removing a method or policy

A method owns its configuration and imports. Remove its package, configuration, and entry point together. Shared code must not import method packages. A policy backend can be removed with its adapter and optional dependencies as long as no method explicitly selects it. This is enforced by lazy backend imports and dependency-direction tests/review.

## Current compatibility layer

- common/base_policy.py FrozenPolicy remains the public compatibility facade for DSRL and RFS. Observation encoding, FlowMatch decoding, and joint action transforms are delegated to smaller components.
- ChunkEnv retains its existing five-tuple API for the current training scripts. ChunkEnvAdapter exposes the new StepResult contract to new code without forcing a simultaneous migration of all trainers.
- The existing policy/ directory remains the FlowMatch model implementation during this migration. policies/ contains runtime adapters, not duplicate model definitions.

## Validation

Run the lightweight contract suite with:

~~~bash
python -m unittest discover -s tests -v
~~~

The contract suite does not replace simulator integration tests. When compatible BC weights and simulator assets are available, run a one-update smoke test for each method. Record missing inputs and report each unrun integration check explicitly.

## Execution and data ownership

FlowMatch returns a batch of executable joint positions in radians with shape
`[B, execute_steps, action_dim]`. The prediction horizon describes the full
model sample. The decoder removes observation history before selecting the
future window. RoboFFT keeps the full sample for matching and selects the
execution window starting at `n_obs_steps - 1`. BORA evaluates the clamped
physical target after mapping it back to the critic's normalized coordinates.

The method selects one batch element for the legacy environment. ChunkEnvAdapter
forwards the physical chunk, reports the actual control frames from info.steps,
and borrows the context-managed environment. Its owner closes the simulator.

Data handling includes demonstration episodes, BORA frame sampling and DSRL
transition replay. An algorithm owns parameter updates. A policy produces
predictions. A method owns data collection and training stages and explicitly
constructs its dependencies. A temporal ActionCodec encodes and decodes action
sequences. A MotionPrior models actions or latent codes conditioned on history.
