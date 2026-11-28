# Architecture and Extension Rules

## Design goals

- Keep the shared kernel small and independent of any paper or robot.
- Keep each policy backend and each paper method in its own module.
- Describe action dimensions and semantics with data, not constants in trainers.
- Each method owns its training and evaluation entry points.
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

Do not modify existing policy adapters or algorithms merely to register a new model. Methods construct the selected backend explicitly.

## Adding a method

Give the paper its own `methods/<method>/` package and entry points. It may compose a policy adapter, one or more algorithms, representations, data sources, and an environment adapter. Keep paper-specific losses, schedules, and training stages inside that method. Each training entry point names its default YAML in `configs/methods/`. The package data directory `dexmani_rl/recipes` links to that directory so the wheel ships the same four recipes.

Do not create a universal BaseTrainer or BaseRLAgent. PPO, SAC, TD3, DAgger, and generative-policy matching have materially different update lifecycles. Methods own rollout and actor loading. Shared evaluation metrics operate on paired results.

## Removing a method or policy

A method owns its configuration and imports. Remove its package, configuration, and entry point together. Shared code must not import concrete methods. The isolation suite blocks each method in turn and runs the other three methods’ import, configuration and CPU update checks. A policy backend can be removed with its adapter and optional dependencies as long as no method explicitly selects it. Fresh Python processes verify lightweight imports of Core, policies and environment adapters, and exercise a separate policy fixture with the FlowMatch package blocked. These checks run using only the source tree and standard library.

## FlowMatch runtime

The network lives in `policies/flowmatch/model/`. The runtime and inference adapter live in
`policies/flowmatch/`. BC construction and weight loading live in `loader.py`. Training and evaluation use `FlowMatchRLRuntime` for observation
encoding, decoding and joint limits. Training scripts consume the five values
from `FlowMatchChunkEnv.step`; `ChunkEnvAdapter` exposes them as a `StepResult`.

## Validation

Run the lightweight contract suite with:

~~~bash
python -m unittest discover -s tests -v
~~~

CPU CI runs this suite, checks dependencies, compiles the sources and invokes all eight method entry points with `--help`. When compatible BC weights and simulator assets are available, run a one-update smoke test for each method. Record missing inputs and report each unrun integration check explicitly.

## Execution and data ownership

FlowMatchPolicyAdapter returns a batch of executable joint positions in radians with shape
`[B, execute_steps, action_dim]`. The prediction horizon describes the full
model sample. The decoder removes observation history before selecting the
future window. RoboFFT keeps the full sample for matching and selects the
execution window starting at `n_obs_steps - 1`. During online collection BORA
uses `JointActionTransform` to bound physical targets. Its critic uses
`envs.action_transform.executable_actions` and the same limits to express
both the base and corrected commands in normalized coordinates.

The method selects one batch element for the simulation environment. ChunkEnvAdapter
forwards the physical chunk, reports the actual control frames from info.steps,
and borrows the context-managed environment. Its owner closes the simulator.

Data handling includes demonstration episodes, BORA frame sampling and DSRL
transition replay. An algorithm owns parameter updates. A policy produces
predictions. A method owns data collection and training stages and explicitly
constructs its dependencies. A temporal ActionCodec encodes and decodes action
sequences. A MotionPrior models actions or latent codes conditioned on history.
