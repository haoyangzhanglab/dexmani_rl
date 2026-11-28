"""Runtime pieces for FlowMatch; kept separate from RL algorithm implementations."""

import torch


class FlowMatchObservationEncoder:
    """Build the observation context and the compact state used by legacy RL agents."""

    def __init__(self, policy) -> None:
        self.policy = policy
        joint_dim = policy.normalization_dims.get("joint_state")
        if type(joint_dim) is not int or joint_dim <= 0:
            raise ValueError("FlowMatch policy requires a positive joint_state dimension")
        self.state_dim = policy.n_obs_steps * (policy.obs_encoder.out_dim + joint_dim)
        self.noise_dim = policy.horizon * policy.action_dim

    @torch.no_grad()
    def encode(self, observation):
        context = self.policy.encode_observations(observation)
        frames = context.unflatten(1, (self.policy.n_obs_steps, -1))
        if frames.shape[2] < 2:
            raise ValueError("FlowMatch context must contain spatial tokens and a joint-state token")
        joint = observation["joint_state"]
        if "joint_state" in self.policy.normalizer.fields:
            joint = self.policy.normalizer["joint_state"].normalize(joint)
        if joint.shape[1:] != (self.policy.n_obs_steps, self.policy.normalization_dims["joint_state"]):
            raise ValueError("joint_state shape does not match the FlowMatch policy configuration")
        state = torch.cat((frames[:, :, :-1].mean(2), joint), dim=-1).flatten(1)
        return context, state


class FlowMatchActionDecoder:
    """Decode latent noise into the executable prefix of a FlowMatch action chunk."""

    def __init__(self, policy, inference_steps: int, decode_batch: int = 64) -> None:
        if type(inference_steps) is not int or inference_steps <= 0:
            raise ValueError("inference_steps must be a positive integer")
        if type(decode_batch) is not int or decode_batch <= 0:
            raise ValueError("decode_batch must be a positive integer")
        self.policy = policy
        self.inference_steps = inference_steps
        self.decode_batch = decode_batch

    @torch.no_grad()
    def decode(self, context, noise):
        expected = self.policy.horizon * self.policy.action_dim
        if noise.ndim != 2 or noise.shape != (len(context), expected):
            raise ValueError(f"Expected flattened FlowMatch noise [B, {expected}], got {tuple(noise.shape)}")
        chunks = []
        for context_batch, noise_batch in zip(context.split(self.decode_batch), noise.split(self.decode_batch)):
            initial_noise = noise_batch.reshape(-1, self.policy.horizon, self.policy.action_dim)
            prediction = self.policy.decode_condition(context_batch, initial_noise, self.inference_steps)
            chunks.append(prediction[:, :self.policy.n_action_steps])
        return torch.cat(chunks, dim=0)
