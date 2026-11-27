"""FlowMatch with uniform Euler integration from noise (0) to action (1)."""
from __future__ import annotations

from typing import Literal
import torch
from torch import nn
from .dit import DiTX, FactorizedActionIO
from .encoder import RGBPointObservationEncoder
from .normalizer import LinearNormalizer


class RectifiedFlow(nn.Module):
    def __init__(self, model, num_inference_steps=10):
        super().__init__()
        self.model = model
        self.num_inference_steps = num_inference_steps

    def predict_action(self, cond, action_shape, inference_steps=None, *, generator=None, initial_noise=None):
        steps = self.num_inference_steps if inference_steps is None else inference_steps
        if type(steps) is not int or steps <= 0:
            raise ValueError('inference_steps must be positive')
        sample = (torch.randn(action_shape, device=cond.device, dtype=torch.float32, generator=generator)
                  if initial_noise is None else initial_noise)
        if tuple(sample.shape) != tuple(action_shape):
            raise ValueError('initial_noise must cover the complete prediction horizon')
        context_kv = self.model.prepare_context_kv(cond) if steps > 1 else None
        times = torch.tensor([i / steps for i in range(steps)], device=cond.device, dtype=torch.float32)
        for t in times:
            velocity = self.model(x=sample, timestep=t.expand(len(cond)), context=cond, context_kv=context_kv)
            sample = sample + (1.0 / steps) * velocity
        return sample


class FlowMatch(nn.Module):
    required_identity_fields = frozenset({"point_cloud", "camera_intrinsic", "camera_extrinsic"})

    def __init__(
        self,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        action_dim: int = 19,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        pc_encoder_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
        n_layers: int = 6,
        hidden_dim: int = 768,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.1,
        timestep_embed_dim: int = 128,
        num_inference_steps: int = 10,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        factorized: bool = False,
    ) -> None:
        super().__init__()
        if action_dim != 19:
            raise ValueError("FlowMatch requires joint action=7+12")

        if not 0 < n_action_steps <= horizon - n_obs_steps + 1:
            raise ValueError('Invalid execution window')

        obs_encoder = RGBPointObservationEncoder(
            pc_dim=pc_dim,
            state_dim=state_dim,
            num_points=num_points,
            n_obs_steps=n_obs_steps,
            pc_encoder_config=pc_encoder_config,
            rgb_backbone_config=rgb_backbone_config,
            rgb_backbone_name=rgb_backbone_name,
            rgb_semantic_sampling=rgb_semantic_sampling,
        )

        backbone = DiTX(
            horizon=horizon,
            action_dim=action_dim,
            n_obs_steps=n_obs_steps,
            obs_token_dim=obs_encoder.out_dim,
            timestep_embed_dim=timestep_embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            action_io=FactorizedActionIO(horizon, hidden_dim) if factorized else None,
        )

        action_decoder = RectifiedFlow(
            model=backbone,
            num_inference_steps=num_inference_steps,
        )

        self.obs_encoder, self.action_decoder = obs_encoder, action_decoder
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.n_action_steps, self.action_dim = n_action_steps, action_dim
        self.normalizer = LinearNormalizer()
        self.normalization_dims: dict[str, int] = {}

    def configure_normalizer(self, feature_dims, feature_shapes):
        self.normalization_dims = dict(feature_dims)
        self.normalizer = LinearNormalizer(feature_shapes)

    def encode_observations(self, obs):
        values = {}
        for key, value in obs.items():
            if value.shape[1] != self.n_obs_steps:
                raise ValueError('Observation history length mismatch')
            if key in self.normalizer.fields:
                value = self.normalizer[key].normalize(value)
            values[key] = value.flatten(0, 1)
        return self.obs_encoder(values)

    def decode_condition(self, condition, initial_noise=None, inference_steps=None, generator=None):
        """Decode the normalized future window, excluding observation history."""
        prediction = self.action_decoder.predict_action(
            condition, (len(condition), self.horizon, self.action_dim),
            inference_steps=inference_steps, initial_noise=initial_noise, generator=generator)
        return prediction[:, self.n_obs_steps - 1:]

    @torch.no_grad()
    def predict_action(self, obs_dict, inference_steps=None, *, generator=None, initial_noise=None):
        prediction = self.decode_condition(
            self.encode_observations(obs_dict), initial_noise, inference_steps, generator)
        return {'future_actions': self.normalizer['action'].unnormalize(prediction)}
