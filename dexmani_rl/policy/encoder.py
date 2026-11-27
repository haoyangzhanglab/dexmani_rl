"""RGB-point scene tokens with late joint-state conditioning."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
import torch
from torch import nn
from .pointcloud import LocalPatchEncoder, SpatialContextEncoder, preprocess_point_cloud
from .fusion import RGBPatchFusion, project_rgb_patches
from .vision import build_rgb_encoder


class StateEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        out_dim: int = 64,
        hidden_dims: Sequence[int] = (64,),
    ):
        super().__init__()
        self.out_dim = out_dim
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RGBPointObservationEncoder(nn.Module):
    def __init__(
        self,
        pc_dim: int = 3,
        state_dim: int = 19,
        num_points: int = 1024,
        n_obs_steps: int = 2,
        pc_encoder_config: dict | None = None,
        rgb_backbone_config: dict | None = None,
        rgb_backbone_name: Literal["dino", "dinov3", "siglip"] = "dino",
        rgb_semantic_sampling: Literal["center", "group_mean"] = "center",
    ) -> None:
        super().__init__()
        if rgb_backbone_name not in ("dino", "dinov3", "siglip"):
            raise ValueError("rgb_backbone_name must be 'dino', 'dinov3' or 'siglip'")
        if rgb_semantic_sampling not in ("center", "group_mean"):
            raise ValueError("rgb_semantic_sampling must be 'center' or 'group_mean'")
        patch_config = dict(pc_encoder_config or {})
        spatial_config = {
            key: patch_config.pop(key)
            for key in ("depth", "num_heads", "mlp_ratio", "rope_scale", "rope_base")
            if key in patch_config
        }
        self.rgb_semantic_sampling = rgb_semantic_sampling
        self.num_points = num_points
        self.patch_encoder = LocalPatchEncoder(input_channels=pc_dim, **patch_config)
        self.rgb_encoder, self.image_processor = build_rgb_encoder(rgb_backbone_name, rgb_backbone_config)
        self.rgb_fusion = RGBPatchFusion(self.rgb_encoder.out_dim, self.patch_encoder.out_dim)
        self.spatial_encoder = SpatialContextEncoder(channels=self.rgb_fusion.out_dim, **spatial_config)
        self.n_obs_steps = n_obs_steps
        self.out_dim = self.spatial_encoder.out_dim
        self.joint_encoder = StateEncoder(state_dim, self.out_dim)
        self.token_type_embedding = nn.Parameter(torch.empty(1, 2, self.out_dim))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        self.token_norm = nn.LayerNorm(self.out_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        points = preprocess_point_cloud(obs["point_cloud"], self.num_points)
        patches = self.patch_encoder(points)
        image_grid, projection = project_rgb_patches(
            obs, points, patches, image_processor=self.image_processor,
            rgb_encoder=self.rgb_encoder, sampling=self.rgb_semantic_sampling,
        )
        tokens = self.rgb_fusion(patches["tokens"], image_grid, projection)
        spatial = self.spatial_encoder(tokens, patches["xyz"])
        joint = self.joint_encoder(obs["joint_state"]).unsqueeze(1)
        spatial = spatial + self.token_type_embedding[:, :1].to(spatial.dtype)
        joint = joint + self.token_type_embedding[:, 1:].to(joint.dtype)
        frames = self.token_norm(torch.cat((spatial, joint), dim=1))
        return frames.unflatten(0, (-1, self.n_obs_steps)).flatten(1, 2)
