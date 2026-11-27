"""Deterministic point sampling, local geometry and spatial attention."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


def build_mlp(
    channels: list[int],
    *,
    normalize: bool = True,
    activate_last: bool = False,
) -> nn.Sequential:
    """Linear layers with optional LayerNorm and activation after hidden layers."""
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(channels, channels[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(channels) - 2 or activate_last:
            if normalize:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def farthest_point_sample(
    pointcloud: torch.Tensor,
    num_samples: int = 1024,
) -> torch.Tensor:
    """Sample original points (K <= N) by XYZ distance, preserving all channels."""
    from pytorch3d.ops import sample_farthest_points

    if pointcloud.size(-1) < 3:
        raise ValueError("pointcloud must have at least three XYZ channels")

    xyz = pointcloud[..., :3]
    sampled_xyz, sample_idx = sample_farthest_points(
        points=xyz,
        K=num_samples,
        random_start_point=False,
    )

    # Reuse native output for XYZ-only input.
    if pointcloud.size(-1) == 3:
        return sampled_xyz.contiguous()
    return index_points(pointcloud, sample_idx).contiguous()


def preprocess_point_cloud(
    pc: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Sample FP32 points with deterministic FPS."""
    pc = pc.float()
    if pc.shape[1] > num_points:
        pc = farthest_point_sample(pc, num_points)
    return pc


def index_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch_size, num_points = points.shape[:2]
    flat = points.reshape(batch_size, num_points, -1)
    expanded = index.reshape(batch_size, -1, 1).expand(-1, -1, flat.shape[-1])
    return flat.gather(1, expanded).reshape(*index.shape, *points.shape[2:])


@torch.no_grad()
def sample_point_patches(
    xyz: torch.Tensor,
    num_patches: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FPS centers and kNN member indices."""
    if xyz.shape[1] < max(num_patches, group_size):
        raise ValueError("N must be at least max(num_patches, group_size)")
    from pytorch3d.ops import knn_points

    centers = farthest_point_sample(xyz, num_patches)
    neighbors = knn_points(p1=centers, p2=xyz, K=group_size, return_sorted=False).idx
    return centers, neighbors


class RotaryPositionEmbedding3D(nn.Module):
    """Continuous metric XYZ RoPE; relative-position encoding, not SE(3) equivariance."""

    def __init__(self, head_dim: int, scale: float = 0.05, base: float = 100.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 6:
            raise ValueError("head_dim must be positive and divisible by 6")
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(base) or base <= 1:
            raise ValueError("RoPE requires finite scale > 0 and base > 1")
        axis_dim = head_dim // 3
        self.scale = scale
        self.register_buffer(
            "inv_freq",
            base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim),
            persistent=False,
        )

    def forward(
        self, xyz: torch.Tensor, *, origin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            # A shared origin improves phase precision without changing pairwise offsets.
            positions = xyz.float()
            if origin is None:
                origin = positions.mean(dim=1, keepdim=True)
            positions = (positions - origin.float()) / self.scale
            angles = positions.unsqueeze(-1) * self.inv_freq.float()
            angles = angles.repeat_interleave(2, dim=-1).flatten(-2)
            return angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        pairs = x.float().unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x.float() * cos + rotated * sin).to(x.dtype)


class LocalGeometryEmbedding(nn.Module):
    """R3D-style PointNet patch context fusion with lightweight point-wise MLPs."""

    def __init__(
        self, input_channels: int, channels: int, geometry_scale: float,
        local_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if not math.isfinite(geometry_scale) or geometry_scale <= 0:
            raise ValueError("geometry_scale must be finite and positive")
        self.geometry_scale = float(geometry_scale)
        self.point_mlp = build_mlp(
            [input_channels, local_hidden_dim, local_hidden_dim], activate_last=True,
        )
        self.point_projection = nn.Linear(local_hidden_dim, local_hidden_dim, bias=False)
        self.context_projection = nn.Linear(local_hidden_dim, local_hidden_dim)
        # The two projections form a single affine map with fan-in 2C.
        bound = (2 * local_hidden_dim) ** -0.5
        for parameter in (self.point_projection.weight, self.context_projection.weight, self.context_projection.bias):
            nn.init.uniform_(parameter, -bound, bound)
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(local_hidden_dim), nn.GELU(), nn.Linear(local_hidden_dim, local_hidden_dim),
        )
        self.projection = nn.Linear(local_hidden_dim, channels)

    def forward(
        self, points: torch.Tensor, centers: torch.Tensor, neighbors: torch.Tensor,
    ) -> torch.Tensor:
        grouped_points = index_points(points, neighbors)
        grouped_points[..., :3].sub_(centers.unsqueeze(2)).div_(self.geometry_scale)
        features = self.point_mlp(grouped_points.to(self.point_mlp[0].weight.dtype))
        context = features.amax(dim=2, keepdim=True)
        features = self.point_projection(features) + self.context_projection(context)
        features = self.context_mlp(features)
        return self.projection(features.amax(dim=2))


class LocalPatchEncoder(nn.Module):
    """Encode local geometry and absolute position at FPS patch centers."""

    def __init__(
        self,
        input_channels: int = 6,
        channels: int = 192,
        num_patches: int = 128,
        group_size: int = 32,
        geometry_scale: float = 0.10,
        local_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if input_channels not in (3, 6):
            raise ValueError("input_channels must be 3 (XYZ) or 6 (XYZRGB)")
        sizes = (channels, num_patches, group_size, local_hidden_dim)
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError("sizes must be positive integers")

        self.input_channels = input_channels
        self.out_dim = channels
        self.num_patches = num_patches
        self.group_size = group_size

        self.geometry_embedding = LocalGeometryEmbedding(
            input_channels, channels, geometry_scale, local_hidden_dim,
        )
        self.position_embedding = build_mlp([3, 128, channels], normalize=False)

    def forward(self, pointcloud: torch.Tensor) -> dict[str, torch.Tensor]:
        if pointcloud.ndim != 3:
            raise ValueError(f"Expected point cloud [B,N,C], got {tuple(pointcloud.shape)}")
        batch_size, _, channels = pointcloud.shape
        if batch_size == 0 or channels < self.input_channels:
            raise ValueError(
                f"Invalid point cloud shape {tuple(pointcloud.shape)}: require B > 0, "
                f"C >= {self.input_channels}"
            )
        points = pointcloud[..., : self.input_channels].float().contiguous()
        xyz = points[..., :3].contiguous()
        with torch.autocast(device_type=pointcloud.device.type, enabled=False):
            centers, neighbors = sample_point_patches(
                xyz,
                self.num_patches,
                self.group_size,
            )

        tokens = self.geometry_embedding(points, centers, neighbors)
        # Preserve absolute metric position alongside relative geometry.
        position = self.position_embedding(centers.to(self.position_embedding[0].weight.dtype))
        tokens = tokens + position.to(tokens.dtype)
        return {"tokens": tokens, "xyz": centers, "neighbor_idx": neighbors}


class SpatialTransformerBlock(nn.Module):
    """Global Pre-LN attention with Q/K normalization before 3D RoPE."""

    def __init__(self, channels: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        hidden = int(channels * mlp_ratio)
        self.ffn = build_mlp([channels, hidden, channels], normalize=False)
        self.attention_gain = nn.Parameter(torch.full((channels,), 0.1))
        self.ffn_gain = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, m, d = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(b, m, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = RotaryPositionEmbedding3D.apply_rotary(self.q_norm(q), cos, sin)
        k = RotaryPositionEmbedding3D.apply_rotary(self.k_norm(k), cos, sin)
        update = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        update = update.transpose(1, 2).reshape(b, m, d)

        tokens = tokens + self.attention_gain.to(tokens.dtype) * self.projection(update)
        return tokens + self.ffn_gain.to(tokens.dtype) * self.ffn(self.norm2(tokens))


class SpatialContextEncoder(nn.Module):
    """Contextualize spatial tokens using metric 3D RoPE."""

    def __init__(
        self, channels: int = 192, depth: int = 2, num_heads: int = 8,
        mlp_ratio: float = 2.0, rope_scale: float = 0.05, rope_base: float = 100.0,
    ) -> None:
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (channels, depth, num_heads)):
            raise ValueError("sizes must be positive integers")
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if int(channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive hidden dimension")
        self.out_dim = channels
        self.rope = RotaryPositionEmbedding3D(channels // num_heads, rope_scale, rope_base)
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(channels, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope(xyz)
        for block in self.blocks:
            tokens = block(tokens, cos, sin)
        return self.norm(tokens)
