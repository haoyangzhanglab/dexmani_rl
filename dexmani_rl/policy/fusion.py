"""Project camera features onto point patches and fuse them with geometry."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple
import torch
import torch.nn.functional as F
from torch import nn
from .pointcloud import index_points

if TYPE_CHECKING:
    from .vision import ImageProcessor, ViTEncoder


class ImageProjection(NamedTuple):
    """FP32 align_corners=False grids [...,2] and Bool validity [...]."""

    grid_xy: torch.Tensor
    valid: torch.Tensor


@torch.no_grad()
def project_to_image(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_transform: torch.Tensor,
    image_hw: tuple[int, int],
) -> ImageProjection:
    """Project XYZ [BT,N,3] using extrinsics from the point frame to OpenCV cameras."""
    height, width = image_hw
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive (height, width)")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [BT,N,3]")
    bt = points.shape[0]
    for name, value, shape in (
        ("intrinsics", intrinsics, (bt, 3, 3)),
        ("extrinsics", extrinsics, (bt, 3, 4)),
        ("image_transform", image_transform, (bt, 3, 3)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    with torch.autocast(device_type=points.device.type, enabled=False):
        points = points.float()
        transform = extrinsics.float()
        camera_intrinsics = intrinsics.float()
        image_affine = image_transform.float()
        # Explicit FP32 products avoid TF32 rounding at image boundaries.
        camera = (points[:, :, None, :] * transform[..., None, :3, :3]).sum(-1)
        camera.add_(transform[..., None, :3, 3])

        homogeneous = (camera[..., None, :] * camera_intrinsics[..., None, :, :]).sum(-1)
        depth = camera[..., 2]
        denominator = homogeneous[..., 2:3]
        valid = (
            torch.isfinite(image_transform).all(dim=(-1, -2))[..., None]
            & torch.isfinite(homogeneous).all(-1)
            & (depth > 1e-6)
            & (denominator[..., 0].abs() > 1e-6)
        )
        # Record validity before division overwrites the homogeneous coordinates.
        pixels = homogeneous[..., :2].div_(torch.where(valid[..., None], denominator, 1))
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
        pixels = (pixels[..., None, :] * image_affine[..., None, :2, :]).sum(-1)
        size = pixels.new_tensor((width, height))
        valid &= ((pixels >= -0.5) & (pixels < size - 0.5)).all(-1)
        grid = (pixels + 0.5).div_(size).mul_(2).sub_(1)
        # A valid pixel just inside the right edge can round to exactly +1.
        grid.clamp_(-1, 1 - torch.finfo(torch.float32).eps / 2)
        return ImageProjection(torch.where(valid[..., None], grid, 0.0), valid)


def project_rgb_patches(
    obs: dict[str, torch.Tensor], points: torch.Tensor, patches: dict[str, torch.Tensor],
    *, image_processor: ImageProcessor, rgb_encoder: ViTEncoder,
    sampling: Literal["center", "group_mean"],
) -> tuple[torch.Tensor, ImageProjection]:
    """Build the RGB feature grid and its projection for point-patch sampling."""
    rgb, image_transform = image_processor.process_images(obs["rgb"])
    height, width = rgb.shape[-2:]
    patch_size = rgb_encoder.patch_size
    # Sampling coordinates assume the patch grid covers the whole processed image.
    if height % patch_size or width % patch_size:
        raise ValueError("RGB input height and width must be divisible by the encoder patch_size")
    patch_grid = (height // patch_size, width // patch_size)
    sample_xyz = patches["xyz"].unsqueeze(2)
    if sampling == "group_mean":
        sample_xyz = index_points(points[..., :3], patches["neighbor_idx"])
    projection = project_to_image(
        points=sample_xyz.flatten(1, 2),
        intrinsics=obs["camera_intrinsic"].unflatten(-1, (3, 3)),
        extrinsics=obs["camera_extrinsic"].unflatten(-1, (3, 4)),
        image_transform=image_transform,
        image_hw=(height, width),
    )
    image_grid = rgb_encoder(rgb).unflatten(1, patch_grid)
    group_shape = sample_xyz.shape[1:3]
    return image_grid, ImageProjection(
        projection.grid_xy.unflatten(1, group_shape),
        projection.valid.unflatten(1, group_shape),
    )


class RGBPatchFusion(nn.Module):
    out_dim = 192

    def __init__(self, image_channels: int, geometry_channels: int):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(image_channels)
        self.semantic_projection = nn.Linear(image_channels, geometry_channels)
        self.geometry_norm = nn.LayerNorm(geometry_channels)
        self.fusion_projection = nn.Linear(2 * geometry_channels, self.out_dim)

    def forward(
        self,
        geometry_tokens: torch.Tensor,
        rgb_grid: torch.Tensor,
        projection: ImageProjection,
    ) -> torch.Tensor:
        grid, valid = projection
        with torch.autocast(device_type=rgb_grid.device.type, enabled=False):
            rgb_input = rgb_grid.permute(0, 3, 1, 2).float()
            if valid.shape[2] == 1:
                sampled = F.grid_sample(
                    rgb_input, grid,
                    mode="bilinear", padding_mode="border", align_corners=False,
                ).squeeze(-1).transpose(1, 2)
                group_valid = valid.squeeze(2)
            else:
                counts = valid.sum(dim=2, keepdim=True)
                group_valid = counts.squeeze(-1) > 0
                counts = counts.clamp_min(1)
                sampled_groups = []
                # Bound temporary FP32 samples while retaining every KNN neighbor.
                for start in range(0, grid.shape[1], 8):
                    neighbor_valid = valid[:, start : start + 8]
                    sampled = F.grid_sample(
                        rgb_input, grid[:, start : start + 8],
                        mode="bilinear", padding_mode="border", align_corners=False,
                    ).permute(0, 2, 3, 1)
                    sampled = torch.where(neighbor_valid.unsqueeze(-1), sampled, 0.0).sum(dim=2)
                    sampled_groups.append(sampled / counts[:, start : start + 8])
                sampled = torch.cat(sampled_groups, dim=1)
        semantic = self.semantic_projection(
            self.rgb_norm(sampled.to(self.rgb_norm.weight.dtype)),
        )
        # Mask after the projection so its bias cannot fill unsupported patches.
        semantic = torch.where(group_valid.unsqueeze(-1), semantic, 0.0)
        geometry = self.geometry_norm(geometry_tokens)
        return self.fusion_projection(torch.cat((geometry, semantic.to(geometry.dtype)), dim=-1))
