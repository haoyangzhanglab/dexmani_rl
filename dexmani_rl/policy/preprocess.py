"""Convert stacked simulator histories into the checkpoint's observation tensors."""
from __future__ import annotations

import numpy as np
import torch


TACTILE_FEATURE_SHAPES = {
    "finger_tactile_force": (5, 3),
    "taxel_tactile_force": (5, 120, 3),
}


FIXED_OBSERVATION_SHAPES: dict[str, tuple[int, ...]] = {
    **TACTILE_FEATURE_SHAPES,
    "joint_state": (19,),
    "eef_pose": (9,),
    "fingertip_xyz": (5, 3),
    "camera_intrinsic": (9,),
    "camera_extrinsic": (12,),
}


POINT_CLOUD_FIELDS = frozenset({"point_cloud", "hand_pointcloud", "object_pointcloud"})


def validate_observation_shape(name: str, shape: tuple[int, ...], *, width: int) -> None:
    """Validate one frame before point-cloud channel selection."""
    if name in FIXED_OBSERVATION_SHAPES:
        expected = FIXED_OBSERVATION_SHAPES[name]
        if shape != expected:
            raise ValueError(f"{name} requires frame shape {expected}, got {shape}")
    elif name in POINT_CLOUD_FIELDS:
        if len(shape) != 2 or not shape[0] or shape[1] < width:
            raise ValueError(f"{name} requires frame shape (N>0, C>={width}), got {shape}")
    elif name == "rgb":
        if len(shape) != 3 or shape[2] != 3 or min(shape[:2]) <= 0:
            raise ValueError(f"RGB requires frame shape (H, W, 3), got {shape}")
    elif name == "proprio" and (len(shape) != 1 or not shape[0]):
        raise ValueError(f"proprio requires one vector per frame, got {shape}")
    elif not shape or shape[-1] != width:
        raise ValueError(f"{name} requires feature width {width}, got frame shape {shape}")


def prepare_observations(observation, *, normalization_dims, device):
    tensors = {}
    for name, width in normalization_dims.items():
        if name == "action":
            continue
        value = observation[name]
        validate_observation_shape(name, value.shape[1:], width=width)
        if name == "rgb":
            if value.dtype != np.uint8:
                raise ValueError("RGB input requires dtype uint8")
            tensor = torch.from_numpy(value).permute(0, 3, 1, 2).contiguous()
        else:
            if name in POINT_CLOUD_FIELDS:
                value = value[..., :width]
            value = np.ascontiguousarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite observation: {name}")
            tensor = torch.from_numpy(value)
        tensors[name] = tensor.unsqueeze(0).to(device)
    return tensors
