import torch
from torch import nn
from .preprocess import FIXED_OBSERVATION_SHAPES, TACTILE_FEATURE_SHAPES


def normalization_feature_shapes(dims, spec, *, required_identity_fields=()):
    shapes = {}
    identity_fields = {"rgb", *TACTILE_FEATURE_SHAPES, *required_identity_fields}
    for key, width in dims.items():
        mode = spec[key]
        if key == "action" and mode != "limits":
            raise ValueError("action must use limits")
        if mode == "identity":
            continue
        if key in identity_fields:
            raise ValueError(f"{key} must use identity")
        if mode not in {"limits", "mean_std"}:
            raise ValueError(f"Invalid normalization mode for {key}: {mode!r}")
        shapes[key] = FIXED_OBSERVATION_SHAPES.get(key, (width,))
    return shapes


class FieldNormalizer(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))

    def validate(self, key="field") -> None:
        if self.scale.shape != self.offset.shape:
            raise ValueError(f"Affine scale and offset for {key} must have identical shapes")
        valid = torch.isfinite(self.scale) & torch.isfinite(self.offset) & (self.scale != 0)
        if not valid.all():
            raise ValueError(f"Invalid affine parameters for {key}: expected finite values and nonzero scale")

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value * self.scale).add_(self.offset)

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.offset).div_(self.scale)


class LinearNormalizer(nn.Module):
    def __init__(self, feature_shapes=None):
        super().__init__()
        self.fields = nn.ModuleDict()
        for key, shape in (feature_shapes or {}).items():
            self.fields[key] = FieldNormalizer(torch.ones(shape), torch.zeros(shape))

    def __getitem__(self, key) -> FieldNormalizer:
        return self.fields[key]

    def validate(self) -> None:
        for key, field in self.fields.items():
            field.validate(key)
