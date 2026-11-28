"""Checkpoint-defined image processing and frozen ViT backbones."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, Dinov2Config, SiglipVisionConfig, SiglipVisionModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)


IMAGENET_STD = (0.229, 0.224, 0.225)


IMAGE_PROCESSOR_PRESETS: dict[str, dict] = {
    "dino": {
        "image_size": (224, 224),
        "image_mean": IMAGENET_MEAN,
        "image_std": IMAGENET_STD,
        "interpolation": "bilinear",
    },
    "siglip": {
        "image_size": (224, 224),
        "image_mean": (0.5, 0.5, 0.5),
        "image_std": (0.5, 0.5, 0.5),
        "interpolation": "bilinear",
    },
}


class ImageProcessor(torch.nn.Module):
    """Convert CHW uint8 or float32 [0,1] RGB batches, then resize and normalize."""

    def __init__(
        self,
        image_size: int | Sequence[int] | None = None,
        image_mean: Sequence[float] = IMAGENET_MEAN,
        image_std: Sequence[float] = IMAGENET_STD,
        interpolation: str = "bilinear",
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(map(int, image_size)) if image_size is not None else None
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32), persistent=False)
        for name, value in (
            ("image_mean", self.image_mean),
            ("image_std", self.image_std),
        ):
            if value.shape != (3,) or not torch.isfinite(value).all():
                raise ValueError(f"{name} requires three finite values")
        if (self.image_std <= 0).any():
            raise ValueError("image_std must be positive")

        self.interpolation = interpolation.lower()
        if self.interpolation not in ("nearest", "bilinear", "bicubic"):
            raise ValueError(f"Unsupported interpolation: {interpolation}")

    def process_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume [BT,3,H,W]; transforms use half-pixel resizing."""
        if images.dtype == torch.uint8:
            images = images.to(torch.float32).div_(255.0)
        output_hw = self.image_size or images.shape[-2:]
        if self.interpolation == "nearest" and tuple(output_hw) != images.shape[-2:]:
            raise ValueError("tracked nearest resize does not use the half-pixel convention")
        # Integer pixel centers with align_corners=False require a half-pixel offset.
        sy, sx = (dst / src for src, dst in zip(images.shape[-2:], output_hw))
        transform = torch.tensor(
            ((sx, 0, (sx - 1) / 2), (0, sy, (sy - 1) / 2), (0, 0, 1)),
            dtype=torch.float32, device=images.device,
        )
        transform = torch.broadcast_to(transform, (images.shape[0], 3, 3))

        if self.image_size is not None and self.image_size != images.shape[-2:]:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode=self.interpolation,
                align_corners=None if self.interpolation == "nearest" else False,
            )
        images = images.sub(self.image_mean.view(1, 3, 1, 1)).div_(self.image_std.view(1, 3, 1, 1))
        return images, transform


class ViTEncoder(nn.Module):
    def __init__(
        self,
        kind: Literal["dino", "dinov3", "siglip"],
        tune_mode: Literal["freeze", "full", "lora"] = "freeze",
        global_token_type: Literal["avg", "cls", "pooler"] = "avg",
        out_dim: int | None = None,
        architecture: dict | None = None,
    ):
        super().__init__()
        token_types = ("avg", "cls", "pooler") if kind == "dino" else ("avg", "pooler")
        if kind == "dinov3":
            token_types = ("avg", "cls")
        if global_token_type not in token_types:
            raise ValueError(f"Unsupported global_token_type: {global_token_type}")

        self.kind = kind
        if kind == "dinov3":
            import timm

            architecture = {"model_name": "vit_small_patch16_dinov3", **(architecture or {})}
            self.backbone = timm.create_model(**architecture, pretrained=False)
            self.patch_size = self.backbone.patch_embed.patch_size[0]
            hidden_dim = self.backbone.num_features
            self.num_prefix_tokens = self.backbone.num_prefix_tokens
        else:
            if kind == "dino":
                config = (
                    Dinov2Config(hidden_size=384, num_attention_heads=6, image_size=518)
                    if architecture is None else AutoConfig.for_model(**architecture)
                )
            elif kind == "siglip":
                if architecture is None:
                    raise ValueError("SigLIP requires an explicit architecture")
                # HF from_dict mutates its input.
                config = SiglipVisionConfig.from_dict(dict(architecture))
                config.vision_use_head = global_token_type == "pooler"
            else:
                raise ValueError(f"Unsupported ViT kind: {kind}")
            config._attn_implementation = "sdpa"
            if kind == "dino":
                self.backbone = AutoModel.from_config(config, torch_dtype=torch.float32)
            else:
                self.backbone = SiglipVisionModel(config)
            self.patch_size = int(config.patch_size)
            hidden_dim = int(config.hidden_size)
            self.num_prefix_tokens = (
                1 + int(getattr(config, "num_register_tokens", 0)) if kind == "dino" else 0
            )
        self.out_dim = hidden_dim if out_dim is None else int(out_dim)

        self.proj = (
            nn.Identity()
            if self.out_dim == hidden_dim
            else nn.Linear(hidden_dim, self.out_dim)
        )
        if tune_mode == "lora":
            from peft import LoraConfig, get_peft_model

            if kind == "dino":
                ffn_targets = (
                    ["weights_in", "weights_out"]
                    if getattr(self.backbone.config, "use_swiglu_ffn", False)
                    else ["fc1", "fc2"]
                )
                targets = ["query", "key", "value", "dense", *ffn_targets]
            elif kind == "dinov3":
                targets = [
                    name for name, module in self.backbone.named_modules()
                    if name.startswith("blocks.") and isinstance(module, nn.Linear)
                ]
            else:
                targets = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=targets, bias="none", use_rslora=True,
            ))
        elif tune_mode not in ("freeze", "full"):
            raise ValueError(f"Unsupported tune_mode: {tune_mode}")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.kind == "dinov3":
            tokens = self.backbone.forward_features(rgb)
        else:
            outputs = self.backbone(pixel_values=rgb, return_dict=True)
            tokens = outputs.last_hidden_state
        return self.proj(tokens[:, self.num_prefix_tokens :])


def build_rgb_encoder(name, config=None):
    config = dict(config or {})
    if name in ('dinov3', 'siglip') and config.get('image_size') is None:
        architecture = config.get('architecture') or {}
        config['image_size'] = architecture.get('img_size' if name == 'dinov3' else 'image_size')
    preset = IMAGE_PROCESSOR_PRESETS['dino' if name == 'dinov3' else name]
    processor_config = {}
    for key, default in preset.items():
        value = config.pop(key, None)
        processor_config[key] = default if value is None else value
    processor = ImageProcessor(**processor_config)
    return ViTEncoder(name, **config), processor
