"""Registry mapping model_type to Grad-CAM target layer paths."""

from __future__ import annotations

import torch.nn as nn

from ..utils.model_utils import resolve_attr_path

MODEL_LAYER_REGISTRY: dict[str, str] = {
    "vit": "vit.encoder.layer[-1].layernorm_before",
}


def _resolve_norm_from_block(block: nn.Module) -> nn.Module | None:
    for attr_name in ("layernorm_before", "norm1", "norm2", "norm", "ln1", "ln2"):
        candidate = getattr(block, attr_name, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def _resolve_last_block(root: nn.Module) -> nn.Module | None:
    for path in ("encoder.layer[-1]", "encoder.layers[-1]", "layers[-1]", "blocks[-1]"):
        block = resolve_attr_path(root, path)
        if isinstance(block, nn.Module):
            return block
    return None


def resolve_gradcam_target_layer(model: nn.Module) -> nn.Module:
    candidate_roots = [model]
    vision_backbone = getattr(model, "vision_backbone", None)
    if isinstance(vision_backbone, nn.Module):
        candidate_roots.append(vision_backbone)

    for root in candidate_roots:
        model_type = getattr(getattr(root, "config", None), "model_type", None)
        if model_type and model_type in MODEL_LAYER_REGISTRY:
            path = MODEL_LAYER_REGISTRY[model_type]
            layer = resolve_attr_path(root, path)
            if isinstance(layer, nn.Module):
                return layer

        if hasattr(root, "vit") and hasattr(root.vit, "encoder"):
            return root.vit.encoder.layer[-1].layernorm_before

        block = _resolve_last_block(root)
        if block is not None:
            layer = _resolve_norm_from_block(block)
            if layer is not None:
                return layer

        fallback_norm = getattr(root, "norm", None)
        if isinstance(fallback_norm, nn.Module):
            return fallback_norm

    raise ValueError(
        "Unsupported model architecture for Grad-CAM. "
        "Pass --gradcam-layer to target a specific layer "
        "(e.g., 'vision_backbone.blocks[-1].norm1')."
    )
