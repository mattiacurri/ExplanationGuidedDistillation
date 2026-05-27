"""Grad-CAM helpers for Vision Transformer classifiers."""

from __future__ import annotations

from math import isqrt
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch import nn

from ..utils.model_utils import training_state_guard
from ._layer_registry import resolve_gradcam_target_layer


class ViTGradCAM:
    """Grad-CAM extractor for HuggingFace ViT-like image classifiers."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_layer: nn.Module | None = None,
    ) -> None:
        # Cache model and optional override for later hook registration.
        self.model = model
        self.target_layer = target_layer
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        target_layer = self.target_layer or resolve_gradcam_target_layer(self.model)

        def forward_hook(module, inputs, output) -> None:  # noqa: ANN001
            del module, inputs
            self.activations = output

        def backward_hook(module, grad_input, grad_output) -> None:  # noqa: ANN001
            del module, grad_input
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    @torch.enable_grad()
    def __call__(
        self,
        pixel_values: Tensor,
        target_class: int | None = None,
        model_forward_kwargs: Dict[str, Tensor] | None = None,
        grid_thw: Tensor | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute normalized Grad-CAM heatmap for a single input image."""
        with training_state_guard(self.model):
            device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(device).requires_grad_(True)

            # Preserve optional model kwargs (e.g., image_grid_thw for VLMs).
            forward_kwargs = model_forward_kwargs or {}
            grid_thw = self._extract_grid_thw(grid_thw, forward_kwargs)

            outputs = self.model(pixel_values=pixel_values, **forward_kwargs)
            logits = outputs.logits

            probabilities = F.softmax(logits, dim=-1)
            predicted_class = logits.argmax(dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            if target_class is None:
                target_class = predicted_class

            target_score = logits[0, target_class]
            target_score.backward()

            if self.activations is None or self.gradients is None:
                msg = "Grad-CAM hooks did not capture activations/gradients"
                raise RuntimeError(msg)

            cam = self._compute_cam(self.activations, self.gradients, grid_thw)

            cam_np = cam.detach().to(torch.float32).cpu().numpy()
            if cam_np.max() > cam_np.min():
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

            return cam_np, predicted_class, confidence

    def _compute_cam(
        self,
        activations: Tensor,
        gradients: Tensor,
        grid_thw: Tensor | None,
    ) -> Tensor:
        """Compute a 2D CAM tensor from captured activations and gradients."""
        # Normalize batch dimension and pick the most likely layout.
        if activations.ndim == 4 and gradients.ndim == 4:
            return self._compute_cam_from_spatial(activations, gradients)

        if activations.ndim == 3 and gradients.ndim == 3:
            activations = activations[0]
            gradients = gradients[0]
            return self._compute_cam_from_tokens(activations, gradients, grid_thw)

        if activations.ndim == 2 and gradients.ndim == 2:
            return self._compute_cam_from_tokens(activations, gradients, grid_thw)

        raise ValueError(
            "Grad-CAM attivazioni/gradienti non compatibili. "
            "Prova a specificare --gradcam-layer su un layer più vicino "
            "ai token (es. vision_backbone.blocks[-1].norm1)."
        )

    def _compute_cam_from_tokens(
        self,
        activations: Tensor,
        gradients: Tensor,
        grid_thw: Tensor | None,
    ) -> Tensor:
        """Compute a CAM map from token-wise activations and gradients."""
        # Remove the CLS token if the sequence length suggests its presence.
        activations, gradients = self._strip_cls_token(
            activations,
            gradients,
            grid_thw,
        )

        weights = gradients.mean(dim=0)
        cam = torch.relu((activations * weights).sum(dim=-1))

        num_patches = cam.shape[0]
        grid_shape = self._resolve_grid_shape(grid_thw, num_patches)
        if grid_shape is not None:
            grid_t, grid_h, grid_w = grid_shape
            cam = cam.reshape(grid_t, grid_h, grid_w)
            # Average over temporal dimension if present.
            if grid_t > 1:
                cam = cam.mean(dim=0)
            else:
                cam = cam[0]
            return cam

        grid_size = isqrt(num_patches)
        if grid_size * grid_size != num_patches:
            raise ValueError(
                "Cannot reshape CAM to a square patch grid. "
                "Use --gradcam-layer to target a token-level layer."
            )
        return cam.reshape(grid_size, grid_size)

    def _compute_cam_from_spatial(
        self,
        activations: Tensor,
        gradients: Tensor,
    ) -> Tensor:
        """Compute a CAM map from spatial feature maps (conv-style)."""
        # Grad-CAM for spatial maps uses channel-wise pooled gradients.
        acts = activations[0]
        grads = gradients[0]

        if acts.ndim != 3 or grads.shape != acts.shape:
            raise ValueError("Grad-CAM spatial tensors have incompatible shapes")

        if acts.shape[0] <= acts.shape[-1]:
            weights = grads.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
            return cam

        weights = grads.mean(dim=(0, 1))
        cam = torch.relu((weights[None, None, :] * acts).sum(dim=-1))
        return cam

    def _strip_cls_token(
        self,
        activations: Tensor,
        gradients: Tensor,
        grid_thw: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Remove CLS token if the token count suggests a CLS + patches layout."""
        if activations.shape[0] != gradients.shape[0]:
            raise ValueError("Grad-CAM token tensors have mismatched lengths")

        token_count = activations.shape[0]
        grid_shape = self._resolve_grid_shape(grid_thw, token_count)
        if grid_shape is not None:
            grid_tokens = grid_shape[0] * grid_shape[1] * grid_shape[2]
            if token_count == grid_tokens + 1:
                return activations[1:], gradients[1:]
            return activations, gradients

        if token_count > 1 and self._is_square(token_count - 1):
            return activations[1:], gradients[1:]
        return activations, gradients

    def _extract_grid_thw(
        self,
        grid_thw: Tensor | None,
        forward_kwargs: Dict[str, Tensor],
    ) -> Tensor | None:
        """Extract the grid_thw tensor from explicit args or forward kwargs."""
        if grid_thw is not None:
            return grid_thw

        for key in ("image_grid_thw", "grid_thw"):
            candidate = forward_kwargs.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
        return None

    def _resolve_grid_shape(
        self,
        grid_thw: Tensor | None,
        token_count: int,
    ) -> tuple[int, int, int] | None:
        """Return (t, h, w) if grid_thw matches the token count."""
        if grid_thw is None:
            return None

        grid = grid_thw.detach().cpu()
        if grid.ndim == 2 and grid.shape[0] == 1:
            grid = grid[0]
        if grid.ndim != 1 or grid.numel() < 3:
            return None

        grid_t = int(grid[-3].item())
        grid_h = int(grid[-2].item())
        grid_w = int(grid[-1].item())
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            return None

        grid_tokens = grid_t * grid_h * grid_w
        if token_count in {grid_tokens, grid_tokens + 1}:
            return grid_t, grid_h, grid_w
        return None

    @staticmethod
    def _is_square(value: int) -> bool:
        """Return True if the input is a perfect square."""
        if value <= 0:
            return False
        root = isqrt(value)
        return root * root == value

