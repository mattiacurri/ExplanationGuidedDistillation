"""Qwen-VL visual adapter backed by a distilled StudentViT.

The adapter replaces ``model.model.visual`` in Qwen2.5-VL generation models.
It accepts Qwen image-processor patch tensors and ``image_grid_thw``, runs the
distilled student ViT on reconstructed image tensors, resizes the student patch
tokens back to the Qwen visual grid, and finally reuses Qwen's patch merger so
the language model receives the usual post-merger image embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import resolve_data_config
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..training.explainable_vit_distillation import StudentViT, load_student_vit
@dataclass(slots=True)
class QwenStudentVisualMetadata:
    """Metadata describing an installed student visual adapter."""

    student_checkpoint_dir: str
    student_model_name: str
    teacher_hidden_size: int
    student_image_size: int
    qwen_patch_size: int
    qwen_temporal_patch_size: int
    qwen_window_size: int
    spatial_merge_size: int

class QwenVisualPassthrough(nn.Module):
    """Qwen visual stub that treats ``pixel_values`` as ready image embeddings."""

    def __init__(self, *, spatial_merge_size: int) -> None:
        super().__init__()
        self.spatial_merge_size = int(spatial_merge_size)
        self.register_buffer("_anchor", torch.empty(0), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """Expose dtype expected by Qwen's ``get_image_features`` helper."""
        return self._anchor.dtype

    @property
    def device(self) -> torch.device:
        """Expose device expected by Qwen's ``get_image_features`` helper."""
        return self._anchor.device

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        return_dict: bool = True,
        **_: Any,
    ) -> BaseModelOutputWithPooling | tuple[torch.Tensor, torch.Tensor]:
        """Return the provided embeddings as Qwen post-merger image features."""
        embeddings = hidden_states.to(device=self.device, dtype=self.dtype)
        output = BaseModelOutputWithPooling(
            last_hidden_state=embeddings,
            pooler_output=embeddings,
        )
        if return_dict:
            return output
        return output.last_hidden_state, output.pooler_output


def install_qwen_visual_passthrough(
    qwen_model: nn.Module,
    *,
    spatial_merge_size: int | None = None,
) -> None:
    """Replace Qwen's visual module with a pass-through over prepared embeddings."""
    qwen_core = getattr(qwen_model, "model", None)
    if qwen_core is None or not hasattr(qwen_core, "visual"):
        raise ValueError("Il modello Qwen non espone model.visual")

    original_visual = qwen_core.visual
    if spatial_merge_size is None:
        spatial_merge_size = int(original_visual.spatial_merge_size)

    param = next(qwen_model.parameters())
    qwen_core.visual = QwenVisualPassthrough(
        spatial_merge_size=spatial_merge_size,
    ).to(device=param.device, dtype=param.dtype)


def build_student_qwen_image_embeddings(
    qwen_model: nn.Module,
    *,
    student_checkpoint_dir: str | Path,
    image: Any,
    image_grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, QwenStudentVisualMetadata]:
    """Run the distilled student on a PIL image and return Qwen-ready embeddings.

    This follows the custom-encoder pattern: compute the visual embeddings outside
    Qwen, then pass them through ``pixel_values`` with a visual pass-through module.
    """
    qwen_core = getattr(qwen_model, "model", None)
    if qwen_core is None or not hasattr(qwen_core, "visual"):
        raise ValueError("Il modello Qwen non espone model.visual")

    original_visual = qwen_core.visual
    param = next(qwen_model.parameters())
    device = param.device
    dtype = param.dtype

    student, student_metadata = load_student_vit(student_checkpoint_dir)
    student.to(device=device, dtype=dtype)
    student.eval()

    data_config = resolve_data_config(
        getattr(student.vit, "pretrained_cfg", None),
        model=student.vit,
    )
    input_size = data_config.get("input_size", (3, 224, 224))
    student_image_size = int(input_size[-1])
    student_mean = tuple(
        float(value) for value in data_config.get("mean", (0.485, 0.456, 0.406))
    )
    student_std = tuple(
        float(value) for value in data_config.get("std", (0.229, 0.224, 0.225))
    )

    student_inputs = pil_image_to_student_tensor(
        image,
        image_size=student_image_size,
        mean=student_mean,
        std=student_std,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        student_tokens = student(student_inputs)
        if student_metadata.get("distillation_target") == "qwen_image_embeddings":
            if student_tokens.size(0) != 1:
                raise ValueError("Eval Qwen supporta una immagine alla volta")
            image_embeddings = student_tokens.squeeze(0)
            metadata = QwenStudentVisualMetadata(
                student_checkpoint_dir=str(student_checkpoint_dir),
                student_model_name=str(student_metadata["student_model_name"]),
                teacher_hidden_size=int(student_metadata["teacher_hidden_size"]),
                student_image_size=student_image_size,
                qwen_patch_size=int(original_visual.patch_size),
                qwen_temporal_patch_size=int(
                    original_visual.patch_embed.temporal_patch_size
                ),
                qwen_window_size=int(original_visual.window_size),
                spatial_merge_size=int(original_visual.spatial_merge_size),
            )
            return image_embeddings, metadata

        grid_thw = image_grid_thw.to(device=device)
        premerge_tokens = student_tokens_to_qwen_premerge(
            student_tokens,
            grid_thw,
            spatial_merge_size=int(original_visual.spatial_merge_size),
        )
        window_index = qwen_window_index(
            grid_thw,
            window_size=int(original_visual.window_size),
            patch_size=int(original_visual.patch_size),
            spatial_merge_size=int(original_visual.spatial_merge_size),
        ).to(device=premerge_tokens.device)
        spatial_merge_unit = int(original_visual.spatial_merge_size) ** 2
        grouped_tokens = premerge_tokens.reshape(
            premerge_tokens.size(0) // spatial_merge_unit,
            spatial_merge_unit,
            premerge_tokens.size(-1),
        )
        windowed_tokens = grouped_tokens[window_index].reshape_as(premerge_tokens)
        image_embeddings = original_visual.merger(windowed_tokens)
        image_embeddings = image_embeddings[torch.argsort(window_index), :]

    metadata = QwenStudentVisualMetadata(
        student_checkpoint_dir=str(student_checkpoint_dir),
        student_model_name=str(student_metadata["student_model_name"]),
        teacher_hidden_size=int(student_metadata["teacher_hidden_size"]),
        student_image_size=student_image_size,
        qwen_patch_size=int(original_visual.patch_size),
        qwen_temporal_patch_size=int(original_visual.patch_embed.temporal_patch_size),
        qwen_window_size=int(original_visual.window_size),
        spatial_merge_size=int(original_visual.spatial_merge_size),
    )
    return image_embeddings, metadata


def pil_image_to_student_tensor(
    image: Any,
    *,
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert a PIL-like RGB image to the normalized student input tensor."""
    image_rgb = image.convert("RGB")
    array = np.asarray(image_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device)
    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    mean_tensor = torch.tensor(mean, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean_tensor) / std_tensor
    return tensor.to(dtype=dtype)


def student_tokens_to_qwen_premerge(
    student_tokens: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Resize student patch tokens per image to Qwen pre-merger token order."""
    batch_size, num_tokens, hidden_size = student_tokens.shape
    side = int(num_tokens**0.5)
    if side * side != num_tokens:
        raise ValueError(
            "Lo student deve produrre una griglia quadrata di patch token, "
            f"ricevuto N={num_tokens}"
        )

    per_image_tokens: list[torch.Tensor] = []
    for idx in range(batch_size):
        grid_t = int(grid_thw[idx, 0].item())
        grid_h = int(grid_thw[idx, 1].item())
        grid_w = int(grid_thw[idx, 2].item())
        token_grid = student_tokens[idx].reshape(side, side, hidden_size)
        token_grid = token_grid.permute(2, 0, 1).unsqueeze(0)
        token_grid = F.interpolate(
            token_grid,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )
        token_grid = token_grid.squeeze(0).permute(1, 2, 0)
        if grid_t > 1:
            token_grid = token_grid.unsqueeze(0).expand(grid_t, -1, -1, -1)
        else:
            token_grid = token_grid.unsqueeze(0)
        per_image_tokens.append(
            spatial_grid_to_qwen_merge_order(
                token_grid,
                spatial_merge_size=spatial_merge_size,
            )
        )

    return torch.cat(per_image_tokens, dim=0)

def spatial_grid_to_qwen_merge_order(
    token_grid: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Convert ``[T,H,W,D]`` grid tokens to Qwen spatial-merge order."""
    if token_grid.ndim != 4:
        raise ValueError("token_grid deve avere shape [T,H,W,D]")
    grid_t, grid_h, grid_w, hidden_size = token_grid.shape
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError(
            "grid_h e grid_w devono essere divisibili per spatial_merge_size"
        )

    return (
        token_grid.reshape(
            grid_t,
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
            hidden_size,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(grid_t * grid_h * grid_w, hidden_size)
    )


def qwen_window_index(
    grid_thw: torch.Tensor,
    *,
    window_size: int,
    patch_size: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Return Qwen2.5-VL's post-merge window order over merger groups."""
    window_index: list[torch.Tensor] = []
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size
    if vit_merger_window_size <= 0:
        raise ValueError("vit_merger_window_size deve essere positivo")

    for grid_t, grid_h, grid_w in grid_thw.detach().cpu().tolist():
        llm_grid_h = int(grid_h) // spatial_merge_size
        llm_grid_w = int(grid_w) // spatial_merge_size
        index = torch.arange(int(grid_t) * llm_grid_h * llm_grid_w).reshape(
            int(grid_t),
            llm_grid_h,
            llm_grid_w,
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            int(grid_t),
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        window_index_id += int(grid_t) * llm_grid_h * llm_grid_w

    return torch.cat(window_index, dim=0).to(device=grid_thw.device)

