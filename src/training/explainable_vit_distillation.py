"""Explainability-aware ViT distillation from a VLM vision teacher.

This module implements the two offline phases described by the distillation
spec:

1. precompute teacher patch tokens plus token-level Grad-CAM weights;
2. train a smaller timm ViT student with global and explanation-weighted
   MSE alignment losses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import DEFAULT_LOG_EVERY_STEPS, MINI_IMAGENET_DATASET_ID
from ..data import load_preprocessed_mini_imagenet
from ..utils import get_device, set_seed
from ..utils.scheduler import create_cosine_warmup_scheduler
from .wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_metrics

if TYPE_CHECKING:
    from ..models.vlm_vit import VLMVisionClassifier

DISTILLATION_SAMPLE_GLOB = "sample_*.pt"
STUDENT_STATE_FILE = "student_state.pt"
STUDENT_METADATA_FILE = "metadata.json"


def _distillation_sample_path(output_dir: Path, sample_index: int) -> Path:
    """Return the stable filename for a precomputed sample position."""
    return output_dir / f"sample_{sample_index:08d}.pt"


def _parse_distillation_sample_index(path: Path) -> int | None:
    """Extract the integer sample position from a sample_XXXXXXXX.pt filename."""
    stem = path.stem
    if not stem.startswith("sample_"):
        return None
    suffix = stem.removeprefix("sample_")
    if not suffix.isdigit():
        return None
    return int(suffix)


@dataclass(slots=True)
class ExplainableDistillationPrecomputeConfig:
    """Configuration for teacher token and Grad-CAM precomputation."""

    output_dir: str
    teacher_model_dir: str | None = None
    teacher_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dataset_id: str = MINI_IMAGENET_DATASET_ID
    preprocessed_dataset_dir: str | None = None
    split: str = "train"
    batch_size: int = 8
    num_workers: int = 2
    max_samples: int = 0
    subset_per_class: int = 0
    class_ids: tuple[int, ...] = ()
    seed: int = 42
    tau: float = 0.05
    target_source: str = "prediction"
    distillation_target: str = "premerge"
    attention_source: str = "gradcam"
    teacher_pooling: str = "auto"
    student_model_name: str = "vit_small_patch16_224"
    trust_remote_code: bool = True
    torch_dtype: torch.dtype | None = None
    overwrite: bool = False
    storage_dtype: str = "float16"
    store_images: bool = False
    use_wandb: bool = False
    wandb_project: str = "compvis-distillation"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None


@dataclass(slots=True)
class ExplainableDistillationTrainConfig:
    """Configuration for student ViT distillation training."""

    precomputed_dir: str
    output_dir: str
    val_precomputed_dir: str | None = None
    test_precomputed_dir: str | None = None
    teacher_model_dir: str | None = None
    student_model_name: str = "vit_small_patch16_224"
    pretrained: bool = False
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    lambda_global: float = 1.0
    lambda_expl: float = 1.0
    lambda_token_mse: float = 0.0
    num_workers: int = 2
    max_grad_norm: float = 1.0
    eval_every: int = 1
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    filter_uniform_attention: bool = False
    seed: int = 42
    use_torch_compile: bool = False
    use_wandb: bool = False
    wandb_project: str = "compvis-distillation"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None


class StudentViT(nn.Module):
    """Small timm ViT wrapper returning patch tokens in the teacher dimension."""

    def __init__(
        self,
        *,
        model_name: str = "vit_small_patch16_224",
        teacher_hidden_size: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.teacher_hidden_size = teacher_hidden_size
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        student_hidden_size = int(getattr(self.vit, "num_features", 0))
        if student_hidden_size <= 0:
            raise ValueError(f"Impossibile inferire num_features per {model_name}")

        if student_hidden_size == teacher_hidden_size:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(student_hidden_size, teacher_hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vit.forward_features(pixel_values)
        if features.ndim != 3:
            raise ValueError(
                "StudentViT richiede token features [B, N, D], "
                f"ricevuto {tuple(features.shape)}"
            )

        patch_tokens = self._strip_prefix_tokens(features)
        return self.projection(patch_tokens)

    def _strip_prefix_tokens(self, features: torch.Tensor) -> torch.Tensor:
        prefix_tokens = int(getattr(self.vit, "num_prefix_tokens", 0))
        if prefix_tokens > 0 and features.size(1) > prefix_tokens:
            return features[:, prefix_tokens:]

        # Fallback for older timm models exposing class_token but not
        # num_prefix_tokens.
        if hasattr(self.vit, "cls_token") and features.size(1) > 1:
            return features[:, 1:]
        return features


class RawImageDataset(Dataset[tuple[Image.Image, int, int, int]]):
    """Dataset wrapper returning raw RGB images, labels, and stable indices."""

    def __init__(
        self,
        hf_dataset: Any,
        *,
        source_indices: list[int] | None = None,
        sample_positions: list[int] | None = None,
    ) -> None:
        self.dataset = hf_dataset
        self.source_indices = source_indices
        self.sample_positions = sample_positions
        if source_indices is not None and len(source_indices) != len(hf_dataset):
            raise ValueError(
                "source_indices deve avere la stessa lunghezza del dataset"
            )
        if sample_positions is not None and len(sample_positions) != len(hf_dataset):
            raise ValueError(
                "sample_positions deve avere la stessa lunghezza del dataset"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, int, int]:
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = int(item["label"]) if "label" in item else -1
        source_index = idx if self.source_indices is None else self.source_indices[idx]
        sample_position = (
            idx if self.sample_positions is None else self.sample_positions[idx]
        )
        return image, label, int(source_index), int(sample_position)


class TeacherStudentCollator:
    """Collate images for both Qwen teacher input and standalone ViT student."""

    def __init__(self, image_processor: Any, student_transform: Any) -> None:
        self.image_processor = image_processor
        self.student_transform = student_transform

    def __call__(
        self,
        batch: list[tuple[Image.Image, int, int, int]],
    ) -> dict[str, torch.Tensor]:
        images, labels, indices, positions = zip(*batch, strict=True)
        encoded = self.image_processor(images=list(images), return_tensors="pt")
        output: dict[str, torch.Tensor] = {
            "teacher_pixel_values": encoded["pixel_values"],
            "student_pixel_values": torch.stack(
                [self.student_transform(image) for image in images],
                dim=0,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "indices": torch.tensor(indices, dtype=torch.long),
            "positions": torch.tensor(positions, dtype=torch.long),
        }

        image_grid_thw = encoded.get("image_grid_thw")
        if image_grid_thw is None:
            image_grid_thw = encoded.get("grid_thw")
        if image_grid_thw is not None:
            output["image_grid_thw"] = image_grid_thw
        return output


class PrecomputedDistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """Load offline distillation samples saved as individual torch records."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        filter_uniform_attention: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob(DISTILLATION_SAMPLE_GLOB))
        if not self.files:
            raise FileNotFoundError(
                f"Nessun sample di distillazione trovato in {self.data_dir}"
            )
        self.metadata = self._load_metadata()
        self._hf_dataset = None
        self._student_transform = None
        self.num_original_files = len(self.files)
        self.num_filtered_uniform = 0
        if filter_uniform_attention:
            self.files, self.num_filtered_uniform = self._filter_uniform_attention(
                self.files
            )
            if not self.files:
                raise ValueError(
                    "Il filtro dei sample con attention uniforme ha rimosso tutto "
                    f"il dataset in {self.data_dir}."
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = torch.load(self.files[idx], map_location="cpu")
        image = sample.get("image")
        if image is None:
            image = self._load_image_from_source_index(sample)
        item = {
            "image": image.to(torch.float32),
            "Z_t": sample["Z_t"].to(torch.float32),
            "A": sample["A"].to(torch.float32),
        }
        if "grid_thw" in sample:
            item["grid_thw"] = sample["grid_thw"].to(torch.long)
        return item

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _load_image_from_source_index(self, sample: dict[str, Any]) -> torch.Tensor:
        source_index = sample.get("source_index")
        if source_index is None:
            raise KeyError(
                "Sample senza 'image' e senza 'source_index': impossibile "
                "ricostruire l'input student."
            )

        config_payload = self.metadata.get("config", {})
        preprocessed_dataset_dir = config_payload.get("preprocessed_dataset_dir")
        dataset_id = config_payload.get("dataset_id")
        split = config_payload.get("split")
        student_model_name = config_payload.get(
            "student_model_name",
            "vit_small_patch16_224",
        )
        if not split:
            raise KeyError("metadata.json non contiene split necessario.")

        if preprocessed_dataset_dir:
            if self._hf_dataset is None:
                self._hf_dataset = load_preprocessed_mini_imagenet(
                    preprocessed_dataset_dir,
                    splits=(split,),
                )[split]
            item = self._hf_dataset[int(source_index)]
            return self._transform_student_image(item["image"], student_model_name)

        if not dataset_id:
            raise KeyError(
                "metadata.json non contiene dataset_id/split necessari per "
                "ricostruire le immagini non salvate."
            )

        if self._hf_dataset is None:
            self._hf_dataset = load_dataset(dataset_id, split=split)

        item = self._hf_dataset[int(source_index)]
        image = item["image"].convert("RGB")
        return self._transform_student_image(image, student_model_name)

    def _transform_student_image(
        self,
        image: Image.Image,
        student_model_name: str,
    ) -> torch.Tensor:
        if self._student_transform is None:
            self._student_transform = _build_student_eval_transform(student_model_name)
        return self._student_transform(image)

    def _filter_uniform_attention(
        self,
        files: list[Path],
    ) -> tuple[list[Path], int]:
        kept_files: list[Path] = []
        filtered = 0
        for path in files:
            sample = torch.load(path, map_location="cpu")
            weights = sample["A"].to(torch.float32)
            if _is_uniform_attention_weights(weights):
                filtered += 1
                continue
            kept_files.append(path)
        return kept_files, filtered


def distillation_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack precomputed records into a training batch."""
    token_counts = {int(item["Z_t"].shape[0]) for item in batch}
    if len(token_counts) != 1:
        raise ValueError(
            "Il batch contiene numeri di token diversi. "
            "Usa immagini/processore con griglia fissa oppure batch_size=1."
        )

    output = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "Z_t": torch.stack([item["Z_t"] for item in batch], dim=0),
        "A": torch.stack([item["A"] for item in batch], dim=0),
    }
    if all("grid_thw" in item for item in batch):
        output["grid_thw"] = torch.stack([item["grid_thw"] for item in batch], dim=0)
    return output


def _build_distillation_loader(
    dataset: PrecomputedDistillationDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=distillation_collate,
    )


def _run_distillation_epoch_eval(
    student: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: ExplainableDistillationTrainConfig,
    *,
    desc: str,
) -> dict[str, float]:
    student.eval()
    running_loss = 0.0
    running_global = 0.0
    running_expl = 0.0
    running_token_mse = 0.0
    total = 0
    progress = tqdm(
        dataloader,
        desc=desc,
        dynamic_ncols=True,
        colour="blue",
    )
    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device)
            Z_t = batch["Z_t"].to(device)
            A = batch["A"].to(device)
            Z_s = student(images)
            loss, parts = distillation_losses(
                Z_s,
                Z_t,
                A,
                lambda_global=config.lambda_global,
                lambda_expl=config.lambda_expl,
                lambda_token_mse=config.lambda_token_mse,
            )

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            running_global += float(parts["loss_global"].item()) * batch_size
            running_expl += float(parts["loss_expl"].item()) * batch_size
            running_token_mse += float(parts["loss_token_mse"].item()) * batch_size
            total += batch_size
            progress.set_postfix(
                loss=f"{running_loss / max(1, total):.4f}",
                global_loss=f"{running_global / max(1, total):.4f}",
                expl_loss=f"{running_expl / max(1, total):.4f}",
            )

    return {
        "loss": running_loss / max(1, total),
        "loss_global": running_global / max(1, total),
        "loss_expl": running_expl / max(1, total),
        "loss_token_mse": running_token_mse / max(1, total),
    }


def _is_uniform_attention_weights(weights: torch.Tensor) -> bool:
    """Return True when a saved attention vector is exactly the uniform fallback."""
    if weights.ndim != 1:
        raise ValueError(f"A deve avere shape [N], ricevuto {tuple(weights.shape)}")
    if weights.numel() == 0:
        return True

    normalized = weights / weights.sum().clamp_min(1e-12)
    uniform = torch.full_like(normalized, 1.0 / normalized.numel())
    return bool(torch.allclose(normalized, uniform, atol=1e-5, rtol=1e-5))


def compute_token_gradcam_weights_list(
    token_list: list[torch.Tensor],
    logits: torch.Tensor,
    *,
    target_class: int | torch.Tensor | None = None,
    tau: float = 0.05,
) -> list[torch.Tensor]:
    """Compute Grad-CAM token weights for ragged per-image token sequences."""
    if tau <= 0:
        raise ValueError("tau deve essere maggiore di 0")
    if logits.ndim != 2 or logits.size(0) != len(token_list):
        raise ValueError("logits deve avere shape [B, C] coerente con token_list")

    for tokens in token_list:
        if tokens.ndim != 2:
            raise ValueError("Ogni elemento di token_list deve avere shape [N, D]")
        if tokens.grad is not None:
            tokens.grad.zero_()

    targets = _resolve_gradcam_targets(logits, target_class)
    score = logits.gather(dim=1, index=targets[:, None]).sum()
    score.backward()

    weights: list[torch.Tensor] = []
    for tokens in token_list:
        grads = tokens.grad
        if grads is None:
            raise RuntimeError("Grad-CAM non ha prodotto gradienti sui token")
        cam = torch.relu((grads.float() * tokens.float()).sum(dim=-1))
        cam = torch.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam_sum = cam.sum(dim=-1, keepdim=True)
        if bool((cam_sum <= 1e-6).all()):
            weights.append(torch.full_like(cam, 1.0 / max(1, cam.numel())))
            continue
        cam = cam / (cam_sum + 1e-6)
        sample_weights = torch.softmax(cam / tau, dim=-1)
        if not torch.isfinite(sample_weights).all():
            sample_weights = torch.full_like(cam, 1.0 / max(1, cam.numel()))
        weights.append(sample_weights)
    return weights


def _resolve_gradcam_targets(
    logits: torch.Tensor,
    target_class: int | torch.Tensor | None,
) -> torch.Tensor:
    if target_class is None:
        return logits.argmax(dim=-1)
    if isinstance(target_class, int):
        return torch.full(
            (logits.size(0),),
            target_class,
            dtype=torch.long,
            device=logits.device,
        )

    targets = target_class.to(device=logits.device, dtype=torch.long)
    if targets.shape != (logits.size(0),):
        raise ValueError("target_class tensor deve avere shape [B]")
    return targets


def distillation_losses(
    Z_s: torch.Tensor,
    Z_t: torch.Tensor,
    A: torch.Tensor,
    *,
    lambda_global: float = 1.0,
    lambda_expl: float = 1.0,
    lambda_token_mse: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute global and explanation-weighted MSE distillation losses."""
    if Z_s.shape != Z_t.shape:
        raise ValueError(
            "Z_s e Z_t devono avere la stessa shape dopo la proiezione: "
            f"{tuple(Z_s.shape)} vs {tuple(Z_t.shape)}"
        )
    if A.shape != Z_t.shape[:2]:
        raise ValueError(
            "A deve avere shape [B, N] coerente con i token: "
            f"{tuple(A.shape)} vs {tuple(Z_t.shape[:2])}"
        )
    Z_s = torch.nan_to_num(Z_s.float(), nan=0.0, posinf=0.0, neginf=0.0)
    Z_t = torch.nan_to_num(Z_t.float(), nan=0.0, posinf=0.0, neginf=0.0)
    A = torch.nan_to_num(A.float(), nan=0.0, posinf=0.0, neginf=0.0)
    A_sum = A.sum(dim=-1, keepdim=True)
    uniform_A = torch.full_like(A, 1.0 / max(1, A.size(-1)))
    A = torch.where(A_sum > 1e-6, A / (A_sum + 1e-6), uniform_A)

    Zs_g = Z_s.mean(dim=1)
    Zt_g = Z_t.mean(dim=1)

    token_mse = (Z_s - Z_t).pow(2).mean(dim=-1)
    loss_global = F.mse_loss(Zs_g, Zt_g)
    loss_expl = (A * token_mse).sum(dim=1).mean()
    loss_token_mse = F.mse_loss(Z_s, Z_t)
    total = (
        lambda_global * loss_global
        + lambda_expl * loss_expl
        + lambda_token_mse * loss_token_mse
    )

    return total, {
        "loss_global": loss_global.detach(),
        "loss_expl": loss_expl.detach(),
        "loss_token_mse": loss_token_mse.detach(),
    }

def teacher_forward_token_list(
    model: "VLMVisionClassifier",
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    pooling: str = "auto",
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run the frozen teacher and return ragged patch tokens plus probe logits."""
    with torch.no_grad():
        backbone_outputs = model._forward_backbone(  # noqa: SLF001
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        token_list = _extract_patch_token_list(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
        )

    probe_tokens = [
        tokens.detach().to(model.classifier.weight.dtype).requires_grad_(True)
        for tokens in token_list
    ]
    pooled = _pool_teacher_probe_tokens(model, probe_tokens, pooling=pooling)
    logits = model.classifier(pooled)
    return probe_tokens, logits


def teacher_forward_token_and_pooler_list(
    model: "VLMVisionClassifier",
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    pooling: str = "auto",
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Return pre-merger probe tokens, post-merger image embeddings, and logits."""
    with torch.no_grad():
        backbone_outputs = model._forward_backbone(  # noqa: SLF001
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        token_list = _extract_patch_token_list(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
        )
        pooler_list = _extract_pooler_token_list(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=int(
                getattr(model.vision_backbone, "spatial_merge_size", 1) or 1
            ),
        )

    probe_tokens = [
        tokens.detach().to(model.classifier.weight.dtype).requires_grad_(True)
        for tokens in token_list
    ]
    pooled = _pool_teacher_probe_tokens(model, probe_tokens, pooling=pooling)
    logits = model.classifier(pooled)
    return probe_tokens, pooler_list, logits


def teacher_forward_token_list_no_probe(
    model: "VLMVisionClassifier",
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Run the frozen teacher and return ragged patch tokens without probe logits."""
    with torch.no_grad():
        backbone_outputs = model._forward_backbone(  # noqa: SLF001
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return _extract_patch_token_list(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
        )


def teacher_forward_token_and_pooler_list_no_probe(
    model: "VLMVisionClassifier",
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return pre-merger tokens and post-merger image embeddings without logits."""
    with torch.no_grad():
        backbone_outputs = model._forward_backbone(  # noqa: SLF001
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        token_list = _extract_patch_token_list(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
        )
        pooler_list = _extract_pooler_token_list(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=int(
                getattr(model.vision_backbone, "spatial_merge_size", 1) or 1
            ),
        )
    return token_list, pooler_list


def _pool_teacher_probe_tokens(
    model: "VLMVisionClassifier",
    token_list: list[torch.Tensor],
    *,
    pooling: str = "auto",
) -> torch.Tensor:
    resolved_pooling = (
        getattr(model, "pooling", "mean") if pooling == "auto" else pooling
    )
    if resolved_pooling == "mean":
        return torch.stack([tokens.mean(dim=0) for tokens in token_list], dim=0)
    if resolved_pooling == "cls":
        return torch.stack([tokens[0] for tokens in token_list], dim=0)

    max_tokens = max(tokens.size(0) for tokens in token_list)
    batch_size = len(token_list)
    hidden_size = token_list[0].size(-1)
    padded = token_list[0].new_zeros(batch_size, max_tokens, hidden_size)
    key_padding_mask = torch.ones(
        batch_size,
        max_tokens,
        dtype=torch.bool,
        device=token_list[0].device,
    )
    for row, tokens in enumerate(token_list):
        token_count = tokens.size(0)
        padded[row, :token_count] = tokens
        key_padding_mask[row, :token_count] = False

    return model._pool_token_batch(  # noqa: SLF001
        padded,
        key_padding_mask=key_padding_mask,
    )

def _extract_patch_token_list(
    outputs: Any,
    *,
    image_grid_thw: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Extract patch tokens as a per-image list, allowing variable grid sizes."""
    hidden = _extract_hidden_tensor(outputs)
    if hidden.ndim == 3:
        return [hidden[idx] for idx in range(hidden.size(0))]
    if hidden.ndim != 2:
        raise ValueError(f"Shape token teacher non supportata: {tuple(hidden.shape)}")
    if image_grid_thw is None:
        raise ValueError("image_grid_thw richiesto per token teacher flatten [T, D]")

    token_counts = image_grid_thw.prod(dim=-1).to(torch.long).tolist()
    total_tokens = int(sum(token_counts))
    if total_tokens != int(hidden.size(0)):
        raise ValueError(
            "Token teacher flatten non coerenti con image_grid_thw: "
            f"{total_tokens} vs {hidden.size(0)}"
        )

    token_list: list[torch.Tensor] = []
    start = 0
    for token_count in token_counts:
        end = start + int(token_count)
        token_list.append(hidden[start:end])
        start = end
    return token_list


def _extract_pooler_token_list(
    outputs: Any,
    *,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    """Extract Qwen post-merger image embeddings as a per-image list."""
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is None and isinstance(outputs, dict):
        pooler = outputs.get("pooler_output")
    if not isinstance(pooler, torch.Tensor):
        raise ValueError("Output teacher non contiene pooler_output post-merger")
    if pooler.ndim == 3:
        return [pooler[idx] for idx in range(pooler.size(0))]
    if pooler.ndim != 2:
        raise ValueError(f"Shape pooler_output non supportata: {tuple(pooler.shape)}")
    if image_grid_thw is None:
        raise ValueError("image_grid_thw richiesto per pooler_output flatten")

    merge_unit = max(1, spatial_merge_size * spatial_merge_size)
    token_counts = (image_grid_thw.prod(dim=-1).to(torch.long) // merge_unit).tolist()
    total_tokens = int(sum(token_counts))
    if total_tokens != int(pooler.size(0)):
        raise ValueError(
            "Token post-merger non coerenti con image_grid_thw: "
            f"{total_tokens} vs {pooler.size(0)}"
        )

    pooler_list: list[torch.Tensor] = []
    start = 0
    for token_count in token_counts:
        end = start + int(token_count)
        pooler_list.append(pooler[start:end])
        start = end
    return pooler_list


def _extract_hidden_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        hidden = outputs
    else:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]

    if not isinstance(hidden, torch.Tensor):
        raise ValueError("Output teacher non contiene token embeddings utilizzabili")
    return hidden


def _infer_student_patch_grid(model_name: str) -> tuple[int, int]:
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    patch_embed = getattr(model, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", None)
    if isinstance(grid_size, tuple) and len(grid_size) == 2:
        return int(grid_size[0]), int(grid_size[1])

    num_patches = int(getattr(patch_embed, "num_patches", 0) or 0)
    root = int(num_patches**0.5)
    if root > 0 and root * root == num_patches:
        return root, root
    raise ValueError(f"Impossibile inferire patch grid per student {model_name}")


def _resample_teacher_tokens_and_weights(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    *,
    grid_thw: torch.Tensor | None,
    target_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize teacher token sequence and CAM weights to the student patch grid."""
    target_h, target_w = target_grid
    target_tokens = target_h * target_w
    if tokens.size(0) == target_tokens:
        weights = weights / (weights.sum() + 1e-6)
        return tokens, weights

    if grid_thw is not None:
        grid = grid_thw.detach().to(device=tokens.device, dtype=torch.long)
        if grid.numel() >= 3:
            grid_t = int(grid[-3].item())
            grid_h = int(grid[-2].item())
            grid_w = int(grid[-1].item())
            if grid_t * grid_h * grid_w == tokens.size(0):
                token_grid = tokens.reshape(grid_t, grid_h, grid_w, tokens.size(-1))
                token_grid = token_grid.mean(dim=0).permute(2, 0, 1).unsqueeze(0)
                resized_tokens = F.interpolate(
                    token_grid,
                    size=target_grid,
                    mode="bilinear",
                    align_corners=False,
                )
                resized_tokens = (
                    resized_tokens.squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(
                        target_tokens,
                        tokens.size(-1),
                    )
                )

                weight_grid = weights.reshape(grid_t, grid_h, grid_w).mean(dim=0)
                weight_grid = weight_grid.unsqueeze(0).unsqueeze(0)
                resized_weights = F.interpolate(
                    weight_grid,
                    size=target_grid,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(target_tokens)
                resized_weights = torch.relu(resized_weights)
                resized_weights = resized_weights / (resized_weights.sum() + 1e-6)
                return resized_tokens, resized_weights

    token_sequence = tokens.transpose(0, 1).unsqueeze(0)
    resized_tokens = (
        F.interpolate(
            token_sequence,
            size=target_tokens,
            mode="linear",
            align_corners=False,
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    weight_sequence = weights.reshape(1, 1, -1)
    resized_weights = F.interpolate(
        weight_sequence,
        size=target_tokens,
        mode="linear",
        align_corners=False,
    ).reshape(target_tokens)
    resized_weights = torch.relu(resized_weights)
    resized_weights = resized_weights / (resized_weights.sum() + 1e-6)
    return resized_tokens, resized_weights


def _build_student_eval_transform(model_name: str) -> Any:
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    data_config = resolve_data_config(
        getattr(model, "pretrained_cfg", None), model=model
    )
    return create_transform(**data_config, is_training=False)


def _select_stratified_subset(
    hf_dataset: Any,
    *,
    per_class: int,
    seed: int,
    class_ids: tuple[int, ...] = (),
    label_key: str = "label",
) -> tuple[Any, list[int]]:
    """Return a balanced subset plus original source indices."""
    class_filter = set(class_ids)
    if per_class <= 0 and not class_filter:
        return hf_dataset, list(range(len(hf_dataset)))

    labels = hf_dataset[label_key]
    indices_by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_id = int(label)
        if class_filter and label_id not in class_filter:
            continue
        indices_by_class.setdefault(label_id, []).append(idx)

    if class_filter:
        missing = sorted(class_filter.difference(indices_by_class))
        if missing:
            raise ValueError(f"Classi non trovate nel dataset: {missing}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    selected_indices: list[int] = []
    for label in sorted(indices_by_class):
        class_indices = indices_by_class[label]
        sample_size = (
            len(class_indices) if per_class <= 0 else min(per_class, len(class_indices))
        )
        if sample_size <= 0:
            continue
        permutation = torch.randperm(len(class_indices), generator=generator)
        selected_indices.extend(
            class_indices[int(position)]
            for position in permutation[:sample_size].tolist()
        )

    if selected_indices:
        shuffle_order = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = [
            selected_indices[int(position)] for position in shuffle_order.tolist()
        ]

    return hf_dataset.select(selected_indices), selected_indices


def _maybe_compile(model: nn.Module, *, enable: bool) -> nn.Module:
    if not enable or not hasattr(torch, "compile"):
        return model
    try:
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        print("torch.compile disabilitato: Triton non disponibile.")
        return model
    try:
        return torch.compile(model)
    except Exception as exc:  # noqa: BLE001
        print(f"torch.compile fallito; proseguo senza compilazione: {exc}")
        return model


def run_explainable_distillation_precompute(
    config: ExplainableDistillationPrecomputeConfig,
) -> dict[str, int | str]:
    """Precompute image tensors, teacher patch tokens, and Grad-CAM weights."""
    if config.target_source not in {"prediction", "label"}:
        raise ValueError("target_source deve essere 'prediction' oppure 'label'")
    if config.distillation_target not in {"premerge", "qwen_image_embeddings"}:
        raise ValueError(
            "distillation_target deve essere 'premerge' oppure 'qwen_image_embeddings'"
        )
    if config.attention_source not in {"gradcam", "uniform"}:
        raise ValueError("attention_source deve essere 'gradcam' oppure 'uniform'")
    if config.attention_source == "gradcam" and not config.teacher_model_dir:
        raise ValueError("attention_source=gradcam richiede --teacher-model-dir")
    storage_dtype = _resolve_storage_dtype(config.storage_dtype)

    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    if config.preprocessed_dataset_dir:
        print(
            "Caricamento dataset preprocessato: "
            f"{config.preprocessed_dataset_dir} (split={config.split})"
        )
        hf_dataset = load_preprocessed_mini_imagenet(
            config.preprocessed_dataset_dir,
            splits=(config.split,),
        )[config.split]
    else:
        print(f"Caricamento dataset: {config.dataset_id} (split={config.split})")
        hf_dataset = load_dataset(config.dataset_id, split=config.split)
    if config.max_samples > 0 and config.subset_per_class > 0:
        raise ValueError(
            "Usa solo uno tra max_samples e subset_per_class: il primo taglia "
            "in ordine, il secondo campiona in modo stratificato."
        )

    source_indices: list[int]
    if config.subset_per_class > 0 or config.class_ids:
        hf_dataset, source_indices = _select_stratified_subset(
            hf_dataset,
            per_class=config.subset_per_class,
            class_ids=config.class_ids,
            seed=config.seed,
        )
        if config.subset_per_class > 0:
            print(
                "Subset stratificato attivo: "
                f"{config.subset_per_class} sample/classe, "
                f"{len(hf_dataset)} sample totali"
            )
        if config.class_ids:
            print(
                "Filtro classi attivo: "
                f"{list(config.class_ids)}, {len(hf_dataset)} sample totali"
            )
    elif config.max_samples > 0:
        max_count = min(config.max_samples, len(hf_dataset))
        source_indices = list(range(max_count))
        hf_dataset = hf_dataset.select(source_indices)
    else:
        source_indices = list(range(len(hf_dataset)))

    student_transform = _build_student_eval_transform(config.student_model_name)
    student_patch_grid = _infer_student_patch_grid(config.student_model_name)
    student_num_tokens = student_patch_grid[0] * student_patch_grid[1]
    print(
        "Student patch grid: "
        f"{student_patch_grid[0]}x{student_patch_grid[1]} "
        f"({student_num_tokens} token)"
    )
    wandb_run = init_wandb_run(
        enabled=config.use_wandb,
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=config,
        extra_config={
            "teacher_model_dir": config.teacher_model_dir,
            "dataset_size": len(hf_dataset),
            "student_patch_grid_h": student_patch_grid[0],
            "student_patch_grid_w": student_patch_grid[1],
            "student_num_tokens": student_num_tokens,
        },
        job_type="distillation-precompute",
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_samples = sorted(output_dir.glob(DISTILLATION_SAMPLE_GLOB))
    if config.overwrite:
        for sample_file in existing_samples:
            sample_file.unlink()
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        existing_samples = []
    else:
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            existing_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            metadata_source_indices = existing_metadata.get("source_indices")
            metadata_config = existing_metadata.get("config", {})
            metadata_checks = {
                "distillation_target": config.distillation_target,
                "student_model_name": config.student_model_name,
                "split": config.split,
            }
            if (
                metadata_source_indices is not None
                and metadata_source_indices != source_indices
            ):
                raise ValueError(
                    f"{metadata_file} ha source_indices diversi dal run corrente. "
                    "Usa --overwrite-precompute per rigenerare la directory."
                )
            for key, expected_value in metadata_checks.items():
                actual_value = metadata_config.get(key, existing_metadata.get(key))
                if actual_value is not None and actual_value != expected_value:
                    raise ValueError(
                        f"{metadata_file} non e' compatibile: {key}="
                        f"{actual_value!r}, atteso {expected_value!r}. "
                        "Usa --overwrite-precompute per rigenerare la directory."
                    )

    expected_positions = set(range(len(source_indices)))
    existing_positions = {
        index
        for sample_file in existing_samples
        if (index := _parse_distillation_sample_index(sample_file)) is not None
    }
    unexpected_positions = sorted(existing_positions - expected_positions)
    if unexpected_positions:
        preview = ", ".join(str(index) for index in unexpected_positions[:5])
        raise ValueError(
            f"{output_dir} contiene sample fuori dal range atteso "
            f"({preview}). Usa --overwrite-precompute per rigenerare la directory."
        )

    missing_positions = sorted(expected_positions - existing_positions)
    if existing_positions:
        print(
            "Resume precompute: "
            f"{len(existing_positions)} sample gia' presenti, "
            f"{len(missing_positions)} mancanti."
        )
    if not missing_positions:
        print(
            f"Precompute gia' completo: {len(existing_positions)} sample in "
            f"{output_dir}"
        )

    if missing_positions:
        teacher_label = config.teacher_model_dir or config.teacher_model_id
        print(f"Caricamento teacher: {teacher_label}")
        from ..models.vlm_vit import (  # Local import keeps pure helpers lightweight.
            build_vlm_vision_classifier,
            load_prepared_vlm_classifier,
            resolve_image_processor,
            validate_image_processor,
        )

        if config.teacher_model_dir:
            _validate_teacher_checkpoint_dir(config.teacher_model_dir)
            teacher, processor, teacher_metadata = load_prepared_vlm_classifier(
                config.teacher_model_dir,
                trust_remote_code=config.trust_remote_code,
                torch_dtype=config.torch_dtype,
            )
        else:
            teacher, processor, teacher_metadata = build_vlm_vision_classifier(
                config.teacher_model_id,
                num_labels=2,
                id2label={0: "dummy_0", 1: "dummy_1"},
                label2id={"dummy_0": 0, "dummy_1": 1},
                pooling="mean",
                feature_target="premerge",
                trust_remote_code=config.trust_remote_code,
                torch_dtype=config.torch_dtype,
            )
        if getattr(teacher_metadata, "feature_target", "premerge") != "premerge":
            raise ValueError(
                "La distillazione non usa il Qwen merger: serve un teacher "
                "finetunato con feature_target=premerge."
            )
        image_processor = resolve_image_processor(processor)
        validate_image_processor(image_processor)
        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False

        hf_dataset_to_compute = hf_dataset.select(missing_positions)
        source_indices_to_compute = [
            source_indices[index] for index in missing_positions
        ]
        dataset = RawImageDataset(
            hf_dataset_to_compute,
            source_indices=source_indices_to_compute,
            sample_positions=missing_positions,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=TeacherStudentCollator(image_processor, student_transform),
        )
    else:
        dataloader = []

    teacher_hidden_size = 0
    num_tokens = 0
    for batch in tqdm(dataloader, desc="precompute", dynamic_ncols=True):
        teacher_pixel_values = batch["teacher_pixel_values"].to(device)
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        teacher.zero_grad(set_to_none=True)
        if config.distillation_target == "qwen_image_embeddings":
            if config.attention_source == "gradcam":
                token_list, pooler_list, logits = teacher_forward_token_and_pooler_list(
                    teacher,
                    pixel_values=teacher_pixel_values,
                    image_grid_thw=image_grid_thw,
                    pooling=config.teacher_pooling,
                )
            else:
                token_list, pooler_list = (
                    teacher_forward_token_and_pooler_list_no_probe(
                        teacher,
                        pixel_values=teacher_pixel_values,
                        image_grid_thw=image_grid_thw,
                    )
                )
                logits = None
        else:
            if config.attention_source == "gradcam":
                token_list, logits = teacher_forward_token_list(
                    teacher,
                    pixel_values=teacher_pixel_values,
                    image_grid_thw=image_grid_thw,
                    pooling=config.teacher_pooling,
                )
            else:
                token_list = teacher_forward_token_list_no_probe(
                    teacher,
                    pixel_values=teacher_pixel_values,
                    image_grid_thw=image_grid_thw,
                )
                logits = None
            pooler_list = []

        if config.attention_source == "uniform":
            weight_list = [
                torch.ones(tokens.size(0), device=tokens.device)
                for tokens in token_list
            ]
        else:
            if config.target_source == "label":
                target_class: torch.Tensor | None = batch["labels"].to(device)
            else:
                target_class = None

            if logits is None:
                raise RuntimeError("Logits mancanti per attention_source=gradcam")
            weight_list = compute_token_gradcam_weights_list(
                token_list,
                logits,
                target_class=target_class,
                tau=config.tau,
            )

        images_cpu = batch["student_pixel_values"].to(torch.float16).cpu()
        labels_cpu = batch["labels"].cpu()
        source_indices_cpu = batch["indices"].cpu()
        positions_cpu = batch["positions"].cpu()

        teacher_hidden_size = (
            int(pooler_list[0].size(-1))
            if config.distillation_target == "qwen_image_embeddings"
            else int(token_list[0].size(-1))
        )
        num_tokens = student_num_tokens

        for row, (tokens, weights) in enumerate(
            zip(token_list, weight_list, strict=True)
        ):
            sample_grid_thw = (
                image_grid_thw[row] if image_grid_thw is not None else None
            )
            if config.distillation_target == "qwen_image_embeddings":
                if sample_grid_thw is None:
                    raise ValueError(
                        "image_grid_thw richiesto per target qwen_image_embeddings"
                    )
                merge_size = int(
                    getattr(teacher.vision_backbone, "spatial_merge_size", 1) or 1
                )
                post_grid_thw = sample_grid_thw.detach().clone()
                post_grid_thw[-2:] = post_grid_thw[-2:] // merge_size
                pooler_tokens = pooler_list[row].detach()
                resized_tokens, _ = _resample_teacher_tokens_and_weights(
                    pooler_tokens,
                    torch.ones(
                        pooler_tokens.size(0),
                        device=pooler_tokens.device,
                        dtype=pooler_tokens.dtype,
                    ),
                    grid_thw=post_grid_thw,
                    target_grid=student_patch_grid,
                )
                _, resized_weights = _resample_teacher_tokens_and_weights(
                    tokens.detach(),
                    weights.detach(),
                    grid_thw=sample_grid_thw,
                    target_grid=student_patch_grid,
                )
            else:
                resized_tokens, resized_weights = _resample_teacher_tokens_and_weights(
                    tokens.detach(),
                    weights.detach(),
                    grid_thw=sample_grid_thw,
                    target_grid=student_patch_grid,
                )
            sample = {
                "Z_t": resized_tokens.to(torch.float32).cpu(),
                "A": resized_weights.to(torch.float32).cpu(),
                "label": labels_cpu[row],
                "source_index": source_indices_cpu[row],
            }
            if sample_grid_thw is not None:
                sample["grid_thw"] = sample_grid_thw.detach().to(torch.long).cpu()
            sample["Z_t"] = sample["Z_t"].to(storage_dtype)
            sample["A"] = sample["A"].to(storage_dtype)
            if config.store_images:
                sample["image"] = images_cpu[row]
            sample_position = int(positions_cpu[row].item())
            torch.save(sample, _distillation_sample_path(output_dir, sample_position))

    if teacher_hidden_size == 0 or num_tokens == 0:
        first_sample = next(
            iter(sorted(output_dir.glob(DISTILLATION_SAMPLE_GLOB))),
            None,
        )
        if first_sample is not None:
            payload = torch.load(first_sample, map_location="cpu")
            teacher_hidden_size = int(payload["Z_t"].shape[-1])
            num_tokens = int(payload["Z_t"].shape[0])
    final_sample_count = len(expected_positions)
    metadata = {
        "num_samples": final_sample_count,
        "num_existing_samples_before_resume": len(existing_positions),
        "num_new_samples": len(missing_positions),
        "teacher_hidden_size": teacher_hidden_size,
        "num_tokens": num_tokens,
        "storage_dtype": config.storage_dtype,
        "store_images": config.store_images,
        "source_indices": source_indices,
        "distillation_target": config.distillation_target,
        "config": _jsonable_dataclass(config),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        f"Precompute completato: {final_sample_count} sample in {output_dir} "
        f"({len(missing_positions)} nuovi)"
    )
    summary = {
        "num_samples": final_sample_count,
        "num_new_samples": len(missing_positions),
        "output_dir": str(output_dir),
    }
    log_wandb_metrics(
        wandb_run,
        {
            "precompute/num_samples": final_sample_count,
            "precompute/num_new_samples": len(missing_positions),
            "precompute/teacher_hidden_size": teacher_hidden_size,
            "precompute/num_tokens": num_tokens,
        },
    )
    finish_wandb_run(wandb_run, summary=summary)
    return summary


def train_explainable_student_vit(
    config: ExplainableDistillationTrainConfig,
) -> dict[str, float | str | int]:
    """Train the student ViT against precomputed teacher tokens and weights."""
    if config.eval_every <= 0:
        raise ValueError("eval_every deve essere maggiore di 0")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience non può essere negativo")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta non può essere negativo")

    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset = PrecomputedDistillationDataset(
        config.precomputed_dir,
        filter_uniform_attention=config.filter_uniform_attention,
    )
    first_sample = dataset[0]
    distillation_target = dataset.metadata.get("distillation_target", "premerge")
    teacher_hidden_size = int(first_sample["Z_t"].shape[-1])
    if distillation_target == "qwen_image_embeddings":
        print("Target distillazione: Qwen image embeddings diretti")
    else:
        print("Target distillazione: pre-merger teacher patch tokens")
    if config.filter_uniform_attention:
        print(
            "Filtro attention uniforme attivo: "
            f"rimossi {dataset.num_filtered_uniform} sample piatti su "
            f"{dataset.num_original_files}"
        )

    val_dataset = (
        PrecomputedDistillationDataset(
            config.val_precomputed_dir,
            filter_uniform_attention=config.filter_uniform_attention,
        )
        if config.val_precomputed_dir
        else None
    )
    test_dataset = (
        PrecomputedDistillationDataset(
            config.test_precomputed_dir,
            filter_uniform_attention=config.filter_uniform_attention,
        )
        if config.test_precomputed_dir
        else None
    )
    if val_dataset is not None:
        print(
            f"Validation precompute: {config.val_precomputed_dir} ({len(val_dataset)} sample)"
        )
    else:
        print(
            "Validation precompute non configurato: il checkpoint best userà "
            "la train loss per compatibilità."
        )
    if test_dataset is not None:
        print(
            f"Test precompute: {config.test_precomputed_dir} ({len(test_dataset)} sample)"
        )

    student = StudentViT(
        model_name=config.student_model_name,
        teacher_hidden_size=teacher_hidden_size,
        pretrained=config.pretrained,
    )
    student.to(device)
    student = _maybe_compile(student, enable=config.use_torch_compile)

    dataloader = _build_distillation_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_dataloader = (
        _build_distillation_loader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        if val_dataset is not None
        else None
    )
    test_dataloader = (
        _build_distillation_loader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        if test_dataset is not None
        else None
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    num_training_steps = len(dataloader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = init_wandb_run(
        enabled=config.use_wandb,
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=config,
        extra_config={
            "num_samples": len(dataset),
            "num_original_samples": dataset.num_original_files,
            "num_filtered_uniform": dataset.num_filtered_uniform,
            "num_val_samples": len(val_dataset) if val_dataset is not None else 0,
            "num_test_samples": len(test_dataset) if test_dataset is not None else 0,
            "filter_uniform_attention": config.filter_uniform_attention,
            "teacher_hidden_size": teacher_hidden_size,
            "distillation_target": distillation_target,
            "num_training_steps": num_training_steps,
            "best_checkpoint_metric": (
                "val/loss" if val_dataloader is not None else "train/loss"
            ),
        },
        job_type="distillation-train",
    )

    last_loss = 0.0
    best_loss = float("inf")
    best_epoch = 0
    best_val_loss: float | None = None
    last_val_metrics: dict[str, float] | None = None
    last_test_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        student.train()
        running_loss = 0.0
        running_global = 0.0
        running_expl = 0.0
        running_token_mse = 0.0
        total = 0
        progress = tqdm(
            dataloader,
            desc=f"{epoch}/{config.epochs} distill",
            dynamic_ncols=True,
            colour="green",
        )

        for step, batch in enumerate(progress, start=1):
            images = batch["image"].to(device)
            Z_t = batch["Z_t"].to(device)
            A = batch["A"].to(device)

            Z_s = student(images)
            loss, parts = distillation_losses(
                Z_s,
                Z_t,
                A,
                lambda_global=config.lambda_global,
                lambda_expl=config.lambda_expl,
                lambda_token_mse=config.lambda_token_mse,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            running_global += float(parts["loss_global"].item()) * batch_size
            running_expl += float(parts["loss_expl"].item()) * batch_size
            running_token_mse += float(parts["loss_token_mse"].item()) * batch_size
            total += batch_size

            progress.set_postfix(
                loss=f"{running_loss / max(1, total):.4f}",
                global_loss=f"{running_global / max(1, total):.4f}",
                expl_loss=f"{running_expl / max(1, total):.4f}",
                token_mse=f"{running_token_mse / max(1, total):.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if (
                step == 1
                or step % DEFAULT_LOG_EVERY_STEPS == 0
                or step == len(dataloader)
            ):
                global_step = (epoch - 1) * len(dataloader) + step
                log_wandb_metrics(
                    wandb_run,
                    {
                        "train/loss_running": running_loss / max(1, total),
                        "train/loss_global_running": running_global / max(1, total),
                        "train/loss_expl_running": running_expl / max(1, total),
                        "train/loss_token_mse_running": running_token_mse
                        / max(1, total),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        last_loss = running_loss / max(1, total)
        epoch_global_loss = running_global / max(1, total)
        epoch_expl_loss = running_expl / max(1, total)
        epoch_token_mse = running_token_mse / max(1, total)
        print(f"Epoch {epoch}/{config.epochs} - Distill Loss: {last_loss:.4f}")
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train/loss": last_loss,
            "train/loss_global": epoch_global_loss,
            "train/loss_expl": epoch_expl_loss,
            "train/loss_token_mse": epoch_token_mse,
        }

        should_eval = epoch % config.eval_every == 0 or epoch == config.epochs
        if should_eval and val_dataloader is not None:
            last_val_metrics = _run_distillation_epoch_eval(
                student,
                val_dataloader,
                device,
                config,
                desc=f"{epoch}/{config.epochs} val",
            )
            print(
                "  Val Loss: "
                f"{last_val_metrics['loss']:.4f} "
                f"- Global: {last_val_metrics['loss_global']:.4f} "
                f"- Expl: {last_val_metrics['loss_expl']:.4f}"
            )
            epoch_metrics.update(
                {
                    "val/loss": last_val_metrics["loss"],
                    "val/loss_global": last_val_metrics["loss_global"],
                    "val/loss_expl": last_val_metrics["loss_expl"],
                    "val/loss_token_mse": last_val_metrics["loss_token_mse"],
                }
            )
        if should_eval and test_dataloader is not None:
            last_test_metrics = _run_distillation_epoch_eval(
                student,
                test_dataloader,
                device,
                config,
                desc=f"{epoch}/{config.epochs} test",
            )
            print(
                "  Test Loss: "
                f"{last_test_metrics['loss']:.4f} "
                f"- Global: {last_test_metrics['loss_global']:.4f} "
                f"- Expl: {last_test_metrics['loss_expl']:.4f}"
            )
            epoch_metrics.update(
                {
                    "test/loss": last_test_metrics["loss"],
                    "test/loss_global": last_test_metrics["loss_global"],
                    "test/loss_expl": last_test_metrics["loss_expl"],
                    "test/loss_token_mse": last_test_metrics["loss_token_mse"],
                }
            )

        log_wandb_metrics(wandb_run, epoch_metrics, step=epoch * len(dataloader))

        checkpoint_metric = (
            last_val_metrics["loss"]
            if val_dataloader is not None and last_val_metrics is not None
            else last_loss
        )
        improved = checkpoint_metric < (best_loss - config.early_stopping_min_delta)
        if should_eval and improved:
            best_loss = checkpoint_metric
            best_epoch = epoch
            if last_val_metrics is not None:
                best_val_loss = last_val_metrics["loss"]
            save_student_vit(
                student,
                output_dir / "best",
                config,
                teacher_hidden_size,
                distillation_target=distillation_target,
            )
            epochs_without_improvement = 0
            metric_name = "val/loss" if last_val_metrics is not None else "train/loss"
            print(
                "  Nuovo miglior student salvato in "
                f"{output_dir / 'best'} ({metric_name}={checkpoint_metric:.4f})"
            )
        elif should_eval and val_dataloader is not None:
            epochs_without_improvement += 1
            if (
                config.early_stopping_patience > 0
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                stopped_early = True
                print(
                    "Early stopping: val/loss non migliora da "
                    f"{epochs_without_improvement} valutazioni."
                )
                break

    save_student_vit(
        student,
        output_dir / "final",
        config,
        teacher_hidden_size,
        distillation_target=distillation_target,
    )
    print(f"Student finale salvato in {output_dir / 'final'}")

    summary = {
        "final_loss": last_loss,
        "best_loss": best_loss,
        "best_checkpoint_metric": (
            "val/loss" if val_dataloader is not None else "train/loss"
        ),
        "best_epoch": best_epoch,
        "num_samples": len(dataset),
        "num_original_samples": dataset.num_original_files,
        "num_filtered_uniform": dataset.num_filtered_uniform,
        "num_val_samples": len(val_dataset) if val_dataset is not None else 0,
        "num_test_samples": len(test_dataset) if test_dataset is not None else 0,
        "stopped_early": int(stopped_early),
        "output_dir": str(output_dir),
    }
    if best_val_loss is not None:
        summary["best_val_loss"] = best_val_loss
    if last_val_metrics is not None:
        summary["final_val_loss"] = last_val_metrics["loss"]
    if last_test_metrics is not None:
        summary["final_test_loss"] = last_test_metrics["loss"]
    finish_wandb_run(wandb_run, summary=summary)
    return summary


def save_student_vit(
    model: nn.Module,
    output_dir: str | Path,
    config: ExplainableDistillationTrainConfig,
    teacher_hidden_size: int,
    *,
    distillation_target: str = "premerge",
) -> None:
    """Save a student checkpoint plus reconstruction metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    torch.save(raw_model.state_dict(), output_path / STUDENT_STATE_FILE)
    metadata = {
        "student_model_name": config.student_model_name,
        "teacher_hidden_size": teacher_hidden_size,
        "distillation_target": distillation_target,
        "pretrained": config.pretrained,
        "config": _jsonable_dataclass(config),
    }
    (output_path / STUDENT_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_student_vit(checkpoint_dir: str | Path) -> tuple[StudentViT, dict[str, Any]]:
    """Load a saved StudentViT checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    metadata = json.loads(
        (checkpoint_path / STUDENT_METADATA_FILE).read_text(encoding="utf-8")
    )
    model = StudentViT(
        model_name=metadata["student_model_name"],
        teacher_hidden_size=int(metadata["teacher_hidden_size"]),
        pretrained=False,
    )
    state_dict = torch.load(checkpoint_path / STUDENT_STATE_FILE, map_location="cpu")
    model.load_state_dict(state_dict)
    return model, metadata


def _validate_teacher_checkpoint_dir(checkpoint_dir: str | Path) -> None:
    """Raise a helpful error when the teacher checkpoint layout is missing."""
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "metadata.json"
    state_path = checkpoint_path / "model_state.pt"
    if metadata_path.exists() and state_path.exists():
        return

    candidates = sorted(
        path.parent
        for path in Path.cwd().glob("**/metadata.json")
        if (path.parent / "model_state.pt").exists()
    )
    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates[:10])
    if len(candidates) > 10:
        candidate_text += f"\n  ... altri {len(candidates) - 10}"
    if not candidate_text:
        candidate_text = "  nessun checkpoint compatibile trovato"

    raise FileNotFoundError(
        "Checkpoint teacher non valido: servono metadata.json e model_state.pt in "
        f"{checkpoint_path}.\nCheckpoint compatibili trovati:\n{candidate_text}"
    )

def _jsonable_dataclass(instance: Any) -> dict[str, Any]:
    payload = asdict(instance)
    dtype = payload.get("torch_dtype")
    if isinstance(dtype, torch.dtype):
        payload["torch_dtype"] = str(dtype).replace("torch.", "")
    return payload


def _resolve_storage_dtype(value: str) -> torch.dtype:
    normalized = value.lower().strip()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        "storage_dtype non valido. Usa uno tra: float16, bfloat16, float32."
    )
