"""Utilities to extract and adapt a VLM vision backbone for classification."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers.modeling_outputs import ImageClassifierOutput

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

from .token_pooling import AdditiveAttentionPooling, CrossAttentionPooling
from ..utils.model_utils import resolve_attr_path

# ---------------------------------------------------------------------------
# VLM Registry — maps known model IDs to deterministic extraction parameters.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VLMRegistryEntry:
    vision_attr_path: str
    hidden_size_attr: str  # "hidden_size" | "embed_dim" | "projection_dim"
    forward_mode: str  # "pixel_values" | "pixel_values_grid"
    # "hidden_states_grid" | "get_image_features"
    preferred_auto_class: (
        str  # "AutoModel" | "AutoModelForVision2Seq" | "AutoModelForCausalLM"
    )


VLM_REGISTRY: dict[str, VLMRegistryEntry] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": VLMRegistryEntry(
        vision_attr_path="visual",
        hidden_size_attr="hidden_size",
        forward_mode="get_image_features",
        preferred_auto_class="AutoModelForVision2Seq",
    ),
}


AUTO_CLASS_MAP: dict[str, type[nn.Module]] = {
    "AutoModel": AutoModel,
    "AutoModelForCausalLM": AutoModelForCausalLM,
}

if AutoModelForVision2Seq is not None:
    AUTO_CLASS_MAP["AutoModelForVision2Seq"] = AutoModelForVision2Seq

_FALLBACK_LOADERS: tuple[type[nn.Module], ...] = tuple(
    loader
    for loader in (AutoModel, AutoModelForVision2Seq, AutoModelForCausalLM)
    if loader is not None
)


def _resolve_auto_class(
    model_id: str,
) -> type[nn.Module] | None:
    """Return the preferred auto-class for a known model, or None."""
    entry = VLM_REGISTRY.get(model_id)
    if entry is not None:
        return AUTO_CLASS_MAP.get(entry.preferred_auto_class)
    return None


# ---------------------------------------------------------------------------
# Legacy candidate list — used only as fallback for unknown models.
# ---------------------------------------------------------------------------

VISION_ATTR_CANDIDATES: tuple[str, ...] = (
    "vision_model",
    "vision_tower",
    "visual",
    "vision_encoder",
    "model.vision_model",
    "model.vision_tower",
    "model.visual",
    "base_model.vision_model",
    "base_model.vision_tower",
    "backbone.vision_model",
)

PREPARED_STATE_FILE = "model_state.pt"
PREPARED_METADATA_FILE = "metadata.json"
PREPARED_PROCESSOR_DIR = "processor"
SUPPORTED_POOLING_MODES = {"cls", "mean", "attention", "cross_attention"}


# ---------------------------------------------------------------------------
# VLMVisionMetadata
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VLMVisionMetadata:
    """Metadata required to rebuild a prepared VLM vision classifier."""

    vlm_model_id: str
    vision_attr_path: str
    pooling: str
    hidden_size: int
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    feature_target: str = "premerge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "vlm_model_id": self.vlm_model_id,
            "vision_attr_path": self.vision_attr_path,
            "pooling": self.pooling,
            "feature_target": self.feature_target,
            "hidden_size": self.hidden_size,
            "num_labels": self.num_labels,
            "id2label": {str(idx): label for idx, label in self.id2label.items()},
            "label2id": self.label2id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VLMVisionMetadata":
        id2label_payload = payload.get("id2label", {})
        label2id_payload = payload.get("label2id", {})

        id2label = {int(idx): str(label) for idx, label in id2label_payload.items()}
        label2id = {str(label): int(idx) for label, idx in label2id_payload.items()}

        return cls(
            vlm_model_id=str(payload["vlm_model_id"]),
            vision_attr_path=str(payload["vision_attr_path"]),
            pooling=str(payload["pooling"]),
            feature_target=str(payload.get("feature_target", "premerge")),
            hidden_size=int(payload["hidden_size"]),
            num_labels=int(payload["num_labels"]),
            id2label=id2label,
            label2id=label2id,
        )


# ---------------------------------------------------------------------------
# Processor helpers
# ---------------------------------------------------------------------------


def resolve_image_processor(processor: Any) -> Any:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        return image_processor
    return processor


def validate_image_processor(image_processor: Any) -> None:
    required_attrs = ("size", "image_mean", "image_std")
    missing = [attr for attr in required_attrs if not hasattr(image_processor, attr)]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Image processor non compatibile: mancano i campi {missing_str}."
        )


# ---------------------------------------------------------------------------
# Vision backbone discovery
# ---------------------------------------------------------------------------


def _choose_fallback_vision_path(model: nn.Module) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        name_lower = name.lower()
        if "vision" not in name_lower and "visual" not in name_lower:
            continue
        num_params = sum(p.numel() for p in module.parameters())
        if num_params == 0:
            continue
        candidates.append((name.count("."), -num_params, name))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def extract_vision_backbone(
    model: nn.Module,
    *,
    vision_attr_path: str | None = None,
    model_id: str | None = None,
) -> tuple[nn.Module, str]:
    """Extract a vision backbone module from a VLM model instance."""
    # 1) Explicitly provided path
    if vision_attr_path is not None:
        module = resolve_attr_path(model, vision_attr_path)
        if isinstance(module, nn.Module):
            return module, vision_attr_path
        raise ValueError(
            "Il percorso vision_attr_path fornito non punta a un modulo valido: "
            f"{vision_attr_path}"
        )

    # 2) Registry lookup
    if model_id and model_id in VLM_REGISTRY:
        path = VLM_REGISTRY[model_id].vision_attr_path
        module = resolve_attr_path(model, path)
        if isinstance(module, nn.Module):
            return module, path

    # 3) Candidate iteration
    for candidate in VISION_ATTR_CANDIDATES:
        module = resolve_attr_path(model, candidate)
        if isinstance(module, nn.Module):
            return module, candidate

    # 4) Heuristic fallback
    fallback_path = _choose_fallback_vision_path(model)
    if fallback_path is None:
        raise ValueError(
            "Impossibile individuare automaticamente il backbone visivo nel VLM. "
            "Passa --vision-attr-path esplicitamente."
        )

    module = resolve_attr_path(model, fallback_path)
    if not isinstance(module, nn.Module):
        raise ValueError(
            f"Il percorso fallback individuato non è un modulo valido: {fallback_path}"
        )
    return module, fallback_path


def infer_hidden_size(
    vision_backbone: nn.Module,
    *,
    model_id: str | None = None,
) -> int:
    """Infer the embedding size produced by a vision backbone."""
    # Registry-directed lookup
    if model_id and model_id in VLM_REGISTRY:
        attr = VLM_REGISTRY[model_id].hidden_size_attr
        config = getattr(vision_backbone, "config", None)
        if config is not None:
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        value = getattr(vision_backbone, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    # Legacy heuristic loop
    config = getattr(vision_backbone, "config", None)
    if config is not None:
        for attr in ("hidden_size", "embed_dim", "projection_dim"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    for attr in ("hidden_size", "embed_dim", "projection_dim"):
        value = getattr(vision_backbone, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    raise ValueError(
        "Impossibile inferire hidden_size dal backbone visivo. "
        "Specifica un modello VLM compatibile o estendi l'inferenza."
    )


def infer_postmerger_hidden_size(vision_backbone: nn.Module) -> int:
    """Infer the hidden size emitted by a Qwen-style visual merger."""
    merger = getattr(vision_backbone, "merger", None)
    if merger is None:
        raise ValueError("Il backbone visivo non espone un modulo merger")

    for module in reversed(list(merger.modules())):
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    raise ValueError("Impossibile inferire hidden_size post-merger dal merger")


# ---------------------------------------------------------------------------
# Output extraction helpers (unchanged logic)
# ---------------------------------------------------------------------------


def _extract_hidden_state(outputs: Any) -> torch.Tensor | None:
    if isinstance(outputs, torch.Tensor):
        return outputs
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state
    if isinstance(outputs, dict):
        hidden = outputs.get("last_hidden_state")
        if isinstance(hidden, torch.Tensor):
            return hidden
        hidden_states = outputs.get("hidden_states")
        if isinstance(hidden_states, (tuple, list)) and hidden_states:
            candidate = hidden_states[-1]
            if isinstance(candidate, torch.Tensor):
                return candidate
    if isinstance(outputs, tuple) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _extract_pooler_output(outputs: Any) -> torch.Tensor | None:
    pooler_output = getattr(outputs, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    if isinstance(outputs, dict):
        pooled = outputs.get("pooler_output")
        if isinstance(pooled, torch.Tensor):
            return pooled
    return None


def _extract_optional_output(outputs: Any, field_name: str) -> Any:
    value = getattr(outputs, field_name, None)
    if value is not None:
        return value
    if isinstance(outputs, dict):
        return outputs.get(field_name)
    return None


def _sanitize_token_tensor(tokens: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(tokens).all():
        return tokens
    return torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Forward mode inference
# ---------------------------------------------------------------------------


def _infer_grid_param_name(func: Callable[..., Any]) -> str | None:
    """Return the grid parameter name supported by a callable, if present."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    parameters = signature.parameters
    if "image_grid_thw" in parameters:
        return "image_grid_thw"
    if "grid_thw" in parameters:
        return "grid_thw"
    return None


def _infer_forward_mode_from_signature(vision_backbone: nn.Module) -> str:
    """Infer the forward mode based on backbone signature heuristics."""
    if hasattr(vision_backbone, "get_image_features"):
        return "get_image_features"

    try:
        forward_signature = inspect.signature(vision_backbone.forward)
    except (TypeError, ValueError):
        return "pixel_values"

    parameters = forward_signature.parameters
    if "hidden_states" in parameters and "grid_thw" in parameters:
        return "hidden_states_grid"
    if "pixel_values" in parameters and (
        "image_grid_thw" in parameters or "grid_thw" in parameters
    ):
        return "pixel_values_grid"
    if "pixel_values" in parameters:
        return "pixel_values"
    return "pixel_values"


def _infer_backbone_forward_mode(
    vision_backbone: nn.Module,
    *,
    model_id: str | None = None,
) -> str:
    """Infer which argument contract should be used for backbone forward."""
    if model_id and model_id in VLM_REGISTRY:
        registry_mode = VLM_REGISTRY[model_id].forward_mode
        if registry_mode == "get_image_features" and not hasattr(
            vision_backbone, "get_image_features"
        ):
            return _infer_forward_mode_from_signature(vision_backbone)
        return registry_mode

    return _infer_forward_mode_from_signature(vision_backbone)


# ---------------------------------------------------------------------------
# VLMVisionClassifier
# ---------------------------------------------------------------------------


class VLMVisionClassifier(nn.Module):
    """Classifier built from a VLM vision backbone plus a linear head."""

    def __init__(
        self,
        *,
        vision_backbone: nn.Module,
        hidden_size: int,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        pooling: str = "cls",
        dropout_prob: float = 0.1,
        model_id: str | None = None,
        feature_target: str = "premerge",
    ) -> None:
        super().__init__()
        if pooling not in SUPPORTED_POOLING_MODES:
            valid_modes = "', '".join(sorted(SUPPORTED_POOLING_MODES))
            raise ValueError(f"pooling deve essere uno tra '{valid_modes}'")
        if feature_target not in {"premerge", "postmerger"}:
            raise ValueError(
                "feature_target deve essere 'premerge' oppure 'postmerger'"
            )
        if num_labels <= 1:
            raise ValueError("num_labels deve essere maggiore di 1")

        self.vision_backbone = vision_backbone
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.pooling = pooling
        self.feature_target = feature_target
        self.token_pooler = self._build_token_pooler(pooling, hidden_size)
        self.id2label = id2label
        self.label2id = label2id
        self.backbone_forward_mode = _infer_backbone_forward_mode(
            vision_backbone,
            model_id=model_id,
        )

        # Determine if backbone supports return_dict — compute once at init.
        try:
            sig = inspect.signature(self.vision_backbone.forward)
            self._supports_return_dict = "return_dict" in sig.parameters
        except (TypeError, ValueError):
            self._supports_return_dict = False

        self._grid_param_name = _infer_grid_param_name(self.vision_backbone.forward)
        self._get_image_features_grid_param = None
        if hasattr(self.vision_backbone, "get_image_features"):
            self._get_image_features_grid_param = _infer_grid_param_name(
                self.vision_backbone.get_image_features
            )

    @staticmethod
    def _build_token_pooler(pooling: str, hidden_size: int) -> nn.Module | None:
        if pooling == "attention":
            return AdditiveAttentionPooling(hidden_size)
        if pooling == "cross_attention":
            return CrossAttentionPooling(hidden_size)
        return None

    def _pool_flattened_tokens(
        self,
        hidden_state: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if image_grid_thw.ndim != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError(
                "image_grid_thw deve avere shape [B, 3], "
                f"ricevuto {tuple(image_grid_thw.shape)}"
            )

        token_counts = image_grid_thw.prod(dim=-1).to(torch.long).tolist()
        total_tokens = int(sum(token_counts))
        if total_tokens != int(hidden_state.size(0)):
            raise ValueError(
                "Numero token non coerente con image_grid_thw: "
                f"expected={total_tokens}, actual={hidden_state.size(0)}"
            )

        pooled_embeddings: list[torch.Tensor] = []
        start = 0
        for token_count in token_counts:
            end = start + token_count
            image_tokens = _sanitize_token_tensor(hidden_state[start:end])
            if self.pooling == "mean":
                pooled_embeddings.append(image_tokens.mean(dim=0))
            else:
                pooled_embeddings.append(image_tokens[0])
            start = end

        return torch.stack(pooled_embeddings, dim=0)

    def _postmerger_token_counts(self, image_grid_thw: torch.Tensor) -> list[int]:
        if image_grid_thw.ndim != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError(
                "image_grid_thw deve avere shape [B, 3], "
                f"ricevuto {tuple(image_grid_thw.shape)}"
            )
        spatial_merge_size = int(
            getattr(self.vision_backbone, "spatial_merge_size", 1) or 1
        )
        merge_unit = max(1, spatial_merge_size * spatial_merge_size)
        return (image_grid_thw.prod(dim=-1).to(torch.long) // merge_unit).tolist()

    def _pool_flattened_tokens_by_counts(
        self,
        hidden_state: torch.Tensor,
        token_counts: list[int],
    ) -> torch.Tensor:
        total_tokens = int(sum(token_counts))
        if total_tokens != int(hidden_state.size(0)):
            raise ValueError(
                "Numero token non coerente: "
                f"expected={total_tokens}, actual={hidden_state.size(0)}"
            )

        pooled_embeddings: list[torch.Tensor] = []
        start = 0
        for token_count in token_counts:
            end = start + int(token_count)
            image_tokens = _sanitize_token_tensor(hidden_state[start:end])
            if self.pooling == "mean":
                pooled_embeddings.append(image_tokens.mean(dim=0))
            else:
                pooled_embeddings.append(image_tokens[0])
            start = end

        return torch.stack(pooled_embeddings, dim=0)

    def _pad_flattened_tokens(
        self,
        hidden_state: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_grid_thw.ndim != 2 or image_grid_thw.size(-1) != 3:
            raise ValueError(
                "image_grid_thw deve avere shape [B, 3], "
                f"ricevuto {tuple(image_grid_thw.shape)}"
            )

        token_counts = image_grid_thw.prod(dim=-1).to(torch.long).tolist()
        total_tokens = int(sum(token_counts))
        if total_tokens != int(hidden_state.size(0)):
            raise ValueError(
                "Numero token non coerente con image_grid_thw: "
                f"expected={total_tokens}, actual={hidden_state.size(0)}"
            )

        batch_size = len(token_counts)
        max_tokens = max(int(count) for count in token_counts)
        hidden_state = _sanitize_token_tensor(hidden_state)
        padded = hidden_state.new_zeros(batch_size, max_tokens, hidden_state.size(-1))
        key_padding_mask = torch.ones(
            batch_size,
            max_tokens,
            dtype=torch.bool,
            device=hidden_state.device,
        )

        start = 0
        for row, token_count in enumerate(token_counts):
            end = start + int(token_count)
            padded[row, : int(token_count)] = hidden_state[start:end]
            key_padding_mask[row, : int(token_count)] = False
            start = end

        return padded, key_padding_mask

    def _pad_flattened_tokens_by_counts(
        self,
        hidden_state: torch.Tensor,
        token_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_tokens = int(sum(token_counts))
        if total_tokens != int(hidden_state.size(0)):
            raise ValueError(
                "Numero token non coerente: "
                f"expected={total_tokens}, actual={hidden_state.size(0)}"
            )

        batch_size = len(token_counts)
        max_tokens = max(int(count) for count in token_counts)
        hidden_state = _sanitize_token_tensor(hidden_state)
        padded = hidden_state.new_zeros(batch_size, max_tokens, hidden_state.size(-1))
        key_padding_mask = torch.ones(
            batch_size,
            max_tokens,
            dtype=torch.bool,
            device=hidden_state.device,
        )

        start = 0
        for row, token_count in enumerate(token_counts):
            end = start + int(token_count)
            padded[row, : int(token_count)] = hidden_state[start:end]
            key_padding_mask[row, : int(token_count)] = False
            start = end

        return padded, key_padding_mask

    def _pool_token_batch(
        self,
        tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = _sanitize_token_tensor(tokens)
        if self.pooling == "mean":
            if key_padding_mask is None:
                return tokens.mean(dim=1)
            valid = (~key_padding_mask).to(dtype=tokens.dtype)
            summed = torch.sum(tokens * valid.unsqueeze(-1), dim=1)
            counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
            return summed / counts

        if self.pooling == "cls":
            return tokens[:, 0]

        if self.token_pooler is None:
            raise RuntimeError(f"Token pooler mancante per pooling={self.pooling}")
        pooler_dtype = next(self.token_pooler.parameters()).dtype
        if tokens.dtype != pooler_dtype:
            tokens = tokens.to(pooler_dtype)
        pooled, _ = self.token_pooler(tokens, key_padding_mask=key_padding_mask)
        return pooled

    def _forward_backbone(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Any:
        if self.backbone_forward_mode == "get_image_features":
            if output_attentions:
                kwargs: dict[str, torch.Tensor | bool] = {
                    "pixel_values": pixel_values,
                    "output_attentions": True,
                }
                if image_grid_thw is not None and self._grid_param_name is not None:
                    kwargs[self._grid_param_name] = image_grid_thw
                if self._supports_return_dict:
                    kwargs["return_dict"] = True
                return self.vision_backbone(**kwargs)

            if not hasattr(self.vision_backbone, "get_image_features"):
                raise AttributeError(
                    "Il backbone visivo non espone get_image_features; "
                    "verifica la modalità di forward configurata."
                )
            kwargs: dict[str, torch.Tensor] = {"pixel_values": pixel_values}
            if (
                image_grid_thw is not None
                and self._get_image_features_grid_param is not None
            ):
                kwargs[self._get_image_features_grid_param] = image_grid_thw
            return self.vision_backbone.get_image_features(**kwargs)

        if self.backbone_forward_mode == "pixel_values_grid":
            if image_grid_thw is None:
                raise ValueError(
                    "Questo backbone richiede image_grid_thw, ma non è stato fornito."
                )
            if self._grid_param_name is None:
                raise ValueError(
                    "Parametro grid_thw non individuato per il backbone visivo."
                )
            kwargs: dict[str, torch.Tensor | bool] = {
                "pixel_values": pixel_values,
                self._grid_param_name: image_grid_thw,
            }
            if self._supports_return_dict:
                kwargs["return_dict"] = True
            if output_attentions:
                kwargs["output_attentions"] = True
            return self.vision_backbone(**kwargs)

        if self.backbone_forward_mode == "hidden_states_grid":
            if image_grid_thw is None:
                raise ValueError(
                    "Questo backbone richiede image_grid_thw, ma non è stato fornito."
                )
            kwargs: dict[str, torch.Tensor | bool] = {
                "hidden_states": pixel_values,
                "grid_thw": image_grid_thw,
            }
            if self._supports_return_dict:
                kwargs["return_dict"] = True
            if output_attentions:
                kwargs["output_attentions"] = True
            return self.vision_backbone(**kwargs)

        kwargs: dict[str, torch.Tensor | bool] = {"pixel_values": pixel_values}
        if self._supports_return_dict:
            kwargs["return_dict"] = True
        if output_attentions:
            kwargs["output_attentions"] = True
        return self.vision_backbone(**kwargs)

    def _pool_backbone_outputs(
        self,
        outputs: Any,
        *,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = _extract_pooler_output(outputs)
        if pooled is not None:
            pooled = _sanitize_token_tensor(pooled)
        if self.feature_target == "postmerger":
            if pooled is None:
                raise ValueError("feature_target=postmerger richiede pooler_output")
            if pooled.ndim == 3:
                return self._pool_token_batch(pooled)
            if pooled.ndim != 2:
                raise ValueError(
                    "Shape pooler_output post-merger non supportata: "
                    f"{tuple(pooled.shape)}"
                )
            if image_grid_thw is None:
                if pooled.size(-1) == self.classifier.in_features:
                    return pooled
                raise ValueError(
                    "image_grid_thw richiesto per pooler_output post-merger flatten"
                )
            token_counts = self._postmerger_token_counts(image_grid_thw)
            if self.pooling in {"cls", "mean"}:
                return self._pool_flattened_tokens_by_counts(pooled, token_counts)
            tokens, key_padding_mask = self._pad_flattened_tokens_by_counts(
                pooled,
                token_counts,
            )
            return self._pool_token_batch(tokens, key_padding_mask=key_padding_mask)

        if (
            self.pooling in {"cls", "mean"}
            and pooled is not None
            and pooled.ndim == 2
            and pooled.size(-1) == self.classifier.in_features
        ):
            return pooled

        hidden_state = _extract_hidden_state(outputs)
        if hidden_state is None:
            raise ValueError(
                "Il backbone visivo non restituisce né token embeddings "
                "né pooler_output utilizzabili."
            )

        if hidden_state.ndim == 2:
            if image_grid_thw is not None:
                if self.pooling in {"cls", "mean"}:
                    return self._pool_flattened_tokens(hidden_state, image_grid_thw)
                tokens, key_padding_mask = self._pad_flattened_tokens(
                    hidden_state,
                    image_grid_thw,
                )
                return self._pool_token_batch(
                    tokens,
                    key_padding_mask=key_padding_mask,
                )
            return _sanitize_token_tensor(hidden_state)

        if hidden_state.ndim != 3:
            raise ValueError(
                "Dimensione output backbone non supportata: "
                f"{tuple(hidden_state.shape)}"
            )

        return self._pool_token_batch(hidden_state)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> ImageClassifierOutput:
        backbone_outputs = self._forward_backbone(
            pixel_values,
            image_grid_thw=image_grid_thw,
        )
        pooled = self._pool_backbone_outputs(
            backbone_outputs,
            image_grid_thw=image_grid_thw,
        )
        classifier_dtype = self.classifier.weight.dtype
        if pooled.dtype != classifier_dtype:
            pooled = pooled.to(classifier_dtype)
        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=_extract_optional_output(backbone_outputs, "hidden_states"),
            attentions=_extract_optional_output(backbone_outputs, "attentions"),
        )


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------


def build_vlm_vision_classifier(
    vlm_model_id: str,
    *,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    pooling: str = "cls",
    dropout_prob: float = 0.1,
    feature_target: str = "premerge",
    vision_attr_path: str | None = None,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype | None = None,
    auto_class: type[nn.Module] | None = None,
) -> tuple[VLMVisionClassifier, Any, VLMVisionMetadata]:
    """Create a Mini-ImageNet classifier from a VLM vision backbone."""

    def _load_vlm_model() -> nn.Module:
        """Load the VLM model using the preferred auto-class or fallback iteration."""
        # 1) Explicit auto_class
        if auto_class is not None:
            return auto_class.from_pretrained(
                vlm_model_id,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )

        # 2) Registry-directed
        registry_class = _resolve_auto_class(vlm_model_id)
        if registry_class is not None:
            try:
                return registry_class.from_pretrained(
                    vlm_model_id,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                )
            except Exception as exc:
                raise ValueError(
                    f"Caricamento VLM fallito con la classe auto registrata "
                    f"'{registry_class.__name__}': {exc}"
                ) from exc

        # 3) Fallback iteration (unknown models only)
        load_errors: list[str] = []
        for loader in _FALLBACK_LOADERS:
            try:
                return loader.from_pretrained(
                    vlm_model_id,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                )
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{loader.__name__}: {exc}")

        error_text = "\n".join(load_errors)
        raise ValueError(
            "Impossibile caricare il VLM con le classi auto supportate. "
            "Dettagli errori:\n"
            f"{error_text}"
        )

    processor = AutoProcessor.from_pretrained(
        vlm_model_id,
        trust_remote_code=trust_remote_code,
    )
    vlm_model = _load_vlm_model()

    vision_backbone, resolved_path = extract_vision_backbone(
        vlm_model,
        vision_attr_path=vision_attr_path,
        model_id=vlm_model_id,
    )
    if feature_target not in {"premerge", "postmerger"}:
        raise ValueError("feature_target deve essere 'premerge' oppure 'postmerger'")
    if feature_target == "postmerger":
        hidden_size = infer_postmerger_hidden_size(vision_backbone)
    else:
        hidden_size = infer_hidden_size(vision_backbone, model_id=vlm_model_id)

    classifier = VLMVisionClassifier(
        vision_backbone=vision_backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        pooling=pooling,
        dropout_prob=dropout_prob,
        model_id=vlm_model_id,
        feature_target=feature_target,
    )

    metadata = VLMVisionMetadata(
        vlm_model_id=vlm_model_id,
        vision_attr_path=resolved_path,
        pooling=pooling,
        hidden_size=hidden_size,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        feature_target=feature_target,
    )

    return classifier, processor, metadata


# ---------------------------------------------------------------------------
# Save / Load prepared classifiers
# ---------------------------------------------------------------------------


def save_prepared_vlm_classifier(
    model: VLMVisionClassifier,
    processor: Any,
    metadata: VLMVisionMetadata,
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / PREPARED_STATE_FILE)

    metadata_path = output_path / PREPARED_METADATA_FILE
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    processor_output_path = output_path / PREPARED_PROCESSOR_DIR
    processor.save_pretrained(processor_output_path)


def load_prepared_vlm_classifier(
    prepared_dir: str | Path,
    *,
    trust_remote_code: bool = True,
    torch_dtype: torch.dtype | None = None,
) -> tuple[VLMVisionClassifier, Any, VLMVisionMetadata]:
    prepared_path = Path(prepared_dir)
    metadata_path = prepared_path / PREPARED_METADATA_FILE
    state_path = prepared_path / PREPARED_STATE_FILE

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata non trovato: {metadata_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State dict non trovato: {state_path}")

    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = VLMVisionMetadata.from_dict(metadata_payload)

    model, _, _ = build_vlm_vision_classifier(
        metadata.vlm_model_id,
        num_labels=metadata.num_labels,
        id2label=metadata.id2label,
        label2id=metadata.label2id,
        pooling=metadata.pooling,
        vision_attr_path=metadata.vision_attr_path,
        feature_target=metadata.feature_target,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    state_dict = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state_dict)

    processor_path = prepared_path / PREPARED_PROCESSOR_DIR
    if processor_path.exists():
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=trust_remote_code,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            metadata.vlm_model_id,
            trust_remote_code=trust_remote_code,
        )

    return model, processor, metadata
