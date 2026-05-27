"""BERTScore helpers for comparing generated text outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BertScoreResult:
    """BERTScore precision/recall/F1, both per item and averaged."""

    precision: list[float]
    recall: list[float]
    f1: list[float]

    @property
    def precision_mean(self) -> float:
        return _mean(self.precision)

    @property
    def recall_mean(self) -> float:
        return _mean(self.recall)

    @property
    def f1_mean(self) -> float:
        return _mean(self.f1)

    def to_dict(self) -> dict[str, float | list[float]]:
        """Return a JSON-serializable representation."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "precision_mean": self.precision_mean,
            "recall_mean": self.recall_mean,
            "f1_mean": self.f1_mean,
        }


def compute_bertscore(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    lang: str = "en",
    model_type: str | None = None,
    batch_size: int = 16,
    device: str | None = None,
    rescale_with_baseline: bool = False,
) -> BertScoreResult:
    """Compute BERTScore for candidate texts against reference texts."""
    if len(candidates) != len(references):
        raise ValueError("candidates e references devono avere la stessa lunghezza")
    if batch_size <= 0:
        raise ValueError("batch_size deve essere maggiore di 0")
    if not candidates:
        return BertScoreResult(precision=[], recall=[], f1=[])

    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError(
            "BERTScore non e' installato. Installa le dipendenze del progetto "
            "o esegui: uv add bert-score"
        ) from exc

    kwargs = {
        "lang": lang,
        "batch_size": batch_size,
        "rescale_with_baseline": rescale_with_baseline,
        "verbose": False,
    }
    if model_type:
        kwargs["model_type"] = model_type
    if device:
        kwargs["device"] = device

    precision, recall, f1 = bert_score(
        list(candidates),
        list(references),
        **kwargs,
    )
    return BertScoreResult(
        precision=_tensor_to_float_list(precision),
        recall=_tensor_to_float_list(recall),
        f1=_tensor_to_float_list(f1),
    )


def _tensor_to_float_list(values) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
