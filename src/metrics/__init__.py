"""Text and model-output metrics used by CompVis evaluation scripts."""

from __future__ import annotations

from .bertscore import BertScoreResult, compute_bertscore

__all__ = [
    "BertScoreResult",
    "compute_bertscore",
]
