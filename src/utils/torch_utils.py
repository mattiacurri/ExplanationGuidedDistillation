"""PyTorch utilities: dtype parsing, VRAM management."""

from __future__ import annotations

import gc

import torch

_VALID_DTYPES = ("auto", "float32", "float16", "bfloat16")
_DTYPE_MAP: dict[str, torch.dtype | None] = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_torch_dtype(value: str) -> torch.dtype | None:
    """Parse a CLI dtype string into a torch dtype or None (auto)."""
    normalized = value.lower().strip()
    if normalized not in _DTYPE_MAP:
        raise ValueError(
            f"torch-dtype non valido. Usa uno tra: {', '.join(_VALID_DTYPES)}."
        )
    return _DTYPE_MAP[normalized]


def cleanup_vram() -> None:
    """Release unreferenced GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
