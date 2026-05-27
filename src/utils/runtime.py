"""Runtime helpers shared across scripts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy and PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
