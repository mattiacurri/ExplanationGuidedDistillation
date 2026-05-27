"""Cosine learning-rate schedule with linear warmup."""

from __future__ import annotations

from math import cos, pi

import torch
from torch.optim.lr_scheduler import LambdaLR


def create_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a cosine LR schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        progress = (current_step - num_warmup_steps) / max(
            1,
            num_training_steps - num_warmup_steps,
        )
        return max(0.0, 0.5 * (1.0 + cos(pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
