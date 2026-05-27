"""Model inspection and state management utilities."""

from __future__ import annotations

from contextlib import contextmanager
import re
from typing import Any

import torch.nn as nn


_INDEX_PATTERN = re.compile(
    r"^(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*))?\[(?P<index>-?\d+)\]$"
)


def resolve_attr_path(root: Any, attr_path: str) -> Any:
    """Resolve a dotted attribute path from an object root.

    Supports optional list-style indexing like "layers[-1]" or "[0]".
    """
    # Walk the attribute chain, handling optional bracket indexing segments.
    current = root
    for part in attr_path.split("."):
        if not part:
            continue

        index_match = _INDEX_PATTERN.match(part)
        if index_match:
            name = index_match.group("name")
            index = int(index_match.group("index"))
            if name:
                if not hasattr(current, name):
                    return None
                current = getattr(current, name)
            try:
                current = current[index]
            except (TypeError, KeyError, IndexError):
                return None
            continue

        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


@contextmanager
def training_state_guard(model: nn.Module):
    """Context manager that saves/restores model training state and zeroes grads."""
    was_training = model.training
    try:
        model.eval()
        yield
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


@contextmanager
def eager_attention_guard(model: nn.Module):
    """Context manager that temporarily switches to eager attention and restores."""
    previous = getattr(model.config, "_attn_implementation", None)
    if previous != "eager" and hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    try:
        yield
    finally:
        if previous is not None and previous != "eager":
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation(previous)


def resolve_processor_output_size(processor: Any) -> int:
    """Resolve image size from a HuggingFace processor configuration."""
    size_config = getattr(processor, "size", None)
    if isinstance(size_config, dict):
        return int(
            size_config.get("height")
            or size_config.get("shortest_edge")
            or size_config.get("width")
            or 224
        )
    if isinstance(size_config, int):
        return size_config
    return 224
