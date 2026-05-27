"""General-purpose utilities for runtime and infrastructure concerns."""

from .model_utils import (
    eager_attention_guard,
    resolve_attr_path,
    resolve_processor_output_size,
    training_state_guard,
)
from .runtime import count_parameters, get_device, set_seed
from .scheduler import create_cosine_warmup_scheduler
from .torch_utils import cleanup_vram, parse_torch_dtype

__all__ = [
    "cleanup_vram",
    "count_parameters",
    "create_cosine_warmup_scheduler",
    "eager_attention_guard",
    "get_device",
    "parse_torch_dtype",
    "resolve_attr_path",
    "resolve_processor_output_size",
    "set_seed",
    "training_state_guard",
]
