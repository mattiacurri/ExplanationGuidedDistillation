"""Small optional Weights & Biases helpers for training runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _to_wandb_config(value: Any) -> Any:
    """Convert common project objects into W&B-friendly config values."""
    if is_dataclass(value):
        return _to_wandb_config(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_wandb_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_wandb_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None = None,
    name: str | None = None,
    mode: str | None = None,
    config: Any | None = None,
    extra_config: Mapping[str, Any] | None = None,
    job_type: str | None = None,
) -> Any | None:
    """Initialize a W&B run when explicitly enabled."""
    if not enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb non installato. Esegui `uv sync` oppure `uv add wandb`."
        ) from exc

    run_config = _to_wandb_config(config) if config is not None else {}
    if extra_config:
        run_config.update(_to_wandb_config(extra_config))

    init_kwargs: dict[str, Any] = {
        "project": project,
        "config": run_config,
    }
    if entity:
        init_kwargs["entity"] = entity
    if name:
        init_kwargs["name"] = name
    if mode:
        init_kwargs["mode"] = mode
    if job_type:
        init_kwargs["job_type"] = job_type

    return wandb.init(**init_kwargs)


def log_wandb_metrics(
    run: Any | None,
    metrics: Mapping[str, Any],
    *,
    step: int | None = None,
) -> None:
    """Log metrics if a W&B run is active."""
    if run is None:
        return
    run.log(_to_wandb_config(metrics), step=step)


def finish_wandb_run(
    run: Any | None,
    *,
    summary: Mapping[str, Any] | None = None,
) -> None:
    """Update summary and close an active W&B run."""
    if run is None:
        return
    if summary:
        run.summary.update(_to_wandb_config(summary))
    run.finish()
