"""Upload a local file as a Weights & Biases artifact with metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local artifact file to Weights & Biases."
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-type", default="report")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--metadata-file", default="")
    parser.add_argument(
        "--mode", default=None, choices=("online", "offline", "disabled")
    )
    return parser.parse_args()


def _parse_metadata(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--metadata-json deve essere un oggetto JSON")
    return payload


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact path non trovato: {path}")

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb non installato. Esegui `uv sync` oppure `uv add wandb`."
        ) from exc

    metadata_text = (
        Path(args.metadata_file).read_text(encoding="utf-8-sig")
        if args.metadata_file
        else args.metadata_json
    )
    metadata = _parse_metadata(metadata_text)
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        mode=args.mode,
        job_type="artifact-upload",
        config=metadata,
    )
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        metadata=metadata,
    )
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    run.finish()
    print(f"Uploaded W&B artifact: {args.artifact_name} <- {path}")


if __name__ == "__main__":
    main()
