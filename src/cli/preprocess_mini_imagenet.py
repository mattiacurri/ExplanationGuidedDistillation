"""Materialize Mini-ImageNet as fixed-size RGB tensors for this project."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from src.config import MINI_IMAGENET_DATASET_ID
from src.data import preprocess_mini_imagenet_split


DEFAULT_OUTPUT_DIR = "runs/mini_imagenet_preprocessed_392"
DEFAULT_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Mini-ImageNet preprocessing."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Mini-ImageNet as RGB uint8 tensors resized with bicubic "
            "interpolation."
        )
    )
    parser.add_argument(
        "--dataset",
        default=MINI_IMAGENET_DATASET_ID,
        help=f"Hugging Face dataset ID (default: {MINI_IMAGENET_DATASET_ID}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=392,
        help="Square resize dimension in pixels (default: 392).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to preprocess (default: train validation test).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing sample files in each selected output split.",
    )
    return parser.parse_args()


def main() -> None:
    """Preprocess the selected dataset splits into local torch records."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    for split in args.splits:
        existing = sorted((output_dir / split).glob("sample_*.pt"))
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{output_dir / split} already contains {len(existing)} samples. "
                "Pass --overwrite or select a different --output-dir."
            )

    for split in args.splits:
        print(f"Loading {args.dataset} split={split}...")
        dataset = load_dataset(args.dataset, split=split)
        label_feature = dataset.features.get("label")
        label_names = getattr(label_feature, "names", None)
        count = preprocess_mini_imagenet_split(
            dataset,
            output_dir=output_dir,
            split=split,
            image_size=args.image_size,
            overwrite=args.overwrite,
            label_names=label_names,
        )
        print(f"Saved {count} samples to {output_dir / split}")

    print(f"Preprocessed dataset ready at {output_dir}")


if __name__ == "__main__":
    main()
