"""CLI to fine-tune a VLM-extracted vision classifier on Mini-ImageNet."""

from __future__ import annotations

import argparse

from src.config import (
    DEFAULT_VLM_FINETUNED_OUTPUT_DIR,
    MINI_IMAGENET_DATASET_ID,
    SMALL_VLM_MODEL_ID,
)
from src.training.finetuning_vlm_vit import (
    VLMVisionFineTuneConfig,
    train_vlm_vision_on_mini_imagenet,
)
from src.utils.torch_utils import parse_torch_dtype


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for VLM-ViT fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen vision module + classifier head su Mini-ImageNet"
    )
    parser.add_argument(
        "--vlm-model",
        type=str,
        default=SMALL_VLM_MODEL_ID,
        help=f"Model ID VLM base (default: {SMALL_VLM_MODEL_ID})",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=MINI_IMAGENET_DATASET_ID,
        help=f"Dataset HF (default: {MINI_IMAGENET_DATASET_ID})",
    )
    parser.add_argument(
        "--preprocessed-dataset-dir",
        type=str,
        default=None,
        help=(
            "Directory dataset Mini-ImageNet preprocessato a dimensione fissa. "
            "Se impostato, sostituisce --dataset per train/validation/test."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_VLM_FINETUNED_OUTPUT_DIR,
        help=f"Directory output (default: {DEFAULT_VLM_FINETUNED_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=("mean", "attention", "cross_attention"),
        default="mean",
        help=(
            "Pooling token visivi: mean, attention MLP o cross-attention "
            "con query learnable (default: mean)."
        ),
    )
    parser.add_argument(
        "--vision-attr-path",
        type=str,
        default=None,
        help=(
            "Percorso attributo del vision module nel VLM "
            "(default: usa il registry per Qwen)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Numero di epoche (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Valuta ogni N epoche (default: 1)",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=0,
        help="Valuta il test ogni N epoche (0 = solo alla fine).",
    )
    parser.add_argument(
        "--subset-per-class",
        type=int,
        default=0,
        help=(
            "Usa un subset stratificato con N campioni per classe (0 = disabilitato)."
        ),
    )
    parser.add_argument(
        "--eval-subset-per-class",
        type=int,
        default=0,
        help=(
            "Usa un subset stratificato anche per validation/test "
            "(0 = valuta split completi)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Tipo numerico per il caricamento del VLM (default: auto)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disabilita trust_remote_code nel caricamento HF.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Abilita logging su Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="compvis-finetuning",
        help="Nome progetto W&B (default: compvis-finetuning).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Entity/team W&B (default: account configurato).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Nome run W&B (default: generato da W&B).",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled"),
        help="Modalità W&B (default: usa configurazione W&B/env).",
    )
    return parser.parse_args()


def main() -> None:
    """Run Mini-ImageNet fine-tuning for the VLM-ViT pipeline."""
    args = parse_args()
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    trust_remote_code = not args.no_trust_remote_code

    config = VLMVisionFineTuneConfig(
        vlm_model_id=args.vlm_model,
        dataset_id=args.dataset,
        preprocessed_dataset_dir=args.preprocessed_dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_workers=args.num_workers,
        eval_every=args.eval_every,
        seed=args.seed,
        subset_per_class=args.subset_per_class,
        eval_subset_per_class=args.eval_subset_per_class,
        test_every=args.test_every,
        pooling=args.pooling,
        vision_attr_path=args.vision_attr_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )

    summary = train_vlm_vision_on_mini_imagenet(config)
    print("\nTraining completato.")
    print(f"Miglior validation accuracy: {summary['best_validation_accuracy']:.4f}")
    if "test_accuracy" in summary:
        print(f"Test accuracy finale: {summary['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
