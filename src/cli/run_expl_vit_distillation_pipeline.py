"""End-to-end CLI for explainability-aware Qwen-ViT distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    DEFAULT_EXPL_DISTILL_PRECOMPUTE_OUTPUT_DIR,
    DEFAULT_EXPL_DISTILLED_STUDENT_OUTPUT_DIR,
    DEFAULT_VLM_FINETUNED_OUTPUT_DIR,
    MINI_IMAGENET_DATASET_ID,
)
from src.training.explainable_vit_distillation import (
    ExplainableDistillationPrecomputeConfig,
    ExplainableDistillationTrainConfig,
    run_explainable_distillation_precompute,
    train_explainable_student_vit,
)
from src.utils.torch_utils import parse_torch_dtype


def _parse_class_ids(value: str | None) -> tuple[int, ...]:
    """Parse comma-separated Mini-ImageNet class ids."""
    if value is None or not value.strip():
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _default_teacher_dir() -> Path:
    """Prefer the run layout used by local Qwen-ViT fine-tuning commands."""
    run_default = Path("runs") / "vlm_qwen25vl3b_finetuned" / "best"
    if (run_default / "metadata.json").exists():
        return run_default
    return Path(DEFAULT_VLM_FINETUNED_OUTPUT_DIR) / "best"


def _default_split_precompute_dir(precomputed_dir: str, split_name: str) -> str:
    return f"{precomputed_dir}_{split_name}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the full distillation pipeline."""
    default_teacher_dir = _default_teacher_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline completa: Qwen-ViT finetuned teacher -> "
            "token Grad-CAM offline -> training ViT-small student"
        )
    )

    parser.add_argument(
        "--teacher-model-dir",
        type=str,
        default=None,
        help=(
            "Checkpoint Qwen-ViT finetuned teacher. Richiesto per "
            f"attention_source=gradcam (default precedente: {default_teacher_dir})."
        ),
    )
    parser.add_argument(
        "--teacher-model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model id Qwen da usare quando non serve un teacher probe.",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default=DEFAULT_EXPL_DISTILL_PRECOMPUTE_OUTPUT_DIR,
        help=(
            "Directory per sample precomputati "
            f"(default: {DEFAULT_EXPL_DISTILL_PRECOMPUTE_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--val-precomputed-dir",
        type=str,
        default=None,
        help=(
            "Directory per sample validation precomputati. "
            "Default: <precomputed-dir>_val."
        ),
    )
    parser.add_argument(
        "--test-precomputed-dir",
        type=str,
        default=None,
        help=(
            "Directory per sample test precomputati. Default: <precomputed-dir>_test."
        ),
    )
    parser.add_argument(
        "--student-output-dir",
        type=str,
        default=DEFAULT_EXPL_DISTILLED_STUDENT_OUTPUT_DIR,
        help=(
            "Directory checkpoint student "
            f"(default: {DEFAULT_EXPL_DISTILLED_STUDENT_OUTPUT_DIR})"
        ),
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
            "Se impostato, sostituisce --dataset nel precompute e nel recupero immagini."
        ),
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument(
        "--student-model",
        type=str,
        default="vit_small_patch16_224",
        help="Nome modello timm student.",
    )

    parser.add_argument("--precompute-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limite opzionale campioni precompute (0 = tutti).",
    )
    parser.add_argument(
        "--subset-per-class",
        type=int,
        default=0,
        help=(
            "Precomputa solo un subset stratificato con N campioni per classe "
            "(0 = disabilitato). La distillazione usa poi solo questi sample. "
            "Non combinare con --max-samples."
        ),
    )
    parser.add_argument(
        "--class-ids",
        type=str,
        default="",
        help="Classi Mini-ImageNet da includere, separate da virgola (es. 0,1,5).",
    )
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument(
        "--target-source",
        choices=("prediction", "label"),
        default="prediction",
        help="Classe target Grad-CAM.",
    )
    parser.add_argument(
        "--attention-source",
        choices=("gradcam", "uniform"),
        default="gradcam",
        help=(
            "Origine dei pesi token A: Grad-CAM dal probe oppure pesi uniformi "
            "per baseline senza probe."
        ),
    )
    parser.add_argument(
        "--distillation-target",
        choices=("premerge", "qwen_image_embeddings"),
        default="premerge",
        help=(
            "Cosa deve imitare lo student: token pre-merger oppure gli embedding "
            "immagine finali da passare direttamente a Qwen."
        ),
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lambda-global", type=float, default=1.0)
    parser.add_argument("--lambda-expl", type=float, default=1.0)
    parser.add_argument("--lambda-token-mse", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "Numero di valutazioni validation senza miglioramento prima dello stop "
            "(default: 5; 0 = disabilitato)."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Miglioramento minimo richiesto su val/loss.",
    )
    parser.add_argument("--pretrained-student", action="store_true")
    parser.add_argument(
        "--filter-uniform-attention",
        action="store_true",
        help="Scarta i sample precomputati con A completamente uniforme.",
    )
    parser.add_argument("--torch-compile", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Tipo numerico per caricamento teacher.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disabilita trust_remote_code nel caricamento HF.",
    )
    parser.add_argument(
        "--overwrite-precompute",
        action="store_true",
        help="Cancella sample precomputati esistenti prima della pipeline.",
    )
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Dtype usato per salvare Z_t e A durante precompute.",
    )
    parser.add_argument(
        "--store-images",
        action="store_true",
        help=(
            "Salva anche le immagini preprocessate nei sample. Default: no, "
            "le immagini vengono ricostruite dal dataset."
        ),
    )
    parser.add_argument(
        "--skip-precompute",
        action="store_true",
        help=(
            "Salta le fasi offline e usa --precomputed-dir, "
            "--val-precomputed-dir e --test-precomputed-dir gia' esistenti."
        ),
    )
    parser.add_argument("--wandb", action="store_true", help="Abilita logging W&B.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="compvis-distillation",
        help="Nome progetto W&B (default: compvis-distillation).",
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled"),
    )
    return parser.parse_args()


def main() -> None:
    """Run precompute and student training as one pipeline."""
    args = parse_args()
    trust_remote_code = not args.no_trust_remote_code
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    val_precomputed_dir = args.val_precomputed_dir or _default_split_precompute_dir(
        args.precomputed_dir,
        "val",
    )
    test_precomputed_dir = args.test_precomputed_dir or _default_split_precompute_dir(
        args.precomputed_dir,
        "test",
    )

    if not args.skip_precompute:
        precompute_jobs = (
            (args.split, args.precomputed_dir, "precompute"),
            (args.val_split, val_precomputed_dir, "precompute-val"),
            (args.test_split, test_precomputed_dir, "precompute-test"),
        )
        for split, output_dir, run_suffix in precompute_jobs:
            precompute_config = ExplainableDistillationPrecomputeConfig(
                teacher_model_dir=args.teacher_model_dir,
                teacher_model_id=args.teacher_model_id,
                output_dir=output_dir,
                dataset_id=args.dataset,
                preprocessed_dataset_dir=args.preprocessed_dataset_dir,
                split=split,
                batch_size=args.precompute_batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                subset_per_class=args.subset_per_class,
                class_ids=_parse_class_ids(args.class_ids),
                seed=args.seed,
                tau=args.tau,
                target_source=args.target_source,
                distillation_target=args.distillation_target,
                attention_source=args.attention_source,
                student_model_name=args.student_model,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                overwrite=args.overwrite_precompute,
                storage_dtype=args.storage_dtype,
                store_images=args.store_images,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_run_name=(
                    f"{args.wandb_run_name}-{run_suffix}"
                    if args.wandb_run_name
                    else None
                ),
                wandb_mode=args.wandb_mode,
            )
            precompute_summary = run_explainable_distillation_precompute(
                precompute_config
            )
            print(
                "\nPrecompute completato: "
                f"{precompute_summary['num_samples']} sample in "
                f"{precompute_summary['output_dir']} (split={split})"
            )
    else:
        print(
            "Precompute saltato. Uso sample esistenti in "
            f"{args.precomputed_dir}, {val_precomputed_dir}, {test_precomputed_dir}"
        )

    train_config = ExplainableDistillationTrainConfig(
        precomputed_dir=args.precomputed_dir,
        output_dir=args.student_output_dir,
        val_precomputed_dir=val_precomputed_dir,
        test_precomputed_dir=test_precomputed_dir,
        teacher_model_dir=args.teacher_model_dir,
        student_model_name=args.student_model,
        pretrained=args.pretrained_student,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lambda_global=args.lambda_global,
        lambda_expl=args.lambda_expl,
        lambda_token_mse=args.lambda_token_mse,
        num_workers=args.num_workers,
        max_grad_norm=args.max_grad_norm,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        filter_uniform_attention=args.filter_uniform_attention,
        seed=args.seed,
        use_torch_compile=args.torch_compile,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=(
            f"{args.wandb_run_name}-train" if args.wandb_run_name else None
        ),
        wandb_mode=args.wandb_mode,
    )
    train_summary = train_explainable_student_vit(train_config)

    print("\nPipeline completata.")
    print(f"Best student loss: {train_summary['best_loss']:.4f}")
    print(f"Final student loss: {train_summary['final_loss']:.4f}")
    if "best_val_loss" in train_summary:
        print(f"Best val loss: {train_summary['best_val_loss']:.4f}")
    if "final_test_loss" in train_summary:
        print(f"Final test loss: {train_summary['final_test_loss']:.4f}")
    print(f"Student best: {Path(args.student_output_dir) / 'best'}")
    print(f"Student final: {Path(args.student_output_dir) / 'final'}")


if __name__ == "__main__":
    main()
