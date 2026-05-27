"""Fine-tune the Qwen vision module with a small classifier head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import (
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LOG_EVERY_STEPS,
    DEFAULT_VLM_FINETUNED_OUTPUT_DIR,
    MINI_IMAGENET_DATASET_ID,
    SMALL_VLM_MODEL_ID,
)
from ..data import MiniImageNetRawDataset, build_label_maps
from ..data import load_preprocessed_mini_imagenet
from ..models.vlm_vit import (
    build_vlm_vision_classifier,
    resolve_image_processor,
    save_prepared_vlm_classifier,
    validate_image_processor,
)
from ..utils import count_parameters, get_device, set_seed
from ..utils.scheduler import create_cosine_warmup_scheduler
from .trainer_utils import evaluate_model, run_training_epoch
from .wandb_utils import finish_wandb_run, init_wandb_run, log_wandb_metrics


@dataclass(slots=True)
class VLMVisionFineTuneConfig:
    """Configuration for Qwen vision-module fine-tuning."""

    vlm_model_id: str = SMALL_VLM_MODEL_ID
    dataset_id: str = MINI_IMAGENET_DATASET_ID
    preprocessed_dataset_dir: str | None = None
    output_dir: str = DEFAULT_VLM_FINETUNED_OUTPUT_DIR
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 4
    eval_every: int = 1
    seed: int = 42
    subset_per_class: int = 0
    eval_subset_per_class: int = 0
    test_every: int = 0
    pooling: str = "mean"
    vision_attr_path: str | None = None
    trust_remote_code: bool = True
    torch_dtype: torch.dtype | None = None
    use_wandb: bool = False
    wandb_project: str = "compvis-finetuning"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str | None = None


class QwenVisionCollator:
    """Collate raw images through the Qwen image processor."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, batch: list[tuple[Any, int]]) -> dict[str, torch.Tensor]:
        images, labels = zip(*batch, strict=True)
        encoded = self.image_processor(images=list(images), return_tensors="pt")

        collated: dict[str, torch.Tensor] = {
            "pixel_values": encoded["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        image_grid_thw = encoded.get("image_grid_thw")
        if image_grid_thw is None:
            image_grid_thw = encoded.get("grid_thw")
        if image_grid_thw is not None:
            collated["image_grid_thw"] = image_grid_thw
        return collated


def _apply_stratified_subset(
    hf_dataset: Any,
    *,
    per_class: int,
    seed: int,
    label_key: str = "label",
) -> Any:
    """Return a stratified subset of a HF dataset."""
    if per_class <= 0:
        return hf_dataset

    labels = hf_dataset[label_key]
    indices_by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(int(label), []).append(idx)

    generator = torch.Generator()
    generator.manual_seed(seed)
    selected_indices: list[int] = []
    for label in sorted(indices_by_class):
        class_indices = indices_by_class[label]
        sample_size = min(per_class, len(class_indices))
        permutation = torch.randperm(len(class_indices), generator=generator)
        selected_indices.extend(
            class_indices[int(position)]
            for position in permutation[:sample_size].tolist()
        )

    if selected_indices:
        shuffle_order = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = [
            selected_indices[int(position)] for position in shuffle_order.tolist()
        ]
    return hf_dataset.select(selected_indices)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    image_grid_thw = batch.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    return pixel_values, labels, image_grid_thw


def _forward_classifier(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel_values, labels, image_grid_thw = _move_batch_to_device(batch, device)
    outputs = model(
        pixel_values=pixel_values,
        labels=labels,
        image_grid_thw=image_grid_thw,
    )
    return outputs.loss, outputs.logits, labels


def _build_loader(
    hf_dataset: Any,
    *,
    image_processor: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        MiniImageNetRawDataset(hf_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=QwenVisionCollator(image_processor),
    )


def train_vlm_vision_on_mini_imagenet(
    config: VLMVisionFineTuneConfig,
) -> dict[str, float | str | int]:
    """Fine-tune Qwen's vision module plus a small classification head."""
    if config.eval_every <= 0:
        raise ValueError("eval_every deve essere maggiore di 0")

    set_seed(config.seed)
    device = get_device()
    print(f"Device: {device}")

    if config.preprocessed_dataset_dir:
        print(f"Caricamento dataset preprocessato: {config.preprocessed_dataset_dir}")
        dataset = load_preprocessed_mini_imagenet(config.preprocessed_dataset_dir)
    else:
        print(f"Caricamento dataset: {config.dataset_id}")
        dataset = load_dataset(config.dataset_id)
    if config.subset_per_class > 0:
        print(
            "Subset stratificato attivo sul train: "
            f"{config.subset_per_class} campioni per classe."
        )
        dataset = dataset.copy()
        dataset["train"] = _apply_stratified_subset(
            dataset["train"],
            per_class=config.subset_per_class,
            seed=config.seed,
        )
    if config.eval_subset_per_class > 0:
        print(
            "Subset stratificato attivo su validation/test: "
            f"{config.eval_subset_per_class} campioni per classe."
        )
        dataset = dataset.copy()
        dataset["validation"] = _apply_stratified_subset(
            dataset["validation"],
            per_class=config.eval_subset_per_class,
            seed=config.seed,
        )
        if "test" in dataset:
            dataset["test"] = _apply_stratified_subset(
                dataset["test"],
                per_class=config.eval_subset_per_class,
                seed=config.seed,
            )

    id2label, label2id = build_label_maps(dataset)
    print(f"Classi: {len(id2label)}")
    print(f"Caricamento VLM: {config.vlm_model_id}")

    model, processor, metadata = build_vlm_vision_classifier(
        config.vlm_model_id,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        pooling=config.pooling,
        dropout_prob=0.0,
        vision_attr_path=config.vision_attr_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=config.torch_dtype,
    )
    image_processor = resolve_image_processor(processor)
    validate_image_processor(image_processor)

    model.to(device)
    for parameter in model.vision_backbone.parameters():
        parameter.requires_grad = False
    print("Backbone Qwen congelata: alleno solo pooling/head di classificazione.")

    trainable, total = count_parameters(model)
    print(f"Vision backbone: {metadata.vision_attr_path}")
    print(f"Pooling: {metadata.pooling}")
    print(f"Parametri: {trainable:,} trainable / {total:,} totali")

    train_loader = _build_loader(
        dataset["train"],
        image_processor=image_processor,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val_loader = _build_loader(
        dataset["validation"],
        image_processor=image_processor,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    test_loader = (
        _build_loader(
            dataset["test"],
            image_processor=image_processor,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )
        if "test" in dataset
        else None
    )

    print(
        "Dimensioni dataset - "
        f"train: {len(dataset['train'])}, "
        f"val: {len(dataset['validation'])}, "
        f"test: {len(dataset['test']) if 'test' in dataset else 0}"
    )

    wandb_run = init_wandb_run(
        enabled=config.use_wandb,
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=config,
        extra_config={
            "num_labels": len(id2label),
            "train_size": len(dataset["train"]),
            "validation_size": len(dataset["validation"]),
            "test_size": len(dataset["test"]) if "test" in dataset else 0,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "resolved_vision_attr_path": metadata.vision_attr_path,
            "resolved_pooling": metadata.pooling,
        },
        job_type="finetune-vlm-vit",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = create_cosine_warmup_scheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = 0.0
    best_epoch = 0
    last_test_metrics: dict[str, float] | None = None
    last_test_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy, progress = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            _forward_classifier,
            epoch=epoch,
            total_epochs=config.epochs,
            log_every=DEFAULT_LOG_EVERY_STEPS,
            max_grad_norm=DEFAULT_GRAD_CLIP_NORM,
            on_log=lambda metrics, epoch=epoch: log_wandb_metrics(
                wandb_run,
                {
                    "train/loss_running": metrics["train_loss_running"],
                    "train/learning_rate": metrics["learning_rate"],
                    "train/epoch": metrics["epoch"],
                },
                step=(epoch - 1) * len(train_loader) + int(metrics["train_step"]),
            ),
        )

        eval_loss = None
        eval_accuracy = None
        if epoch % config.eval_every == 0:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                _forward_classifier,
                show_progress=True,
                desc=f"{epoch}/{config.epochs} eval",
                colour="blue",
            )
            eval_loss = val_metrics["loss"]
            eval_accuracy = val_metrics["accuracy"]

        if progress is not None:
            postfix: dict[str, str] = {"train_loss": f"{train_loss:.4f}"}
            if eval_loss is not None:
                postfix["eval_loss"] = f"{eval_loss:.4f}"
            progress.set_postfix(postfix)
            progress.close()

        tqdm.write(
            f"Epoch {epoch}/{config.epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_accuracy:.4f}"
        )
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
        }
        if eval_loss is not None and eval_accuracy is not None:
            tqdm.write(f"  Val Loss: {eval_loss:.4f} - Val Acc: {eval_accuracy:.4f}")
            epoch_metrics["validation/loss"] = eval_loss
            epoch_metrics["validation/accuracy"] = eval_accuracy
            if eval_accuracy > best_val_accuracy:
                best_val_accuracy = eval_accuracy
                best_epoch = epoch
                save_path = output_dir / "best"
                save_prepared_vlm_classifier(model, processor, metadata, save_path)
                print(
                    "  Nuovo miglior modello salvato in "
                    f"{save_path} (acc={eval_accuracy:.4f})"
                )

        log_wandb_metrics(wandb_run, epoch_metrics, step=epoch * len(train_loader))

        if (
            config.test_every > 0
            and test_loader is not None
            and epoch % config.test_every == 0
        ):
            last_test_metrics = evaluate_model(
                model,
                test_loader,
                device,
                _forward_classifier,
                show_progress=True,
                desc=f"{epoch}/{config.epochs} test",
                colour="magenta",
            )
            last_test_epoch = epoch
            tqdm.write(
                "  Test Loss: "
                f"{last_test_metrics['loss']:.4f} "
                f"- Test Acc: {last_test_metrics['accuracy']:.4f}"
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "test/loss": last_test_metrics["loss"],
                    "test/accuracy": last_test_metrics["accuracy"],
                    "test/epoch": epoch,
                },
                step=epoch * len(train_loader),
            )

    final_path = output_dir / "final"
    save_prepared_vlm_classifier(model, processor, metadata, final_path)
    print(f"\nModello finale salvato in {final_path}")
    print(f"Miglior validation accuracy: {best_val_accuracy:.4f}")

    summary: dict[str, float | str | int] = {
        "best_validation_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "output_dir": str(output_dir),
    }

    if test_loader is not None:
        if last_test_metrics is None or last_test_epoch != config.epochs:
            last_test_metrics = evaluate_model(
                model,
                test_loader,
                device,
                _forward_classifier,
                show_progress=True,
                desc=f"{config.epochs}/{config.epochs} test",
                colour="magenta",
            )
        print(
            "Test Loss: "
            f"{last_test_metrics['loss']:.4f} "
            f"- Test Acc: {last_test_metrics['accuracy']:.4f}"
        )
        summary["test_accuracy"] = last_test_metrics["accuracy"]
        log_wandb_metrics(
            wandb_run,
            {
                "test/final_loss": last_test_metrics["loss"],
                "test/final_accuracy": last_test_metrics["accuracy"],
            },
            step=config.epochs * len(train_loader),
        )

    finish_wandb_run(wandb_run, summary=summary)
    return summary
