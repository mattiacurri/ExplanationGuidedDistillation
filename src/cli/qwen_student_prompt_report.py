"""Generate a teacher-vs-student Qwen prompt report on Mini-ImageNet images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from timm.data import resolve_data_config
from transformers import AutoProcessor

from src.cli.eval_qwen_student_visual import (
    _generate_from_inputs,
    _load_qwen_model,
    _prepare_inputs,
)
from src.data import load_preprocessed_mini_imagenet
from src.models.qwen_student_visual import (
    QwenVisualPassthrough,
    pil_image_to_student_tensor,
)
from src.training.explainable_vit_distillation import load_student_vit
from src.utils import get_device
from src.utils.torch_utils import parse_torch_dtype


NO_IMAGE_MARKERS = (
    "cannot see any image",
    "can't see any image",
    "haven't provided any image",
    "no image attached",
    "please upload an image",
    "provide an image",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Qwen student prompt report."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera un report JSON teacher-vs-student Qwen su un sottoinsieme "
            "stratificato di Mini-ImageNet."
        )
    )
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--preprocessed-dataset-dir", required=True)
    parser.add_argument("--student-checkpoint-dir", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt", default="Describe the image.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_no_image_answer(text: str) -> bool:
    """Return True when Qwen's answer says it cannot see the image."""
    lowered = text.lower()
    return any(marker in lowered for marker in NO_IMAGE_MARKERS)


def select_indices_by_class(dataset, *, samples_per_class: int, seed: int) -> list[int]:
    """Select a deterministic stratified subset using the repo's torch RNG path."""
    labels = dataset["label"]
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []
    for label in sorted(by_class):
        candidates = by_class[label]
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected.extend(candidates[int(pos)] for pos in order[:samples_per_class])
    return selected


def main() -> None:
    """Generate teacher/student answers and write the report JSON."""
    args = parse_args()
    dtype = parse_torch_dtype(args.torch_dtype)
    device = get_device()

    datasets = load_preprocessed_mini_imagenet(
        args.preprocessed_dataset_dir,
        splits=(args.split,),
    )
    dataset = datasets[args.split]
    selected_indices = select_indices_by_class(
        dataset,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    processor = AutoProcessor.from_pretrained(args.qwen_model)
    teacher = (
        _load_qwen_model(
            args.qwen_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    student_qwen = (
        _load_qwen_model(
            args.qwen_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    merge_size = int(student_qwen.model.visual.spatial_merge_size)
    student_qwen.model.visual = QwenVisualPassthrough(
        spatial_merge_size=merge_size,
    ).to(device=device, dtype=next(student_qwen.parameters()).dtype)

    student, metadata = load_student_vit(args.student_checkpoint_dir)
    student.to(device=device, dtype=next(student_qwen.parameters()).dtype).eval()
    data_config = resolve_data_config(
        getattr(student.vit, "pretrained_cfg", None),
        model=student.vit,
    )
    student_image_size = int(data_config.get("input_size", (3, 224, 224))[-1])
    student_mean = tuple(
        float(x) for x in data_config.get("mean", (0.485, 0.456, 0.406))
    )
    student_std = tuple(float(x) for x in data_config.get("std", (0.229, 0.224, 0.225)))

    results = []
    for row, dataset_idx in enumerate(selected_indices, start=1):
        item = dataset[dataset_idx]
        image = item["image"]
        label = int(item["label"])
        print(f"[{row}/{len(selected_indices)}] class={label:03d} idx={dataset_idx}")

        with torch.no_grad():
            teacher_inputs = _prepare_inputs(processor, image, args.prompt, device)
            teacher_text = _generate_from_inputs(
                teacher,
                processor,
                teacher_inputs,
                max_new_tokens=args.max_new_tokens,
            )

            student_inputs = _prepare_inputs(processor, image, args.prompt, device)
            student_tensor = pil_image_to_student_tensor(
                image,
                image_size=student_image_size,
                mean=student_mean,
                std=student_std,
                device=device,
                dtype=next(student_qwen.parameters()).dtype,
            )
            student_inputs["pixel_values"] = student(student_tensor).squeeze(0)
            student_text = _generate_from_inputs(
                student_qwen,
                processor,
                student_inputs,
                max_new_tokens=args.max_new_tokens,
            )

        results.append(
            {
                "split": args.split,
                "dataset_index": int(dataset_idx),
                "source_index": int(item.get("source_index", dataset_idx)),
                "label": label,
                "teacher": teacher_text,
                "student": student_text,
                "outputs": {
                    "teacher": teacher_text,
                    "student": student_text,
                },
                "student_no_image": is_no_image_answer(student_text),
            }
        )

    per_class = {}
    for item in results:
        bucket = per_class.setdefault(
            str(item["label"]),
            {"total": 0, "student_no_image": 0},
        )
        bucket["total"] += 1
        bucket["student_no_image"] += int(item["student_no_image"])

    report = {
        "qwen_model": args.qwen_model,
        "preprocessed_dataset_dir": args.preprocessed_dataset_dir,
        "student_checkpoint_dir": args.student_checkpoint_dir,
        "student_metadata": metadata,
        "split": args.split,
        "prompt": args.prompt,
        "samples_per_class": args.samples_per_class,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "num_images": len(results),
        "student_no_image_count": sum(
            int(item["student_no_image"]) for item in results
        ),
        "per_class": per_class,
        "results": results,
    }
    output_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Report salvato in {output_path}")
    print(
        f"Student no-image: {report['student_no_image_count']}/{report['num_images']}"
    )


if __name__ == "__main__":
    main()
