"""Add negative visual baselines to an existing Qwen generation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from timm.data import resolve_data_config
from transformers import AutoProcessor

from src.cli.eval_qwen_student_visual import (
    _generate_from_inputs,
    _load_qwen_model,
    _prepare_inputs,
)
from src.cli.qwen_student_prompt_report import is_no_image_answer
from src.data import load_preprocessed_mini_imagenet
from src.models.qwen_student_visual import (
    QwenVisualPassthrough,
    pil_image_to_student_tensor,
)
from src.training.explainable_vit_distillation import load_student_vit
from src.utils import get_device
from src.utils.torch_utils import parse_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggiunge baseline visuali negative a un report Qwen gia' esistente, "
            "senza rigenerare teacher/student."
        )
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--qwen-model", default="")
    parser.add_argument("--preprocessed-dataset-dir", default="")
    parser.add_argument("--student-checkpoint-dir", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--results-key", default="results")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=("no_image", "mismatched"),
        default=["no_image", "mismatched"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rigenera anche baseline gia' presenti nel report.",
    )
    parser.add_argument("--mismatch-offset", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--torch-dtype", default="")
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Il report deve essere un oggetto JSON")
    return payload


def _get_required_str(
    *,
    cli_value: str,
    report: dict[str, Any],
    report_key: str,
    label: str,
) -> str:
    value = cli_value or str(report.get(report_key, "") or "")
    if not value:
        raise ValueError(f"{label} mancante: passalo via CLI o salvalo nel report")
    return value


def _result_dataset_index(item: dict[str, Any]) -> int:
    if "dataset_index" not in item:
        raise ValueError("Una riga report non contiene dataset_index")
    return int(item["dataset_index"])


def _ensure_outputs(item: dict[str, Any]) -> dict[str, Any]:
    outputs = item.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
        for key in ("teacher", "student"):
            if isinstance(item.get(key), str):
                outputs[key] = item[key]
        item["outputs"] = outputs
    return outputs


def _pick_mismatched_index(
    row_pos: int,
    rows: list[dict[str, Any]],
    *,
    offset: int,
) -> int:
    if len(rows) < 2:
        raise ValueError("La baseline mismatched richiede almeno due righe")
    offset = offset % len(rows)
    if offset == 0:
        offset = 1
    other_pos = (row_pos + offset) % len(rows)
    return _result_dataset_index(rows[other_pos])


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path)
    report = _load_report(report_path)
    results = report.get(args.results_key)
    if not isinstance(results, list):
        raise ValueError(f"Campo report non valido: {args.results_key!r}")
    rows = [item for item in results if isinstance(item, dict)]
    if not rows:
        raise ValueError("Nessuna riga valida nel report")

    qwen_model = _get_required_str(
        cli_value=args.qwen_model,
        report=report,
        report_key="qwen_model",
        label="qwen model",
    )
    dataset_dir = _get_required_str(
        cli_value=args.preprocessed_dataset_dir,
        report=report,
        report_key="preprocessed_dataset_dir",
        label="dataset preprocessato",
    )
    student_dir = _get_required_str(
        cli_value=args.student_checkpoint_dir,
        report=report,
        report_key="student_checkpoint_dir",
        label="checkpoint student",
    )
    split = args.split or str(report.get("split", "test"))
    prompt = args.prompt or str(report.get("prompt", "Describe the image."))
    max_new_tokens = args.max_new_tokens or int(report.get("max_new_tokens", 128))
    torch_dtype = args.torch_dtype or str(report.get("torch_dtype", "float16"))

    dtype = parse_torch_dtype(torch_dtype)
    device = get_device()
    datasets = load_preprocessed_mini_imagenet(dataset_dir, splits=(split,))
    dataset = datasets[split]

    processor = AutoProcessor.from_pretrained(qwen_model)
    qwen = (
        _load_qwen_model(
            qwen_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    merge_size = int(qwen.model.visual.spatial_merge_size)
    qwen.model.visual = QwenVisualPassthrough(spatial_merge_size=merge_size).to(
        device=device,
        dtype=next(qwen.parameters()).dtype,
    )

    student, metadata = load_student_vit(student_dir)
    student.to(device=device, dtype=next(qwen.parameters()).dtype).eval()
    data_config = resolve_data_config(
        getattr(student.vit, "pretrained_cfg", None),
        model=student.vit,
    )
    student_image_size = int(data_config.get("input_size", (3, 224, 224))[-1])
    student_mean = tuple(
        float(x) for x in data_config.get("mean", (0.485, 0.456, 0.406))
    )
    student_std = tuple(float(x) for x in data_config.get("std", (0.229, 0.224, 0.225)))

    added = {"no_image": 0, "mismatched": 0}
    for pos, item in enumerate(rows):
        dataset_idx = _result_dataset_index(item)
        dataset_item = dataset[dataset_idx]
        image = dataset_item["image"]
        outputs = _ensure_outputs(item)
        baselines = item.setdefault("baselines", {})
        if not isinstance(baselines, dict):
            raise ValueError("Campo item.baselines non valido")

        with torch.no_grad():
            inputs = _prepare_inputs(processor, image, prompt, device)
            if "no_image" in args.baselines and (
                args.force or "no_image" not in baselines
            ):
                no_image_inputs = dict(inputs)
                token_count = int(
                    student(
                        pil_image_to_student_tensor(
                            image,
                            image_size=student_image_size,
                            mean=student_mean,
                            std=student_std,
                            device=device,
                            dtype=next(qwen.parameters()).dtype,
                        )
                    ).shape[1]
                )
                hidden_size = int(metadata["teacher_hidden_size"])
                no_image_inputs["pixel_values"] = torch.zeros(
                    (token_count, hidden_size),
                    device=device,
                    dtype=next(qwen.parameters()).dtype,
                )
                text = _generate_from_inputs(
                    qwen,
                    processor,
                    no_image_inputs,
                    max_new_tokens=max_new_tokens,
                )
                baselines["no_image"] = text
                outputs["baseline_no_image"] = text
                item["baseline_no_image"] = text
                item["baseline_no_image_no_image"] = is_no_image_answer(text)
                added["no_image"] += 1

            if "mismatched" in args.baselines and (
                args.force or "mismatched" not in baselines
            ):
                other_idx = _pick_mismatched_index(
                    pos,
                    rows,
                    offset=args.mismatch_offset,
                )
                other_image = dataset[other_idx]["image"]
                other_tensor = pil_image_to_student_tensor(
                    other_image,
                    image_size=student_image_size,
                    mean=student_mean,
                    std=student_std,
                    device=device,
                    dtype=next(qwen.parameters()).dtype,
                )
                mismatch_inputs = dict(inputs)
                mismatch_inputs["pixel_values"] = student(other_tensor).squeeze(0)
                text = _generate_from_inputs(
                    qwen,
                    processor,
                    mismatch_inputs,
                    max_new_tokens=max_new_tokens,
                )
                baselines["mismatched"] = text
                outputs["baseline_mismatched"] = text
                item["baseline_mismatched"] = text
                item["baseline_mismatched_source_index"] = int(other_idx)
                added["mismatched"] += 1

    report["preprocessed_dataset_dir"] = dataset_dir
    report["prompt"] = prompt
    report["max_new_tokens"] = max_new_tokens
    report["torch_dtype"] = torch_dtype
    report.setdefault("evaluation_extensions", {})
    report["evaluation_extensions"]["negative_baselines"] = {
        "baselines": args.baselines,
        "mismatch_offset": args.mismatch_offset,
        "added": added,
    }

    output_path = (
        report_path
        if args.in_place
        else (Path(args.output_report) if args.output_report else None)
    )
    if output_path is None:
        print(
            json.dumps(report["evaluation_extensions"]["negative_baselines"], indent=2)
        )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Report aggiornato: {output_path}")
    print(f"Baseline aggiunte: {added}")


if __name__ == "__main__":
    main()
