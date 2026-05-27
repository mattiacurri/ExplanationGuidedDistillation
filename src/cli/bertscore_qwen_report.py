"""Compute BERTScore from an existing Qwen teacher-vs-student report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.metrics import compute_bertscore


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for report-level BERTScore evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Calcola BERTScore da un report JSON gia' generato, senza "
            "rigenerare le risposte Qwen."
        )
    )
    parser.add_argument("--report-path", required=True, help="Report JSON di input.")
    parser.add_argument(
        "--output-report",
        default="",
        help="Path per salvare il report arricchito. Se vuoto stampa solo le metriche.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Aggiorna direttamente --report-path con i campi BERTScore.",
    )
    parser.add_argument(
        "--candidate-key",
        default="student",
        help="Campo candidate dentro ogni item results[].",
    )
    parser.add_argument(
        "--reference-key",
        default="teacher",
        help="Campo reference dentro ogni item results[].",
    )
    parser.add_argument(
        "--results-key",
        default="results",
        help="Campo del report che contiene la lista degli esempi.",
    )
    parser.add_argument(
        "--exclude-no-image",
        action="store_true",
        help="Esclude righe con student_no_image=true.",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--model-type",
        default="",
        help="Encoder BERTScore esplicito, es. microsoft/deberta-xlarge-mnli.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        default="",
        help="Device BERTScore opzionale, es. cuda o cpu.",
    )
    parser.add_argument("--rescale-with-baseline", action="store_true")
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Il report deve essere un oggetto JSON")
    return payload


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _select_pairs(
    results: list[Any],
    *,
    candidate_key: str,
    reference_key: str,
    exclude_no_image: bool,
) -> tuple[list[int], list[str], list[str]]:
    row_indices: list[int] = []
    candidates: list[str] = []
    references: list[str] = []

    for idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        if exclude_no_image and bool(item.get("student_no_image")):
            continue
        candidate = item.get(candidate_key)
        reference = item.get(reference_key)
        if not isinstance(candidate, str) or not isinstance(reference, str):
            continue
        row_indices.append(idx)
        candidates.append(candidate)
        references.append(reference)

    return row_indices, candidates, references


def _per_class_metrics(
    results: list[Any],
    row_indices: list[int],
    f1: list[float],
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[float]] = {}
    for row_idx, score in zip(row_indices, f1, strict=True):
        item = results[row_idx]
        label = (
            str(item.get("label", "unknown")) if isinstance(item, dict) else "unknown"
        )
        buckets.setdefault(label, []).append(score)

    return {
        label: {"count": len(scores), "f1_mean": _mean(scores)}
        for label, scores in sorted(buckets.items())
    }


def main() -> None:
    """Compute and optionally persist BERTScore metrics for a report."""
    args = parse_args()
    report_path = Path(args.report_path)
    report = _load_report(report_path)

    results = report.get(args.results_key)
    if not isinstance(results, list):
        raise ValueError(f"Campo report non valido: {args.results_key!r}")

    row_indices, candidates, references = _select_pairs(
        results,
        candidate_key=args.candidate_key,
        reference_key=args.reference_key,
        exclude_no_image=args.exclude_no_image,
    )
    if not candidates:
        raise ValueError("Nessuna coppia candidate/reference valida trovata nel report")

    scores = compute_bertscore(
        candidates,
        references,
        lang=args.lang,
        model_type=args.model_type or None,
        batch_size=args.batch_size,
        device=args.device or None,
        rescale_with_baseline=args.rescale_with_baseline,
    )
    metrics = {
        "candidate_key": args.candidate_key,
        "reference_key": args.reference_key,
        "num_pairs": len(candidates),
        "excluded_no_image": bool(args.exclude_no_image),
        "lang": args.lang,
        "model_type": args.model_type or None,
        "rescale_with_baseline": bool(args.rescale_with_baseline),
        "precision_mean": scores.precision_mean,
        "recall_mean": scores.recall_mean,
        "f1_mean": scores.f1_mean,
        "per_class": _per_class_metrics(results, row_indices, scores.f1),
    }

    for score_idx, row_idx in enumerate(row_indices):
        item = results[row_idx]
        if not isinstance(item, dict):
            continue
        item["bertscore"] = {
            "precision": scores.precision[score_idx],
            "recall": scores.recall[score_idx],
            "f1": scores.f1[score_idx],
        }

    report["bertscore"] = metrics

    print("[BERTScore report]")
    print(f"Pairs: {metrics['num_pairs']}")
    print(f"Precision: {metrics['precision_mean']:.4f}")
    print(f"Recall: {metrics['recall_mean']:.4f}")
    print(f"F1: {metrics['f1_mean']:.4f}")

    output_path = None
    if args.in_place:
        output_path = report_path
    elif args.output_report:
        output_path = Path(args.output_report)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report aggiornato: {output_path}")


if __name__ == "__main__":
    main()
