"""Aggregate multi-judge blind preferences in a Qwen generation report."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BLIND_CLASSES = ("student_win", "teacher_win", "tie")
INVALID_RATIONALE_RE = re.compile(
    r"did not return|requested structured|no rationale|invalid|error|exception|traceback",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcola majority vote, agreement, Fleiss' kappa e confusion matrix "
            "judge-vs-judge dai blind judgments salvati in report.llm_judges."
        )
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--results-key", default="results")
    parser.add_argument(
        "--namespace",
        choices=("llm_judges", "vlm_judges"),
        default="llm_judges",
        help="Namespace report/item da aggregare.",
    )
    parser.add_argument(
        "--mode",
        default="blind",
        help="Modo da aggregare dentro ogni judge, es. blind o vision_blind.",
    )
    parser.add_argument(
        "--judges",
        nargs="*",
        default=[],
        help="Judge ids da usare. Default: tutti i judge con blind disponibile.",
    )
    parser.add_argument(
        "--legacy-judge-id",
        default="qwen3_8b",
        help=(
            "Judge id da usare per migrare i campi legacy llm_blind_preference "
            "quando il vecchio Qwen judge e' gia' nel report."
        ),
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help=(
            "Aggrega solo righe in cui tutti i judge selezionati hanno output "
            "strutturato valido e senza fallback parser."
        ),
    )
    parser.add_argument(
        "--output-key-suffix",
        default="",
        help="Suffisso opzionale da aggiungere alla chiave aggregata nel report.",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Il report deve essere un oggetto JSON")
    return payload


def _available_judges(
    report: dict[str, Any],
    *,
    namespace: str,
    mode: str,
) -> list[str]:
    judges = report.get(namespace, {})
    if not isinstance(judges, dict):
        return []
    return sorted(
        judge_id
        for judge_id, payload in judges.items()
        if isinstance(payload, dict) and isinstance(payload.get(mode), dict)
    )


def _backfill_legacy_blind_judge(
    report: dict[str, Any],
    *,
    results: list[Any],
    judge_id: str,
) -> bool:
    if "llm_blind_preference" not in report:
        return False
    judges = report.setdefault("llm_judges", {})
    if not isinstance(judges, dict):
        raise ValueError("Campo report.llm_judges non valido")
    judge_payload = judges.setdefault(judge_id, {})
    if not isinstance(judge_payload, dict):
        raise ValueError(f"Campo report.llm_judges.{judge_id} non valido")
    judge_payload.setdefault(
        "blind", {**report["llm_blind_preference"], "judge_id": judge_id}
    )

    migrated = False
    for item in results:
        if not isinstance(item, dict):
            continue
        legacy = item.get("llm_blind_preference")
        if not isinstance(legacy, dict):
            continue
        item_judges = item.setdefault("llm_judges", {})
        if not isinstance(item_judges, dict):
            raise ValueError("Campo item.llm_judges non valido")
        item_judge = item_judges.setdefault(judge_id, {})
        if not isinstance(item_judge, dict):
            raise ValueError(f"Campo item.llm_judges.{judge_id} non valido")
        item_judge.setdefault("blind", legacy)
        migrated = True
    return migrated


def _winner_for(
    item: dict[str, Any],
    judge_id: str,
    *,
    namespace: str,
    mode: str,
) -> str | None:
    judges = item.get(namespace)
    if not isinstance(judges, dict):
        return None
    judge_payload = judges.get(judge_id)
    if not isinstance(judge_payload, dict):
        return None
    blind = judge_payload.get(mode)
    if not isinstance(blind, dict):
        return None
    winner = str(blind.get("winner", "")).lower()
    if winner in BLIND_CLASSES:
        return winner
    return None


def _valid_payload_for(
    item: dict[str, Any],
    judge_id: str,
    *,
    namespace: str,
    mode: str,
) -> bool:
    judges = item.get(namespace)
    if not isinstance(judges, dict):
        return False
    judge_payload = judges.get(judge_id)
    if not isinstance(judge_payload, dict):
        return False
    blind = judge_payload.get(mode)
    if not isinstance(blind, dict):
        return False
    preference = str(blind.get("preference", "")).strip()
    if preference not in {"A", "B", "tie"}:
        return False
    winner = str(blind.get("winner", "")).lower()
    if winner not in BLIND_CLASSES:
        return False
    rationale = str(blind.get("rationale", "") or "")
    if INVALID_RATIONALE_RE.search(rationale):
        return False
    return True


def _majority_vote(votes: dict[str, str]) -> str:
    counts = Counter(votes.values())
    if not counts:
        return "invalid"
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "no_majority"
    return most_common[0][0]


def _fleiss_kappa(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "kappa": 0.0,
            "num_items": 0,
            "num_judges": 0,
            "p_bar": 0.0,
            "p_e": 0.0,
        }
    num_judges = len(next(iter(rows)).values())
    if num_judges < 2:
        return {
            "kappa": 0.0,
            "num_items": len(rows),
            "num_judges": num_judges,
            "p_bar": 0.0,
            "p_e": 0.0,
        }

    category_totals = Counter()
    p_i_values: list[float] = []
    for votes in rows:
        counts = Counter(votes.values())
        category_totals.update(counts)
        p_i = (sum(count * count for count in counts.values()) - num_judges) / (
            num_judges * (num_judges - 1)
        )
        p_i_values.append(float(p_i))

    total_ratings = len(rows) * num_judges
    p_j = {
        category: category_totals[category] / total_ratings
        for category in BLIND_CLASSES
    }
    p_bar = sum(p_i_values) / len(p_i_values)
    p_e = sum(value * value for value in p_j.values())
    kappa = 0.0 if p_e >= 1.0 else (p_bar - p_e) / (1.0 - p_e)
    return {
        "kappa": float(kappa),
        "num_items": len(rows),
        "num_judges": num_judges,
        "p_bar": float(p_bar),
        "p_e": float(p_e),
        "category_priors": p_j,
    }


def _pairwise_confusion(
    rows: list[dict[str, str]], judges: list[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for judge_a, judge_b in itertools.combinations(judges, 2):
        matrix = {
            class_a: {class_b: 0 for class_b in BLIND_CLASSES}
            for class_a in BLIND_CLASSES
        }
        total = 0
        agree = 0
        for votes in rows:
            vote_a = votes[judge_a]
            vote_b = votes[judge_b]
            matrix[vote_a][vote_b] += 1
            total += 1
            agree += int(vote_a == vote_b)
        output[f"{judge_a}__vs__{judge_b}"] = {
            "judge_a": judge_a,
            "judge_b": judge_b,
            "total": total,
            "agreement": float(agree / total) if total else 0.0,
            "matrix": matrix,
        }
    return output


def _mean_pairwise_agreement(rows: list[dict[str, str]], judges: list[str]) -> float:
    agreements = []
    for judge_a, judge_b in itertools.combinations(judges, 2):
        total = 0
        agree = 0
        for votes in rows:
            total += 1
            agree += int(votes[judge_a] == votes[judge_b])
        if total:
            agreements.append(agree / total)
    return float(sum(agreements) / len(agreements)) if agreements else 0.0


def main() -> None:
    args = parse_args()
    report_path = Path(args.report_path)
    report = _load_report(report_path)
    results = report.get(args.results_key)
    if not isinstance(results, list):
        raise ValueError(f"Campo report non valido: {args.results_key!r}")

    migrated_legacy = _backfill_legacy_blind_judge(
        report,
        results=results,
        judge_id=args.legacy_judge_id,
    )
    judges = args.judges or _available_judges(
        report,
        namespace=args.namespace,
        mode=args.mode,
    )
    if len(judges) < 2:
        raise ValueError(
            "Servono almeno due judge con blind judgments sotto report.llm_judges"
        )

    complete_rows: list[dict[str, str]] = []
    judged_indices: list[int] = []
    majority_counts = Counter()
    all_agree = 0
    for row_idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        votes = {
            judge_id: _winner_for(
                item,
                judge_id,
                namespace=args.namespace,
                mode=args.mode,
            )
            for judge_id in judges
        }
        if any(value is None for value in votes.values()):
            continue
        if args.valid_only and not all(
            _valid_payload_for(
                item,
                judge_id,
                namespace=args.namespace,
                mode=args.mode,
            )
            for judge_id in judges
        ):
            continue
        typed_votes = {key: str(value) for key, value in votes.items()}
        complete_rows.append(typed_votes)
        judged_indices.append(row_idx)
        majority = _majority_vote(typed_votes)
        majority_counts[majority] += 1
        item["multi_judge_blind"] = {
            "judges": typed_votes,
            "majority": majority,
        }
        all_agree += int(len(set(typed_votes.values())) == 1)

    if not complete_rows:
        raise ValueError(
            "Nessuna riga contiene blind judgments completi per i judge scelti"
        )

    output_key = (
        "multi_judge_blind"
        if args.namespace == "llm_judges" and args.mode == "blind"
        else f"multi_judge_{args.mode}"
    )
    if args.output_key_suffix:
        output_key = f"{output_key}_{args.output_key_suffix}"
    report[output_key] = {
        "judges": judges,
        "namespace": args.namespace,
        "mode": args.mode,
        "valid_only": bool(args.valid_only),
        "classes": BLIND_CLASSES,
        "num_items": len(complete_rows),
        "complete_row_indices": judged_indices,
        "majority_counts": dict(majority_counts),
        "all_judges_agreement": float(all_agree / len(complete_rows)),
        "mean_pairwise_agreement": _mean_pairwise_agreement(complete_rows, judges),
        "fleiss": _fleiss_kappa(complete_rows),
        "pairwise_confusion": _pairwise_confusion(complete_rows, judges),
        "migrated_legacy_judge": migrated_legacy,
    }

    summary = report[output_key]
    print("[Multi-judge blind]")
    print(f"Judges: {', '.join(judges)}")
    print(f"Items: {summary['num_items']}")
    print(f"Majority: {summary['majority_counts']}")
    print(f"All-judge agreement: {summary['all_judges_agreement']:.3f}")
    print(f"Mean pairwise agreement: {summary['mean_pairwise_agreement']:.3f}")
    print(f"Fleiss kappa: {summary['fleiss']['kappa']:.3f}")

    output_path = (
        report_path
        if args.in_place
        else (Path(args.output_report) if args.output_report else None)
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report aggiornato: {output_path}")


if __name__ == "__main__":
    main()
