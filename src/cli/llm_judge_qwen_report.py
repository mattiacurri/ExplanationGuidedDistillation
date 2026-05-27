"""Judge Qwen teacher-vs-student report answers with an LLM backend."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from src.judging import JudgeExample, PreferenceExample


DEFAULT_HF_JUDGE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_OPENROUTER_JUDGE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
DEFAULT_OPENCODE_JUDGE_MODEL = "opencode/qwen3.6-plus-free"
DEFAULT_LMSTUDIO_JUDGE_MODEL = "zai-org/glm-4.6v-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggiunge LLM-as-a-judge a un report Qwen teacher-vs-student."
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("hf", "gemini", "openrouter", "opencode", "lmstudio"),
        default="hf",
    )
    parser.add_argument(
        "--judge-id",
        default="",
        help=(
            "Nome stabile con cui salvare questo judge sotto llm_judges. "
            "Default: derivato da backend e modello."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ricalcola anche i judgment gia' presenti per questo judge/mode.",
    )
    parser.add_argument("--exclude-no-image", action="store_true")
    parser.add_argument(
        "--student-key",
        "--candidate-key",
        dest="student_key",
        default="student",
        help="Campo JSON che contiene la risposta dello student.",
    )
    parser.add_argument(
        "--teacher-key",
        "--reference-key",
        dest="teacher_key",
        default="teacher",
        help="Campo JSON che contiene la risposta del teacher.",
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--results-key", default="results")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("direct", "reverse", "blind"),
        default=["direct"],
        help=(
            "Judge modes to run. direct scores student vs teacher; reverse "
            "scores teacher vs student; blind asks A/B/tie with hidden roles."
        ),
    )
    parser.add_argument(
        "--blind-seed",
        type=int,
        default=42,
        help="Seed for deterministic A/B shuffling in blind preference mode.",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_HF_JUDGE_MODEL,
        help="Modello locale Hugging Face usato dal backend hf.",
    )
    parser.add_argument("--hf-device", default="auto")
    parser.add_argument("--hf-torch-dtype", default="auto")
    parser.add_argument(
        "--hf-quantization",
        choices=("bnb4", "none"),
        default="bnb4",
        help="Quantizzazione HF locale. Default: bitsandbytes 4-bit NF4.",
    )
    parser.add_argument("--hf-max-new-tokens", type=int, default=160)
    parser.add_argument("--hf-trust-remote-code", action="store_true")
    parser.add_argument(
        "--gemini-model",
        default="gemini-3.1-flash-lite",
        help="Modello Gemini piccolo/economico usato dal backend gemini.",
    )
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--gemini-usage-log",
        default="",
        help="File JSONL separato in cui salvare usage_metadata delle chiamate Gemini.",
    )
    parser.add_argument(
        "--openrouter-model",
        default=DEFAULT_OPENROUTER_JUDGE_MODEL,
        help="Modello OpenRouter usato dal backend openrouter.",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="API key OpenRouter. Default: env OPENROUTER_API_KEY.",
    )
    parser.add_argument("--openrouter-temperature", type=float, default=0.0)
    parser.add_argument("--openrouter-max-tokens", type=int, default=512)
    parser.add_argument("--openrouter-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--openrouter-site-url",
        default=None,
        help="Valore opzionale per header HTTP-Referer.",
    )
    parser.add_argument(
        "--openrouter-app-name",
        default="CompVis evaluation",
        help="Valore opzionale per header X-Title.",
    )
    parser.add_argument(
        "--opencode-model",
        default=DEFAULT_OPENCODE_JUDGE_MODEL,
        help="Modello OpenCode nel formato provider/model.",
    )
    parser.add_argument(
        "--opencode-timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout per ogni chiamata opencode run.",
    )
    parser.add_argument("--lmstudio-model", default=DEFAULT_LMSTUDIO_JUDGE_MODEL)
    parser.add_argument("--lmstudio-base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--lmstudio-max-tokens", type=int, default=256)
    parser.add_argument("--lmstudio-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--lmstudio-auto-load",
        action="store_true",
        help=(
            "Se server/modello LM Studio non sono live, avvia il server e "
            "carica il modello con lms prima di giudicare."
        ),
    )
    parser.add_argument(
        "--lmstudio-load-model-key",
        default="",
        help=("Model key da passare a 'lms load'. Default: uguale a --lmstudio-model."),
    )
    parser.add_argument(
        "--lmstudio-load-identifier",
        default="",
        help=(
            "Identificatore opzionale da assegnare al modello caricato. "
            "Se usato, deve coincidere con --lmstudio-model."
        ),
    )
    parser.add_argument("--lmstudio-gpu", default="max")
    parser.add_argument("--lmstudio-context-length", type=int, default=32768)
    parser.add_argument("--lmstudio-parallel", type=int, default=1)
    parser.add_argument("--lmstudio-server-port", type=int, default=1234)
    parser.add_argument("--lmstudio-server-bind", default="127.0.0.1")
    parser.add_argument("--lmstudio-load-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "judge"


def _judge_id(args: argparse.Namespace) -> str:
    if args.judge_id.strip():
        return _slugify(args.judge_id)
    if args.backend == "gemini":
        model = args.gemini_model
    elif args.backend == "openrouter":
        model = args.openrouter_model
    elif args.backend == "opencode":
        model = args.opencode_model
    elif args.backend == "lmstudio":
        model = args.lmstudio_model
    else:
        model = args.hf_model
    return _slugify(f"{args.backend}_{model}")


def _model_name(args: argparse.Namespace) -> str:
    if args.backend == "gemini":
        return args.gemini_model
    if args.backend == "openrouter":
        return args.openrouter_model
    if args.backend == "opencode":
        return args.opencode_model
    if args.backend == "lmstudio":
        return args.lmstudio_model
    return args.hf_model


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Il report deve essere un oggetto JSON")
    return payload


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_judge(args: argparse.Namespace):
    if args.backend == "gemini":
        from src.judging import GeminiJudge, GeminiJudgeConfig

        return GeminiJudge(
            GeminiJudgeConfig(
                model=args.gemini_model,
                api_key=args.gemini_api_key,
                max_output_tokens=args.gemini_max_output_tokens,
                usage_log_path=args.gemini_usage_log,
            )
        )
    if args.backend == "openrouter":
        from src.judging import OpenRouterJudge, OpenRouterJudgeConfig

        return OpenRouterJudge(
            OpenRouterJudgeConfig(
                model=args.openrouter_model,
                api_key=args.openrouter_api_key,
                temperature=args.openrouter_temperature,
                max_tokens=args.openrouter_max_tokens,
                timeout_seconds=args.openrouter_timeout_seconds,
                site_url=args.openrouter_site_url,
                app_name=args.openrouter_app_name,
            )
        )
    if args.backend == "opencode":
        from src.judging import OpenCodeJudge, OpenCodeJudgeConfig

        return OpenCodeJudge(
            OpenCodeJudgeConfig(
                model=args.opencode_model,
                cwd=str(Path.cwd()),
                timeout_seconds=args.opencode_timeout_seconds,
            )
        )
    if args.backend == "lmstudio":
        from src.judging.lmstudio_runtime import (
            LMStudioRuntimeConfig,
            ensure_lmstudio_model,
        )
        from src.judging import LMStudioJudge, LMStudioJudgeConfig

        ensure_lmstudio_model(
            LMStudioRuntimeConfig(
                model=args.lmstudio_model,
                base_url=args.lmstudio_base_url,
                auto_load=args.lmstudio_auto_load,
                load_model_key=args.lmstudio_load_model_key,
                load_identifier=args.lmstudio_load_identifier,
                gpu=args.lmstudio_gpu,
                context_length=args.lmstudio_context_length,
                parallel=args.lmstudio_parallel,
                server_port=args.lmstudio_server_port,
                server_bind=args.lmstudio_server_bind,
                timeout_seconds=args.lmstudio_load_timeout_seconds,
                probe_timeout_seconds=min(30.0, args.lmstudio_timeout_seconds),
            )
        )
        return LMStudioJudge(
            LMStudioJudgeConfig(
                model=args.lmstudio_model,
                base_url=args.lmstudio_base_url,
                max_tokens=args.lmstudio_max_tokens,
                timeout_seconds=args.lmstudio_timeout_seconds,
            )
        )
    from src.judging import HuggingFaceJudge, HuggingFaceJudgeConfig

    return HuggingFaceJudge(
        HuggingFaceJudgeConfig(
            model_id=args.hf_model,
            torch_dtype=args.hf_torch_dtype,
            device=args.hf_device,
            load_in_4bit=args.hf_quantization == "bnb4",
            max_new_tokens=args.hf_max_new_tokens,
            trust_remote_code=args.hf_trust_remote_code,
        )
    )


def _iter_report_rows(
    results: list[Any],
    *,
    candidate_key: str,
    reference_key: str,
    exclude_no_image: bool,
    limit: int,
):
    emitted = 0
    for row_idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        if exclude_no_image and bool(item.get("student_no_image")):
            continue
        if not isinstance(item.get(candidate_key), str):
            continue
        if not isinstance(item.get(reference_key), str):
            continue
        yield row_idx, item
        emitted += 1
        if limit > 0 and emitted >= limit:
            return


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _summarize_judge(
    results: list[Any],
    judged_indices: list[int],
    *,
    payload_key: str,
) -> dict[str, Any]:
    scores: list[float] = []
    verdict_counts = {"pass": 0, "partial": 0, "fail": 0}
    per_class_scores: dict[str, list[float]] = {}
    for row_idx in judged_indices:
        item = results[row_idx]
        if not isinstance(item, dict):
            continue
        judge_payload = item.get(payload_key)
        if not isinstance(judge_payload, dict):
            continue
        score = float(judge_payload.get("score", 0.0))
        verdict = str(judge_payload.get("verdict", "")).lower()
        scores.append(score)
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1
        label = str(item.get("label", "unknown"))
        per_class_scores.setdefault(label, []).append(score)

    return {
        "num_judged": len(scores),
        "score_mean": _mean(scores),
        "verdict_counts": verdict_counts,
        "per_class": {
            label: {"count": len(bucket), "score_mean": _mean(bucket)}
            for label, bucket in sorted(per_class_scores.items())
        },
    }
def _summarize_blind_preference_for_judge(
    results: list[Any],
    *,
    judge_id: str,
) -> dict[str, Any]:
    counts = {
        "student_win": 0,
        "teacher_win": 0,
        "tie": 0,
        "invalid": 0,
    }
    per_class_counts: dict[str, dict[str, int]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        judges = item.get("llm_judges")
        if not isinstance(judges, dict):
            continue
        judge_payload = judges.get(judge_id)
        if not isinstance(judge_payload, dict):
            continue
        payload = judge_payload.get("blind")
        if not isinstance(payload, dict):
            continue
        winner = str(payload.get("winner", "")).lower()
        if winner == "candidate_win":
            winner = "student_win"
        elif winner == "reference_win":
            winner = "teacher_win"
        if winner not in counts:
            winner = "invalid"
        counts[winner] += 1
        label = str(item.get("label", "unknown"))
        bucket = per_class_counts.setdefault(
            label,
            {"student_win": 0, "teacher_win": 0, "tie": 0, "invalid": 0},
        )
        bucket[winner] += 1
    total = sum(counts.values())
    rates = {
        key: (float(value / total) if total else 0.0) for key, value in counts.items()
    }
    return {
        "num_judged": total,
        "counts": counts,
        "rates": rates,
        "per_class": {
            label: {
                "count": sum(bucket.values()),
                "counts": bucket,
                "rates": {
                    key: (
                        float(value / sum(bucket.values()))
                        if sum(bucket.values())
                        else 0.0
                    )
                    for key, value in bucket.items()
                },
            }
            for label, bucket in sorted(per_class_counts.items())
        },
    }


def _ensure_item_judge_namespace(item: dict[str, Any], judge_id: str) -> dict[str, Any]:
    judges = item.setdefault("llm_judges", {})
    if not isinstance(judges, dict):
        raise ValueError("Campo item.llm_judges non valido")
    judge_payload = judges.setdefault(judge_id, {})
    if not isinstance(judge_payload, dict):
        raise ValueError(f"Campo item.llm_judges.{judge_id} non valido")
    return judge_payload


def _copy_item_payloads_to_judge_namespace(
    results: list[Any],
    judged_indices: list[int],
    *,
    source_key: str,
    judge_id: str,
    mode: str,
) -> None:
    for row_idx in judged_indices:
        item = results[row_idx]
        if not isinstance(item, dict):
            continue
        payload = item.get(source_key)
        if isinstance(payload, dict):
            _ensure_item_judge_namespace(item, judge_id)[mode] = payload


def _ensure_report_judge_namespace(
    report: dict[str, Any],
    judge_id: str,
) -> dict[str, Any]:
    judges = report.setdefault("llm_judges", {})
    if not isinstance(judges, dict):
        raise ValueError("Campo report.llm_judges non valido")
    judge_payload = judges.setdefault(judge_id, {})
    if not isinstance(judge_payload, dict):
        raise ValueError(f"Campo report.llm_judges.{judge_id} non valido")
    return judge_payload


def _run_direct_judge(
    *,
    judge: Any,
    rows: list[tuple[int, dict[str, Any]]],
    report: dict[str, Any],
    args: argparse.Namespace,
    payload_key: str,
    candidate_key: str,
    reference_key: str,
    label: str,
) -> list[int]:
    judged_indices: list[int] = []
    for pos, (row_idx, item) in enumerate(rows, start=1):
        row_label = item.get("label", "unknown")
        print(f"[{pos}/{len(rows)}] {label} idx={row_idx} class={row_label}")
        result = judge.judge(
            JudgeExample(
                reference_answer=item[reference_key],
                candidate_answer=item[candidate_key],
                prompt=str(item.get(args.prompt_key, report.get("prompt", "")) or ""),
                label=row_label,
                dataset_index=item.get("dataset_index"),
            )
        )
        item[payload_key] = result.to_dict()
        judged_indices.append(row_idx)
    return judged_indices


def _run_blind_preference(
    *,
    judge: Any,
    rows: list[tuple[int, dict[str, Any]]],
    report: dict[str, Any],
    args: argparse.Namespace,
    judge_id: str,
    save_callback: Any | None = None,
) -> list[int]:
    judged_indices: list[int] = []
    for pos, (row_idx, item) in enumerate(rows, start=1):
        row_label = item.get("label", "unknown")
        rng = random.Random(args.blind_seed + row_idx)
        candidate_is_a = bool(rng.getrandbits(1))
        if candidate_is_a:
            answer_a = item[args.student_key]
            answer_b = item[args.teacher_key]
            role_a = "student"
            role_b = "teacher"
        else:
            answer_a = item[args.teacher_key]
            answer_b = item[args.student_key]
            role_a = "teacher"
            role_b = "student"
        print(f"[{pos}/{len(rows)}] blind preference idx={row_idx} class={row_label}")
        result = judge.prefer(
            PreferenceExample(
                answer_a=answer_a,
                answer_b=answer_b,
                prompt=str(item.get(args.prompt_key, report.get("prompt", "")) or ""),
                label=row_label,
                dataset_index=item.get("dataset_index"),
            )
        )
        preferred = result.preference
        if preferred == "A":
            winner = role_a
        elif preferred == "B":
            winner = role_b
        else:
            winner = "tie"
        if winner == "student":
            winner = "student_win"
        elif winner == "teacher":
            winner = "teacher_win"
        payload = result.to_dict()
        payload.update(
            {
                "answer_a_role": role_a,
                "answer_b_role": role_b,
                "winner": winner,
                "student_key": args.student_key,
                "teacher_key": args.teacher_key,
            }
        )
        item["llm_blind_preference"] = payload
        _ensure_item_judge_namespace(item, judge_id)["blind"] = payload
        judged_indices.append(row_idx)
        if save_callback is not None:
            save_callback()
    return judged_indices


def main() -> None:
    args = parse_args()
    judge_id = _judge_id(args)
    report_path = Path(args.report_path)
    report = _load_report(report_path)
    results = report.get(args.results_key)
    if not isinstance(results, list):
        raise ValueError(f"Campo report non valido: {args.results_key!r}")

    judge = _build_judge(args)
    output_path = None
    if args.in_place:
        output_path = report_path
    elif args.output_report:
        output_path = Path(args.output_report)

    def save_progress() -> None:
        if output_path is not None:
            _write_report(output_path, report)

    rows = list(
        _iter_report_rows(
            results,
            candidate_key=args.student_key,
            reference_key=args.teacher_key,
            exclude_no_image=args.exclude_no_image,
            limit=args.limit,
        )
    )
    if args.modes == ["blind"] and not args.force:
        original_count = len(rows)
        rows = [
            (row_idx, item)
            for row_idx, item in rows
            if not (
                isinstance(item.get("llm_judges"), dict)
                and isinstance(item["llm_judges"].get(judge_id), dict)
                and isinstance(item["llm_judges"][judge_id].get("blind"), dict)
            )
        ]
        if len(rows) != original_count:
            print(
                f"Resume: salto {original_count - len(rows)} blind judgments "
                f"gia' presenti per {judge_id}."
            )

    if "direct" in args.modes:
        judged_indices = _run_direct_judge(
            judge=judge,
            rows=rows,
            report=report,
            args=args,
            payload_key="llm_judge",
            candidate_key=args.student_key,
            reference_key=args.teacher_key,
            label="direct judge",
        )
        report["llm_judge"] = {
            "backend": args.backend,
            "model": _model_name(args),
            "mode": "direct",
            "student_key": args.student_key,
            "teacher_key": args.teacher_key,
            "candidate_role": "student",
            "reference_role": "teacher",
            "excluded_no_image": bool(args.exclude_no_image),
            **_summarize_judge(results, judged_indices, payload_key="llm_judge"),
        }
        _copy_item_payloads_to_judge_namespace(
            results,
            judged_indices,
            source_key="llm_judge",
            judge_id=judge_id,
            mode="direct",
        )
        _ensure_report_judge_namespace(report, judge_id)["direct"] = {
            **report["llm_judge"],
            "judge_id": judge_id,
        }

    if "reverse" in args.modes:
        judged_indices = _run_direct_judge(
            judge=judge,
            rows=rows,
            report=report,
            args=args,
            payload_key="llm_judge_reverse",
            candidate_key=args.teacher_key,
            reference_key=args.student_key,
            label="reverse judge",
        )
        report["llm_judge_reverse"] = {
            "backend": args.backend,
            "model": _model_name(args),
            "mode": "reverse",
            "student_key": args.student_key,
            "teacher_key": args.teacher_key,
            "candidate_role": "teacher",
            "reference_role": "student",
            "excluded_no_image": bool(args.exclude_no_image),
            **_summarize_judge(
                results,
                judged_indices,
                payload_key="llm_judge_reverse",
            ),
        }
        _copy_item_payloads_to_judge_namespace(
            results,
            judged_indices,
            source_key="llm_judge_reverse",
            judge_id=judge_id,
            mode="reverse",
        )
        _ensure_report_judge_namespace(report, judge_id)["reverse"] = {
            **report["llm_judge_reverse"],
            "judge_id": judge_id,
        }

    if "blind" in args.modes:
        judged_indices = _run_blind_preference(
            judge=judge,
            rows=rows,
            report=report,
            args=args,
            judge_id=judge_id,
            save_callback=save_progress,
        )
        report["llm_blind_preference"] = {
            "backend": args.backend,
            "model": _model_name(args),
            "mode": "blind",
            "student_key": args.student_key,
            "teacher_key": args.teacher_key,
            "blind_seed": args.blind_seed,
            "excluded_no_image": bool(args.exclude_no_image),
            **_summarize_blind_preference_for_judge(results, judge_id=judge_id),
        }
        _copy_item_payloads_to_judge_namespace(
            results,
            judged_indices,
            source_key="llm_blind_preference",
            judge_id=judge_id,
            mode="blind",
        )
        _ensure_report_judge_namespace(report, judge_id)["blind"] = {
            **report["llm_blind_preference"],
            "judge_id": judge_id,
        }

    print("[LLM judge report]")
    if "llm_judge" in report:
        print(
            "Direct: "
            f"{report['llm_judge']['num_judged']} judged, "
            f"mean={report['llm_judge']['score_mean']:.3f}"
        )
    if "llm_judge_reverse" in report:
        print(
            "Reverse: "
            f"{report['llm_judge_reverse']['num_judged']} judged, "
            f"mean={report['llm_judge_reverse']['score_mean']:.3f}"
        )
    if "llm_blind_preference" in report:
        print(f"Blind preference: {report['llm_blind_preference']['counts']}")

    if output_path is not None:
        _write_report(output_path, report)
        print(f"Report aggiornato: {output_path}")


if __name__ == "__main__":
    main()
