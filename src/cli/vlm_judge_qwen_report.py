"""Judge Qwen teacher-vs-student answers with image-grounded VLM backends."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from src.data import load_preprocessed_mini_imagenet
from src.judging import PreferenceExample


DEFAULT_GEMINI_VISION_JUDGE_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENCODE_VISION_JUDGE_MODEL = "opencode/qwen3.6-plus-free"
DEFAULT_HF_VISION_JUDGE_MODEL = "zai-org/GLM-4.6V-Flash"
DEFAULT_LMSTUDIO_VISION_JUDGE_MODEL = "zai-org/glm-4.6v-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggiunge VLM-as-a-judge image-grounded a un report Qwen."
    )
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--output-report", default="")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("gemini", "opencode", "hf_vision", "lmstudio"),
        required=True,
    )
    parser.add_argument("--judge-id", required=True)
    parser.add_argument("--preprocessed-dataset-dir", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--results-key", default="results")
    parser.add_argument("--student-key", default="student")
    parser.add_argument("--teacher-key", default="teacher")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--blind-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_VISION_JUDGE_MODEL)
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--gemini-usage-log",
        default="",
        help="File JSONL separato in cui salvare usage_metadata delle chiamate Gemini.",
    )

    parser.add_argument("--opencode-model", default=DEFAULT_OPENCODE_VISION_JUDGE_MODEL)
    parser.add_argument("--opencode-timeout-seconds", type=float, default=240.0)

    parser.add_argument("--hf-vision-model", default=DEFAULT_HF_VISION_JUDGE_MODEL)
    parser.add_argument("--hf-vision-device", default="auto")
    parser.add_argument("--hf-vision-torch-dtype", default="auto")
    parser.add_argument(
        "--hf-vision-quantization", choices=("bnb4", "none"), default="bnb4"
    )
    parser.add_argument("--hf-vision-max-new-tokens", type=int, default=256)
    parser.add_argument("--hf-vision-no-trust-remote-code", action="store_true")

    parser.add_argument("--lmstudio-model", default=DEFAULT_LMSTUDIO_VISION_JUDGE_MODEL)
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
        from src.judging import LMStudioVisionJudge, LMStudioVisionJudgeConfig

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
        return LMStudioVisionJudge(
            LMStudioVisionJudgeConfig(
                model=args.lmstudio_model,
                base_url=args.lmstudio_base_url,
                max_tokens=args.lmstudio_max_tokens,
                timeout_seconds=args.lmstudio_timeout_seconds,
            )
        )
    from src.judging import HuggingFaceVisionJudge, HuggingFaceVisionJudgeConfig

    return HuggingFaceVisionJudge(
        HuggingFaceVisionJudgeConfig(
            model_id=args.hf_vision_model,
            torch_dtype=args.hf_vision_torch_dtype,
            device=args.hf_vision_device,
            load_in_4bit=args.hf_vision_quantization == "bnb4",
            max_new_tokens=args.hf_vision_max_new_tokens,
            trust_remote_code=not args.hf_vision_no_trust_remote_code,
        )
    )


def _model_name(args: argparse.Namespace) -> str:
    if args.backend == "gemini":
        return args.gemini_model
    if args.backend == "opencode":
        return args.opencode_model
    if args.backend == "lmstudio":
        return args.lmstudio_model
    return args.hf_vision_model


def _dataset_dir(args: argparse.Namespace, report: dict[str, Any]) -> str:
    value = args.preprocessed_dataset_dir or str(
        report.get("preprocessed_dataset_dir", "") or ""
    )
    if value:
        return value
    return str(Path("runs") / "mini_imagenet_preprocessed_392")


def _winner_from_preference(preference: str, role_a: str, role_b: str) -> str:
    if preference == "A":
        winner = role_a
    elif preference == "B":
        winner = role_b
    else:
        winner = "tie"
    if winner == "student":
        return "student_win"
    if winner == "teacher":
        return "teacher_win"
    return winner


def _ensure_vlm_namespace(item: dict[str, Any], judge_id: str) -> dict[str, Any]:
    judges = item.setdefault("vlm_judges", {})
    if not isinstance(judges, dict):
        raise ValueError("Campo item.vlm_judges non valido")
    payload = judges.setdefault(judge_id, {})
    if not isinstance(payload, dict):
        raise ValueError(f"Campo item.vlm_judges.{judge_id} non valido")
    return payload


def _summarize(results: list[Any], *, judge_id: str) -> dict[str, Any]:
    counts = {"student_win": 0, "teacher_win": 0, "tie": 0, "invalid": 0}
    for item in results:
        if not isinstance(item, dict):
            continue
        judges = item.get("vlm_judges")
        if not isinstance(judges, dict):
            continue
        judge_payload = judges.get(judge_id)
        if not isinstance(judge_payload, dict):
            continue
        payload = judge_payload.get("vision_blind")
        if not isinstance(payload, dict):
            continue
        winner = str(payload.get("winner", "")).lower()
        if winner not in counts:
            winner = "invalid"
        counts[winner] += 1
    total = sum(counts.values())
    return {
        "num_judged": total,
        "counts": counts,
        "rates": {
            key: (float(value / total) if total else 0.0)
            for key, value in counts.items()
        },
    }


def main() -> None:
    args = parse_args()
    judge_id = _slugify(args.judge_id)
    report_path = Path(args.report_path)
    report = _load_report(report_path)
    results = report.get(args.results_key)
    if not isinstance(results, list):
        raise ValueError(f"Campo report non valido: {args.results_key!r}")

    split = args.split or str(report.get("split", "test"))
    dataset_dir = _dataset_dir(args, report)
    dataset = load_preprocessed_mini_imagenet(dataset_dir, splits=(split,))[split]

    output_path = (
        report_path
        if args.in_place
        else (Path(args.output_report) if args.output_report else None)
    )

    def save_progress() -> None:
        if output_path is not None:
            _write_report(output_path, report)

    judge = _build_judge(args)
    prompt = str(
        report.get(args.prompt_key, "Describe the image.") or "Describe the image."
    )

    rows: list[tuple[int, dict[str, Any]]] = []
    for row_idx, item in enumerate(results):
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get(args.student_key), str):
            continue
        if not isinstance(item.get(args.teacher_key), str):
            continue
        existing = item.get("vlm_judges")
        if (
            not args.force
            and isinstance(existing, dict)
            and isinstance(existing.get(judge_id), dict)
            and isinstance(existing[judge_id].get("vision_blind"), dict)
        ):
            continue
        rows.append((row_idx, item))
        if args.limit > 0 and len(rows) >= args.limit:
            break

    if len(rows) < len([r for r in results if isinstance(r, dict)]):
        print(f"Resume: {len(rows)} righe da giudicare per {judge_id}.")

    judged = []
    for pos, (row_idx, item) in enumerate(rows, start=1):
        rng = random.Random(args.blind_seed + row_idx)
        student_is_a = bool(rng.getrandbits(1))
        if student_is_a:
            answer_a = item[args.student_key]
            answer_b = item[args.teacher_key]
            role_a = "student"
            role_b = "teacher"
        else:
            answer_a = item[args.teacher_key]
            answer_b = item[args.student_key]
            role_a = "teacher"
            role_b = "student"

        dataset_idx = int(item["dataset_index"])
        image = dataset[dataset_idx]["image"]
        image_input: Any = image
        if args.backend == "opencode":
            image_path = Path(dataset.files[dataset_idx])
            image_input = image_path

        print(
            f"[{pos}/{len(rows)}] vision blind idx={row_idx} class={item.get('label')}"
        )
        result = judge.prefer_with_image(
            PreferenceExample(
                answer_a=answer_a,
                answer_b=answer_b,
                prompt=prompt,
                label=item.get("label"),
                dataset_index=dataset_idx,
            ),
            image_input,
        )
        payload = result.to_dict()
        payload.update(
            {
                "answer_a_role": role_a,
                "answer_b_role": role_b,
                "winner": _winner_from_preference(result.preference, role_a, role_b),
                "student_key": args.student_key,
                "teacher_key": args.teacher_key,
                "dataset_index": dataset_idx,
            }
        )
        _ensure_vlm_namespace(item, judge_id)["vision_blind"] = payload
        judged.append(row_idx)
        save_progress()

    report.setdefault("vlm_judges", {})
    report["vlm_judges"][judge_id] = {
        "backend": args.backend,
        "model": _model_name(args),
        "mode": "vision_blind",
        "judge_id": judge_id,
        "preprocessed_dataset_dir": dataset_dir,
        "split": split,
        **_summarize(results, judge_id=judge_id),
    }

    print("[VLM judge report]")
    print(f"Judge: {judge_id}")
    print(f"Judged this run: {len(judged)}")
    print(f"Counts: {report['vlm_judges'][judge_id]['counts']}")

    if output_path is not None:
        _write_report(output_path, report)
        print(f"Report aggiornato: {output_path}")


if __name__ == "__main__":
    main()
