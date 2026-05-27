"""Gemini LLM-as-a-judge backend using google-genai."""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    JUDGE_OUTPUT_SCHEMA,
    PREFERENCE_OUTPUT_SCHEMA,
    JudgeExample,
    JudgeResult,
    PreferenceExample,
    PreferenceResult,
    build_judge_prompt,
    build_preference_prompt,
    build_vision_preference_prompt,
    parse_judge_json,
    parse_preference_json,
)


DEFAULT_GEMINI_JUDGE_MODEL = "gemini-3.1-flash-lite"


def _load_dotenv_if_present() -> None:
    env_path = os.getcwd()
    while True:
        candidate = os.path.join(env_path, ".env")
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8-sig") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(
                        key.strip(),
                        value.strip().strip('"').strip("'"),
                    )
            return
        parent = os.path.dirname(env_path)
        if parent == env_path:
            return
        env_path = parent


@dataclass(frozen=True)
class GeminiJudgeConfig:
    """Configuration for Gemini API judge calls."""

    model: str = DEFAULT_GEMINI_JUDGE_MODEL
    api_key: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 256
    max_retries: int = 6
    retry_sleep_seconds: float = 10.0
    usage_log_path: str = ""


class GeminiJudge:
    """Run Gemini as a JSON judge."""

    def __init__(self, config: GeminiJudgeConfig | None = None) -> None:
        self.config = config or GeminiJudgeConfig()
        _load_dotenv_if_present()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai non e' installato. Installa le dipendenze del "
                "progetto o esegui: uv add google-genai"
            ) from exc
        self._types = types
        self.client = genai.Client(
            api_key=self.config.api_key or os.environ.get("GEMINI_API_KEY")
        )

    def judge(self, example: JudgeExample) -> JudgeResult:
        prompt = build_judge_prompt(example)
        response = self._generate_content(
            usage_context={
                "mode": "direct",
                "label": example.label,
                "dataset_index": example.dataset_index,
            },
            model=self.config.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=JUDGE_OUTPUT_SCHEMA,
            ),
        )
        return parse_judge_json(response.text or "")

    def prefer(self, example: PreferenceExample) -> PreferenceResult:
        prompt = build_preference_prompt(example)
        response = self._generate_content(
            usage_context={
                "mode": "blind",
                "label": example.label,
                "dataset_index": example.dataset_index,
            },
            model=self.config.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=PREFERENCE_OUTPUT_SCHEMA,
            ),
        )
        return parse_preference_json(response.text or "")

    def prefer_with_image(
        self,
        example: PreferenceExample,
        image,
    ) -> PreferenceResult:
        prompt = build_vision_preference_prompt(example)
        response = self._generate_content(
            usage_context={
                "mode": "vision_blind",
                "label": example.label,
                "dataset_index": example.dataset_index,
            },
            model=self.config.model,
            contents=[image, prompt],
            config=self._types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=PREFERENCE_OUTPUT_SCHEMA,
            ),
        )
        return parse_preference_json(response.text or "")

    def _generate_content(
        self, *, usage_context: dict[str, Any] | None = None, **kwargs
    ):
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.models.generate_content(**kwargs)
                self._log_usage(response, usage_context=usage_context)
                return response
            except Exception as exc:
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                message = str(exc)
                is_transient = status_code in {429, 500, 502, 503, 504} or any(
                    marker in message
                    for marker in ("429", "500", "502", "503", "504", "UNAVAILABLE")
                )
                if not is_transient:
                    raise
                if attempt >= self.config.max_retries:
                    raise
                sleep_for = self.config.retry_sleep_seconds * (attempt + 1)
                print(
                    f"Gemini transient error {status_code}; retry "
                    f"{attempt + 1}/{self.config.max_retries} in {sleep_for:.1f}s"
                )
                time.sleep(sleep_for)
        assert last_exc is not None
        raise last_exc

    def _log_usage(
        self,
        response: Any,
        *,
        usage_context: dict[str, Any] | None,
    ) -> None:
        if not self.config.usage_log_path:
            return
        path = Path(self.config.usage_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        usage = getattr(response, "usage_metadata", None)
        payload = {
            "timestamp_unix": time.time(),
            "provider": "gemini",
            "model": self.config.model,
            "context": usage_context or {},
            "usage_metadata": _to_jsonable(usage),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if hasattr(value, "to_json_dict"):
        return _to_jsonable(value.to_json_dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)
