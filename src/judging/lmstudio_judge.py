"""LM Studio OpenAI-compatible text judge backend."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from .base import (
    JudgeExample,
    JudgeResult,
    PreferenceExample,
    PreferenceResult,
    build_judge_prompt,
    build_preference_prompt,
    parse_judge_json,
    parse_preference_json,
)


DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LMSTUDIO_JUDGE_MODEL = "zai-org/glm-4.6v-flash"


@dataclass(frozen=True)
class LMStudioJudgeConfig:
    """Configuration for LM Studio text judge calls."""

    model: str = DEFAULT_LMSTUDIO_JUDGE_MODEL
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: float = 180.0
    max_retries: int = 3
    retry_sleep_seconds: float = 5.0


class LMStudioJudge:
    """Run a local LM Studio model as a JSON text judge."""

    def __init__(self, config: LMStudioJudgeConfig | None = None) -> None:
        self.config = config or LMStudioJudgeConfig()

    def judge(self, example: JudgeExample) -> JudgeResult:
        raw_text = self._generate_json(build_judge_prompt(example))
        try:
            return parse_judge_json(raw_text)
        except ValueError:
            return JudgeResult(
                score=1.0,
                verdict="fail",
                rationale=(
                    "LM Studio judge did not return the requested structured "
                    "score/rationale output."
                ),
                raw_text=raw_text,
            )

    def prefer(self, example: PreferenceExample) -> PreferenceResult:
        raw_text = self._generate_json(build_preference_prompt(example))
        try:
            return parse_preference_json(raw_text)
        except ValueError:
            return PreferenceResult(
                preference="tie",
                rationale=(
                    "LM Studio judge did not return the requested structured "
                    "preference/rationale output."
                ),
                raw_text=raw_text,
            )

    def _generate_json(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "preference": {
                                "type": "string",
                                "enum": ["A", "B", "tie"],
                            },
                            "score": {"type": "number"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["rationale"],
                        "additionalProperties": True,
                    },
                },
            },
        }
        response = self._post_with_retries(payload)
        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Risposta LM Studio senza choices: {body}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError(f"Risposta LM Studio senza message: {body}")
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                return content
        if isinstance(content, list):
            text = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
                and part.get("type") in {"text", "output_text"}
            )
            if text.strip():
                return text
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content
        raise ValueError(f"Risposta LM Studio senza contenuto testuale: {body}")

    def _post_with_retries(self, payload: dict[str, Any]) -> requests.Response:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        last_response: requests.Response | None = None
        for attempt in range(self.config.max_retries + 1):
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise requests.HTTPError(
                        f"{exc}; body={response.text[:1000]}",
                        response=response,
                    ) from exc
                return response
            last_response = response
            if attempt >= self.config.max_retries:
                response.raise_for_status()
            sleep_for = self.config.retry_sleep_seconds * (attempt + 1)
            print(
                f"LM Studio transient error {response.status_code}; retry "
                f"{attempt + 1}/{self.config.max_retries} in {sleep_for:.1f}s"
            )
            time.sleep(sleep_for)
        assert last_response is not None
        last_response.raise_for_status()
        return last_response
