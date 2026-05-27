"""LM Studio OpenAI-compatible vision-language judge backend."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .base import (
    PreferenceExample,
    PreferenceResult,
    build_vision_preference_prompt,
    parse_preference_json,
)


DEFAULT_LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LMSTUDIO_VISION_MODEL = "zai-org/glm-4.6v-flash"


@dataclass(frozen=True)
class LMStudioVisionJudgeConfig:
    """Configuration for LM Studio VLM judge calls."""

    model: str = DEFAULT_LMSTUDIO_VISION_MODEL
    base_url: str = DEFAULT_LMSTUDIO_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: float = 180.0
    max_retries: int = 3
    retry_sleep_seconds: float = 5.0


class LMStudioVisionJudge:
    """Run a local LM Studio VLM as an image-grounded A/B preference judge."""

    def __init__(self, config: LMStudioVisionJudgeConfig | None = None) -> None:
        self.config = config or LMStudioVisionJudgeConfig()

    def prefer_with_image(
        self,
        example: PreferenceExample,
        image: Image.Image | str | Path,
    ) -> PreferenceResult:
        raw_text = self._generate_json(
            build_vision_preference_prompt(example),
            _image_to_data_url(image),
        )
        try:
            return parse_preference_json(raw_text)
        except ValueError:
            return PreferenceResult(
                preference="tie",
                rationale=(
                    "LM Studio vision judge did not return the requested "
                    "structured preference/rationale output."
                ),
                raw_text=raw_text,
            )

    def _generate_json(self, prompt: str, image_data_url: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "vision_preference",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "preference": {
                                "type": "string",
                                "enum": ["A", "B", "tie"],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["preference", "rationale"],
                        "additionalProperties": False,
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


def _image_to_data_url(image: Image.Image | str | Path) -> str:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.open(image).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
