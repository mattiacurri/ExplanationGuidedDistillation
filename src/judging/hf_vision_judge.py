"""Local Hugging Face vision-language judge backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from src.utils.torch_utils import parse_torch_dtype

from .base import (
    PreferenceExample,
    PreferenceResult,
    build_vision_preference_prompt,
    parse_preference_json,
)


DEFAULT_HF_VISION_JUDGE_MODEL = "zai-org/GLM-4.6V-Flash"


@dataclass(frozen=True)
class HuggingFaceVisionJudgeConfig:
    """Configuration for a local VLM judge."""

    model_id: str = DEFAULT_HF_VISION_JUDGE_MODEL
    torch_dtype: str = "auto"
    device: str = "auto"
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    trust_remote_code: bool = True


class HuggingFaceVisionJudge:
    """Run a local HF VLM as an image-grounded A/B preference judge."""

    def __init__(self, config: HuggingFaceVisionJudgeConfig | None = None) -> None:
        self.config = config or HuggingFaceVisionJudgeConfig()
        dtype = parse_torch_dtype(self.config.torch_dtype)
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.config.load_in_4bit:
            compute_dtype = dtype or _default_4bit_compute_dtype()
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = (
                "auto" if self.config.device == "auto" else {"": self.config.device}
            )
        elif dtype is not None:
            kwargs["dtype"] = dtype

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            **kwargs,
        )
        if not self.config.load_in_4bit and self.config.device != "auto":
            self.model.to(torch.device(self.config.device))
        elif not self.config.load_in_4bit and torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        self.model.eval()

    @torch.no_grad()
    def prefer_with_image(
        self,
        example: PreferenceExample,
        image: Image.Image | str | Path,
    ) -> PreferenceResult:
        prompt = build_vision_preference_prompt(example)
        pil_image = _load_image(image)
        raw_text = self._generate(prompt, pil_image)
        try:
            return parse_preference_json(raw_text)
        except ValueError:
            return PreferenceResult(
                preference="tie",
                rationale=(
                    "HF vision judge did not return the requested structured "
                    "preference/rationale output."
                ),
                raw_text=raw_text,
            )

    def _generate(self, prompt: str, image: Image.Image) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1] :]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def _load_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _default_4bit_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
