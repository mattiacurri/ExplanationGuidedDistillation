"""Hugging Face local LLM-as-a-judge backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.torch_utils import parse_torch_dtype

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


DEFAULT_HF_JUDGE_MODEL = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class HuggingFaceJudgeConfig:
    """Configuration for a small local Hugging Face judge model."""

    model_id: str = DEFAULT_HF_JUDGE_MODEL
    torch_dtype: str = "auto"
    device: str = "auto"
    load_in_4bit: bool = True
    max_new_tokens: int = 160
    trust_remote_code: bool = False


class HuggingFaceJudge:
    """Run a local causal LM from Hugging Face as a JSON judge."""

    def __init__(self, config: HuggingFaceJudgeConfig | None = None) -> None:
        self.config = config or HuggingFaceJudgeConfig()
        dtype = parse_torch_dtype(self.config.torch_dtype)
        kwargs = {}
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        if not self.config.load_in_4bit and self.config.device != "auto":
            self.model.to(torch.device(self.config.device))
        elif not self.config.load_in_4bit and torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        self.model.eval()

    @torch.no_grad()
    def judge(self, example: JudgeExample) -> JudgeResult:
        prompt = build_judge_prompt(example)
        raw_text = self._generate(prompt)
        try:
            return parse_judge_json(raw_text)
        except ValueError:
            return JudgeResult(
                score=1.0,
                verdict="fail",
                rationale=(
                    "Judge did not return the requested structured "
                    "score/rationale output."
                ),
                raw_text=raw_text,
            )

    @torch.no_grad()
    def prefer(self, example: PreferenceExample) -> PreferenceResult:
        prompt = build_preference_prompt(example)
        raw_text = self._generate(prompt)
        try:
            return parse_preference_json(raw_text)
        except ValueError:
            return PreferenceResult(
                preference="tie",
                rationale=(
                    "Judge did not return the requested structured "
                    "preference/rationale output."
                ),
                raw_text=raw_text,
            )

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if self.config.model_id.startswith("Qwen/Qwen3"):
                template_kwargs["enable_thinking"] = False
            try:
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs,
                )
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[-1]
        raw_text = self.tokenizer.decode(
            generated[0, prompt_len:],
            skip_special_tokens=True,
        )
        return raw_text


def _default_4bit_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
