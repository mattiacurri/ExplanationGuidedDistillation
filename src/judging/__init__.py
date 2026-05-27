"""LLM-as-a-judge utilities for generated-answer evaluation."""

from __future__ import annotations

__all__ = [
    "GeminiJudge",
    "GeminiJudgeConfig",
    "HuggingFaceJudge",
    "HuggingFaceJudgeConfig",
    "HuggingFaceVisionJudge",
    "HuggingFaceVisionJudgeConfig",
    "OpenRouterJudge",
    "OpenRouterJudgeConfig",
    "OpenCodeJudge",
    "OpenCodeJudgeConfig",
    "LMStudioVisionJudge",
    "LMStudioVisionJudgeConfig",
    "LMStudioJudge",
    "LMStudioJudgeConfig",
    "JudgeExample",
    "JudgeResult",
    "PreferenceExample",
    "PreferenceResult",
    "build_judge_prompt",
    "build_preference_prompt",
    "build_vision_preference_prompt",
    "parse_judge_json",
    "parse_preference_json",
]
