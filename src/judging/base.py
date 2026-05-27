"""Shared prompt and parsing logic for LLM-as-a-judge backends."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from textwrap import dedent
from typing import Any


JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 1,
            "maximum": 5,
            "description": "Quality score from 1 to 5.",
        },
        "rationale": {
            "type": "string",
            "description": "Short reason explaining the score.",
        },
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}


PREFERENCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "preference": {
            "type": "string",
            "enum": ["A", "B", "tie"],
            "description": "Preferred answer: A, B, or tie.",
        },
        "rationale": {
            "type": "string",
            "description": "Short reason explaining the preference.",
        },
    },
    "required": ["preference", "rationale"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeExample:
    """One reference answer and one candidate answer to judge."""

    reference_answer: str
    candidate_answer: str
    prompt: str = ""
    label: int | str | None = None
    dataset_index: int | None = None


@dataclass(frozen=True)
class PreferenceExample:
    """One blind A/B preference comparison."""

    answer_a: str
    answer_b: str
    prompt: str = ""
    label: int | str | None = None
    dataset_index: int | None = None


@dataclass(frozen=True)
class JudgeResult:
    """Structured result returned by an LLM judge."""

    score: float
    verdict: str
    rationale: str
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = float(payload["score"])
        return payload


@dataclass(frozen=True)
class PreferenceResult:
    """Structured result returned by a blind preference judge."""

    preference: str
    rationale: str
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_judge_prompt(example: JudgeExample) -> str:
    """Build a compact JSON-only prompt for answer quality judgment."""
    context_lines = []
    if example.prompt:
        context_lines.append(f"Original visual prompt: {example.prompt}")
    if example.label is not None:
        context_lines.append(f"Dataset label id: {example.label}")
    if example.dataset_index is not None:
        context_lines.append(f"Dataset index: {example.dataset_index}")
    context = "\n".join(context_lines) if context_lines else "No extra metadata."

    return dedent(
        f"""
        You are judging whether a candidate vision-language answer preserves
        the same visual meaning as a reference answer.

        Use only the reference answer and the candidate answer. Do not
        reward extra unsupported details. Penalize contradictions, missing main
        objects/actions, and generic "I cannot see the image" style answers.

        Score rubric:
        5 = equivalent or better, all key visual content preserved
        4 = mostly equivalent, only minor omissions or wording differences
        3 = partially aligned, main topic present but important details missing
        2 = weakly aligned, only vague or incidental overlap
        1 = wrong, contradictory, empty, or no-image answer

        Return ONLY one valid JSON object with exactly these fields:
        {{"score": number, "rationale": "short reason explaining the score"}}

        Structured output rules:
        - score must be a number from 1 to 5.
        - rationale must explain why that score was assigned.
        - do not include markdown, comments, verdict, or extra fields.

        Context:
        {context}

        Reference answer:
        {example.reference_answer}

        Candidate answer:
        {example.candidate_answer}
        """
    ).strip()


def build_preference_prompt(example: PreferenceExample) -> str:
    """Build a compact JSON-only prompt for blind A/B preference judgment."""
    context_lines = []
    if example.prompt:
        context_lines.append(f"Original visual prompt: {example.prompt}")
    if example.label is not None:
        context_lines.append(f"Dataset label id: {example.label}")
    if example.dataset_index is not None:
        context_lines.append(f"Dataset index: {example.dataset_index}")
    context = "\n".join(context_lines) if context_lines else "No extra metadata."

    return dedent(
        f"""
        You are comparing two vision-language answers to the same image prompt.

        The answers are anonymized. Do not assume either answer is the teacher
        or the student. Choose the answer that is more visually specific,
        accurate, and complete. Penalize contradictions, hallucinated details,
        missing main objects/actions, and generic "I cannot see the image"
        style answers. Choose tie only when both answers are similarly good or
        similarly weak.

        Return ONLY one valid JSON object with exactly these fields:
        {{"preference": "A" | "B" | "tie", "rationale": "short reason"}}

        Structured output rules:
        - preference must be exactly "A", "B", or "tie".
        - rationale must explain why that preference was assigned.
        - do not include markdown, comments, or extra fields.

        Context:
        {context}

        Answer A:
        {example.answer_a}

        Answer B:
        {example.answer_b}
        """
    ).strip()


def build_vision_preference_prompt(example: PreferenceExample) -> str:
    """Build a JSON-only prompt for image-grounded A/B judgment."""
    context_lines = []
    if example.prompt:
        context_lines.append(f"Original visual prompt: {example.prompt}")
    if example.label is not None:
        context_lines.append(f"Dataset label id: {example.label}")
    if example.dataset_index is not None:
        context_lines.append(f"Dataset index: {example.dataset_index}")
    context = "\n".join(context_lines) if context_lines else "No extra metadata."

    return dedent(
        f"""
        You are comparing two vision-language answers against the provided
        image. The answers are anonymized. Do not assume either answer is the
        teacher or the student.

        Use the image as the source of truth. Choose the answer that is more
        visually faithful, specific, accurate, and complete for the image.
        Penalize hallucinated objects, wrong attributes, unsupported species or
        class names, missing main objects/actions, contradictions, and generic
        "I cannot see the image" style answers. Choose tie only when both
        answers are similarly faithful or similarly weak.

        Return ONLY one valid JSON object with exactly these fields:
        {{"preference": "A" | "B" | "tie", "rationale": "short reason"}}

        Structured output rules:
        - preference must be exactly "A", "B", or "tie".
        - rationale must explain the image-grounded reason.
        - do not include markdown, comments, or extra fields.

        Context:
        {context}

        Answer A:
        {example.answer_a}

        Answer B:
        {example.answer_b}
        """
    ).strip()


def parse_judge_json(raw_text: str) -> JudgeResult:
    """Parse structured judge output and normalize score/verdict fields."""
    parsed = _loads_json_or_score(raw_text)
    if isinstance(parsed, dict):
        score = _clamp_score(float(parsed.get("score", 1.0)))
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "partial", "fail"}:
            verdict = _verdict_from_score(score)
        rationale = str(parsed.get("rationale", "")).strip()
        if not rationale:
            rationale = "No rationale returned."
    else:
        score = _clamp_score(float(parsed))
        verdict = _verdict_from_score(score)
        rationale = "Judge returned only a numeric score."
    return JudgeResult(
        score=score, verdict=verdict, rationale=rationale, raw_text=raw_text
    )


def parse_preference_json(raw_text: str) -> PreferenceResult:
    """Parse structured blind-preference output."""
    payload = _loads_json_object(raw_text)
    preference = str(payload.get("preference", "")).strip()
    if preference.upper() in {"A", "B"}:
        preference = preference.upper()
    elif preference.lower() in {"tie", "draw", "equal"}:
        preference = "tie"
    else:
        raise ValueError(f"Preference must be A, B, or tie: {raw_text}")
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        rationale = "No rationale returned."
    return PreferenceResult(
        preference=preference,
        rationale=rationale,
        raw_text=raw_text,
    )


def _loads_json_or_score(raw_text: str) -> dict[str, Any] | float:
    text = raw_text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is not None:
            payload = json.loads(match.group(0))
        else:
            score_match = re.search(r"(?<!\d)([1-5](?:\.\d+)?)(?!\d)", text)
            if score_match is None:
                raise ValueError(
                    f"Judge did not return JSON or a score: {raw_text}"
                ) from None
            return float(score_match.group(1))
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, str):
        score_match = re.search(r"(?<!\d)([1-5](?:\.\d+)?)(?!\d)", payload)
        if score_match is not None:
            return float(score_match.group(1))
    raise ValueError(f"Judge JSON must be an object or score: {raw_text}")


def _loads_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Judge did not return JSON: {raw_text}") from None
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError(f"Judge JSON must be an object: {raw_text}")
    return payload


def _clamp_score(score: float) -> float:
    return max(1.0, min(5.0, score))


def _verdict_from_score(score: float) -> str:
    if score >= 4.0:
        return "pass"
    if score >= 2.5:
        return "partial"
    return "fail"
