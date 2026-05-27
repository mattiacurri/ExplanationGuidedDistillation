"""Utilities for ensuring an LM Studio OpenAI-compatible endpoint is live."""

from __future__ import annotations

import shutil
import subprocess
import time
import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LMStudioRuntimeConfig:
    """Configuration for starting/loading LM Studio models via ``lms``."""

    model: str
    base_url: str
    auto_load: bool = False
    load_model_key: str = ""
    load_identifier: str = ""
    gpu: str = "max"
    context_length: int = 32768
    parallel: int = 1
    server_port: int = 1234
    server_bind: str = "127.0.0.1"
    timeout_seconds: float = 300.0
    probe_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 2.0


def ensure_lmstudio_model(config: LMStudioRuntimeConfig) -> None:
    """Ensure the server is reachable and the requested model answers a probe."""
    if _model_is_live(config):
        _probe_chat_completion(config)
        return

    if not config.auto_load:
        raise RuntimeError(
            "LM Studio endpoint raggiungibile ma modello non live, oppure server "
            f"spento. Modello richiesto: {config.model!r}. Usa "
            "--lmstudio-auto-load per avviare/caricare via lms."
        )

    _ensure_lms_available()
    _ensure_server_started(config)
    if not _model_is_live(config):
        _load_model(config)
    _wait_until_model_live(config)
    _probe_chat_completion(config)


def _ensure_lms_available() -> None:
    if shutil.which("lms") is None and shutil.which("lms.exe") is None:
        raise FileNotFoundError(
            "Comando lms non trovato nel PATH. Installa/configura LM Studio CLI."
        )


def _ensure_server_started(config: LMStudioRuntimeConfig) -> None:
    if _server_is_reachable(config):
        return
    command = [
        "lms",
        "server",
        "start",
        "--port",
        str(config.server_port),
        "--bind",
        config.server_bind,
    ]
    print(
        f"LM Studio server non raggiungibile; avvio su "
        f"{config.server_bind}:{config.server_port}."
    )
    _run_lms(command, timeout=config.timeout_seconds)
    _wait_until_server_reachable(config)


def _load_model(config: LMStudioRuntimeConfig) -> None:
    model_key = config.load_model_key or config.model
    command = [
        "lms",
        "load",
        model_key,
        "--gpu",
        config.gpu,
        "--context-length",
        str(config.context_length),
        "--parallel",
        str(config.parallel),
        "-y",
    ]
    if config.load_identifier:
        command.extend(["--identifier", config.load_identifier])
    print(
        "LM Studio modello non live; carico "
        f"{model_key!r}"
        + (f" come {config.load_identifier!r}." if config.load_identifier else ".")
    )
    _run_lms(command, timeout=config.timeout_seconds)


def _run_lms(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Comando LM Studio fallito: {' '.join(command)}\n"
            f"stdout={completed.stdout[-2000:]}\n"
            f"stderr={completed.stderr[-2000:]}"
        )
    return completed


def _server_is_reachable(config: LMStudioRuntimeConfig) -> bool:
    try:
        response = requests.get(
            config.base_url.rstrip("/") + "/models",
            timeout=config.probe_timeout_seconds,
        )
    except requests.RequestException:
        return False
    return response.ok


def _model_is_live(config: LMStudioRuntimeConfig) -> bool:
    loaded_ids = _loaded_model_ids()
    if loaded_ids is not None:
        return config.model in loaded_ids
    try:
        response = requests.get(
            config.base_url.rstrip("/") + "/models",
            timeout=config.probe_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    return config.model in _model_ids(payload)


def _loaded_model_ids() -> set[str] | None:
    if shutil.which("lms") is None and shutil.which("lms.exe") is None:
        return None
    try:
        completed = _run_lms(["lms", "ps", "--json"], timeout=60.0)
    except (RuntimeError, subprocess.SubprocessError):
        return None
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return None
    if not isinstance(payload, list):
        return None
    ids: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("identifier", "modelKey", "selectedVariant"):
            value = item.get(key)
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


def _model_ids(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    data = payload.get("data")
    if not isinstance(data, list):
        return set()
    ids: set[str] = set()
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.add(item["id"])
    return ids


def _wait_until_server_reachable(config: LMStudioRuntimeConfig) -> None:
    deadline = time.monotonic() + config.timeout_seconds
    while time.monotonic() < deadline:
        if _server_is_reachable(config):
            return
        time.sleep(config.poll_interval_seconds)
    raise TimeoutError(f"LM Studio server non raggiungibile: {config.base_url}")


def _wait_until_model_live(config: LMStudioRuntimeConfig) -> None:
    deadline = time.monotonic() + config.timeout_seconds
    while time.monotonic() < deadline:
        if _model_is_live(config):
            return
        time.sleep(config.poll_interval_seconds)
    raise TimeoutError(f"Modello LM Studio non visibile in /models: {config.model!r}")


def _probe_chat_completion(config: LMStudioRuntimeConfig) -> None:
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON object and nothing else: "
                    '{"preference":"tie","rationale":"probe ok"}'
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "lmstudio_probe",
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
    response = requests.post(
        config.base_url.rstrip("/") + "/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=config.probe_timeout_seconds,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"{exc}; body={response.text[:1000]}",
            response=response,
        ) from exc
    body = response.json()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Probe LM Studio senza choices: {body}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Probe LM Studio senza message: {body}")
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    if not (
        (isinstance(content, str) and content.strip())
        or (isinstance(reasoning_content, str) and reasoning_content.strip())
    ):
        raise RuntimeError(f"Probe LM Studio senza contenuto: {body}")
