from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from comfy_agent_prompter.models import AppConfig


def load_app_config(
    config_path: str | Path,
    *,
    objective_override: str | None = None,
    reference_image_paths_override: list[str] | None = None,
) -> AppConfig:
    load_dotenv()

    config_file = Path(config_path).resolve()
    raw_config = json.loads(config_file.read_text(encoding="utf-8"))
    config = AppConfig.model_validate(raw_config)

    workflow_path = _resolve_relative(config_file, config.comfyui.workflow_path)
    mapping_path = _resolve_relative(config_file, config.comfyui.mapping_path)
    reference_image_paths = [
        _resolve_relative(config_file, image_path)
        for image_path in (reference_image_paths_override or config.task.reference_image_paths)
    ]

    config = config.model_copy(
        update={
            "comfyui": config.comfyui.model_copy(
                update={
                    "workflow_path": str(workflow_path),
                    "mapping_path": str(mapping_path),
                }
            ),
            "task": config.task.model_copy(
                update={
                    "objective": objective_override or config.task.objective,
                    "reference_image_paths": [str(path) for path in reference_image_paths],
                }
            ),
        }
    )

    providers = {}
    for role, provider in config.providers.items():
        extra_headers = dict(provider.extra_headers)
        if "openrouter.ai" in provider.base_url:
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            title = os.getenv("OPENROUTER_X_TITLE")
            if referer and "HTTP-Referer" not in extra_headers:
                extra_headers["HTTP-Referer"] = referer
            if title and "X-Title" not in extra_headers:
                extra_headers["X-Title"] = title
        providers[role] = provider.model_copy(
            update={
                "extra_headers": extra_headers,
                "model": _resolve_provider_model_override(role, provider.model),
            }
        )

    return config.model_copy(update={"providers": providers})


def resolve_api_key(raw_key: str | None, env_name: str | None) -> str | None:
    if raw_key:
        return raw_key
    if env_name:
        return os.getenv(env_name)
    return None


def _resolve_relative(config_file: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (config_file.parent / candidate).resolve()


def _resolve_provider_model_override(role: str, default_model: str) -> str:
    env_name = {
        "agent": "CAP_PROMPTER_MODEL",
        "judge": "CAP_JUDGE_MODEL",
    }.get(role)
    if not env_name:
        return default_model
    return os.getenv(env_name) or default_model
