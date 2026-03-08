from __future__ import annotations

from typing import Any

import httpx

from comfy_agent_prompter.config import resolve_api_key
from comfy_agent_prompter.models import ModelEndpointConfig


class OpenAICompatibleClient:
    def __init__(self, config: ModelEndpointConfig) -> None:
        self.config = config

    async def complete(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self._endpoint_url("/chat/completions"),
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()

        choices = body.get("choices", [])
        if not choices:
            raise ValueError("Model response did not contain any choices.")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content, body

        if isinstance(content, list):
            text_chunks = [part.get("text", "") for part in content if part.get("type") == "text"]
            return "\n".join(chunk for chunk in text_chunks if chunk), body

        raise ValueError("Model response content was not text.")

    async def list_models(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(self._endpoint_url("/models"), headers=self._headers())
            response.raise_for_status()
            return response.json()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = resolve_api_key(self.config.api_key, self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _endpoint_url(self, suffix: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{suffix}"

