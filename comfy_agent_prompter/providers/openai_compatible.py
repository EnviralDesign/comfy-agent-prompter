from __future__ import annotations

import json
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

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        self._endpoint_url("/chat/completions"),
                        headers=self._headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    body = response.json()
            except httpx.HTTPError as exc:
                last_error = RuntimeError(
                    f"{self._describe_target()} request to /chat/completions failed: {exc}"
                )
                if attempt == 3:
                    raise last_error from exc
                continue

            choices = body.get("choices", [])
            if not choices:
                last_error = ValueError("Model response did not contain any choices.")
                if attempt == 3:
                    raise last_error
                continue

            message = choices[0].get("message", {})
            content = message.get("content", "")
            extracted_text = self._extract_text_content(content)
            if extracted_text:
                return extracted_text, body

            if isinstance(content, dict):
                return json.dumps(content), body

            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal:
                return refusal, body

            finish_reason = choices[0].get("finish_reason")
            reasoning_present = bool(message.get("reasoning") or message.get("reasoning_details"))
            last_error = ValueError(
                "Model response content was not text. "
                f"content_type={type(content).__name__}, "
                f"message_keys={sorted(message.keys())}, "
                f"finish_reason={finish_reason}, "
                f"reasoning_present={reasoning_present}, "
                f"attempt={attempt}/3"
            )
            if attempt == 3:
                raise last_error

        if last_error is not None:
            raise last_error
        raise RuntimeError("Model completion failed unexpectedly without an error.")

    async def list_models(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(self._endpoint_url("/models"), headers=self._headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{self._describe_target()} request to /models failed: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = resolve_api_key(self.config.api_key, self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.config.extra_headers)
        return headers

    def _endpoint_url(self, suffix: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{suffix}"

    def _describe_target(self) -> str:
        return (
            f"Provider '{self.config.label}' "
            f"(model={self.config.model}, base_url={self.config.base_url.rstrip('/')})"
        )

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            for key in ("text", "content", "value"):
                value = content.get(key)
                extracted = self._extract_text_content(value)
                if extracted:
                    return extracted
            return ""

        if isinstance(content, list):
            text_chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    if part:
                        text_chunks.append(part)
                    continue
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type")
                if part_type in {"text", "output_text", "input_text"}:
                    extracted = self._extract_text_content(part.get("text"))
                    if extracted:
                        text_chunks.append(extracted)
                    continue

                for key in ("text", "content", "value"):
                    extracted = self._extract_text_content(part.get(key))
                    if extracted:
                        text_chunks.append(extracted)
                        break

            return "\n".join(chunk for chunk in text_chunks if chunk)

        return ""
