from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from comfy_agent_prompter.comfy.workflow import apply_workflow_mapping, render_uploaded_value
from comfy_agent_prompter.files import read_json
from comfy_agent_prompter.models import AgentPlan, AppConfig, WorkflowMapping

StatusCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class ComfyUiClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = config.comfyui.base_url.rstrip("/")
        self.timeout = config.comfyui.request_timeout_ms / 1000

    async def health_check(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            return response.json()

    async def generate(
        self,
        plan: AgentPlan,
        reference_image_path: str | None,
        status_callback: StatusCallback | None = None,
    ) -> tuple[str, bytes]:
        workflow = read_json(self.config.comfyui.workflow_path)
        mapping = WorkflowMapping.model_validate(read_json(self.config.comfyui.mapping_path))

        uploaded_value = None
        if reference_image_path and mapping.reference_image is not None:
            await self._emit_status(
                status_callback,
                "reference_upload_started",
                {"reference_image_path": str(Path(reference_image_path).resolve())},
            )
            upload_payload = await self.upload_image(reference_image_path)
            uploaded_value = render_uploaded_value(mapping.reference_image, upload_payload)
            await self._emit_status(
                status_callback,
                "reference_upload_completed",
                {"uploaded_name": upload_payload["name"], "uploaded_subfolder": upload_payload.get("subfolder", "")},
            )

        prompt = apply_workflow_mapping(
            workflow=workflow,
            mapping=mapping,
            plan=plan,
            defaults=self.config.generation_defaults,
            reference_image_value=uploaded_value,
        )
        await self._emit_status(status_callback, "prompt_prepared", {})

        prompt_id = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/prompt",
                json={"prompt": prompt, "prompt_id": prompt_id, "client_id": prompt_id},
            )
            response.raise_for_status()
        await self._emit_status(status_callback, "prompt_submitted", {"prompt_id": prompt_id})

        image_ref = await self._wait_for_image(prompt_id, status_callback)
        image_bytes = await self._download_image(image_ref)
        return image_ref["filename"], image_bytes

    async def upload_image(self, image_path: str) -> dict[str, str]:
        file_path = Path(image_path)
        files = {
            "image": (file_path.name, file_path.read_bytes(), "application/octet-stream"),
            "type": (None, "input"),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/upload/image", files=files)
            response.raise_for_status()
            payload = response.json()
        return {
            "name": payload["name"],
            "subfolder": payload.get("subfolder", ""),
            "type": payload.get("type", "input"),
        }

    async def _wait_for_image(
        self,
        prompt_id: str,
        status_callback: StatusCallback | None = None,
    ) -> dict[str, str]:
        poll_interval = self.config.comfyui.poll_interval_ms / 1000
        elapsed_seconds = 0.0
        await self._emit_status(status_callback, "waiting_for_output", {"prompt_id": prompt_id})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while True:
                response = await client.get(f"{self.base_url}/history/{prompt_id}")
                response.raise_for_status()
                payload = response.json()
                prompt_payload = payload.get(prompt_id)
                if prompt_payload:
                    for output in prompt_payload.get("outputs", {}).values():
                        images = output.get("images", [])
                        if images:
                            await self._emit_status(
                                status_callback,
                                "output_ready",
                                {"prompt_id": prompt_id, "filename": images[0].get("filename", "")},
                            )
                            return images[0]

                elapsed_seconds += poll_interval
                if elapsed_seconds and elapsed_seconds % 15 < poll_interval:
                    await self._emit_status(
                        status_callback,
                        "still_waiting",
                        {"prompt_id": prompt_id, "elapsed_seconds": int(elapsed_seconds)},
                    )
                await asyncio.sleep(poll_interval)

    async def _download_image(self, image_ref: dict[str, str]) -> bytes:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.base_url}/view",
                params={
                    "filename": image_ref["filename"],
                    "subfolder": image_ref.get("subfolder", ""),
                    "type": image_ref.get("type", "output"),
                },
            )
            response.raise_for_status()
            return response.content

    async def _emit_status(
        self,
        status_callback: StatusCallback | None,
        stage: str,
        data: dict[str, Any],
    ) -> None:
        if status_callback is None:
            return
        await status_callback(stage, data)
