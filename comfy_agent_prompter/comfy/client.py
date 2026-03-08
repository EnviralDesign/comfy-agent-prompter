from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import httpx

from comfy_agent_prompter.comfy.workflow import apply_workflow_mapping, render_uploaded_value
from comfy_agent_prompter.files import read_json
from comfy_agent_prompter.models import AgentPlan, AppConfig, WorkflowMapping


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

    async def generate(self, plan: AgentPlan, reference_image_path: str | None) -> tuple[str, bytes]:
        workflow = read_json(self.config.comfyui.workflow_path)
        mapping = WorkflowMapping.model_validate(read_json(self.config.comfyui.mapping_path))

        uploaded_value = None
        if reference_image_path and mapping.reference_image is not None:
            upload_payload = await self.upload_image(reference_image_path)
            uploaded_value = render_uploaded_value(mapping.reference_image, upload_payload)

        prompt = apply_workflow_mapping(
            workflow=workflow,
            mapping=mapping,
            plan=plan,
            defaults=self.config.generation_defaults,
            reference_image_value=uploaded_value,
        )

        prompt_id = str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/prompt",
                json={"prompt": prompt, "prompt_id": prompt_id, "client_id": prompt_id},
            )
            response.raise_for_status()

        image_ref = await self._wait_for_image(prompt_id)
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

    async def _wait_for_image(self, prompt_id: str) -> dict[str, str]:
        poll_interval = self.config.comfyui.poll_interval_ms / 1000
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
                            return images[0]
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

