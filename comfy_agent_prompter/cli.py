from __future__ import annotations

import asyncio
import json

import typer
import uvicorn

from comfy_agent_prompter.app_state import run_store, runner
from comfy_agent_prompter.comfy.client import ComfyUiClient
from comfy_agent_prompter.config import load_app_config
from comfy_agent_prompter.providers.openai_compatible import OpenAICompatibleClient

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("comfy_agent_prompter.main:app", host=host, port=port, reload=False)


@app.command()
def run(
    config: str = typer.Option(..., "--config"),
    objective: str | None = typer.Option(None, "--objective"),
    reference: list[str] = typer.Option(None, "--reference"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    async def _run() -> None:
        run_id = await runner.run_inline(
            config_path=config,
            objective_override=objective,
            reference_image_paths_override=reference or None,
        )
        detail = await run_store.get_run(run_id)
        if detail is None:
            raise RuntimeError("Run completed but could not be loaded from the run store.")
        payload = detail.model_dump(mode="json")
        if json_output:
            print(json.dumps(payload, indent=2))
            return

        print(f"run_id: {detail.run_id}")
        print(f"status: {detail.status}")
        print(f"accepted: {detail.accepted}")
        print(f"iterations: {detail.iteration_count}")
        print(f"judge_count: {detail.judge_count}")
        print(f"stop_reason: {detail.stop_reason}")
        print(f"final_image_path: {detail.final_image_path}")
        if detail.error:
            print(f"error: {detail.error}")

    asyncio.run(_run())


@app.command()
def doctor(config: str = typer.Option(..., "--config")) -> None:
    async def _doctor() -> None:
        app_config = load_app_config(config)
        comfy = ComfyUiClient(app_config)
        agent = OpenAICompatibleClient(app_config.providers["agent"])

        print(f"workflow: {app_config.comfyui.workflow_path}")
        print(f"mapping: {app_config.comfyui.mapping_path}")
        print(f"objective: {app_config.task.objective}")

        comfy_health = await comfy.health_check()
        print("comfyui: ok")
        print(f"  devices: {len(comfy_health.get('devices', []))}")

        agent_models = await agent.list_models()
        print("agent provider: ok")
        print(f"  model_count: {len(agent_models.get('data', []))}")

        if app_config.loop.enable_judge and "judge" in app_config.providers:
            judge = OpenAICompatibleClient(app_config.providers["judge"])
            judge_models = await judge.list_models()
            print("judge provider: ok")
            print(f"  model_count: {len(judge_models.get('data', []))}")

    asyncio.run(_doctor())

