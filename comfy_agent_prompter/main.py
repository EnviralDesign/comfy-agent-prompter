from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from comfy_agent_prompter.app_state import run_store, runner


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "ui" / "templates"))


class StartRunRequest(BaseModel):
    config_path: str
    objective_override: str | None = None
    reference_image_paths_override: list[str] = Field(default_factory=list)


class ConfigPresetResponse(BaseModel):
    path: str
    objective: str
    reference_image_paths: list[str] = Field(default_factory=list)


def create_app() -> FastAPI:
    app = FastAPI(title="Comfy Agent Prompter")
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "ui" / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/api/runs")
    async def start_run(payload: StartRunRequest) -> dict[str, str]:
        run_id = await runner.start_background(
            config_path=payload.config_path,
            objective_override=payload.objective_override,
            reference_image_paths_override=payload.reference_image_paths_override or None,
        )
        return {"run_id": run_id}

    @app.get("/api/configs")
    async def list_config_presets() -> list[dict]:
        config_dir = BASE_DIR.parent / "examples"
        presets: list[ConfigPresetResponse] = []

        for config_file in sorted(config_dir.glob("*.json")):
            raw = json.loads(config_file.read_text(encoding="utf-8"))
            if not {"comfyui", "providers", "task"}.issubset(raw.keys()):
                continue
            task = raw.get("task", {})
            presets.append(
                ConfigPresetResponse(
                    path=str(Path("examples") / config_file.name).replace("\\", "/"),
                    objective=task.get("objective", ""),
                    reference_image_paths=task.get("reference_image_paths", []),
                )
            )

        return [preset.model_dump(mode="json") for preset in presets]

    @app.get("/api/runs")
    async def list_runs() -> list[dict]:
        runs = await run_store.list_runs()
        return [run.model_dump(mode="json") for run in runs]

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        run = await run_store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return run.model_dump(mode="json")

    @app.websocket("/ws/runs/{run_id}")
    async def run_events(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        queue = await run_store.subscribe(run_id)
        try:
            current = await run_store.get_run(run_id)
            if current is not None:
                await websocket.send_json(
                    {
                        "type": "snapshot",
                        "run": current.model_dump(mode="json"),
                    }
                )

            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await run_store.unsubscribe(run_id, queue)

    return app


app = create_app()
