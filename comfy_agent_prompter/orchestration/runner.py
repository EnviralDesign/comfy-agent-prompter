from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

from comfy_agent_prompter.comfy.client import ComfyUiClient
from comfy_agent_prompter.config import load_app_config
from comfy_agent_prompter.files import bytes_to_data_url, ensure_dir, path_to_data_url, write_json
from comfy_agent_prompter.models import AgentPlan, IterationSnapshot
from comfy_agent_prompter.prompts import (
    build_agent_messages,
    build_judge_messages,
    parse_agent_plan,
    parse_judge_result,
)
from comfy_agent_prompter.providers.openai_compatible import OpenAICompatibleClient
from comfy_agent_prompter.services.run_store import RunStore


class OrchestrationRunner:
    def __init__(self, run_store: RunStore) -> None:
        self.run_store = run_store

    async def start_background(
        self,
        *,
        config_path: str,
        objective_override: str | None = None,
        reference_image_paths_override: list[str] | None = None,
    ) -> str:
        config = load_app_config(
            config_path,
            objective_override=objective_override,
            reference_image_paths_override=reference_image_paths_override,
        )
        run_id = uuid.uuid4().hex[:12]
        await self.run_store.create_run(run_id, config.task.objective, str(Path(config_path).resolve()))
        await self.run_store.append_event(run_id, "run.created", "Run queued.")
        asyncio.create_task(
            self._execute(
                run_id=run_id,
                config_path=config_path,
                objective_override=objective_override,
                reference_image_paths_override=reference_image_paths_override,
            )
        )
        return run_id

    async def run_inline(
        self,
        *,
        config_path: str,
        objective_override: str | None = None,
        reference_image_paths_override: list[str] | None = None,
    ) -> str:
        config = load_app_config(
            config_path,
            objective_override=objective_override,
            reference_image_paths_override=reference_image_paths_override,
        )
        run_id = uuid.uuid4().hex[:12]
        await self.run_store.create_run(run_id, config.task.objective, str(Path(config_path).resolve()))
        await self._execute(
            run_id=run_id,
            config_path=config_path,
            objective_override=objective_override,
            reference_image_paths_override=reference_image_paths_override,
        )
        return run_id

    async def _execute(
        self,
        *,
        run_id: str,
        config_path: str,
        objective_override: str | None,
        reference_image_paths_override: list[str] | None,
    ) -> None:
        try:
            config = load_app_config(
                config_path,
                objective_override=objective_override,
                reference_image_paths_override=reference_image_paths_override,
            )
            await self.run_store.update_run(run_id, status="running")
            await self.run_store.append_event(run_id, "run.started", "Run started.")

            reference_data_urls = [path_to_data_url(path) for path in config.task.reference_image_paths]
            output_dir = ensure_dir(Path("runs") / f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{run_id}")
            await self.run_store.update_run(run_id, output_dir=str(output_dir.resolve()))

            agent_client = OpenAICompatibleClient(config.providers["agent"])
            judge_client = (
                OpenAICompatibleClient(config.providers["judge"])
                if config.loop.enable_judge and "judge" in config.providers
                else None
            )
            if config.loop.enable_judge and judge_client is None:
                raise ValueError("Judge is enabled in config but no judge provider was configured.")
            comfy_client = ComfyUiClient(config)

            await self.run_store.append_event(run_id, "comfy.ready", "ComfyUI client configured.")

            snapshots: list[IterationSnapshot] = []
            accepted = False
            stop_reason = "max_iterations_reached"

            for iteration_index in range(1, config.loop.max_iterations + 1):
                await self.run_store.append_event(
                    run_id,
                    "agent.requested",
                    f"Requesting plan for iteration {iteration_index}.",
                    iteration=iteration_index,
                )

                agent_messages = build_agent_messages(config, snapshots, reference_data_urls)
                agent_text, _ = await agent_client.complete(agent_messages)
                plan: AgentPlan = parse_agent_plan(agent_text, config)

                await self.run_store.append_event(
                    run_id,
                    "agent.planned",
                    f"Agent produced prompt for iteration {iteration_index}.",
                    iteration=iteration_index,
                    prompt=plan.prompt,
                )

                primary_reference = config.task.reference_image_paths[0] if config.task.reference_image_paths else None
                await self.run_store.append_event(
                    run_id,
                    "comfy.executing",
                    f"ComfyUI execution started for iteration {iteration_index}.",
                    iteration=iteration_index,
                )
                filename, image_bytes = await comfy_client.generate(plan, primary_reference)
                image_path = output_dir / f"iteration-{iteration_index:03d}-{filename}"
                image_path.write_bytes(image_bytes)
                image_data_url = bytes_to_data_url(image_bytes, suffix=image_path.suffix or ".png")

                await self.run_store.append_event(
                    run_id,
                    "image.generated",
                    f"ComfyUI finished iteration {iteration_index}.",
                    iteration=iteration_index,
                    image_path=str(image_path.resolve()),
                )

                judge_accept = None
                judge_feedback = None
                judge_score = None

                if judge_client is not None:
                    judge_messages = build_judge_messages(config, plan, image_data_url, reference_data_urls)
                    judge_text, _ = await judge_client.complete(judge_messages)
                    judge_result = parse_judge_result(judge_text)
                    judge_accept = judge_result.accept
                    judge_feedback = judge_result.feedback
                    judge_score = judge_result.score

                    await self.run_store.append_event(
                        run_id,
                        "judge.completed",
                        f"Judge evaluated iteration {iteration_index}.",
                        iteration=iteration_index,
                        accept=judge_result.accept,
                        feedback=judge_result.feedback,
                    )

                    if judge_result.accept:
                        accepted = True
                        stop_reason = "judge_accepted"
                elif plan.is_satisfied and iteration_index >= config.loop.min_iterations_before_self_stop:
                    accepted = True
                    stop_reason = "agent_self_stopped"

                snapshot = IterationSnapshot(
                    index=iteration_index,
                    prompt=plan.prompt,
                    negative_prompt=plan.negative_prompt,
                    width=plan.width,
                    height=plan.height,
                    steps=plan.steps,
                    cfg_scale=plan.cfg_scale,
                    seed=plan.seed,
                    self_critique=plan.self_critique,
                    notes_to_judge=plan.notes_to_judge,
                    judge_accept=judge_accept,
                    judge_feedback=judge_feedback,
                    judge_score=judge_score,
                    image_path=str(image_path.resolve()),
                    image_data_url=image_data_url,
                )
                snapshots.append(snapshot)
                await self.run_store.append_iteration(run_id, snapshot)

                if accepted:
                    break

            manifest = {
                "run_id": run_id,
                "accepted": accepted,
                "stop_reason": stop_reason,
                "objective": config.task.objective,
                "quality_bar": config.task.quality_bar,
                "config_path": str(Path(config_path).resolve()),
                "iterations": [
                    snapshot.model_dump(mode="json", exclude={"image_data_url"}) for snapshot in snapshots
                ],
            }
            write_json(output_dir / "manifest.json", manifest)

            await self.run_store.update_run(
                run_id,
                status="succeeded",
                accepted=accepted,
                stop_reason=stop_reason,
            )
            await self.run_store.append_event(run_id, "run.finished", f"Run finished: {stop_reason}.")
        except Exception as exc:  # noqa: BLE001
            await self.run_store.update_run(run_id, status="failed", error=str(exc))
            await self.run_store.append_event(run_id, "run.failed", "Run failed.", error=str(exc))
