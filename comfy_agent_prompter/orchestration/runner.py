from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path

from comfy_agent_prompter.comfy.client import ComfyUiClient
from comfy_agent_prompter.config import load_app_config
from comfy_agent_prompter.files import bytes_to_data_url, ensure_dir, path_to_data_url, read_json, write_json
from comfy_agent_prompter.models import AgentPlan, IterationSnapshot, WorkflowMapping
from comfy_agent_prompter.prompts import (
    build_agent_messages,
    build_agent_selection_messages,
    build_judge_messages,
    parse_agent_selection,
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
            workflow_mapping = WorkflowMapping.model_validate(read_json(config.comfyui.mapping_path))

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
            frontier_snapshot: IterationSnapshot | None = None
            accepted = False
            judge_round_limit = config.loop.max_judge_rounds or config.loop.max_iterations
            global_iteration_index = 0
            stop_reason = "max_judge_rounds_reached"

            async def replace_snapshot(updated_snapshot: IterationSnapshot) -> None:
                for idx, item in enumerate(snapshots):
                    if item.index == updated_snapshot.index:
                        snapshots[idx] = updated_snapshot
                        await self.run_store.update_iteration(run_id, updated_snapshot)
                        return
                raise KeyError(f"Iteration snapshot {updated_snapshot.index} was not found.")

            for judge_round in range(1, judge_round_limit + 1):
                await self.run_store.append_event(
                    run_id,
                    "judge.round.started",
                    f"Judge round {judge_round} started.",
                    judge_round=judge_round,
                    max_judge_rounds=judge_round_limit,
                )

                round_candidates: list[IterationSnapshot] = []
                requested_judge_early = False

                for round_iteration in range(1, config.loop.max_agent_iterations_per_round + 1):
                    global_iteration_index += 1
                    await self.run_store.append_event(
                        run_id,
                        "agent.requested",
                        f"Requesting plan for iteration {global_iteration_index}.",
                        iteration=global_iteration_index,
                        judge_round=judge_round,
                        round_iteration=round_iteration,
                        max_judge_rounds=judge_round_limit,
                        max_agent_iterations_per_round=config.loop.max_agent_iterations_per_round,
                        provider_label=config.providers["agent"].label,
                        provider_model=config.providers["agent"].model,
                        provider_base_url=config.providers["agent"].base_url,
                    )

                    agent_messages = build_agent_messages(
                        config,
                        snapshots,
                        reference_data_urls,
                        judge_round=judge_round,
                        max_judge_rounds=judge_round_limit,
                        round_iteration=round_iteration,
                        frontier=frontier_snapshot,
                    )
                    agent_text, _ = await agent_client.complete(agent_messages)
                    plan: AgentPlan = parse_agent_plan(agent_text, config)

                    await self.run_store.append_event(
                        run_id,
                        "agent.planned",
                        f"Agent produced prompt for iteration {global_iteration_index}.",
                        iteration=global_iteration_index,
                        judge_round=judge_round,
                        round_iteration=round_iteration,
                        prompt=plan.prompt,
                        ready_for_judge=plan.ready_for_judge,
                    )

                    primary_reference = (
                        config.task.reference_image_paths[0] if config.task.reference_image_paths else None
                    )
                    await self.run_store.append_event(
                        run_id,
                        "comfy.executing",
                        f"ComfyUI execution started for iteration {global_iteration_index}.",
                        iteration=global_iteration_index,
                        judge_round=judge_round,
                        round_iteration=round_iteration,
                    )

                    async def report_comfy_status(stage: str, data: dict[str, object]) -> None:
                        event_type = f"comfy.{stage}"
                        message = {
                            "prompt_prepared": f"ComfyUI prompt prepared for iteration {global_iteration_index}.",
                            "prompt_submitted": f"ComfyUI accepted iteration {global_iteration_index} into its queue.",
                            "waiting_for_output": f"Waiting for ComfyUI output for iteration {global_iteration_index}.",
                            "still_waiting": (
                                "ComfyUI is still running iteration "
                                f"{global_iteration_index} ({data.get('elapsed_seconds', 0)}s elapsed)."
                            ),
                            "output_ready": f"ComfyUI reported an image for iteration {global_iteration_index}.",
                            "reference_upload_started": (
                                f"Uploading reference image for iteration {global_iteration_index}."
                            ),
                            "reference_upload_completed": (
                                f"Reference image uploaded for iteration {global_iteration_index}."
                            ),
                        }.get(stage, f"ComfyUI status update for iteration {global_iteration_index}: {stage}.")
                        await self.run_store.append_event(
                            run_id,
                            event_type,
                            message,
                            iteration=global_iteration_index,
                            judge_round=judge_round,
                            round_iteration=round_iteration,
                            **data,
                        )

                    filename, image_bytes = await comfy_client.generate(
                        plan,
                        primary_reference,
                        status_callback=report_comfy_status,
                    )
                    image_path = output_dir / f"iteration-{global_iteration_index:03d}-{filename}"
                    image_path.write_bytes(image_bytes)
                    image_data_url = bytes_to_data_url(image_bytes, suffix=image_path.suffix or ".png")

                    await self.run_store.append_event(
                        run_id,
                        "image.generated",
                        f"ComfyUI finished iteration {global_iteration_index}.",
                        iteration=global_iteration_index,
                        judge_round=judge_round,
                        round_iteration=round_iteration,
                        image_path=str(image_path.resolve()),
                    )

                    snapshot = IterationSnapshot(
                        index=global_iteration_index,
                        judge_round=judge_round,
                        round_iteration=round_iteration,
                        prompt=plan.prompt,
                        negative_prompt=(
                            plan.negative_prompt
                            if workflow_mapping.negative_prompt is not None
                            else config.generation_defaults.negative_prompt
                        ),
                        width=(
                            plan.width
                            if workflow_mapping.width is not None
                            else config.generation_defaults.width
                        ),
                        height=(
                            plan.height
                            if workflow_mapping.height is not None
                            else config.generation_defaults.height
                        ),
                        steps=(
                            plan.steps
                            if workflow_mapping.steps is not None
                            else config.generation_defaults.steps
                        ),
                        cfg_scale=(
                            plan.cfg_scale
                            if workflow_mapping.cfg_scale is not None
                            else config.generation_defaults.cfg_scale
                        ),
                        seed=(
                            plan.seed if workflow_mapping.seed is not None else config.generation_defaults.seed
                        ),
                        self_critique=plan.self_critique,
                        notes_to_judge=plan.notes_to_judge,
                        image_path=str(image_path.resolve()),
                        image_data_url=image_data_url,
                    )
                    snapshots.append(snapshot)
                    round_candidates.append(snapshot)
                    await self.run_store.append_iteration(run_id, snapshot)

                    if judge_client is None:
                        if plan.is_satisfied and global_iteration_index >= config.loop.min_iterations_before_self_stop:
                            accepted = True
                            stop_reason = "agent_self_stopped"
                            snapshot = snapshot.model_copy(update={"selected_as_frontier": True})
                            await replace_snapshot(snapshot)
                            frontier_snapshot = snapshot
                            break
                    elif (
                        plan.ready_for_judge
                        and round_iteration >= config.loop.min_agent_iterations_before_judge
                    ):
                        requested_judge_early = True
                        await self.run_store.append_event(
                            run_id,
                            "agent.ready_for_judge",
                            (
                                f"Agent requested judge review after iteration {global_iteration_index} "
                                f"in judge round {judge_round}."
                            ),
                            iteration=global_iteration_index,
                            judge_round=judge_round,
                            round_iteration=round_iteration,
                        )
                        break

                if accepted:
                    break

                if judge_client is None:
                    continue

                if not round_candidates:
                    continue

                if len(round_candidates) == 1:
                    selection_rationale = (
                        "Only one candidate was generated in this judge round."
                        if requested_judge_early
                        else "Only one candidate was available for judge review."
                    )
                    selected_snapshot = round_candidates[0].model_copy(
                        update={
                            "selected_for_judge": True,
                            "selection_rationale": selection_rationale,
                        }
                    )
                else:
                    await self.run_store.append_event(
                        run_id,
                        "agent.selecting_candidate",
                        f"Agent is selecting the best candidate from judge round {judge_round}.",
                        judge_round=judge_round,
                        candidate_iterations=[candidate.index for candidate in round_candidates],
                    )
                    selection_messages = build_agent_selection_messages(
                        config,
                        round_candidates,
                        reference_data_urls,
                        judge_round=judge_round,
                        max_judge_rounds=judge_round_limit,
                        frontier=frontier_snapshot,
                    )
                    selection_text, _ = await agent_client.complete(selection_messages)
                    selection = parse_agent_selection(selection_text, round_candidates)
                    chosen = next(
                        candidate
                        for candidate in round_candidates
                        if candidate.index == selection.selected_iteration_index
                    )
                    selected_snapshot = chosen.model_copy(
                        update={
                            "selected_for_judge": True,
                            "selection_rationale": selection.rationale,
                            "notes_to_judge": selection.notes_to_judge or chosen.notes_to_judge,
                        }
                    )

                if frontier_snapshot is not None and frontier_snapshot.selected_as_frontier:
                    demoted_frontier = frontier_snapshot.model_copy(update={"selected_as_frontier": False})
                    frontier_snapshot = demoted_frontier
                    await replace_snapshot(demoted_frontier)

                selected_snapshot = selected_snapshot.model_copy(update={"selected_as_frontier": True})
                await replace_snapshot(selected_snapshot)
                frontier_snapshot = selected_snapshot

                await self.run_store.append_event(
                    run_id,
                    "agent.selected_candidate",
                    f"Agent selected iteration {selected_snapshot.index} for judge review.",
                    iteration=selected_snapshot.index,
                    judge_round=judge_round,
                    round_iteration=selected_snapshot.round_iteration,
                    selection_rationale=selected_snapshot.selection_rationale,
                )

                await self.run_store.append_event(
                    run_id,
                    "judge.requested",
                    f"Requesting judge evaluation for iteration {selected_snapshot.index}.",
                    iteration=selected_snapshot.index,
                    judge_round=judge_round,
                    round_iteration=selected_snapshot.round_iteration,
                    provider_label=config.providers["judge"].label,
                    provider_model=config.providers["judge"].model,
                    provider_base_url=config.providers["judge"].base_url,
                    judge_text_history_turns=config.loop.judge_text_history_turns,
                    judge_image_history_turns=config.loop.judge_image_history_turns,
                    max_judge_rounds=judge_round_limit,
                )
                judge_messages = build_judge_messages(
                    config,
                    [item for item in snapshots if item.index != selected_snapshot.index],
                    selected_snapshot,
                    reference_data_urls,
                    judge_round=judge_round,
                    max_judge_rounds=judge_round_limit,
                )
                judge_text, _ = await judge_client.complete(judge_messages)
                judge_result = parse_judge_result(judge_text)

                selected_snapshot = selected_snapshot.model_copy(
                    update={
                        "judge_accept": judge_result.accept,
                        "judge_feedback": judge_result.feedback,
                        "judge_score": judge_result.score,
                        "judge_must_fix": judge_result.must_fix,
                    }
                )
                await replace_snapshot(selected_snapshot)
                frontier_snapshot = selected_snapshot

                await self.run_store.append_event(
                    run_id,
                    "judge.completed",
                    f"Judge evaluated iteration {selected_snapshot.index}.",
                    iteration=selected_snapshot.index,
                    judge_round=judge_round,
                    round_iteration=selected_snapshot.round_iteration,
                    accept=judge_result.accept,
                    score=judge_result.score,
                    feedback=judge_result.feedback,
                    must_fix=judge_result.must_fix,
                )

                if judge_result.accept:
                    accepted = True
                    stop_reason = "judge_accepted"
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
