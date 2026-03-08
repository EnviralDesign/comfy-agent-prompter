from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime

from comfy_agent_prompter.models import IterationSnapshot, RunDetail, RunEvent, RunSummary


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunDetail] = {}
        self._subscribers: dict[str, set[asyncio.Queue[dict]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def create_run(self, run_id: str, objective: str, config_path: str) -> RunDetail:
        async with self._lock:
            now = datetime.now(UTC)
            detail = RunDetail(
                run_id=run_id,
                status="queued",
                created_at=now,
                updated_at=now,
                objective=objective,
                config_path=config_path,
            )
            self._runs[run_id] = detail
            return detail

    async def list_runs(self) -> list[RunSummary]:
        async with self._lock:
            return [
                RunSummary.model_validate(detail.model_dump())
                for detail in sorted(self._runs.values(), key=lambda item: item.created_at, reverse=True)
            ]

    async def get_run(self, run_id: str) -> RunDetail | None:
        async with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else RunDetail.model_validate(run.model_dump())

    async def update_run(self, run_id: str, **fields: object) -> None:
        async with self._lock:
            current = self._runs[run_id]
            self._runs[run_id] = current.model_copy(
                update={"updated_at": datetime.now(UTC), **fields}
            )

    async def append_event(self, run_id: str, event_type: str, message: str, **data: object) -> None:
        event = RunEvent(timestamp=datetime.now(UTC), type=event_type, message=message, data=data)
        async with self._lock:
            current = self._runs[run_id]
            self._runs[run_id] = current.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "events": [*current.events, event],
                }
            )

        await self._broadcast(run_id, {"type": "event", "event": event.model_dump(mode="json")})

    async def append_iteration(self, run_id: str, snapshot: IterationSnapshot) -> None:
        async with self._lock:
            current = self._runs[run_id]
            iterations = [*current.iterations, snapshot]
            judge_count = sum(1 for item in iterations if item.selected_for_judge)
            self._runs[run_id] = current.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "iterations": iterations,
                    "iteration_count": len(iterations),
                    "judge_count": judge_count,
                    "final_image_path": snapshot.image_path,
                }
            )

        await self._broadcast(
            run_id,
            {"type": "iteration", "iteration": snapshot.model_dump(mode="json")},
        )

    async def update_iteration(self, run_id: str, snapshot: IterationSnapshot) -> None:
        async with self._lock:
            current = self._runs[run_id]
            iterations = [
                snapshot if item.index == snapshot.index else item
                for item in current.iterations
            ]
            judge_count = sum(1 for item in iterations if item.selected_for_judge)
            self._runs[run_id] = current.model_copy(
                update={
                    "updated_at": datetime.now(UTC),
                    "iterations": iterations,
                    "judge_count": judge_count,
                    "final_image_path": iterations[-1].image_path if iterations else current.final_image_path,
                }
            )

        await self._broadcast(
            run_id,
            {"type": "iteration", "iteration": snapshot.model_dump(mode="json")},
        )

    async def subscribe(self, run_id: str) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        async with self._lock:
            self._subscribers[run_id].add(queue)
        return queue

    async def unsubscribe(self, run_id: str, queue: asyncio.Queue[dict]) -> None:
        async with self._lock:
            self._subscribers[run_id].discard(queue)

    async def _broadcast(self, run_id: str, payload: dict) -> None:
        for queue in list(self._subscribers.get(run_id, set())):
            await queue.put(payload)
