from __future__ import annotations

from comfy_agent_prompter.orchestration.runner import OrchestrationRunner
from comfy_agent_prompter.services.run_store import RunStore

run_store = RunStore()
runner = OrchestrationRunner(run_store)

