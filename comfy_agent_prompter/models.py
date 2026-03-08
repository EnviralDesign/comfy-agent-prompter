from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelEndpointConfig(BaseModel):
    label: str
    base_url: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float = 0.4
    max_tokens: int = 1200
    response_format: Literal["text", "json_object"] = "json_object"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    system_prompt: str | None = None


class WorkflowInputBinding(BaseModel):
    node_id: str
    input_name: str
    value_mode: Literal["raw", "filename", "subfolder_filename"] = "raw"


class WorkflowMapping(BaseModel):
    positive_prompt: WorkflowInputBinding
    negative_prompt: WorkflowInputBinding | None = None
    width: WorkflowInputBinding | None = None
    height: WorkflowInputBinding | None = None
    steps: WorkflowInputBinding | None = None
    cfg_scale: WorkflowInputBinding | None = None
    seed: WorkflowInputBinding | None = None
    filename_prefix: WorkflowInputBinding | None = None
    reference_image: WorkflowInputBinding | None = None


class GenerationDefaults(BaseModel):
    width: int = 1024
    height: int = 1024
    steps: int = 28
    cfg_scale: float = 4.5
    negative_prompt: str = ""
    seed: int | None = None
    filename_prefix: str = "cap"


class LoopConfig(BaseModel):
    enable_judge: bool = True
    max_iterations: int = 8
    max_judge_rounds: int | None = None
    max_agent_iterations_per_round: int = 3
    min_agent_iterations_before_judge: int = 1
    agent_text_history_turns: int = 4
    agent_image_history_turns: int = 2
    judge_text_history_turns: int = 3
    judge_image_history_turns: int = 2
    min_iterations_before_self_stop: int = 2
    save_artifacts: bool = True


class ComfyUiConfig(BaseModel):
    base_url: str
    workflow_path: str
    mapping_path: str
    poll_interval_ms: int = 1500
    request_timeout_ms: int = 120000


class TaskConfig(BaseModel):
    objective: str
    quality_bar: str = "Match the requested composition and style cleanly, with obvious defects fixed."
    reference_image_paths: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    comfyui: ComfyUiConfig
    providers: dict[str, ModelEndpointConfig]
    loop: LoopConfig = Field(default_factory=LoopConfig)
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    task: TaskConfig


class AgentPlan(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    is_satisfied: bool = False
    ready_for_judge: bool = False
    self_critique: str | None = None
    notes_to_judge: str | None = None


class JudgeResult(BaseModel):
    accept: bool
    score: float | None = None
    feedback: str
    must_fix: list[str] = Field(default_factory=list)


class AgentSelectionDecision(BaseModel):
    selected_iteration_index: int
    rationale: str
    notes_to_judge: str | None = None


class RunEvent(BaseModel):
    timestamp: datetime
    type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class IterationSnapshot(BaseModel):
    index: int
    judge_round: int = 1
    round_iteration: int = 1
    prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    self_critique: str | None = None
    notes_to_judge: str | None = None
    selected_for_judge: bool = False
    selected_as_frontier: bool = False
    selection_rationale: str | None = None
    judge_accept: bool | None = None
    judge_feedback: str | None = None
    judge_score: float | None = None
    judge_must_fix: list[str] = Field(default_factory=list)
    image_path: str
    image_data_url: str | None = None


class RunSummary(BaseModel):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    accepted: bool = False
    objective: str
    config_path: str
    output_dir: str | None = None
    final_image_path: str | None = None
    iteration_count: int = 0
    judge_count: int = 0
    stop_reason: str | None = None
    error: str | None = None


class RunDetail(RunSummary):
    events: list[RunEvent] = Field(default_factory=list)
    iterations: list[IterationSnapshot] = Field(default_factory=list)
