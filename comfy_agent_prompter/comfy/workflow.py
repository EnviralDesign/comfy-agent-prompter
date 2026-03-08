from __future__ import annotations

from copy import deepcopy
from typing import Any

from comfy_agent_prompter.models import AgentPlan, GenerationDefaults, WorkflowInputBinding, WorkflowMapping


def apply_workflow_mapping(
    workflow: dict[str, Any],
    mapping: WorkflowMapping,
    plan: AgentPlan,
    defaults: GenerationDefaults,
    reference_image_value: str | None,
) -> dict[str, Any]:
    prompt = deepcopy(workflow)

    _set_value(prompt, mapping.positive_prompt, plan.prompt)

    if mapping.negative_prompt is not None:
        _set_value(prompt, mapping.negative_prompt, plan.negative_prompt or defaults.negative_prompt)

    if mapping.width is not None and plan.width is not None:
        _set_value(prompt, mapping.width, plan.width)

    if mapping.height is not None and plan.height is not None:
        _set_value(prompt, mapping.height, plan.height)

    if mapping.steps is not None and plan.steps is not None:
        _set_value(prompt, mapping.steps, plan.steps)

    if mapping.cfg_scale is not None and plan.cfg_scale is not None:
        _set_value(prompt, mapping.cfg_scale, plan.cfg_scale)

    if mapping.seed is not None:
        seed_value = plan.seed if plan.seed is not None else defaults.seed
        if seed_value is not None:
            _set_value(prompt, mapping.seed, seed_value)

    if mapping.filename_prefix is not None:
        _set_value(prompt, mapping.filename_prefix, defaults.filename_prefix)

    if mapping.reference_image is not None and reference_image_value is not None:
        _set_value(prompt, mapping.reference_image, reference_image_value)

    return prompt


def render_uploaded_value(binding: WorkflowInputBinding, upload_payload: dict[str, str]) -> str:
    if binding.value_mode == "filename":
        return upload_payload["name"]
    if binding.value_mode == "subfolder_filename":
        subfolder = upload_payload.get("subfolder", "")
        return f"{subfolder}/{upload_payload['name']}" if subfolder else upload_payload["name"]
    return upload_payload["name"]


def _set_value(workflow: dict[str, Any], binding: WorkflowInputBinding, value: Any) -> None:
    node = workflow.get(binding.node_id)
    if node is None:
        raise KeyError(f"Workflow node {binding.node_id!r} was not found.")

    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise KeyError(f"Workflow node {binding.node_id!r} did not expose an inputs object.")

    inputs[binding.input_name] = value
