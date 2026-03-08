from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from comfy_agent_prompter.json_utils import extract_json_object
from comfy_agent_prompter.models import AgentPlan, AppConfig, IterationSnapshot, JudgeResult

DEFAULT_AGENT_SYSTEM_PROMPT = """
You are an image prompting agent controlling a text-to-image workflow.
Your job is to iteratively improve the generated image until it matches the objective and quality bar.
Use the reference images as targets, not as vague inspiration.
Respond with a single JSON object only.

Required JSON fields:
- prompt: string
- negative_prompt: string
- width: integer
- height: integer
- steps: integer
- cfg_scale: number
- seed: integer or null
- is_satisfied: boolean
- self_critique: string
- notes_to_judge: string
""".strip()

DEFAULT_JUDGE_SYSTEM_PROMPT = """
You are an impartial image quality judge.
You do not optimize for kindness or shortness. You decide whether the image meets the objective.
Be strict and compare the candidate image against the quality bar and any reference image.
Respond with a single JSON object only.

Required JSON fields:
- accept: boolean
- score: number between 0 and 1
- feedback: string
- must_fix: string[]
""".strip()


def build_agent_messages(
    config: AppConfig,
    iterations: list[IterationSnapshot],
    reference_data_urls: list[str],
) -> list[dict[str, Any]]:
    recent_text = iterations[-config.loop.agent_text_history_turns :]
    recent_images = iterations[-config.loop.agent_image_history_turns :]

    text_lines = [
        f"Objective: {config.task.objective}",
        f"Quality bar: {config.task.quality_bar}",
        f"Current iteration number: {len(iterations) + 1}",
    ]

    if recent_text:
        text_lines.append("Recent iteration summaries:")
        for item in recent_text:
            judge_line = (
                f"judge_accept={item.judge_accept}, judge_feedback={item.judge_feedback}"
                if item.judge_feedback
                else "judge not used yet"
            )
            text_lines.append(
                f"- Iteration {item.index}: prompt={item.prompt!r}; "
                f"self_critique={item.self_critique!r}; {judge_line}"
            )
    else:
        text_lines.append("No prior iterations exist yet.")

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_lines)}]

    for index, data_url in enumerate(reference_data_urls, start=1):
        content.append({"type": "text", "text": f"Reference image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    for item in recent_images:
        if item.image_data_url:
            content.append({"type": "text", "text": f"Recent generated image from iteration {item.index}:"})
            content.append({"type": "image_url", "image_url": {"url": item.image_data_url}})

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": config.providers["agent"].system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT,
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def build_judge_messages(
    config: AppConfig,
    plan: AgentPlan,
    generated_image_data_url: str,
    reference_data_urls: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "\n".join(
                [
                    f"Objective: {config.task.objective}",
                    f"Quality bar: {config.task.quality_bar}",
                    f"Agent prompt: {plan.prompt}",
                    f"Agent self critique: {plan.self_critique or 'n/a'}",
                    f"Agent notes to judge: {plan.notes_to_judge or 'n/a'}",
                ]
            ),
        }
    ]

    for index, data_url in enumerate(reference_data_urls, start=1):
        content.append({"type": "text", "text": f"Reference image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    content.append({"type": "text", "text": "Candidate image:"})
    content.append({"type": "image_url", "image_url": {"url": generated_image_data_url}})

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": config.providers["judge"].system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT,
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def parse_agent_plan(raw_text: str, config: AppConfig) -> AgentPlan:
    payload = AgentPlan.model_validate_json(extract_json_object(raw_text))
    return payload.model_copy(
        update={
            "negative_prompt": payload.negative_prompt or config.generation_defaults.negative_prompt,
            "width": payload.width or config.generation_defaults.width,
            "height": payload.height or config.generation_defaults.height,
            "steps": payload.steps or config.generation_defaults.steps,
            "cfg_scale": payload.cfg_scale or config.generation_defaults.cfg_scale,
        }
    )


def parse_judge_result(raw_text: str) -> JudgeResult:
    try:
        return JudgeResult.model_validate_json(extract_json_object(raw_text))
    except ValidationError as exc:
        raise ValueError(f"Judge response could not be parsed: {exc}") from exc

