from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from comfy_agent_prompter.json_utils import extract_json_object
from comfy_agent_prompter.models import (
    AgentPlan,
    AgentSelectionDecision,
    AppConfig,
    IterationSnapshot,
    JudgeResult,
)

DEFAULT_AGENT_SYSTEM_PROMPT = """
You are a junior prompt engineer and technical artist controlling a text-to-image workflow.
You are imaginative, methodical, and hungry to impress a seasoned art director.
You enjoy exploration, troubleshooting, and discovering which prompt components actually move the image closer to the target.
Your job is to iteratively improve the generated image until it matches the objective and quality bar.
Use the reference images as targets, not as vague inspiration.
You operate in judge rounds and inner exploration tries.
Within a judge round, you are free to explore locally before escalating back to the judge.
Think like a technically-minded studio artist:
- Earlier judge rounds should make coarse silhouette, composition, pose, and mood moves.
- Later judge rounds should make finer corrective moves and preserve confirmed wins.
- Inside one judge round, test hypotheses instead of just paraphrasing the last judge feedback.
- Preserve what is already working and isolate which prompt changes help versus hurt.
- Start broad and simple, then layer additions iteratively.
- Do not change many variables at once unless you are intentionally resetting from a bad local optimum.
- If stuck, temporarily reduce or simplify the prompt to identify which phrases are helping versus distracting the image model.
- Treat prompt wording like troubleshooting: change a small cluster of ideas, observe the effect, then keep or discard that change.
- Compare candidate prompts against each other like experiments, not like a single linear essay.
- Keep a curious, exploratory mindset. When results are weak, keep probing instead of rushing to the judge.

The inner-try numbers have three meanings:
- Minimum inner tries before judge handoff: earliest legal handoff, not a recommendation to stop.
- Target inner tries before judge handoff: the normal depth of local exploration you should aim for before escalating.
- Hard cap inner tries before judge handoff: the maximum local tries allowed in this judge round.

Do not describe reaching the target depth as "budget exhausted." The budget is only exhausted at the hard cap.
Set `ready_for_judge` to true only when one of these is true:
- `candidate_strong`: you have a candidate that is genuinely worth presenting now.
- `blocked`: the local search is genuinely stuck and further nearby prompt changes are unlikely to help.
- `plateau`: you explored to at least the target depth and the local improvements have flattened.
- `hard_cap_reached`: you are at the final allowed inner try for this round.

If the current results are still obviously subpar and the hard cap has not been reached, prefer more local exploration.
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
- ready_for_judge: boolean
- handoff_reason: "candidate_strong" | "plateau" | "blocked" | "hard_cap_reached" | null
- self_critique: string
- notes_to_judge: string
""".strip()

DEFAULT_AGENT_SELECTION_SYSTEM_PROMPT = """
You are selecting the best candidate from the agent's local exploration batch.
Pick the one candidate that should move forward to the judge.
Prefer candidates that preserve existing wins, fix the highest-leverage issues, and avoid regressions.
Do not average candidates together. Choose one actual candidate from the batch.
Respond with a single JSON object only.

Required JSON fields:
- selected_iteration_index: integer
- rationale: string
- notes_to_judge: string
""".strip()

DEFAULT_JUDGE_SYSTEM_PROMPT = """
You are a seasoned art director judging iterative image work.
You have strong taste, broad visual memory, and a calm, demanding studio-director mindset.
Before deciding, ruminate on the problem like an art director reviewing a junior artist's explorations:
composition, silhouette, pose, value hierarchy, material read, lighting, atmosphere, stylization, and whether the work
is converging toward the brief or drifting into an easier but wrong local optimum.
You do not optimize for kindness or shortness. You decide whether the image meets the objective.
Be strict and compare the candidate image against the quality bar and any reference image.
You are not grading in isolation. Use the recent trajectory to detect local optima, regressions, repeated mistakes,
and whether the prompting agent is wasting iterations.
Act like a demanding but useful art director: preserve momentum, call out regressions clearly, and focus feedback on the
highest-leverage fixes given the remaining iteration budget.
Favor clear art-direction feedback over vague praise. Tell the agent what to preserve, what to cut, and what to push.
When the agent appears stuck, take a more guided approach: offer concrete ideas for prompt changes, structural simplifications,
prompt-component substitutions, negative prompt moves, or search-strategy resets that could unstick the next round.
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
    *,
    judge_round: int,
    max_judge_rounds: int,
    round_iteration: int,
    frontier: IterationSnapshot | None,
) -> list[dict[str, Any]]:
    recent_text = iterations[-config.loop.agent_text_history_turns :]
    recent_images = iterations[-config.loop.agent_image_history_turns :]
    remaining_rounds = max(max_judge_rounds - judge_round, 0)

    text_lines = [
        f"Objective: {config.task.objective}",
        f"Quality bar: {config.task.quality_bar}",
        f"Current generation number: {len(iterations) + 1}",
        f"Current judge round: {judge_round} of {max_judge_rounds}",
        f"Remaining judge rounds after this one: {remaining_rounds}",
        (
            "Current inner exploration try: "
            f"{round_iteration} of {config.loop.max_agent_iterations_per_round}"
        ),
        (
            "Minimum inner tries before judge handoff: "
            f"{config.loop.min_agent_iterations_before_judge}"
        ),
        (
            "Target inner tries before judge handoff: "
            f"{config.loop.target_agent_iterations_per_round}"
        ),
        (
            "Hard cap inner tries before judge handoff: "
            f"{config.loop.max_agent_iterations_per_round}"
        ),
    ]

    if frontier is not None:
        text_lines.extend(
            [
                "Current frontier from the last judged handoff:",
                (
                    f"- iteration={frontier.index}, judge_round={frontier.judge_round}, "
                    f"judge_score={frontier.judge_score}, selected_for_judge={frontier.selected_for_judge}"
                ),
                f"- frontier prompt={frontier.prompt!r}",
                f"- frontier judge_feedback={frontier.judge_feedback or 'n/a'}",
                f"- frontier must_fix={frontier.judge_must_fix or []}",
            ]
        )

    current_round_candidates = [item for item in iterations if item.judge_round == judge_round]
    if current_round_candidates:
        text_lines.append("Current round exploration so far:")
        for item in current_round_candidates:
            text_lines.append(
                f"- Iteration {item.index} / try {item.round_iteration}: "
                f"prompt={item.prompt!r}; self_critique={item.self_critique!r}"
            )

    if recent_text:
        text_lines.append("Recent iteration summaries:")
        for item in recent_text:
            judge_line = (
                (
                    f"judge_accept={item.judge_accept}, judge_score={item.judge_score}, "
                    f"judge_feedback={item.judge_feedback}, judge_must_fix={item.judge_must_fix}"
                )
                if item.judge_feedback or item.judge_accept is not None
                else "judge not used yet"
            )
            text_lines.append(
                f"- Iteration {item.index} (judge_round={item.judge_round}, round_try={item.round_iteration}): "
                f"prompt={item.prompt!r}; self_critique={item.self_critique!r}; {judge_line}"
            )
    else:
        text_lines.append("No prior iterations exist yet.")

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_lines)}]

    for index, data_url in enumerate(reference_data_urls, start=1):
        content.append({"type": "text", "text": f"Reference image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    if frontier is not None and frontier.image_data_url:
        content.append({"type": "text", "text": f"Current frontier image from iteration {frontier.index}:"})
        content.append({"type": "image_url", "image_url": {"url": frontier.image_data_url}})

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


def build_agent_selection_messages(
    config: AppConfig,
    round_candidates: list[IterationSnapshot],
    reference_data_urls: list[str],
    *,
    judge_round: int,
    max_judge_rounds: int,
    frontier: IterationSnapshot | None,
) -> list[dict[str, Any]]:
    remaining_rounds = max(max_judge_rounds - judge_round, 0)
    text_lines = [
        f"Objective: {config.task.objective}",
        f"Quality bar: {config.task.quality_bar}",
        f"Current judge round: {judge_round} of {max_judge_rounds}",
        f"Remaining judge rounds after this one: {remaining_rounds}",
        "Choose the single best candidate from this batch to present to the judge.",
    ]

    if frontier is not None:
        text_lines.extend(
            [
                "Current judged frontier to beat or preserve:",
                f"- frontier iteration={frontier.index}, judge_score={frontier.judge_score}",
                f"- frontier judge_feedback={frontier.judge_feedback or 'n/a'}",
                f"- frontier must_fix={frontier.judge_must_fix or []}",
            ]
        )

    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(text_lines)}]

    for index, data_url in enumerate(reference_data_urls, start=1):
        content.append({"type": "text", "text": f"Reference image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    for candidate in round_candidates:
        content.append(
            {
                "type": "text",
                "text": "\n".join(
                    [
                        f"Candidate iteration {candidate.index} (inner try {candidate.round_iteration}):",
                        f"Prompt: {candidate.prompt}",
                        f"Self critique: {candidate.self_critique or 'n/a'}",
                        f"Notes to judge: {candidate.notes_to_judge or 'n/a'}",
                    ]
                ),
            }
        )
        if candidate.image_data_url:
            content.append({"type": "image_url", "image_url": {"url": candidate.image_data_url}})

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": DEFAULT_AGENT_SELECTION_SYSTEM_PROMPT,
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def build_judge_messages(
    config: AppConfig,
    iterations: list[IterationSnapshot],
    candidate: IterationSnapshot,
    reference_data_urls: list[str],
    *,
    judge_round: int,
    max_judge_rounds: int,
) -> list[dict[str, Any]]:
    remaining_rounds = max(max_judge_rounds - judge_round, 0)
    recent_text = iterations[-config.loop.judge_text_history_turns :]
    recent_images = iterations[-config.loop.judge_image_history_turns :]

    text_lines = [
        f"Objective: {config.task.objective}",
        f"Quality bar: {config.task.quality_bar}",
        f"Current judge round: {judge_round}",
        f"Maximum judge rounds allowed: {max_judge_rounds}",
        f"Remaining judge rounds after this decision: {remaining_rounds}",
        f"Selected candidate iteration: {candidate.index}",
        f"Selected candidate inner try: {candidate.round_iteration}",
        f"Agent prompt: {candidate.prompt}",
        f"Agent self critique: {candidate.self_critique or 'n/a'}",
        f"Agent notes to judge: {candidate.notes_to_judge or 'n/a'}",
        f"Agent selection rationale: {candidate.selection_rationale or 'n/a'}",
    ]

    if recent_text:
        text_lines.append("Recent trajectory summaries:")
        for item in recent_text:
            judge_line = (
                f"accepted={item.judge_accept}, score={item.judge_score}, feedback={item.judge_feedback}"
                if item.judge_feedback or item.judge_accept is not None
                else "judge not used yet"
            )
            text_lines.append(
                f"- Iteration {item.index}: prompt={item.prompt!r}; "
                f"self_critique={item.self_critique!r}; {judge_line}"
            )
    else:
        text_lines.append("No prior iteration history exists yet.")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "\n".join(text_lines),
        }
    ]

    for index, data_url in enumerate(reference_data_urls, start=1):
        content.append({"type": "text", "text": f"Reference image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    for item in recent_images:
        if item.image_data_url and item.index != candidate.index:
            content.append({"type": "text", "text": f"Recent generated image from iteration {item.index}:"})
            content.append({"type": "image_url", "image_url": {"url": item.image_data_url}})

    content.append({"type": "text", "text": "Candidate image:"})
    content.append({"type": "image_url", "image_url": {"url": candidate.image_data_url}})

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


def parse_agent_selection(raw_text: str, round_candidates: list[IterationSnapshot]) -> AgentSelectionDecision:
    payload = AgentSelectionDecision.model_validate_json(extract_json_object(raw_text))
    valid_indexes = {candidate.index for candidate in round_candidates}
    if payload.selected_iteration_index not in valid_indexes:
        raise ValueError(
            f"Agent selection referenced iteration {payload.selected_iteration_index}, "
            f"but only {sorted(valid_indexes)} were available."
        )
    return payload


def parse_judge_result(raw_text: str) -> JudgeResult:
    try:
        return JudgeResult.model_validate_json(extract_json_object(raw_text))
    except ValidationError as exc:
        raise ValueError(f"Judge response could not be parsed: {exc}") from exc
