# Comfy Agent Prompter

`comfy-agent-prompter` is a Python-first harness for iterative text-to-image prompting against ComfyUI.
It is built around two roles:

- A prompting agent that sees the target objective, optional reference images, and recent local history.
- A judge that evaluates each candidate image independently and either accepts it or sends it back with feedback.

This repository now includes:

- A shared orchestration core for CLI and web usage.
- A FastAPI server with a live browser UI.
- A batch CLI for benchmark-style runs.
- An OpenAI-compatible model transport that works with OpenRouter and LM Studio.
- A declarative ComfyUI workflow mapping layer so the real workflow can be swapped in without changing code.

## Current status

The app is bootstrapped end-to-end and now includes an initial real workflow:

- [`workflows/comfy_klein_image_gen_api_512.json`](/C:/repos/comfy-agent-prompter/workflows/comfy_klein_image_gen_api_512.json)
- [`workflows/comfy_klein_image_gen_api_512.mapping.json`](/C:/repos/comfy-agent-prompter/workflows/comfy_klein_image_gen_api_512.mapping.json)

There is still one important limitation with this specific workflow: it is text-to-image only. The agent and judge can both see external reference images, but this workflow itself does not yet take a reference image node as input.

## Why this structure

The visual UI is the primary surface right now, so the project is built as:

- `FastAPI` for the backend and live API/websocket layer.
- A small vanilla web UI for watching runs in real time.
- A shared runner so the same logic can be used for UI-driven runs and automated CLI benchmarks.

The loop keeps text history and image history as separate rolling windows. That matters because multimodal context is provider-specific and often much more expensive than plain text context. The current implementation does not replay the entire conversation back to the model. Instead it synthesizes recent local state so cheaper models can stay on task without carrying the full historical loop forever.

The runner now has two levels of control:

- Judge rounds: the expensive checkpoints where the judge evaluates progress.
- Inner agent tries: the local exploration steps the prompter can take before escalating back to the judge.

At the end of each judge round, the prompter selects the best candidate from its local batch and only that candidate is sent to the judge.

## Quick start

1. Copy [`.env.example`](/C:/repos/comfy-agent-prompter/.env.example) to `.env` and set `OPENROUTER_API_KEY` if you want to use OpenRouter.
2. Optionally set `CAP_PROMPTER_MODEL` and `CAP_JUDGE_MODEL` in `.env` if you want to override the JSON config's model IDs without editing the config file.
3. Update [`examples/config.example.json`](/C:/repos/comfy-agent-prompter/examples/config.example.json) with the provider base URLs and any reference image paths you want.
4. If you swap workflows later, update the `workflow_path` and `mapping_path` in that config file.
5. Run:

```powershell
uv sync
uv run cap serve
```

Then open [http://127.0.0.1:8765](http://127.0.0.1:8765).

By default the server binds to `0.0.0.0:8765`.

## CLI usage

Run a benchmark-style job:

```powershell
uv run cap run --config examples/config.example.json
```

Override the objective or reference list:

```powershell
uv run cap run --config examples/config.example.json --objective "Match the reference poster but make it rainy"
uv run cap run --config examples/config.example.json --reference C:\path\to\ref1.png --reference C:\path\to\ref2.png
```

Connectivity check:

```powershell
uv run cap doctor --config examples/config.example.json
```

## Web UI

The UI shows:

- Run list and status.
- A transcript-style conversation view for prompter, judge, and image outputs.
- A raw run trace for lower-level debugging.
- Summary stats such as iteration count, judge count, acceptance state, and stop reason.

## Configuration model

The config file is JSON and contains:

- `providers.agent`
- `providers.judge`
- `comfyui.workflow_path`
- `comfyui.mapping_path`
- `loop.*` tuning knobs
- `generation_defaults.*`
- `task.objective`
- `task.reference_image_paths`

Environment overrides:

- `CAP_PROMPTER_MODEL` overrides `providers.agent.model`
- `CAP_JUDGE_MODEL` overrides `providers.judge.model`

Important loop knobs:

- `loop.max_judge_rounds`
- `loop.max_agent_iterations_per_round`
- `loop.target_agent_iterations_per_round`
- `loop.min_agent_iterations_before_judge`
- `loop.agent_text_history_turns`
- `loop.agent_image_history_turns`
- `loop.judge_text_history_turns`
- `loop.judge_image_history_turns`

Seed behavior:

- If `generation_defaults.seed` is set, that exact seed is used for the whole run.
- If it is omitted, the harness chooses one random seed at run start and keeps it fixed for the whole run.

This keeps prompt exploration from being confounded by per-iteration seed changes.

Both providers are expected to expose OpenAI-compatible endpoints. For the first slice this means:

- LM Studio: local `base_url` like `http://127.0.0.1:1234/v1`
- OpenRouter: `https://openrouter.ai/api/v1`

## Workflow mapping

The workflow mapping file tells the harness which ComfyUI node inputs can be changed by the prompting agent.
That keeps the code generic and avoids hardcoding node IDs or assumptions about one graph.

Supported fields in the mapping right now:

- `positive_prompt`
- `negative_prompt`
- `width`
- `height`
- `steps`
- `cfg_scale`
- `seed`
- `filename_prefix`
- `reference_image`

The `reference_image` path currently assumes the workflow reads a single uploaded image input. The current Klein workflow does not expose that yet, so reference images are only used by the prompting agent and the judge for evaluation and steering.

## Next handoff needed from you

The next useful pass from here is:

- Validate one complete run locally against your actual ComfyUI setup.
- Confirm the model IDs you want for the agent and judge.
- Decide whether we want to keep this workflow text-only for now or move immediately to a workflow that accepts a reference image input node.
