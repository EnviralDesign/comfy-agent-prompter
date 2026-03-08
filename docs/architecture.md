# Architecture

## Goal

Support two modes over one shared core:

- Interactive visual runs where the user watches the loop in real time.
- Automated benchmark runs where the user mostly cares about stats and the final output.

## Core loop

1. Build agent context from:
   - Objective
   - Quality bar
   - Optional reference images
   - Recent text history
   - Recent generated images
2. Ask the prompting agent for a JSON generation plan.
3. Apply that plan to a ComfyUI workflow template via a mapping file.
4. Run ComfyUI and collect the output image.
5. If the judge is enabled, ask the judge to accept or reject the result.
6. Either stop or continue iterating.
7. Persist artifacts and emit live events.

## Stack

- Backend: FastAPI
- CLI: Typer
- HTTP client: httpx
- Config/data models: Pydantic
- Packaging/runtime: uv
- Frontend: server-rendered shell plus vanilla JS with websocket updates

## Why separate text and image history windows

Multimodal context handling differs across providers and models.
Even when an API presents images cleanly as content parts, the real cost and context behavior are not uniform.
Because of that, the runner tracks:

- `agent_text_history_turns`
- `agent_image_history_turns`

This lets the system stay useful for cheap local models that benefit from a small local gradient of recent attempts without forcing full replay of every past image and every past message.

## Current limitations

- The workflow mapper currently targets one uploaded reference image at most.
- The ComfyUI client picks the first output image it finds.
- The live event store is in-memory and single-process.
- The UI is intentionally minimal and optimized for visibility, not polish or auth.

## Near-term phases

### Phase 1

Bootstrap the platform:

- Shared runner
- FastAPI app
- CLI
- OpenAI-compatible providers
- ComfyUI workflow mapping
- Live UI

### Phase 2

Wire the real workflow and validate one successful run:

- Replace placeholder workflow
- Tune mapping against actual node IDs
- Verify LM Studio and OpenRouter model behavior
- Refine prompts and stop conditions

### Phase 3

Make the benchmark mode more serious:

- Structured run summaries
- CSV or JSONL exports
- Repeated runs across prompt/judge/provider combinations
- Better aggregate statistics

### Phase 4

Expand workflow support:

- Multiple reference images
- ControlNet and other extra inputs
- Alternative workflow families
- Better image selection when multiple outputs are generated

