const state = {
  selectedRunId: null,
  socket: null,
  configMap: new Map(),
};

async function fetchConfigs() {
  const response = await fetch("/api/configs");
  const configs = await response.json();
  renderConfigOptions(configs);
}

async function fetchRuns() {
  const response = await fetch("/api/runs");
  const runs = await response.json();
  renderRunList(runs);
  if (!state.selectedRunId && runs.length > 0) {
    await selectRun(runs[0].run_id);
  }
}

function renderConfigOptions(configs) {
  state.configMap = new Map(configs.map((config) => [config.path, config]));
  const select = document.querySelector("#config-path");
  const previous = window.localStorage.getItem("cap:selected-config");

  select.innerHTML = "";

  for (const config of configs) {
    const option = document.createElement("option");
    option.value = config.path;
    option.textContent = config.path;
    select.appendChild(option);
  }

  const nextValue =
    (previous && state.configMap.has(previous) && previous) ||
    (configs.length > 0 ? configs[0].path : "");

  if (nextValue) {
    select.value = nextValue;
    applyConfigDefaults(nextValue);
  }
}

function applyConfigDefaults(configPath) {
  const config = state.configMap.get(configPath);
  if (!config) {
    return;
  }

  window.localStorage.setItem("cap:selected-config", configPath);
  document.querySelector("#objective-override").value = config.objective || "";
  document.querySelector("#reference-paths").value = (config.reference_image_paths || []).join(", ");
}

function renderRunList(runs) {
  const container = document.querySelector("#run-list");
  container.innerHTML = "";

  if (runs.length === 0) {
    container.innerHTML = `<p class="empty">No runs yet.</p>`;
    return;
  }

  for (const run of runs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `run-card ${run.run_id === state.selectedRunId ? "active" : ""}`;
    button.innerHTML = `
      <span class="run-card-top">
        <strong>${run.run_id}</strong>
        <span class="status status-${run.status}">${run.status}</span>
      </span>
      <span class="run-card-body">${run.objective}</span>
      <span class="run-card-footer">${run.iteration_count} iterations • ${run.judge_count} judge calls</span>
    `;
    button.addEventListener("click", () => selectRun(run.run_id));
    container.appendChild(button);
  }
}

async function selectRun(runId) {
  state.selectedRunId = runId;
  const response = await fetch(`/api/runs/${runId}`);
  const run = await response.json();
  renderRun(run);
  connectSocket(runId);
  await fetchRuns();
}

function connectSocket(runId) {
  if (state.socket) {
    state.socket.close();
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  state.socket = new WebSocket(`${protocol}://${window.location.host}/ws/runs/${runId}`);

  state.socket.addEventListener("message", async (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "snapshot") {
      renderRun(payload.run);
      return;
    }

    const response = await fetch(`/api/runs/${runId}`);
    const run = await response.json();
    renderRun(run);
    await fetchRuns();
  });
}

function renderRun(run) {
  document.querySelector("#run-title").textContent = `${run.run_id} • ${run.status}`;
  document.querySelector("#run-meta").textContent = `${run.objective} • ${run.config_path}`;

  const stats = [
    ["Accepted", String(run.accepted)],
    ["Iterations", String(run.iteration_count)],
    ["Judge calls", String(run.judge_count)],
    ["Stop reason", run.stop_reason || "n/a"],
  ];

  const statGrid = document.querySelector("#stat-grid");
  statGrid.innerHTML = stats
    .map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");

  renderTranscript(run);
  renderTrace(run);
}

function renderTranscript(run) {
  const transcript = document.querySelector("#transcript");
  const messages = buildTranscript(run);

  transcript.innerHTML = "";
  if (messages.length === 0) {
    transcript.innerHTML = `<p class="empty">No conversation yet.</p>`;
    return;
  }

  for (const message of messages) {
    const item = document.createElement("article");
    item.className = `message message-${message.role}`;

    const body = [];
    if (message.title) {
      body.push(`<div class="message-title">${escapeHtml(message.title)}</div>`);
    }
    if (message.text) {
      body.push(`<p>${escapeHtml(message.text).replaceAll("\n", "<br>")}</p>`);
    }
    if (message.imageDataUrl) {
      body.push(
        `<img src="${message.imageDataUrl}" alt="${escapeHtml(message.imageAlt || "Generated image")}" />`,
      );
    }
    if (message.meta && message.meta.length > 0) {
      body.push(
        `<div class="message-meta">${message.meta.map((entry) => `<span>${escapeHtml(entry)}</span>`).join("")}</div>`,
      );
    }

    item.innerHTML = `
      <div class="message-top">
        <strong>${escapeHtml(message.speaker)}</strong>
        <span>${new Date(message.timestamp).toLocaleTimeString()}</span>
      </div>
      <div class="message-body">
        ${body.join("")}
      </div>
    `;
    transcript.appendChild(item);
  }
}

function renderTrace(run) {
  const eventLog = document.querySelector("#events");
  eventLog.innerHTML = "";

  for (const event of run.events.slice().reverse()) {
    const item = document.createElement("article");
    item.className = "event";
    item.innerHTML = `
      <div class="event-top">
        <strong>${event.type}</strong>
        <span>${new Date(event.timestamp).toLocaleTimeString()}</span>
      </div>
      <p>${escapeHtml(event.message)}</p>
      <pre>${escapeHtml(JSON.stringify(event.data, null, 2))}</pre>
    `;
    eventLog.appendChild(item);
  }
}

function buildTranscript(run) {
  const messages = [
    {
      role: "system",
      speaker: "System",
      title: "Objective",
      text: run.objective,
      meta: [
        `config: ${run.config_path}`,
        `status: ${run.status}`,
      ],
      timestamp: run.created_at,
    },
  ];

  const eventTimes = new Map();
  for (const event of run.events) {
    const iteration = event.data?.iteration;
    if (iteration) {
      eventTimes.set(`${event.type}:${iteration}`, event.timestamp);
    }

    if (
      event.type === "run.failed" ||
      event.type === "run.finished" ||
      event.type === "judge.round.started" ||
      event.type === "agent.ready_for_judge" ||
      event.type === "agent.selecting_candidate" ||
      event.type === "agent.selected_candidate" ||
      event.type === "comfy.executing" ||
      event.type === "comfy.prompt_submitted" ||
      event.type === "comfy.waiting_for_output" ||
      event.type === "comfy.still_waiting"
    ) {
      messages.push({
        role: "system",
        speaker: "System",
        title: event.type,
        text: event.message,
        meta: formatEventMeta(event),
        timestamp: event.timestamp,
      });
    }
  }

  for (const iteration of run.iterations) {
    messages.push({
      role: "agent",
      speaker: "Prompter",
      title: `Round ${iteration.judge_round} • Try ${iteration.round_iteration} • Iteration ${iteration.index}`,
      text: iteration.prompt,
      meta: compactMeta([
        iteration.self_critique ? `self-critique: ${iteration.self_critique}` : null,
        iteration.notes_to_judge ? `notes to judge: ${iteration.notes_to_judge}` : null,
        iteration.selected_for_judge ? "selected for judge" : null,
        iteration.selected_as_frontier ? "frontier" : null,
        iteration.selection_rationale ? `selection: ${iteration.selection_rationale}` : null,
      ]),
      timestamp:
        eventTimes.get(`agent.planned:${iteration.index}`) ||
        eventTimes.get(`agent.requested:${iteration.index}`) ||
        run.updated_at,
    });

    if (iteration.image_data_url) {
      messages.push({
        role: "image",
        speaker: "ComfyUI",
        title: `Rendered image for iteration ${iteration.index}`,
        text: null,
        imageDataUrl: iteration.image_data_url,
        imageAlt: `Iteration ${iteration.index}`,
        meta: compactMeta([
          `round ${iteration.judge_round}`,
          `try ${iteration.round_iteration}`,
          iteration.image_path ? `saved: ${iteration.image_path}` : null,
        ]),
        timestamp:
          eventTimes.get(`image.generated:${iteration.index}`) ||
          run.updated_at,
      });
    }

    if (iteration.judge_feedback || iteration.judge_accept !== null) {
      messages.push({
        role: "judge",
        speaker: "Judge",
        title: `Iteration ${iteration.index} ${iteration.judge_accept ? "accepted" : "feedback"}`,
        text: iteration.judge_feedback || (iteration.judge_accept ? "Accepted." : "No feedback returned."),
        meta: compactMeta([
          iteration.judge_score !== null ? `score: ${iteration.judge_score}` : null,
          iteration.judge_accept !== null ? `accepted: ${iteration.judge_accept}` : null,
          iteration.judge_must_fix?.length ? `must-fix: ${iteration.judge_must_fix.join(" | ")}` : null,
        ]),
        timestamp:
          eventTimes.get(`judge.completed:${iteration.index}`) ||
          eventTimes.get(`judge.requested:${iteration.index}`) ||
          run.updated_at,
      });
    }
  }

  return messages.sort((left, right) => new Date(left.timestamp) - new Date(right.timestamp));
}

function formatEventMeta(event) {
  return compactMeta([
    event.data?.iteration ? `iteration: ${event.data.iteration}` : null,
    event.data?.judge_round ? `judge round: ${event.data.judge_round}` : null,
    event.data?.round_iteration ? `inner try: ${event.data.round_iteration}` : null,
    event.data?.provider_label ? `provider: ${event.data.provider_label}` : null,
    event.data?.provider_model ? `model: ${event.data.provider_model}` : null,
    event.data?.max_judge_rounds ? `max judge rounds: ${event.data.max_judge_rounds}` : null,
    event.data?.max_agent_iterations_per_round
      ? `max inner tries: ${event.data.max_agent_iterations_per_round}`
      : null,
    event.data?.elapsed_seconds ? `elapsed: ${event.data.elapsed_seconds}s` : null,
    event.data?.selection_rationale ? `selection: ${event.data.selection_rationale}` : null,
    event.data?.must_fix?.length ? `must-fix: ${event.data.must_fix.join(" | ")}` : null,
    event.data?.error ? `error: ${event.data.error}` : null,
  ]);
}

function compactMeta(values) {
  return values.filter(Boolean);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

document.querySelector("#run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const configPath = document.querySelector("#config-path").value.trim();
  const objectiveValue = document.querySelector("#objective-override").value.trim();
  const references = document.querySelector("#reference-paths").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const configDefaults = state.configMap.get(configPath);
  const defaultObjective = (configDefaults?.objective || "").trim();
  const defaultReferences = configDefaults?.reference_image_paths || [];
  const objectiveOverride = objectiveValue === defaultObjective ? null : objectiveValue || null;
  const referenceOverrides = arraysEqual(references, defaultReferences) ? [] : references;

  const response = await fetch("/api/runs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      config_path: configPath,
      objective_override: objectiveOverride || null,
      reference_image_paths_override: referenceOverrides,
    }),
  });

  const payload = await response.json();
  await fetchRuns();
  await selectRun(payload.run_id);
});

document.querySelector("#refresh-runs").addEventListener("click", fetchRuns);
document.querySelector("#config-path").addEventListener("change", (event) => {
  applyConfigDefaults(event.target.value);
});

function arraysEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

async function init() {
  await fetchConfigs();
  await fetchRuns();
}

init();
