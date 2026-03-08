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
  document.querySelector("#run-meta").textContent = run.objective;

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
      <p>${event.message}</p>
      <pre>${JSON.stringify(event.data, null, 2)}</pre>
    `;
    eventLog.appendChild(item);
  }

  const iterations = document.querySelector("#iterations");
  iterations.innerHTML = "";
  for (const iteration of run.iterations.slice().reverse()) {
    const card = document.createElement("article");
    card.className = "iteration";
    card.innerHTML = `
      <header>
        <strong>Iteration ${iteration.index}</strong>
        <span>${iteration.judge_accept === true ? "accepted" : iteration.judge_accept === false ? "rejected" : "unjudged"}</span>
      </header>
      <p><strong>Prompt</strong><br>${escapeHtml(iteration.prompt)}</p>
      <p><strong>Judge</strong><br>${escapeHtml(iteration.judge_feedback || "n/a")}</p>
      ${iteration.image_path ? `<img src="${iteration.image_data_url}" alt="Iteration ${iteration.index}" />` : ""}
    `;
    iterations.appendChild(card);
  }
}

function escapeHtml(value) {
  return value
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
