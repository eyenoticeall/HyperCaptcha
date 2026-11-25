const byId = (id) => document.getElementById(id);
const googleStatus = byId("google-status");
const modelStatus = byId("model-status");
const next = byId("next");
const scoreButton = byId("run-score");
const challengeButton = byId("run-challenge");
const imageForm = byId("image-form");
const promptInput = byId("prompt");
const layoutSelect = byId("layout");
const observationForm = byId("observation-form");
const parameters = new URLSearchParams(location.hash.slice(1));
const requestedSource = parameters.get("source");
const source = ["camoufox", "chrome"].includes(requestedSource) ? requestedSource : "browser";
const companionTrial = parameters.get("trial");

let trialId;
let config;
let policyChallenge;
let widget;
let widgetId;
let verifying = false;
let modelBusy = false;
let googleDone = false;
let started;
let pointerMoves = 0;
let pointerDowns = 0;
let keyPresses = 0;
let focusChanges = 0;
let imageRounds = 0;

function syncPromptPlaceholder() {
  promptInput.placeholder = layoutSelect.value === "4"
    ? "Select all squares with traffic lights"
    : "Select all images with bicycles";
}

function startTrial() {
  trialId = crypto.randomUUID();
  byId("trial-id").textContent = trialId;
  next.hidden = true;
  byId("google-result").hidden = true;
  byId("model-result").hidden = true;
  byId("observation-status").hidden = true;
  observationForm.reset();
  byId("companion").href = "/#" + new URLSearchParams({ trial: trialId, source });
  modelStatus.hidden = true;
  googleDone = false;
  started = performance.now();
  pointerMoves = 0;
  pointerDowns = 0;
  keyPresses = 0;
  focusChanges = 0;
  imageRounds = 0;
  if (config.score_based) {
    scoreButton.disabled = false;
    googleStatus.textContent = "Request an assessment to score this browser interaction.";
  } else if (policyChallenge) {
    challengeButton.disabled = false;
    googleStatus.textContent = "Request a challenge with your Google policy key.";
  } else {
    if (widgetId !== undefined) widget.reset(widgetId);
    googleStatus.textContent = config.test
      ? "Check the box to test token verification."
      : "Check the box. Google decides whether to show an image challenge.";
  }
}

function show(target, record) {
  target.textContent = JSON.stringify(record, null, 2);
  target.hidden = false;
}

async function post(path, fields) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trial_id: trialId, source, ...fields }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}

async function verify(token) {
  if (verifying || googleDone) return;
  verifying = true;
  if (policyChallenge) challengeButton.disabled = true;
  googleStatus.textContent = "Verifying with Google…";
  try {
    const result = await post("/verify", { token });
    show(byId("google-result"), result);
    const verdict = result.result;
    googleStatus.textContent = verdict.error || (verdict.valid
      ? `Token verified${verdict.score === null ? "." : ` · Google score ${verdict.score}`}`
      : "The token did not pass verification.");
  } catch (error) {
    googleStatus.textContent = error.message;
  } finally {
    verifying = false;
    googleDone = true;
    if (policyChallenge) challengeButton.disabled = false;
    next.hidden = false;
  }
}

async function askModel(fields) {
  modelBusy = true;
  modelStatus.hidden = false;
  modelStatus.textContent = "Waiting for the model…";
  byId("model-result").hidden = true;
  try {
    const result = await post("/model", fields);
    show(byId("model-result"), result);
    const answer = result.result;
    const prediction = answer.prediction;
    if (prediction?.type === "tiles") {
      imageRounds += 1;
      modelStatus.textContent = prediction.value.length
        ? `Round ${imageRounds} · Select tiles ${prediction.value.join(", ")}.`
        : `Round ${imageRounds} · The model selected no tiles.`;
    } else if (prediction?.type === "legitimacy_score") {
      modelStatus.textContent = `Model estimate · ${prediction.value}`;
    } else {
      modelStatus.textContent = answer.error || answer.detail || "The model refused the request.";
    }
  } catch (error) {
    modelStatus.textContent = error.message;
  } finally {
    modelBusy = false;
  }
}

function loadGoogle() {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const file = config.enterprise ? "enterprise.js" : "api.js";
    const url = new URL(`https://www.google.com/recaptcha/${file}`);
    url.searchParams.set("render", config.score_based || policyChallenge ? config.site_key : "explicit");
    url.searchParams.set("onload", "hypercaptchaGoogleReady");
    window.hypercaptchaGoogleReady = () => {
      clearTimeout(timer);
      resolve();
    };
    script.onerror = () => {
      clearTimeout(timer);
      reject(new Error("Could not load Google reCAPTCHA. Check the browser’s network access."));
    };
    const timer = setTimeout(() => reject(new Error("Google reCAPTCHA did not load within 20 seconds.")), 20_000);
    script.src = url;
    script.async = true;
    document.head.append(script);
  });
}

function readPng(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",", 2)[1]);
    reader.onerror = () => reject(new Error("Could not read the screenshot."));
    reader.readAsDataURL(file);
  });
}

document.addEventListener("pointermove", () => pointerMoves++);
document.addEventListener("pointerdown", () => pointerDowns++);
document.addEventListener("keydown", () => keyPresses++);
window.addEventListener("focus", () => focusChanges++);
window.addEventListener("blur", () => focusChanges++);

layoutSelect.addEventListener("change", syncPromptPlaceholder);

next.addEventListener("click", () => {
  if (verifying || modelBusy) return;
  if (policyChallenge) {
    location.reload();
    return;
  }
  imageForm.reset();
  syncPromptPlaceholder();
  startTrial();
});

observationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = byId("observation-status");
  const button = observationForm.querySelector("button");
  button.disabled = true;
  status.hidden = false;
  try {
    await post("/observe", { outcome: byId("observation").value });
    status.textContent = "Outcome recorded.";
    next.hidden = false;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

scoreButton.addEventListener("click", async (event) => {
  scoreButton.disabled = true;
  googleStatus.textContent = "Requesting a Google assessment…";
  const trace = {
    action: config.action,
    elapsed_ms: Math.round(performance.now() - started),
    pointer_moves: pointerMoves,
    pointer_downs: pointerDowns,
    key_presses: keyPresses,
    focus_changes: focusChanges,
    activation: event.detail === 0 ? "keyboard or script" : "pointer",
  };
  try {
    const token = await widget.execute(config.site_key, { action: config.action });
    next.disabled = true;
    await verify(token);
    if (config.provider) await askModel({ trace });
  } catch (error) {
    googleStatus.textContent = `Google could not generate a token: ${error.message || error}`;
    scoreButton.disabled = false;
  } finally {
    next.disabled = false;
  }
});

imageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (modelBusy) return;
  const ask = byId("ask");
  ask.disabled = true;
  try {
    const file = byId("image").files[0];
    if (file.size > 10 * 1024 * 1024) throw new Error("Choose a PNG under 10 MiB.");
    await askModel({
      prompt: byId("prompt").value,
      layout: Number(byId("layout").value),
      image: await readPng(file),
    });
  } catch (error) {
    modelStatus.hidden = false;
    modelStatus.textContent = error.message;
  } finally {
    ask.disabled = false;
  }
});

async function init() {
  const response = await fetch("/config");
  if (!response.ok) throw new Error("Could not load the lab configuration.");
  config = await response.json();
  policyChallenge = config.mode === "policy-challenge";
  const version = config.score_based ? "v3" : "v2";
  byId("mode").textContent = policyChallenge
    ? "reCAPTCHA image challenge"
    : `reCAPTCHA ${version}${config.enterprise ? " Enterprise" : ""}`;
  byId("context").textContent = config.test ? "Google public test key · Integration smoke test" : "Live verification with your Google key";
  byId("model-info").textContent = config.provider
    ? (config.model_name || `${config.provider} · ${config.model}`)
    : "Start the server with --provider, --model, or --model-config to enable model requests.";
  syncPromptPlaceholder();
  const vision = Boolean(!config.score_based && !config.test && config.provider && config.supports_vision);
  if (companionTrial) {
    document.body.classList.add("companion-only");
    trialId = companionTrial;
    byId("trial-id").textContent = trialId;
    imageForm.hidden = !vision;
    return;
  }
  if (config.score_based) scoreButton.hidden = false;
  if (policyChallenge) {
    challengeButton.classList.add("g-recaptcha");
    challengeButton.dataset.sitekey = config.site_key;
    challengeButton.dataset.action = config.action;
    challengeButton.dataset.callback = "hypercaptchaChallengeComplete";
    challengeButton.dataset.expiredCallback = "hypercaptchaChallengeExpired";
    challengeButton.dataset.errorCallback = "hypercaptchaChallengeError";
    window.hypercaptchaChallengeComplete = verify;
    window.hypercaptchaChallengeExpired = () => {
      if (!googleDone) {
        challengeButton.disabled = false;
        googleStatus.textContent = "The challenge expired. Request a challenge again.";
      }
    };
    window.hypercaptchaChallengeError = () => {
      if (!googleDone) {
        challengeButton.disabled = false;
        googleStatus.textContent = "Google could not complete the challenge. Check the network or reload.";
      }
    };
    challengeButton.hidden = false;
    challengeButton.addEventListener("click", () => {
      if (googleDone || verifying) return;
      googleStatus.textContent = "Waiting for Google’s challenge…";
      next.hidden = false;
    });
  }
  await loadGoogle();
  widget = config.enterprise ? window.grecaptcha.enterprise : window.grecaptcha;
  startTrial();
  if (!config.score_based && !policyChallenge) {
    widgetId = widget.render("widget", {
      sitekey: config.site_key,
      callback: verify,
      "expired-callback": () => { if (!googleDone) googleStatus.textContent = "The challenge expired. Check the box again."; },
      "error-callback": () => { if (!googleDone) googleStatus.textContent = "Google could not complete the challenge. Check the network or reload."; },
    });
  }
  observationForm.hidden = config.test || config.score_based;
  imageForm.hidden = !vision;
  byId("companion").hidden = !vision;
  if (vision) {
    byId("model-info").textContent += " · Apply the selected tiles in the Google challenge.";
  } else if (!config.score_based && config.provider && !config.supports_vision) {
    byId("model-info").textContent += " · Choose a vision backend for image challenges.";
  }
}

init().catch((error) => { googleStatus.textContent = error.message; });
