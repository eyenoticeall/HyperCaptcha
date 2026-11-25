const byId = (id) => document.getElementById(id);
const status = byId("status");
const next = byId("next");
const requestedSource = new URLSearchParams(location.hash.slice(1)).get("source");
const source = ["camoufox", "chrome"].includes(requestedSource) ? requestedSource : "browser";

let trialId;
let config;
let widgetId;
let verifying = false;
let done = false;

function startTrial() {
  trialId = crypto.randomUUID();
  byId("trial-id").textContent = trialId;
  byId("verdict").hidden = true;
  next.hidden = true;
  done = false;
  // What the widget last reported, for a runner to read.
  document.body.dataset.widget = "ready";
  if (widgetId !== undefined) window.hcaptcha.reset(widgetId);
  status.textContent = config.test
    ? "Check the box to test token verification."
    : "Check the box. hCaptcha decides which challenge to show.";
}

async function verify(token) {
  if (verifying || done) return;
  verifying = true;
  status.textContent = "Verifying with hCaptcha…";
  try {
    const response = await fetch("/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trial_id: trialId, source, token }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request failed.");
    byId("verdict").textContent = JSON.stringify(data, null, 2);
    byId("verdict").hidden = false;
    const verdict = data.result;
    status.textContent = verdict.error
      || (verdict.valid ? "Token verified." : "The token did not pass verification.");
  } catch (error) {
    status.textContent = error.message;
  } finally {
    verifying = false;
    done = true;
    next.hidden = false;
  }
}

function loadHcaptcha() {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const url = new URL("https://js.hcaptcha.com/1/api.js");
    url.searchParams.set("render", "explicit");
    url.searchParams.set("onload", "hypercaptchaReady");
    url.searchParams.set("recaptchacompat", "off");
    url.searchParams.set("hl", "en");
    const timer = setTimeout(() => reject(new Error("hCaptcha did not load within 20 seconds.")), 20_000);
    window.hypercaptchaReady = () => {
      clearTimeout(timer);
      resolve();
    };
    script.onerror = () => {
      clearTimeout(timer);
      reject(new Error("Could not load hCaptcha. Check the browser’s network access."));
    };
    script.src = url;
    script.async = true;
    document.head.append(script);
  });
}

function report(state, message) {
  return (detail) => {
    document.body.dataset.widget = typeof detail === "string" ? `${state}:${detail}` : state;
    if (message && !done) status.textContent = typeof detail === "string" ? `${message} (${detail})` : message;
  };
}

next.addEventListener("click", () => {
  if (!verifying) startTrial();
});

async function init() {
  const response = await fetch("/config");
  if (!response.ok) throw new Error("Could not load the lab configuration.");
  config = await response.json();
  byId("context").textContent = config.test
    ? "hCaptcha public test key · Integration smoke test"
    : "Live verification with your hCaptcha key";
  byId("model-info").textContent = config.provider
    ? (config.model_name || `${config.provider} · ${config.model}`)
    : "Start the server with --provider, --model, or --model-config to enable model requests.";
  await loadHcaptcha();
  startTrial();
  widgetId = window.hcaptcha.render("widget", {
    sitekey: config.site_key,
    callback: verify,
    "open-callback": report("open"),
    "close-callback": report("closed", "The challenge was closed. Check the box again."),
    "expired-callback": report("expired", "The token expired. Check the box again."),
    "chalexpired-callback": report("challenge-expired", "The challenge expired. Check the box again."),
    "error-callback": report("error", "hCaptcha could not complete the challenge."),
  });
}

init().catch((error) => { status.textContent = error.message; });
