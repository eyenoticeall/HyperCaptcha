# Setup Guide

This guide covers installing prerequisites, configuring researcher-owned CAPTCHA keys and model provider credentials, and understanding HyperCaptcha's localhost-only security model.

> **Research scope:** HyperCaptcha runs exclusively against local test pages served on loopback (`127.0.0.1`) using official vendor test keys or your own registered CAPTCHA site keys. See [SECURITY.md](../SECURITY.md) before running live trials.

## 1. Prerequisites

### Rust 1.88+

HyperCaptcha uses Rust edition 2024 (`rust-version = "1.88"`). Install or update Rust via [rustup](https://rustup.rs/):

```sh
rustup update stable
rustc --version   # expect 1.88.0 or later
```

Build and run the offline test suite to verify your toolchain:

```sh
cargo test --locked
```

### Python 3.11+ & `uv`

Browser runners (`scripts/`) and the offline evaluation toolkit (`eval/`) require **Python 3.11 or later**. [uv](https://docs.astral.sh/uv/) is recommended and resolves optional extras defined in `pyproject.toml`:

| Extra / Group | Packages Installed | Used For |
| --- | --- | --- |
| *(base)* | `numpy`, `httpx` | `hypercaptcha-eval` offline dataset, labeling server, recording, fitting, sweep, export, and summary tools |
| `browser` | `playwright==1.62.0` | Automated Chrome trials (`scripts/chrome_lab.py`, `scripts/hcaptcha_lab.py`) |
| `camoufox` | `camoufox==0.5.9` | Manual/automated Camoufox trials (`scripts/camoufox_lab.py`, `scripts/chrome_lab.py --browser camoufox`) |
| `dev` | `pytest` | Running the offline Python test suite (`uv run --extra dev pytest`) |

With `uv` installed, verify the Python environment:

```sh
uv run --extra dev pytest
```

If you prefer standard `pip` inside a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[browser,camoufox,dev]"
```

## 2. Browser Installation

### Google Chrome (for Playwright runners)

`scripts/chrome_lab.py` (default `--browser chrome`) and `scripts/hcaptcha_lab.py` launch Chrome in headless mode via Playwright's `chrome` channel. Either install system **Google Chrome** or have Playwright install the Chrome channel binary:

```sh
uv run --extra browser playwright install chrome
```

### Camoufox (for manual, score, and `--browser camoufox` trials)

Fetch the pinned Camoufox browser binary once:

```sh
uv run --extra camoufox python -m camoufox fetch
```

Verify Camoufox against the public Google v2 test key (with the Rust server running in another terminal):

```sh
uv run --extra camoufox scripts/camoufox_lab.py --smoke
```

## 3. Credential-Free Smoke Tests

Before creating live keys, verify both services with their official public integration test keys (neither test key ever opens a real image challenge):

```sh
# Google reCAPTCHA v2 public test key -> http://localhost:8765
cargo run --locked -- serve --mode v2 --google-test-key

# hCaptcha public test key -> http://hypercaptcha.localhost:8765
cargo run --locked -- serve --captcha hcaptcha --hcaptcha-test-key
```

Open the printed URL in your browser and click the checkbox to confirm token verification succeeds locally.

## 4. Google reCAPTCHA Live Keys

Create a dedicated research key for each integration mode you intend to study and allow `localhost` on the key. Keep verification secrets in your shell environment and restart the server when switching modes.

| `--mode` | Key Type | Verification API | Required Environment Variables |
| --- | --- | --- | --- |
| `v2` | Checkbox | Classic SiteVerify (`api.js`) | `RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET_KEY` |
| `v2-enterprise` | Checkbox | Cloud Assessments (`enterprise.js`) | `RECAPTCHA_SITE_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_API_KEY` *(or `GOOGLE_CLOUD_ACCESS_TOKEN`)* |
| `v3` | Score | Classic SiteVerify (`api.js`) | `RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET_KEY` |
| `v3-enterprise` | Score | Cloud Assessments (`enterprise.js`) | `RECAPTCHA_SITE_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_API_KEY` *(or `GOOGLE_CLOUD_ACCESS_TOKEN`)* |
| `policy-challenge` | Policy-based challenge | Cloud Assessments (`enterprise.js`) | `RECAPTCHA_SITE_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_API_KEY` *(or `GOOGLE_CLOUD_ACCESS_TOKEN`)* |

> **Note on Google Cloud migration:** Classic keys may be managed under Google Cloud depending on your account migration status. Record the exact key type, security preference, and integration mode alongside any experimental results.

### Creating a Dedicated `policy-challenge` Key with `gcloud`

To trigger an image challenge on every request (for deterministic visual trials), create a dedicated **policy-based challenge** key in a Google Cloud project with billing and the reCAPTCHA Enterprise API enabled, with `--default-score-threshold=1.0`:

```sh
gcloud auth login
export GOOGLE_CLOUD_PROJECT="your-research-project"
gcloud services enable recaptchaenterprise.googleapis.com --project="$GOOGLE_CLOUD_PROJECT"

export RECAPTCHA_SITE_KEY="$(gcloud recaptcha keys create \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --display-name="HyperCaptcha localhost image trials" \
  --web \
  --domains=localhost \
  --integration-type=policy-based-challenge \
  --default-score-threshold=1.0 \
  --security-preference=security \
  --format='value(name.basename())')"
```

Authenticate Assessments API requests with either a persistent `GOOGLE_CLOUD_API_KEY` or a short-lived OAuth access token (restart the lab server when a short-lived token expires):

```sh
export GOOGLE_CLOUD_ACCESS_TOKEN="$(gcloud auth print-access-token)"
mkdir -p reports
cargo run --locked -- serve --mode policy-challenge --log reports/image-challenges.jsonl
```

For naturally triggered checkbox trials (`scripts/chrome_lab.py`), create a `v2` or `v2-enterprise` checkbox key (`--security-preference=security`) rather than `policy-challenge`, so natural checkbox-only passes and image challenges are measured separately.

## 5. hCaptcha Live Keys & Hostname Setup

hCaptcha's public test key never displays an image challenge. To collect or test real hCaptcha 3×3 grid challenges:

1. Create a sitekey in your hCaptcha dashboard and set its behavior to **Always Challenge**.
2. Export `HCAPTCHA_SITE_KEY` and `HCAPTCHA_SECRET_KEY` in your shell.
3. Because hCaptcha refuses widgets hosted on `localhost` or `127.0.0.1`, HyperCaptcha defaults to serving the hCaptcha lab at **`http://hypercaptcha.localhost:8765`** — modern browsers resolve `.localhost` subdomains to `127.0.0.1` without editing `/etc/hosts`.
4. If hCaptcha rejects `hypercaptcha.localhost` for a live sitekey, add a local hosts entry and pass `--hostname`:

```sh
# Add to /etc/hosts:
# 127.0.0.1 hypercaptcha.test

mkdir -p reports
cargo run --release --locked -- serve \
  --captcha hcaptcha \
  --hostname hypercaptcha.test \
  --provider vllm \
  --log reports/hcaptcha.jsonl
```

Then pass the matching `--url http://hypercaptcha.test:8765` to `scripts/hcaptcha_lab.py`. Both `scripts/chrome_lab.py` and `scripts/hcaptcha_lab.py` read capability flags (`supports_vision`, `requires_vision_check`) from `GET /config`, so `vllm`, `liquid`, `custom` (`--model-config`), and `jev` (with `--jev-vision average` on hCaptcha) all work with `--submit`.

## 6. Model Provider Credentials (`.env.example`)

Copy [`.env.example`](../.env.example) to `.env` (or per-provider files such as `.env.vllm`, `.env.liquid` — all `.env*` files other than `.env.example` are gitignored) and load them into your shell before starting the server. HyperCaptcha intentionally **never** auto-loads `.env` files from disk so credentials stay explicit:

```sh
cp .env.example .env
# Edit .env with your keys, then export into the current shell:
set -a && source .env && set +a
```

### Decision & Vision Providers

| Provider (`--provider`) | Built-in Models (`--model`) | API Key Env Var(s) | Optional Base URL Env Var (Default) | Notes |
| --- | --- | --- | --- | --- |
| `vllm` | `vLLM/d3-ultra` | `VLLM_SR_API_KEY` *(plus comma-separated `VLLM_SR_EXTRA_API_KEYS`)* | `VLLM_SR_BASE_URL` (`https://app.vllm-sr.ai/v1`) | Extra keys rotate round-robin across parallel requests and hedged retries to increase concurrent capacity (`16` reqs/key). |
| `liquid` | `Liquid/d1.2`, `Liquid/d1.2-ultra` | `LIQUID_API_KEY` | `LIQUID_BASE_URL` (`https://api.liquid.ai/decisions/v1`) | Positional-image dialect; enforced patch-token billing guard; retries `429`/`5xx` up to `3×`. |
| `jev` | `typesafe/Jev` (`jev-1.14.0`, `jev-latest`) | `TYPESAFE_API_KEY` *(or `JEV_API_KEY`)* | `JEV_BASE_URL` (`https://api.typesafe.ai/v1`) | Requires a vision-capable System One endpoint; verified automatically by the red/blue vision control check before live trials. |
| `custom` | Any model via `--model-config <path.toml>` | Configured by `endpoint.api_key_env` *(and optional `extra_api_keys_env`)* | Configured by `endpoint.base_url` *(overridable via `--base-url`)* | Supports both `systemone-assets` and `systemone-images` dialects without editing Rust. See [ADDING_MODELS.md](ADDING_MODELS.md). |
| `openai-chat` | Any vision-capable chat model (`--model` required) | `OPENAI_API_KEY` | `OPENAI_BASE_URL` (`https://api.openai.com/v1`) | Existing baseline for v2 image grids and v3 score fixtures. Supports `--max-output-tokens`. |
| `openai-completions` | Any text completions model (`--model` required) | `OPENAI_API_KEY` | `OPENAI_BASE_URL` (`https://api.openai.com/v1`) | Text-only baseline for v3 score fixtures on OpenAI-compatible `/completions` servers. |

Notes on provider flags:
- Passing a provider-qualified name such as `--model vLLM/d3-ultra` or `--model Liquid/d1.2-ultra` selects the provider automatically (case-insensitive), so `--provider` may be omitted. On `--captcha hcaptcha`, only `vLLM/d3-ultra` is built-in; use `--model-config` or `--provider <provider> --model <id> --decision-rule average` (or `--provider jev --jev-vision average`) for other models.
- Custom base URLs must include the API version prefix (usually `/v1`).
- Starting the server without `--provider`, `--model`, or `--model-config` runs in **CAPTCHA-only mode** (useful for capturing datasets or recording human baselines).

## 7. Localhost-Only Security Model & Data Hygiene

HyperCaptcha enforces several local boundaries by design:

- **Loopback binding only:** The HTTP server binds strictly to `127.0.0.1:<port>` and never listens on external interfaces.
- **Strict `Host` and `Origin` checks:** Every request must match `Host: <page_host>:<port>`, and every `POST` request must match `Origin: http://<page_host>:<port>`. Requests from other hostnames or origins receive HTTP `403`, preventing DNS rebinding and cross-origin access to your local API keys.
- **Hardened HTTP headers & body cap:** Every response sets `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`, and request bodies are capped at `16 MiB` (`1 MiB` per base64 replacement tile).
- **Redacted server logs:** JSONL server logs (`--log`) record trial IDs, verification outcomes, tile probabilities, decision rule metadata, latency, and token usage, while omitting API keys, CAPTCHA tokens, raw screenshots, and raw telemetry payloads.
- **Gitignored local artifacts:** Store live logs, captured PNGs, human labels, and offline recordings inside `/reports/`, `/datasets/`, or `/recordings/` — never commit credentials or raw CAPTCHA imagery. See [SECURITY.md](../SECURITY.md) for vulnerability reporting and disclosure guidance.
