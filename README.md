<p align="center">
  <img src="docs/assets/banner.png" alt="HyperCaptcha — Open-source research harness for multimodal CAPTCHA evaluation" width="100%" />
</p>

# HyperCaptcha

HyperCaptcha is an open-source Rust and Python research harness for evaluating **multimodal decision models** on real CAPTCHA widgets (Google reCAPTCHA and hCaptcha) and offline saved grids. It combines a strict localhost lab server, declarative TOML model configs, automated Camoufox runners, and a standalone Python evaluation toolkit (`hypercaptcha-eval`) for labeling, recording, fitting, sweeping, and verifying decision rules.

> **Research Scope & Responsible Use:** HyperCaptcha binds strictly to loopback (`127.0.0.1`) and is intended only for empirical research using researcher-owned CAPTCHA keys, official vendor test keys (`--google-test-key` / `--hcaptcha-test-key`), or offline saved grids. Do **not** use this repository against third-party websites without explicit written authorization. See [SECURITY.md](SECURITY.md) · Licensed under [Apache-2.0](LICENSE).

---

## Benchmark Results

Offline 4×4 exact-grid accuracy replayed through the Rust server, comparing **`vLLM/d3-ultra`** across its four `--effort` levels (`low`, `medium`, `high` default, `max`) against **`Liquid/d1.2-ultra`**, **`Liquid/d1.2`**, and earlier `v1`/live **`typesafe/Jev`** reference runs:

<p align="center">
  <img src="docs/assets/results-model-comparison.png" alt="vLLM/d3-ultra effort levels vs Liquid/d1.2-ultra, Liquid/d1.2, and typesafe/Jev on offline 4x4 benchmarks and live 50-attempt snapshot" width="100%" />
</p>

<p align="center">
  <img src="docs/assets/results-effort-frontier.png" alt="Accuracy vs input token budget Pareto frontier and vLLM/d3-ultra provider compute latency across low, medium, high, and max effort levels" width="100%" />
</p>

- **`--effort high` is the default sweet spot for `vLLM/d3-ultra`:** It reaches **`82.9%` (`281/339`)**, within 1 grid of `max` (`83.2%`, `282/339`), while saving `4,404` input tokens per 4×4 round and cutting median provider compute+queue wait from `556 ms` to `282 ms`.
- **Scope & label caveats:** `v2` (`339` grids) and `v1` (`311` grids) are distinct human label revisions and must not be compared raw across versions. The 4×4 box veto, `--effort` ladder, `Liquid/*` models, and `hCaptcha` live pass rates are offline-only / unmeasured on live CAPTCHA verdicts.
- **Full empirical analysis:** See [docs/RESULTS.md](docs/RESULTS.md) for dataset genealogy, component ablations, negative findings, live reCAPTCHA runs, and transport tuning, plus the historical Nov 15 live review in [docs/BENCHMARK_REPORT_2026-11-15.md](docs/BENCHMARK_REPORT_2026-11-15.md).

---

## Using HyperCaptcha

**Prerequisites:** **Rust 1.88+**, **Python 3.11+**, and **[uv](https://docs.astral.sh/uv/)** (or `pip`). Full installation and CAPTCHA key setup instructions live in [docs/SETUP.md](docs/SETUP.md).

### 1. Try It Locally Without Credentials

Inspect built-in models, validate TOML configs, and boot a credential-free smoke lab using official public test keys:

```sh
# List built-in models, required env vars, default decision rules, and effort tiers
cargo run --locked -- models

# Validate one or more model configs offline
cargo run --locked -- validate-config models/recaptcha/vllm/d3-ultra.toml

# Start a Google reCAPTCHA v2 test-key lab (defaults to --mode v2 -> http://localhost:8765)
cargo run --locked -- serve --google-test-key

# Or start an hCaptcha test-key lab (defaults to --captcha hcaptcha -> http://hypercaptcha.localhost:8765)
cargo run --locked -- serve --hcaptcha-test-key
```

### 2. Run Automated Camoufox Trials with a Decision Model

Copy `.env.example` to `.env`, add your localhost CAPTCHA key and model provider key (for example `VLLM_SR_API_KEY`, `LIQUID_API_KEY`, or `TYPESAFE_API_KEY`), and pass `--env-file .env` explicitly (shell env vars always take precedence):

```sh
cp .env.example .env

cargo run --release --locked -- serve \
  --env-file .env \
  --mode v2-enterprise \
  --model vLLM/d3-ultra \
  --log reports/vllm-server.jsonl
```

`--effort` defaults to `high` (`low`, `medium`, `high`, and `max` are all supported). In a second terminal, fetch the pinned Camoufox browser once and run trials:

```sh
uv run --extra browser python -m camoufox fetch
uv run --extra browser python scripts/recaptcha_lab.py --trials 5 --submit --output reports/vllm-live
uv run hypercaptcha-eval summarize reports/vllm-live/trials.jsonl
```

#### Browser Runner Quick Reference

Both runners use Camoufox (installed via `--extra browser`), query `GET /config` capability flags automatically, and support `vllm`, `liquid`, `jev`, `custom` (`--model-config`), and `openai-chat`:

| Script | Default `--output` | Automated Flags | Special Modes |
| --- | --- | --- | --- |
| `scripts/recaptcha_lab.py` | `reports/recaptcha` | `--trials`, `--submit`, `--max-rounds`, `--max-stages`, `--no-early-tiles`, `--fast`, `--headed` | `--interactive`, `--smoke`, `--score` *(v3/v3-enterprise)* |
| `scripts/hcaptcha_lab.py` | `reports/hcaptcha` | `--trials`, `--submit`, `--max-rounds`, `--max-stages`, `--max-refreshes`, `--headed` | `--interactive`, `--smoke` |

### 3. Tune & Export Your Own Decision Model Offline

`hypercaptcha-eval` turns captured runs into a fitted, validated TOML config and deterministic parity fixture without touching live CAPTCHA services during optimization:

```sh
# 1. Deduplicate captured runs into a portable manifest
uv run hypercaptcha-eval dataset reports/capture-run -o datasets/4x4/dataset.jsonl --layout 4

# 2. Label grids in the local browser UI (127.0.0.1 only)
uv run hypercaptcha-eval label --dataset datasets/4x4/dataset.jsonl --labels datasets/4x4/labels.jsonl

# 3. Record raw two-view scores, skip probabilities, and boxes (auto-discovers target/release/hypercaptcha)
cargo build --release --locked
uv run hypercaptcha-eval record --dataset datasets/4x4/dataset.jsonl -o recordings/my-model.jsonl \
  --model-config models/examples/custom-decision-model.toml --passes 3

# 4. Fit logistic weights, sweep effort tiers, and export + validate TOML & parity fixture
uv run hypercaptcha-eval fit --dataset datasets/4x4/dataset.jsonl --labels datasets/4x4/labels.jsonl \
  --recordings recordings/my-model.jsonl --layout 4 -o recordings/fit.json
uv run hypercaptcha-eval sweep --dataset datasets/4x4/dataset.jsonl --labels datasets/4x4/labels.jsonl \
  --recordings recordings/my-model.jsonl --fit recordings/fit.json -o recordings/sweep.json
uv run hypercaptcha-eval export --base-config models/examples/custom-decision-model.toml \
  --sweep recordings/sweep.json --fit recordings/fit.json \
  -o my-model.toml --fixture tests/fixtures/my_model_4x4.json.gz --recordings recordings/my-model.jsonl
cargo run --locked -- validate-config my-model.toml
```

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for the complete `schema_version = 1` TOML schema, capability flags, and custom-provider walkthrough.

---

## Developing HyperCaptcha

### Architecture & Repository Layout

Requests flow from the local lab page or Camoufox runner (`scripts/`, `web/`) to the localhost Rust server (`src/server.rs`), which loads declarative `models/**/*.toml` `WireProfile` configs (`src/captcha/mod.rs`, `src/plan.rs`) and executes parallel multi-view + box-veto fan-out (`src/providers.rs`, `src/boxes.rs`). Pixel-exact preprocessing lives in `src/pillow.rs`, positional-image translation and patch-token billing guards live in `src/liquid.rs`, and hCaptcha canvas segmentation lives in `src/captcha/hcaptcha/grid.rs`.

```text
src/
  main.rs                 # CLI entry point (`serve`, `models`, `validate-config`)
  envfile.rs              # Opt-in `--env-file` parser & precedence overlay
  server.rs               # Localhost-only HTTP server, Host/Origin guards, JSONL logging
  providers.rs            # WireProfile-driven parallel fan-out, log-odds scoring, warming & hedging
  plan.rs                 # Owned/deserializable 4x4 effort plans, gap rule, box veto rule
  boxes.rs                # 9-ordinal, every-instance, and 16-window box requests + incremental Vote
  pillow.rs               # Byte-exact Pillow bilinear resizer & libjpeg baseline 4:2:0 q90 encoder
  liquid.rs               # Positional-image dialect rewriter & 1.5 tokens/patch billing guard
  captcha/                # Service profiles (recaptcha/, hcaptcha/ + canvas grid segmentation)
models/                   # Embedded schema_version=1 TOML configs + custom-decision-model.toml starter
eval/                     # Standalone Python toolkit (`hypercaptcha-eval`: dataset/label/record/fit/sweep/export/summarize)
scripts/                  # Camoufox browser runners (`recaptcha_lab.py`, `hcaptcha_lab.py`, `_common.py`)
web/                      # Embedded local lab UI (`recaptcha/`, `hcaptcha/`, `lab.css`)
tests/                    # Rust integration tests, Python eval tests, image-free `.json.gz` parity fixtures
docs/                     # Setup, empirical results, adding models, architecture, and historical report
```

### Local Verification Suite

All checks below run completely offline (no API keys or live browsers required) and match `.github/workflows/ci.yml`:

```sh
# Rust formatting, lints, unit/integration tests, and offline fixture parity
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- models >/dev/null
cargo run --locked -- validate-config models/recaptcha/*/*.toml models/hcaptcha/*/*.toml models/examples/custom-decision-model.toml

# Python runner syntax/CLI checks and offline eval + Rust-parity tests (5,288/5,288 exact fixture checks)
uv run --extra browser python -m py_compile scripts/_common.py scripts/recaptcha_lab.py scripts/hcaptcha_lab.py
uv run --extra browser python scripts/recaptcha_lab.py --help >/dev/null
uv run --extra browser python scripts/hcaptcha_lab.py --help >/dev/null
uv run --extra dev pytest -v
```

### Common Extension Paths

- **Evaluate an unmeasured model on an existing provider:** Run `hypercaptcha serve --provider vllm --model <model-id> --decision-rule average` (or `--record` to capture raw views/boxes for offline fitting) without modifying code.
- **Integrate a custom decision-model API:** Copy [models/examples/custom-decision-model.toml](models/examples/custom-decision-model.toml), configure `[endpoint]` (`systemone-assets` or `systemone-images` dialect), validate with `hypercaptcha validate-config`, and pass `--model-config <path.toml>` to `serve` or `hypercaptcha-eval record`.
- **Ship a built-in model & parity fixture:** Add its TOML under `models/<service>/<provider>/<model>.toml`, register it via `include_str!` in `src/captcha/<service>/mod.rs`, generate an image-free/label-free `.json.gz` fixture under `tests/fixtures/` via `hypercaptcha-eval export --fixture`, and verify both `cargo test --locked` and `uv run --extra dev pytest -v`.
- **Add a new wire dialect or CAPTCHA service:** Extend `WireDialect` / `WireProfile` in [src/providers.rs](src/providers.rs) and [src/captcha/mod.rs](src/captcha/mod.rs) (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md#adding-a-new-wire-dialect-in-rust)).
- **Data hygiene:** Keep raw captures, labels, logs, and `.env` files in gitignored paths (`/reports/`, `/datasets/`, `/recordings/`); never commit CAPTCHA imagery, human labels, or credentials. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Documentation

| Guide | What It Covers |
| --- | --- |
| [docs/SETUP.md](docs/SETUP.md) | Rust/Python/Camoufox installation, Google reCAPTCHA & hCaptcha key setup, `.env` / `--env-file`, localhost security |
| [docs/RESULTS.md](docs/RESULTS.md) | Full verified empirical report (`v0`/`v1`/`v2` datasets, effort ladders, negative findings, live runs, unmeasured scope) |
| [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) | Unmeasured models, custom `schema_version = 1` TOML configs, full `hypercaptcha-eval` loop, adding wire dialects |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Server routes, JSONL schemas, decision & box-veto pipeline, `pillow.rs`, `liquid.rs`, hCaptcha canvas segmentation |
| [docs/BENCHMARK_REPORT_2026-11-15.md](docs/BENCHMARK_REPORT_2026-11-15.md) | Historical November 15, 2026 live 50-attempt `Jev` vs. `d3-ultra` failure review |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [LICENSE](LICENSE) | Development workflow, parity-fixture rules, responsible-research scope, Apache-2.0 license |
