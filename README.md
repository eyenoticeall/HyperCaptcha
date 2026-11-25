# HyperCaptcha

HyperCaptcha is an open-source Rust and Python research harness for evaluating **multimodal decision models** on real CAPTCHA widgets (Google reCAPTCHA and hCaptcha) and saved offline grids. It pairs a strict localhost lab server with declarative TOML model configurations, automated Chrome/Camoufox runners, and a standalone Python evaluation toolkit (`hypercaptcha-eval`) for labeling, recording, fitting, sweeping, and verifying decision rules.

> **Research Scope & Responsible Use:** HyperCaptcha is a **localhost-only research lab** (`127.0.0.1`) intended for empirical evaluation using researcher-owned CAPTCHA keys, official vendor test keys (`--google-test-key` / `--hcaptcha-test-key`), or offline saved grids. Do **not** use this repository to target or automate against third-party websites without explicit written authorization. See [SECURITY.md](SECURITY.md). Licensed under [Apache-2.0](LICENSE).

---

## Headline Empirical Results

All offline 4×4 scores below (except `typesafe/Jev`, which was evaluated on earlier `v1` labels) are measured on the **`v2` corrected benchmark** (`339` graded grids = `311` square-by-square human-labelled + `28` live-graded) replayed through the lab server. Liquid scores are 3-pass averages; tokens include the 4×4 skip question (`~300` fewer without skip). Full methodology, caveats, and live breakdowns live in [docs/RESULTS.md](docs/RESULTS.md).

| Model | Label Ver. | Effort / Mode | Requests | Input Tokens | Right of 339 | Right of 311 |
| --- | :---: | --- | ---: | ---: | ---: | ---: |
| **`vLLM/d3-ultra`** | `v2` | `low` (1 pass @ 506 px + gap rule) | 6 | 6,475 | 243 (71.7%) | 233 (74.9%) |
| **`vLLM/d3-ultra`** | `v2` | `medium` (`low` + 9 whole boxes) | 7 | 7,915 | 274 (80.8%) | 250 (80.4%) |
| **`vLLM/d3-ultra`** | `v2` | `high` (2 passes + 9 whole boxes) | 11 | 12,351 | **281 (82.9%)** | **257 (82.6%)** |
| **`vLLM/d3-ultra`** *(default)* | `v2` | `max` (`high` + 16 window boxes) | 12 | 16,755 | **282 (83.2%)** | **258 (83.0%)** |
| **`Liquid/d1.2-ultra`** | `v2` | `low` / `medium` / `high`–`max` | 6 / 7 / 11 | 7.3k / 8.7k / 14.1k | 232 / 242 / **245** (72.3%) | 218 / 224 / **226** (72.6%) |
| **`Liquid/d1.2`** | `v2` | `low` / `medium`–`max` | 6 / 7 | 6.8k / 8.1k | 200 / **236** (69.6%) | 188 / **217** (69.7%) |
| **`typesafe/Jev`** *(ref)* | `v1`* | `scene-and-crop` (0.50) / OOF 5-wt + boxes | 1 / 27 | 7.0k / 34.7k | — | 166 (53.4%) / 214 (68.8%) |

*\*Do not compare `v1` and `v2` directly: on `v1`, pre-box `vLLM/d3-ultra` scored `251/311` (`80.7%`) and the initial `0.3` box rule scored `257/311` (`82.6%`).*

**Live reCAPTCHA snapshots (pre-box / pre-effort):**
- **50-attempt head-to-head (equal-weight 50/50 averaging, Camoufox):** `vLLM/d3-ultra` verified **`40/50` (80.0%, 95% CI 67.0–88.8%)** vs. `typesafe/Jev` **`34/50` (68.0%, CI 54.2–79.2%)**; median click → pass `4.26 s` vs. `3.28 s` ([BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)).
- **200-attempt `--max-stages 10` (`d3-ultra` logistic + 2-pass + skip, pre-box):** **`160/169` (94.7%)** challenged attempts eventually solved (`104` on first answer; median click → pass **`2.7 s`**, static 3×3 fast-pacing first-answer median **`1.7 s`**). A subsequent `297`-attempt capture run verified **`215/217` (99.1%)**.
- **Unmeasured live:** The 4×4 box rule, `--effort` ladder, `Liquid/*` models, and `hCaptcha` live pass rates have **not** been measured on live service verdicts yet (see [docs/RESULTS.md](docs/RESULTS.md#7-what-has-not-been-measured-live-yet)).

### Five Key Empirical Takeaways

1. **Multi-view log-odds scoring dominates equal averaging:** Neighbor-aware logistic combination lifted `vLLM/d3-ultra` from `59% → 69%` on `v0` 4×4 (`87% → 91%` static 3×3), and adding a second close-up pass, skip gate, and box veto reaches `82.9–83.2%` on `339` `v2`-labelled 4×4 grids.
2. **`--effort high` is the sweet spot for `vLLM/d3-ultra`:** It scores within `1` grid of `max` (`281` vs. `282/339`) while saving `4,404` input tokens and avoiding the `16`-window box wait (`282 ms` vs. `556 ms` estimated provider compute+queue median).
3. **Boxes work as a veto, not as features or rescues:** Exempting high-confidence picks (`≥ 90–95.3%`) and vetoing unsupported lower-confidence picks gained `+10` to `+36` grids, whereas box logistic features, unselected-tile rescue, and cross-model ensembles all failed offline validation.
4. **Live failures on dynamic 3×3 and 5-grid skip are not tile-accuracy bottlenecks:** On blind-reviewed dynamic 3×3 challenges, `d3-ultra` answered every round right in **`~72%`** of challenges while Google accepted only **`26.0%`** (`34/131`), and 5-grid skip rejections showed no significant grid-accuracy gap (`1.00` vs. `1.42` wrong grids, `p ≈ 0.25`).
5. **Exact wire & pixel reproducibility matters:** Changing JPEG subsampling or resize filters shifted `d3-ultra` box outcomes on `1–2/60` grids (`src/pillow.rs` matches Pillow + libjpeg 4:2:0 q90 byte-for-byte on `5,661/5,661` box pictures), and Liquid silently bills text-only tokens if images are sent as named `assets` (`src/liquid.rs` rewrites to positional `images` and enforces a `1.5` tokens/patch floor).

---

## 5-Minute Quickstart

Requires **Rust 1.88+**, **Python 3.11+**, and **[uv](https://docs.astral.sh/uv/)** (or `pip`). Full installation instructions live in [docs/SETUP.md](docs/SETUP.md).

### 1. Verify Locally Without Credentials

```sh
# Run offline Rust unit, integration, and 4x4 fixture parity tests
cargo test --locked

# Run offline Python eval & Rust-parity tests (5,288/5,288 exact fixture checks)
uv run --extra dev pytest -v

# Start the localhost smoke lab with Google's public test key (never challenges)
cargo run --locked -- serve --mode v2 --google-test-key
# Open http://localhost:8765 and click the checkbox
```

For hCaptcha's public test key, run `cargo run --locked -- serve --captcha hcaptcha --hcaptcha-test-key` and open `http://hypercaptcha.localhost:8765`.

### 2. Run a Live Localhost Trial with a Decision Model

Copy `.env.example` to `.env`, add your own localhost CAPTCHA key and model provider key, and export them into your shell:

```sh
cp .env.example .env
# Edit .env with RECAPTCHA_SITE_KEY / verification credentials and VLLM_SR_API_KEY (or LIQUID_API_KEY / TYPESAFE_API_KEY)
set -a && source .env && set +a

mkdir -p reports
cargo run --release --locked -- serve \
  --mode v2-enterprise \
  --model vLLM/d3-ultra \
  --effort high \
  --log reports/vllm-server.jsonl
```

In a second terminal, run automated checkbox trials in headless Chrome (or `--browser camoufox` with `--extra camoufox`):

```sh
uv run --extra browser scripts/chrome_lab.py --trials 3 --submit --output reports/vllm-live
uv run hypercaptcha-eval summarize reports/vllm-live/trials.jsonl
```

- `scripts/chrome_lab.py` and `scripts/hcaptcha_lab.py` query `GET /config` capability flags (`supports_vision`, `requires_vision_check`), so `vllm`, `liquid`, `jev`, `custom` (`--model-config`), and `openai-chat` all work with `--submit`.
- Early-tile compositing on dynamic 3×3 grids is **on by default** (`--no-early-tiles` disables it).

### 3. Record, Fit, Sweep & Export Your Own Decision Model Offline

```sh
# 1. Deduplicate captured runs into a portable manifest
uv run hypercaptcha-eval dataset reports/capture-run -o datasets/4x4/dataset.jsonl --layout 4

# 2. Label grids in the local browser UI (127.0.0.1 only)
uv run hypercaptcha-eval label --dataset datasets/4x4/dataset.jsonl --labels datasets/4x4/labels.jsonl

# 3. Record raw two-view scores, skip probabilities, and boxes (--record)
cargo build --release --locked
uv run hypercaptcha-eval record --dataset datasets/4x4/dataset.jsonl -o recordings/my-model.jsonl \
  --binary target/release/hypercaptcha --model-config models/examples/custom-decision-model.toml --passes 3

# 4. Fit logistic weights, sweep effort tiers, and export a validated TOML + parity fixture
uv run hypercaptcha-eval fit --dataset datasets/4x4/dataset.jsonl --labels datasets/4x4/labels.jsonl \
  --recordings recordings/my-model.jsonl --layout 4 -o recordings/fit.json
uv run hypercaptcha-eval sweep --dataset datasets/4x4/dataset.jsonl --labels datasets/4x4/labels.jsonl \
  --recordings recordings/my-model.jsonl --fit recordings/fit.json -o recordings/sweep.json
uv run hypercaptcha-eval export --service recaptcha --name MyProvider/my-model --provider custom \
  --model-id my-model-v1 --dialect systemone-assets --base-url https://api.example.com/v1 \
  --api-key-env MY_DECISION_API_KEY --sweep recordings/sweep.json --fit recordings/fit.json \
  -o my-model.toml --fixture tests/fixtures/my_model_4x4.json.gz --recordings recordings/my-model.jsonl
```

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for the complete schema, capability flags, and custom-provider walkthrough.

---

## Architecture & Repository Layout

```text
src/
  main.rs                 # CLI entry point (`hypercaptcha serve`)
  server.rs               # Localhost-only HTTP server, Host/Origin checks, JSONL logging
  providers.rs            # WireProfile-driven parallel fan-out, log-odds scoring, warming & hedging
  plan.rs                 # Owned/deserializable 4x4 effort plans, gap rule, box rule
  boxes.rs                # 9-ordinal, every-instance, and 16-window box requests + incremental Vote
  pillow.rs               # Byte-exact Pillow bilinear resizer & libjpeg baseline 4:2:0 q90 encoder
  liquid.rs               # Positional-image dialect rewriter & 1.5 tokens/patch billing guard
  captcha/                # Service profiles (recaptcha/, hcaptcha/ + canvas grid segmentation)
models/                   # Embedded schema_version=1 TOML configs + custom-decision-model.toml example
eval/                     # Standalone Python toolkit (`hypercaptcha-eval`: dataset/label/record/fit/sweep/export/summarize)
scripts/                  # Browser runners (`chrome_lab.py`, `camoufox_lab.py`, `hcaptcha_lab.py`)
web/                      # Embedded local lab UI (`recaptcha/`, `hcaptcha/`, `lab.css`)
tests/                    # Rust integration tests, Python eval tests, image-free `.json.gz` parity fixtures
```

---

## Documentation

- [docs/SETUP.md](docs/SETUP.md) — Rust/Python/browser installation, Google reCAPTCHA & hCaptcha key setup, `.env.example`, and localhost security model
- [docs/RESULTS.md](docs/RESULTS.md) — Full verified empirical report (`v0`/`v1`/`v2` datasets, effort ladders, negative findings, live benchmarks, hCaptcha status, unmeasured live scope)
- [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) — Running unmeasured models, writing custom TOML configs, full offline tuning & parity-fixture loop, and adding new wire dialects
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Server routes, JSONL schemas, decision pipeline, `pillow.rs`, `liquid.rs`, hCaptcha canvas segmentation, and `eval/` internals
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) — Historical November 15, 2026 live 50-attempt `Jev` vs. `d3-ultra` failure review
- [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [LICENSE](LICENSE)
