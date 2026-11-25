# Adding & Tuning Decision Models

HyperCaptcha separates **model configuration** from **Rust harness code**. Every built-in model lives in a declarative TOML file (`schema_version = 1`) under `models/`, and any new or custom decision-model API can be evaluated, recorded, fitted, swept, and loaded via `--model-config <path>` without touching Rust.

Built-in configurations:

- `models/recaptcha/vllm/d3-ultra.toml` (`vLLM/d3-ultra`)
- `models/recaptcha/liquid/d1.2-ultra.toml` (`Liquid/d1.2-ultra`)
- `models/recaptcha/liquid/d1.2.toml` (`Liquid/d1.2`)
- `models/recaptcha/typesafe/jev.toml` (`typesafe/Jev`)
- `models/hcaptcha/vllm/d3-ultra.toml` (`vLLM/d3-ultra` on hCaptcha 3×3)
- `models/examples/custom-decision-model.toml` (template for custom decision APIs)

---

## Option A: Run a New Model on an Existing Provider (`vllm`, `liquid`, `jev`)

By default, HyperCaptcha refuses to apply another model's fitted logistic weights or effort levels to an unmeasured model ID (because decision-model score calibration varies widely across models). You have two ways to run an unmeasured model on a built-in provider immediately:

### 1. Evaluate with Equal or Weighted Averaging (`--decision-rule average`)

```sh
# Try an unmeasured model on vLLM or Liquid using equal 50/50 view averaging
VLLM_SR_API_KEY=... cargo run --release --locked -- serve \
  --mode v2-enterprise \
  --provider vllm \
  --model my-org/candidate-model \
  --decision-rule average

# Or supply custom whole-grid weights (implies --decision-rule average)
LIQUID_API_KEY=... cargo run --release --locked -- serve \
  --mode v2-enterprise \
  --provider liquid \
  --model d1.3-preview \
  --whole-grid-weight-3x3 0.5 --whole-grid-weight-4x4 0.35
```

### 2. Record Full Unpruned Outputs for Offline Fitting (`--record`)

`--record` (backward-compatible alias `--record-4x4`) queries every signal any effort tier can use — whole grid, both close-up passes (`capture` and `506` px), skip presence question, and every supported box request (`whole`, `windows`, and `every` if supported) — waits for all replies, logs them all, and leaves the answer unmodified by rules. On 3×3 grids it records standard two-view scores using equal averaging:

```sh
LIQUID_API_KEY=... cargo run --release --locked -- serve \
  --mode v2 --google-test-key \
  --provider liquid \
  --model d1.3-preview \
  --record
```

On **hCaptcha** (`--captcha hcaptcha`), `vllm`, `liquid`, `custom`, and `jev` (with `--jev-vision average`, `--record`, or `--model-config`) are all supported.

### Provider-Agnostic CLI Flags & Aliases

| Preferred Flag | Backward-Compatible Alias | Purpose |
| --- | --- | --- |
| `--decision-rule <average\|logistic>` | `--vllm-decision` | Choose weighted averaging vs. fitted logistic rule |
| `--whole-grid-weight-3x3 <0..1>` | `--vllm-whole-grid-weight-3x3` | Whole-grid weight on 3×3 (implies `average`) |
| `--whole-grid-weight-4x4 <0..1>` | `--vllm-whole-grid-weight-4x4` | Whole-grid weight on 4×4 (implies `average`) |
| `--box-wait` | `--vllm-box-wait` | Wait for and log all box replies instead of early-returning |
| `--close-ups-per-request <2..=9>` | `--vllm-close-ups-per-request` | Max tile crops per close-up request (default `4`) |
| `--cold-connections` | `--vllm-cold-connections` | Disable startup connection warming and idle keep-warm pings |
| `--record` | `--record-4x4` | Record all scoring passes, skip check, and supported box asks |

---

## Option B: Add a Custom Decision-Model API via TOML (`--model-config`)

If you host or call a System One–compatible decision model on another endpoint, copy [`models/examples/custom-decision-model.toml`](../models/examples/custom-decision-model.toml) and load it with `--model-config`:

```toml
schema_version = 1
service = "recaptcha"                  # "recaptcha" | "hcaptcha"
name = "MyProvider/my-decision-model"  # Display/CLI name (<provider>/<model>)
provider = "custom"                    # "vllm" | "liquid" | "jev" | "custom"
model_id = "my-decision-model-v1"      # Wire model ID sent in request JSON

# Required when provider = "custom".
# Optional for built-in providers (inherits provider defaults; specified fields override).
[endpoint]
dialect = "systemone-assets"           # "systemone-assets" | "systemone-images"
base_url = "https://api.example.com/v1"
api_key_env = "MY_DECISION_API_KEY"
# extra_api_keys_env = "MY_DECISION_EXTRA_API_KEYS"
image_format = "jpeg"                  # "jpeg" | "png"
crop_filter = "bilinear"               # "bilinear" | "nearest"
gzip = false
max_images = 16                        # e.g. 8 for positional-images, 16 for named-assets
multi_state_windows = true             # false => 16 parallel single-state window box requests
supports_every_box = false             # true => model supports type="boxes" multi-instance query
return_meta = false
hedge_on_stall = false                 # enable ONLY for deterministic models
warm_connections = false
min_patch_tokens = false               # true => enforce 1.5 input tokens per 32x32 image patch floor
# extra_headers = { "X-Custom-Header" = "example" }

# Optional before tuning (defaults to two-view averaging on 3x3 and Plan::PLAIN on 4x4).
[tuning]
default_decision = "average"           # "logistic" | "average"

[tuning.average_4x4]
close_ups = ["capture"]                # "capture" or positive integer longer-side px (e.g. 506)
select_from = 0.5
# skip_below = 0.7
```

Run it directly (CLI flags such as `--base-url`, `--effort`, `--decision-rule`, `--tile-threshold`, `--cold-connections`, and `--record` still override the TOML where applicable):

```sh
MY_DECISION_API_KEY=... cargo run --release --locked -- serve \
  --mode v2-enterprise \
  --model-config models/examples/custom-decision-model.toml \
  --decision-rule average
```

### Wire Dialects (`endpoint.dialect`)

| Dialect | Wire Format | Image Reference Style | Typical Providers |
| --- | --- | --- | --- |
| `systemone-assets` | Named `"assets": { "<id>": { "type": "image", "format": "...", "data_base64": "..." } }` map | Referenced by asset key name inside `state` and `questions` | vLLM Semantic Router, TypeSafe Jev |
| `systemone-images` | Ordered `"images": ["data:image/jpeg;base64,...", ...]` array | Referenced positionally (`"image 1"` in object values, `"PHOTO is image 2."` in instruction lists), rewritten by `src/liquid.rs` | Liquid Decisions API |

### `[endpoint]` Capability Flags Reference

| Field | Type | Default (`custom`) | Behavior |
| --- | --- | --- | --- |
| `dialect` | string | *(required)* | `"systemone-assets"` or `"systemone-images"`. |
| `base_url` | string | *(required)* | API root including version prefix (e.g., `https://api.example.com/v1`); HyperCaptcha appends `/systemone`. |
| `api_key_env` | string | *(required)* | Environment variable holding the primary Bearer API key. |
| `extra_api_keys_env` | string | none | Optional env var with comma-separated extra keys rotated round-robin across concurrent requests and retries. |
| `image_format` | string | `"jpeg"` | `"jpeg"` (q90) or `"png"` for scoring requests (box requests always use byte-exact Pillow JPEG q90). |
| `crop_filter` | string | `"bilinear"` | `"bilinear"` or `"nearest"` resampling for 4×4 tile close-ups. |
| `gzip` | bool | `false` | Gzip-compress JSON request bodies (`Content-Encoding: gzip`). Set `false` for endpoints that reject compressed bodies. |
| `max_images` | int (`>= 1`) | `16` (`systemone-assets`) / `8` (`systemone-images`) | Maximum pictures allowed in a single request (`8` for Liquid-style endpoints). Automatically caps close-ups per batch to `max_images - 1`. |
| `multi_state_windows` | bool | `true` (`systemone-assets`) / `false` (`systemone-images`) | `true` packs all 16 window box queries into one multi-state request; `false` sends 16 parallel single-state requests. |
| `supports_every_box` | bool | `false` | `true` enables the single-question `type = "boxes"` multi-instance query (`boxes.every`). Rust rejects configs that set `boxes.every` while `supports_every_box = false`. |
| `return_meta` | bool | `false` | Sends `options.return_meta = true` to collect provider queue/compute timings in `provider_meta`. |
| `hedge_on_stall` | bool | `false` | Enables hedged duplicate requests at `1.5 s`, `4.0 s`, and `8.0 s`. **Only enable for deterministic models**; disable for stochastic models. |
| `warm_connections` | bool | `false` | Pre-opens and keeps warm HTTP/1.1 sockets with a noise-image practice request ahead of rounds. |
| `min_patch_tokens` | bool | `false` | Rejects any image-bearing reply billed fewer than `ceil(1.5 * ceil(w/32) * ceil(h/32))` input tokens per picture, catching endpoints that silently drop image attachments. |
| `extra_headers` | table | `{}` | Static HTTP headers added to every API request (e.g., `{ "TypeSafe-Danger-Zone-Images" = "true" }`). |

### Validation Rules Enforced by Rust

When loading any built-in or external TOML config, `src/captcha/mod.rs` strictly validates:
- `schema_version == 1` and unknown keys are rejected (`#[serde(deny_unknown_fields)]`).
- `service` matches `--captcha` (`recaptcha` or `hcaptcha`), and `provider` is a decision provider (`vllm`, `liquid`, `jev`, `custom`).
- For `provider = "custom"`, `[endpoint]` must supply `dialect`, `base_url`, and `api_key_env`.
- `base_url` is a valid HTTP/HTTPS URL, all `extra_headers` have valid HTTP header names/values, and `max_images >= 1`.
- All weights are finite, all probability cutoffs/thresholds (`cutoff`, `select_from`, `skip_below`, `gap.*`, `boxes.*`) lie in `0.0..=1.0`, `close_ups` is non-empty with positive pixel sizes (or `"capture"`), and `boxes.every` is only allowed when `supports_every_box = true`.

---

## Full Offline Tuning & Parity-Fixture Loop (`hypercaptcha-eval`)

Use the standalone Python toolkit (`hypercaptcha-eval`, also runnable as `python -m eval`) to go from raw captured lab runs to a tuned TOML config and a deterministic Rust parity test.

### Step 1: Build a Deduplicated Dataset Manifest (`dataset`)

Scan one or more `chrome_lab.py` or `hcaptcha_lab.py` run directories. Grids are deduplicated by image SHA-256 and grouped by `trial_id` (`group_id`) so grids from the same challenge never leak across cross-validation folds:

```sh
uv run hypercaptcha-eval dataset reports/capture-run-1 reports/capture-run-2 \
  -o datasets/recaptcha_4x4/dataset.jsonl \
  --layout 4 --service recaptcha
```

### Step 2: Label Grids Locally (`label`)

Launch the standalone `127.0.0.1`-only browser labeling UI (supports 3-state `target` / `sliver` / `empty` square marking, `unclear` flag, keyboard navigation, and resume):

```sh
uv run hypercaptcha-eval label \
  --dataset datasets/recaptcha_4x4/dataset.jsonl \
  --labels datasets/recaptcha_4x4/labels.jsonl \
  --port 8787 --layout 4
```

To measure same-day or inter-rater repeatability on a blind subset (recommended before trusting small parameter gains):

```sh
uv run hypercaptcha-eval label \
  --repeat-check datasets/recaptcha_4x4/labels_pass1.jsonl \
                 datasets/recaptcha_4x4/labels_pass2.jsonl
```

### Step 3: Record Raw Model Outputs (`record`)

Build the Rust release binary, then record all views, skip probabilities, and box replies across your dataset. Use `--passes 1` for deterministic models and `--passes 3` for stochastic models (like Liquid or Jev). Output is resumable per `(grid_id, pass_index)` pair and strips `grid_png` automatically:

```sh
cargo build --release --locked

MY_DECISION_API_KEY=... uv run hypercaptcha-eval record \
  --dataset datasets/recaptcha_4x4/dataset.jsonl \
  -o recordings/my-model-4x4.jsonl \
  --binary target/release/hypercaptcha \
  --captcha recaptcha \
  --model-config models/examples/custom-decision-model.toml \
  --passes 3 --workers 2 --retries 5
```

- For a built-in provider without a custom TOML, pass `--provider liquid --model d1.3-preview` instead of `--model-config`.
- To reuse an already-running loopback server (started with `--record`), pass `--server-url http://localhost:8765`.
- Add `--cold-connections` to disable connection warming on the spawned server, or `--grid-max-side-4x4 <px>` to cap input resolution.

### Step 4: Fit Logistic Weights with Grouped K-Fold CV (`fit`)

Fits standardized ridge logistic regression for 3×3 (`[whole_lo, crop_lo, intercept]`) and/or 4×4 (`[whole_lo, crop_lo, nb_whole_lo, nb_crop_lo, intercept]` plus a 2-feature ablation) across `capture`, `506`, and `avg` close-up modes, printing single-view baselines alongside in-sample and grouped out-of-fold accuracy:

```sh
uv run hypercaptcha-eval fit \
  --dataset datasets/recaptcha_4x4/dataset.jsonl \
  --labels datasets/recaptcha_4x4/labels.jsonl \
  --recordings recordings/my-model-4x4.jsonl \
  --layout 4 --folds 5 --seed 7 --lam 1.0 \
  -o recordings/fit_4x4.json
```

Add `--slivers-empty` to check sensitivity when optional slivers are treated as empty squares.

### Step 5: Sweep 4×4 Effort Tiers (`sweep`)

Runs a vectorized NumPy parameter sweep over close-up pass modes, cutoffs, skip thresholds, gap rules, and every recorded box ask (`whole_grid`, `every`, `windows`). Reports in-sample accuracy, grouped K-fold OOF accuracy, cross-pass stability and pass-to-pass flip rates (when `--passes >= 2`), token/request estimates, and paired grouped bootstrap 95% confidence intervals, selecting plateau centers for `low`, `medium`, `high`, and `max`:

```sh
uv run hypercaptcha-eval sweep \
  --dataset datasets/recaptcha_4x4/dataset.jsonl \
  --labels datasets/recaptcha_4x4/labels.jsonl \
  --recordings recordings/my-model-4x4.jsonl \
  --fit recordings/fit_4x4.json \
  --folds 5 --seed 7 --boot 4000 \
  -o recordings/sweep_4x4.json \
  --markdown recordings/sweep_4x4.md
```

Useful options: `--weights w0,w1,w2,w3,b` to hold weights fixed, `--fast` for a coarse smoke sweep, and `--no-collapse` to prevent higher effort tiers from collapsing onto cheaper ones when OOF does not improve.

### Step 6: Export Validated TOML & Parity Fixture (`export`)

Generates a `tomllib`-validated `schema_version = 1` TOML config and (optionally) a deterministic (`mtime = 0`), **image-free and label-free** `.json.gz` parity fixture matching the exact format of `tests/fixtures/*.json.gz`:

```sh
uv run hypercaptcha-eval export \
  --service recaptcha \
  --name "MyProvider/my-decision-model" \
  --provider custom \
  --model-id "my-decision-model-v1" \
  --default-decision logistic \
  --dialect systemone-assets \
  --base-url "https://api.example.com/v1" \
  --api-key-env MY_DECISION_API_KEY \
  --image-format jpeg \
  --crop-filter bilinear \
  --max-images 16 \
  --multi-state-windows \
  --no-supports-every-box \
  --no-gzip \
  --sweep recordings/sweep_4x4.json \
  --fit recordings/fit_4x4.json \
  -o models/recaptcha/myprovider/my-decision-model.toml \
  --fixture tests/fixtures/myprovider_my_decision_model_4x4.json.gz \
  --recordings recordings/my-model-4x4.jsonl \
  --dataset datasets/recaptcha_4x4/dataset.jsonl \
  --pass-index 0
```

### Step 7: Verify in Rust & Python

Test loading the exported TOML immediately with `--model-config`:

```sh
MY_DECISION_API_KEY=... cargo run --release --locked -- serve \
  --mode v2 --google-test-key \
  --model-config models/recaptcha/myprovider/my-decision-model.toml \
  --effort high
```

If contributing a new built-in model to HyperCaptcha:
1. Place its TOML under `models/<service>/<provider>/<model>.toml` and register it with `include_str!` in `src/captcha/<service>/mod.rs`.
2. Add its image-free, label-free `.json.gz` fixture under `tests/fixtures/` and add a fixture-replay parity assertion in `src/providers.rs` (and `tests/test_eval.py`).
3. Confirm all checks pass:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
uv run --extra dev pytest -v
```

### Step 8: Summarize Live Trials (`summarize`)

After running live trials with `scripts/chrome_lab.py` or `scripts/hcaptcha_lab.py`, compute Wilson 95% confidence intervals, layout/challenge breakdowns, timelines, and token usage:

```sh
uv run hypercaptcha-eval summarize reports/my-model-live/trials.jsonl \
  --group-by-condition --markdown reports/my-model-live/summary.md
```

---

## Adding a New Wire Dialect in Rust

If a decision-model provider does not speak either `systemone-assets` or `systemone-images` (for example, a proprietary JSON payload or multimodal logprob format):

1. **Extend `WireDialect` in `src/captcha/mod.rs`:** Add a new enum variant with its kebab-case `#[serde(rename = "...")]` name (and update `VALID_DIALECTS` in `eval/schema.py` if exporting configs from Python).
2. **Add a clean dialect adapter module:** Follow `src/liquid.rs` as a template — keep wire translation and any response sanity checks (such as token-floor guards) isolated in a dedicated module with thorough unit tests.
3. **Dispatch on `WireProfile.dialect` in `src/providers.rs`:** Handle encoding and response decoding for `/vision-check`, whole-grid & close-up scoring, skip presence queries, and box queries without adding `provider == ...` special cases.
4. **Add mock-server integration tests in `tests/model_routes.rs`:** Exercise the new dialect end-to-end against a local mock HTTP server (`GET /config`, `POST /vision-check`, `POST /model`, and effort-level execution) and run `cargo test --locked`.
