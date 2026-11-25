# Contributing to HyperCaptcha

Contributions that improve reproducibility, model evaluation coverage, server reliability, and documentation are welcome. Before submitting changes, please review [SECURITY.md](SECURITY.md) for the localhost-only research scope and data-handling policy.

## Development Checks

### Rust Server & Decision Harness

Run the full Rust verification suite before opening a pull request:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
cargo run --locked -- models >/dev/null
cargo run --locked -- validate-config models/recaptcha/*/*.toml models/hcaptcha/*/*.toml models/examples/custom-decision-model.toml
```

### Python Lab & Evaluation Toolkit

Verify browser runner scripts under `scripts/` and run the offline Python evaluation and parity test suite:

```bash
uv run --extra browser python -m py_compile scripts/_common.py scripts/recaptcha_lab.py scripts/hcaptcha_lab.py
uv run --extra browser python scripts/recaptcha_lab.py --help >/dev/null
uv run --extra browser python scripts/hcaptcha_lab.py --help >/dev/null
uv run --extra dev pytest -v
```

## Contributing a New Decision-Model Configuration

Built-in decision-model configurations (endpoint capabilities, multi-view weights, effort levels, thresholds, and box-veto rules) live as declarative TOML files (`schema_version = 1`) under `models/<service>/<provider>/<model>.toml`. Inspect the shipped models offline with `cargo run --locked -- models`, and see [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for the full schema, capability flags, and offline tuning loop:

1. Prototype and validate your configuration externally with `cargo run --locked -- validate-config <path.toml>` and `--model-config <path.toml>` (or record an unmeasured model on an existing provider with `--record` / `--decision-rule average`).
2. Use `hypercaptcha-eval` (`dataset` -> `label` -> `record` -> `fit` -> `sweep` -> `export --base-config <starter.toml>`) to fit logistic weights, sweep effort tiers, and export a validated TOML file plus an image-free parity fixture.
3. To add a built-in model, place its TOML under `models/<service>/<provider>/<model>.toml`, register it via `include_str!` in `src/captcha/<service>/mod.rs`, and verify it appears in `cargo run --locked -- models` and passes `cargo run --locked -- validate-config` (only edit `src/providers.rs` / `src/captcha/mod.rs` if adding a new wire dialect beyond `systemone-assets` and `systemone-images`).
4. Document the empirical evaluation methodology, dataset split, label version, token/request budgets, and trade-offs rather than cherry-picked runs.
5. Keep offline experiments, raw captures, labels, and live logs in gitignored paths (`/reports/`, `/datasets/`, `/recordings/`), never in git history.

## Adding Parity Fixtures

Regression and parity tests in `src/providers.rs`, `tests/model_routes.rs`, and `tests/test_eval.py` validate exact decision-rule parity without hitting live APIs or bundling CAPTCHA imagery:

- Generate compressed fixtures with `hypercaptcha-eval export --fixture tests/fixtures/<provider>_<model>_4x4.json.gz ...`.
- **Image-free and label-free requirement**: Fixtures must contain only sanitized model score/box outputs and expected tile picks — never raw CAPTCHA images, human ground-truth labels, third-party URLs, API keys, or proprietary datasets.
- Add or update parity assertions in Rust and Python, and confirm both `cargo test --locked` and `uv run --extra dev pytest -v` pass.
