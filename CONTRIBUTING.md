# Contributing to HyperCaptcha

Contributions that improve reproducibility, model evaluation coverage, server reliability, and documentation are welcome. Before submitting changes, please review [SECURITY.md](SECURITY.md) for the localhost-only research scope and data-handling policy.

## Development Checks

### Rust Server & Decision Harness

Run the full Rust verification suite before opening a pull request:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
```

### Python Lab & Evaluation Scripts

Verify Python scripts under `scripts/` and run any Python evaluation or unit tests in your local virtual environment:

```bash
python3 -m py_compile scripts/chrome_lab.py scripts/camoufox_lab.py scripts/hcaptcha_lab.py
pytest
```

## Contributing a New Decision-Model Configuration

Decision-model profiles (prompts, multi-view weights, effort levels, thresholds, and box-veto rules) live under `src/captcha/<service>/<provider>/<model>.rs`:

1. Add a model module under the appropriate service and provider directory (for example, `src/captcha/recaptcha/<provider>/<model>.rs`) defining its `Model` spec, weights, and effort levels.
2. Register the model in `src/captcha/<service>/mod.rs` and wire provider routing in `src/providers.rs` if adding a new provider endpoint.
3. Document the empirical evaluation methodology, dataset split, token/request budgets, and trade-offs rather than cherry-picked runs.
4. Keep offline experiments and live logs in gitignored paths (`/reports/`, `/datasets/`), never in git history.

## Adding Parity Fixtures

Regression and parity tests in `tests/model_routes.rs` validate exact decision-rule behavior without hitting live APIs or bundling CAPTCHA imagery:

- Place compressed fixtures under `tests/fixtures/` (for example, `tests/fixtures/<provider>_<model>_4x4.json.gz`).
- **Image-free and label-free requirement**: Fixtures must contain only sanitized model score/box outputs and synthetic decision inputs — never raw CAPTCHA images, third-party URLs, API keys, or proprietary datasets.
- Add or update test cases in `tests/model_routes.rs` to assert exact tile selection, score parity, and effort-level behavior, and confirm `cargo test --locked` passes.
