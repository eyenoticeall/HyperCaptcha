# Security Policy

## Research Scope & Responsible Use

HyperCaptcha is a localhost-only browser and model evaluation harness intended for empirical research on visual decision models, calibration, latency, and verification behavior.

- **Localhost-only lab**: The harness serves its own local HTML test pages (`web/recaptcha/`, `web/hcaptcha/`) on loopback (`127.0.0.1` / `localhost`) using either researcher-owned CAPTCHA site/secret keys or official vendor test keys (`--test-keys`).
- **No third-party targeting**: Do not use this repository to target, automate against, or bypass protections on third-party websites or services you do not own or have explicit written authorization to test.
- **Authorized experimentation**: Users are responsible for complying with the terms of service of their CAPTCHA providers, model APIs, and applicable laws.

## Credential & Data Handling Policy

- **Never commit secrets**: Do not commit `.env`, `.env.*` (other than `.env.example`), API keys, site/secret keys, bearer tokens, or cookies to this repository.
- **Never commit raw CAPTCHA datasets or recordings**: Keep local captures, screenshots, video recordings, raw human labels, and live run logs inside gitignored directories (`/reports/`, `/datasets/`, `/recordings/`, `/references/`).
- **Sanitized test fixtures only**: Committed regression fixtures in `tests/fixtures/` must remain image-free and label-free (containing only sanitized request/score payloads and synthetic decision traces).

## Reporting a Vulnerability

If you discover a security vulnerability in HyperCaptcha (for example, accidental credential exposure, unsafe request handling, SSRF/path traversal in the local lab server, or dependency vulnerabilities):

1. **Do not open a public GitHub issue** with exploit details or sensitive information.
2. Report the issue privately via GitHub Security Advisories (**Security -> Advisories -> Report a vulnerability**) on this repository, or contact the maintainers privately.
3. Include a clear description of the issue, reproduction steps, affected files/versions, and any suggested mitigation.

Maintainers will acknowledge receipt promptly, assess severity, and coordinate disclosure once a fix is available.
