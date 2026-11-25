"""Shared helpers for HyperCaptcha's Camoufox-based local lab runners."""

import argparse
import ipaddress
import json
import math
import socket
from contextlib import contextmanager
from urllib.parse import urlsplit
from uuid import uuid4

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as BrowserTimeout
except ImportError:

    class PlaywrightError(Exception):
        """Fallback error type when Playwright is not installed."""

    class BrowserTimeout(PlaywrightError):
        """Fallback timeout error type when Playwright is not installed."""


BROWSER = "camoufox"
SOURCE = "camoufox"

MODEL_WAIT_JS = """() => {
    const result = document.querySelector('#model-result');
    if (result && !result.hidden && result.textContent.trim().length > 0) {
        return {kind: 'result', text: result.textContent};
    }
    const status = document.querySelector('#model-status');
    if (
        status && !status.hidden && status.textContent
        && status.textContent !== 'Waiting for the model…'
    ) {
        return {kind: 'error', text: status.textContent};
    }
    const nextButton = document.querySelector('#next');
    if (nextButton && !nextButton.hidden && !nextButton.disabled) {
        return {kind: 'error', text: (status && status.textContent) || 'Model request failed.'};
    }
    return null;
}"""


def positive_int(value):
    """Parse a strictly positive integer CLI argument."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def non_negative_int(value):
    """Parse a non-negative integer CLI argument."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


def positive_float(value):
    """Parse a finite strictly positive float CLI argument."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return number


def is_local_hostname(hostname, resolver=socket.getaddrinfo):
    """Return whether `hostname` is localhost, a `.localhost` subdomain, or resolves only to loopback."""
    if not hostname:
        return False
    host = hostname.rstrip(".").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        pass
    try:
        infos = resolver(host, None)
    except OSError:
        return False
    addresses = [info[4][0] for info in infos if info and len(info) >= 5 and info[4]]
    if not addresses:
        return False
    try:
        return all(ipaddress.ip_address(addr.split("%", 1)[0]).is_loopback for addr in addresses)
    except ValueError:
        return False


def validate_lab_url(raw_url, resolver=socket.getaddrinfo):
    """Validate and normalize `--url`, ensuring it targets the local lab server."""
    parts = urlsplit(raw_url)
    try:
        _ = parts.port
    except ValueError:
        return None
    if (
        parts.scheme != "http"
        or not is_local_hostname(parts.hostname, resolver=resolver)
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        return None
    return raw_url.rstrip("/")


def validate_captcha_config(settings, expected_captcha):
    """Verify `/config['captcha']` matches `expected_captcha`, returning an error string on mismatch."""
    actual = settings.get("captcha")
    if actual == expected_captcha:
        return None
    if expected_captcha == "recaptcha" and actual == "hcaptcha":
        return (
            "The lab server is running hCaptcha. Use scripts/hcaptcha_lab.py instead, "
            "or restart the server for reCAPTCHA."
        )
    if expected_captcha == "hcaptcha" and actual == "recaptcha":
        return (
            "The lab server is running reCAPTCHA. Use scripts/recaptcha_lab.py instead, "
            "or restart the server with --captcha hcaptcha."
        )
    return f"Expected a {expected_captcha} lab server (captcha={expected_captcha!r}), got {actual!r}."


def supports_vision(settings, *, recaptcha_compat=False):
    """Return whether `/config` indicates a vision-capable backend."""
    flag = settings.get("supports_vision")
    if flag is not None:
        return bool(flag)
    provider = settings.get("provider")
    if recaptcha_compat:
        return provider not in (None, "openai-completions")
    return provider in ("jev", "vllm", "liquid", "custom")


def requires_vision_check(settings):
    """Return whether `/config` requires a `/vision-check` before submitting."""
    flag = settings.get("requires_vision_check")
    if flag is not None:
        return bool(flag)
    return settings.get("provider") in ("jev", "vllm", "liquid", "custom")


def resolve_model_name(settings, record):
    """Return the trial's display model name from `/config` or model replies."""
    model_name = settings.get("model_name")
    if model_name:
        return model_name
    for image_round in record.get("rounds", ()):
        model = image_round.get("model")
        if isinstance(model, dict) and model.get("model_name"):
            return model["model_name"]
    provider = settings.get("provider")
    model = settings.get("model")
    if provider and model:
        return f"{provider}/{model}"
    return model or None


def post(page, path, trial_id, fields):
    """POST JSON to a lab endpoint with matching Origin, trial_id, and source."""
    origin = urlsplit(page.url)._replace(path="", query="", fragment="").geturl()
    response = page.request.post(
        f"{origin}/{path}",
        headers={"Origin": origin},
        data={
            "trial_id": trial_id,
            "source": urlsplit(page.url).fragment.removeprefix("source=") or SOURCE,
            **fields,
        },
        timeout=125_000,
    )
    try:
        result = response.json()
    except ValueError as error:
        raise RuntimeError(f"{path} returned HTTP {response.status}") from error
    if not response.ok:
        raise RuntimeError(result.get("error", f"{path} returned HTTP {response.status}"))
    return result


def fetch_lab_config(browser, url):
    """Fetch and parse `/config` from the local lab server."""
    with browser.new_context() as context:
        response = context.request.get(f"{url}/config")
        if not response.ok:
            raise RuntimeError(f"The lab returned HTTP {response.status}.")
        return response.json()


def run_vision_check(browser, url, output_dir, *, source=SOURCE):
    """Run `/vision-check`, log it to `vision-checks.jsonl`, print it, and return `(check_id, check)`."""
    vision_check_id = str(uuid4())
    with browser.new_context() as context:
        page = context.new_page()
        page.goto(f"{url}/config#source={source}", wait_until="commit")
        check = post(page, "vision-check", vision_check_id, {})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vision-checks.jsonl").open("a", encoding="utf-8") as log:
        print(json.dumps(check), file=log)
    print(json.dumps(check), flush=True)
    return vision_check_id, check


def model_reply_failed(model_record):
    """Return whether a `/model` response did not produce an answered prediction."""
    if not isinstance(model_record, dict):
        return True
    result = model_record.get("result")
    if not isinstance(result, dict):
        return True
    return result.get("status") != "answered"


def format_model_http_error(http_error, status_text):
    """Format a non-OK `/model` HTTP error for stderr."""
    detail = (status_text or "").strip()
    if http_error is not None:
        status, body_error = http_error
        message = body_error or detail or "Model request failed."
        return f"Model request failed (HTTP {status}): {message}"
    return f"Model request failed: {detail or 'Unknown error.'}"


@contextmanager
def open_camoufox(headless: bool = True):
    """Launch Camoufox with Playwright's sync API."""
    try:
        from camoufox.sync_api import Camoufox
    except ImportError as error:
        raise RuntimeError(
            "Camoufox is not installed. Run with `uv run --extra browser ...` "
            "and fetch the browser once with `uv run --extra browser python -m camoufox fetch`."
        ) from error
    with Camoufox(headless=headless) as browser:
        yield browser


def run_interactive(browser, url, *, heading, source=SOURCE):
    """Open the lab page in a headed Camoufox window and wait until closed."""
    with browser.new_context(locale="en-US") as context:
        page = context.new_page()
        print(f"Camoufox browser {browser.version}", flush=True)
        response = page.goto(f"{url}/#source={source}", wait_until="domcontentloaded")
        if response is None or not response.ok:
            raise RuntimeError("The lab did not return a successful response.")
        page.get_by_role("heading", name=heading).wait_for()
        print("Camoufox is open. Close the lab tab or press Ctrl-C to exit.", flush=True)
        try:
            page.wait_for_event("close", timeout=0)
        except KeyboardInterrupt:
            pass


def run_smoke(browser, url, *, heading, click_checkbox, result_selector, source=SOURCE):
    """Verify a public test-key widget once, print the JSON verdict, and return the record."""
    with browser.new_context(locale="en-US") as context:
        page = context.new_page()
        print(f"Camoufox browser {browser.version}", flush=True)
        response = page.goto(f"{url}/#source={source}", wait_until="domcontentloaded")
        if response is None or not response.ok:
            raise RuntimeError("The lab did not return a successful response.")
        page.get_by_role("heading", name=heading).wait_for()
        click_checkbox(page)
        result = page.locator(result_selector)
        result.wait_for(state="visible", timeout=35_000)
        record = json.loads(result.inner_text())
        print(json.dumps(record, indent=2), flush=True)
        return record
