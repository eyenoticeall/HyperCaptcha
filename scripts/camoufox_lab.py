"""Run the local reCAPTCHA lab in Camoufox."""

import argparse
import json
from urllib.parse import urlsplit

from camoufox.sync_api import Camoufox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8765")
    run = parser.add_mutually_exclusive_group()
    run.add_argument(
        "--score",
        action="store_true",
        help="Click the score-mode button once, print the Google result, and exit.",
    )
    run.add_argument(
        "--smoke",
        action="store_true",
        help="Verify the public Google v2 test widget once and exit.",
    )
    args = parser.parse_args()
    url = urlsplit(args.url)
    if (
        url.scheme != "http"
        or url.hostname != "localhost"
        or url.username is not None
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        parser.error("--url must be the localhost URL printed by the lab server.")

    with Camoufox(headless=False) as browser:
        page = browser.new_page()
        print(f"Camoufox browser {browser.version}", flush=True)
        response = page.goto(f"{args.url}#source=camoufox", wait_until="domcontentloaded")
        if response is None or not response.ok:
            parser.error("The lab did not return a successful response.")
        page.get_by_role("heading", name="Google widget").wait_for()
        if args.score or args.smoke:
            config_response = page.request.get(f"{args.url.rstrip('/')}/config")
            if not config_response.ok:
                parser.error("The lab config could not be loaded.")
            config = config_response.json()
            if args.smoke:
                if not config["test"]:
                    parser.error("--smoke requires --mode v2 --google-test-key.")
                page.frame_locator('iframe[title="reCAPTCHA"]').get_by_role(
                    "checkbox"
                ).click(timeout=45000)
            elif config["score_based"]:
                page.get_by_role("button", name="Request assessment").click(timeout=25000)
            else:
                parser.error("--score requires --mode v3 or --mode v3-enterprise.")
            result = page.locator("#google-result")
            result.wait_for(state="visible", timeout=35000)
            record = json.loads(result.inner_text())
            print(json.dumps(record, indent=2))
            if args.score and config["provider"]:
                model = page.locator("#model-result")
                model.wait_for(state="visible", timeout=75000)
                model_record = json.loads(model.inner_text())
                print(json.dumps(model_record, indent=2))
                if model_record["result"]["status"] == "api_error":
                    raise SystemExit(1)
            if not record["result"].get("valid"):
                raise SystemExit(1)
            return
        print("Camoufox is open. Close the lab tab or press Ctrl-C to exit.", flush=True)
        try:
            page.wait_for_event("close", timeout=0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
