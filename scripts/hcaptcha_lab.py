"""Capture or answer hCaptcha grid challenges on the local lab page.

Only hCaptcha's grid task is handled: nine pictures and one instruction. hCaptcha also serves
tasks of other kinds (dragging a piece, clicking a point, following a moving object). Those are
passed over with the widget's own refresh button and counted, so they show in the results apart
from wrong answers.

hCaptcha draws the pictures on a canvas, so the page has no element per picture. The runner
sends a screenshot of the canvas to the lab, which finds the nine pictures, and clicks the
middle of each picture the model selects.
"""

import argparse
import base64
import json
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

from playwright.sync_api import TimeoutError as BrowserTimeout
from playwright.sync_api import sync_playwright

CHECKBOX_FRAME = 'iframe[title*="checkbox"]'
CHALLENGE_FRAME = 'iframe[title*="hCaptcha challenge"]'


def post(page, path, trial_id, fields):
    origin = urlsplit(page.url)._replace(path="", query="", fragment="").geturl()
    response = page.request.post(
        f"{origin}/{path}",
        headers={"Origin": origin},
        data={
            "trial_id": trial_id,
            "source": urlsplit(page.url).fragment.removeprefix("source="),
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


class Challenge:
    """The open challenge, read from the widget's frame."""

    def __init__(self, page):
        self.page = page
        self.outer = page.locator(CHALLENGE_FRAME)
        self.frame = page.frame_locator(CHALLENGE_FRAME)
        self.prompt = self.frame.locator("#prompt-question")
        self.header = self.frame.locator(".challenge-header")
        self.canvas = self.frame.locator(".challenge-view canvas")
        self.button = self.frame.locator(".button-submit")
        self.refresh = self.frame.locator(".refresh.button")
        self.error = self.frame.locator(".display-error")

    def is_open(self):
        return self.outer.is_visible() and self.prompt.is_visible()

    def instruction(self):
        return " ".join(self.prompt.inner_text().split())

    def button_label(self):
        """The button's spoken label, such as "Skip Challenge, page 1 of 2"."""
        return self.button.get_attribute("aria-label") or ""

    def pages(self):
        """The page shown and how many the challenge has."""
        words = self.button_label().replace(",", " ").split()
        if "page" in words and "of" in words:
            at = words.index("page")
            try:
                return int(words[at + 1]), int(words[at + 3])
            except (IndexError, ValueError):
                pass
        return 1, 1

    def examples(self):
        """How many example pictures sit beside the instruction."""
        return self.frame.locator(".challenge-header .examples .example-wrapper > *").count()

    def task_area(self):
        """The part of the canvas below the instruction, in page coordinates."""
        canvas = self.canvas.bounding_box()
        header = self.header.bounding_box()
        if canvas is None:
            raise RuntimeError("The challenge has no canvas.")
        top = canvas["y"]
        if header is not None and header["height"] > 0:
            top = max(top, header["y"] + header["height"])
        return {
            "x": canvas["x"],
            "y": top,
            "width": canvas["width"],
            "height": canvas["y"] + canvas["height"] - top,
        }

    def settled_screenshot(self, poll, patience=3.0):
        """A screenshot of the task area once two in a row are the same. A task that keeps
        moving never settles; its last screenshot is returned with `settled` false."""
        deadline = monotonic() + patience
        previous = None
        while True:
            area = self.task_area()
            shot = self.page.screenshot(clip=area)
            if shot == previous:
                return shot, area, True
            if monotonic() >= deadline:
                return shot, area, False
            previous = shot
            self.page.wait_for_timeout(poll)


def wait_for(page, condition, seconds, poll):
    """The first true value `condition` returns, or None when the time runs out."""
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        value = condition()
        if value:
            return value
        page.wait_for_timeout(poll)
    return None


def read_task(challenge, page, trial_id, poll, stale=None, patience=5.0):
    """Waits for a task to be drawn and asks the lab for its pictures.

    A challenge shows its instruction before its pictures, and a screenshot taken in between
    holds no grid. So does one taken while the previous page is still on the canvas (`stale`).
    The lab is asked again until it finds a grid or the time runs out, and only then is the
    task taken for one of another kind."""
    deadline = monotonic() + patience
    while True:
        shot, area, settled = challenge.settled_screenshot(poll)
        if shot != stale:
            found = post(page, "pictures", trial_id, {"image": base64.b64encode(shot).decode()})
            if found["result"]["status"] == "grid" or monotonic() >= deadline:
                return shot, area, settled, found
        elif monotonic() >= deadline:
            return shot, area, settled, {"result": {"status": "no_grid", "detail": "unchanged"}}
        page.wait_for_timeout(max(poll, 150))


def verdict_on_page(page):
    result = page.locator("#verdict")
    return json.loads(result.inner_text()) if result.is_visible() else None


def widget_state(page):
    return page.evaluate("document.body.dataset.widget || ''")


def run_trial(page, args, url, record):
    page.goto(f"{url}/#source={args.browser}", wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#trial-id').textContent.length > 0")
    trial_id = page.locator("#trial-id").inner_text()
    record["trial_id"] = trial_id
    poll = 50 if args.fast else 200
    challenge = Challenge(page)
    started = monotonic()
    page.frame_locator(CHECKBOX_FRAME).locator("#checkbox").click(timeout=35_000)

    def elapsed():
        return round((monotonic() - started) * 1000)

    def opened():
        verdict = verdict_on_page(page)
        if verdict is not None:
            return ("verdict", verdict)
        state = widget_state(page)
        if state.startswith("error"):
            return ("error", state)
        return ("challenge", None) if challenge.is_open() else None

    first = wait_for(page, opened, 25, poll)
    if first is None:
        raise RuntimeError("hCaptcha neither returned a token nor showed a challenge.")
    kind, value = first
    if kind == "error":
        record.update(outcome="widget_error", detail=value)
        return
    if kind == "verdict":
        valid = value["result"].get("valid")
        record.update(
            outcome="checkbox_only" if valid else "verification_error",
            hcaptcha=value,
            checkbox_to_verification_ms=elapsed(),
        )
        return

    directory = args.output / trial_id
    directory.mkdir()
    record["checkbox_to_challenge_ms"] = elapsed()
    rounds = record["rounds"]
    passed_over = record.setdefault("passed_over", [])
    # A challenge is judged once, after its last page. `stages` counts the challenges answered
    # and `rejections` the ones hCaptcha judged wrong.
    stages = rejections = 0
    # The canvas as it was left when the last page was answered.
    stale = None
    while True:
        if len(rounds) >= args.max_rounds:
            record["outcome"] = "round_limit"
            return
        instruction = challenge.instruction()
        shot, area, settled, found = read_task(challenge, page, trial_id, poll, stale)
        stale = None
        number = len(rounds) + len(passed_over) + 1
        (directory / f"task-{number:02d}.png").write_bytes(shot)
        image = base64.b64encode(shot).decode()
        if found["result"]["status"] != "grid":
            if challenge.pages()[0] > 1:
                # Part of this challenge is answered already. Refreshing would throw it away.
                record["outcome"] = "grid_lost_mid_challenge"
                return
            # Another kind of task. Ask hCaptcha for a different one, as a person may.
            passed_over.append({"prompt": instruction, "settled": settled, "ms": elapsed()})
            if len(passed_over) > args.max_refreshes:
                record["outcome"] = "no_grid_task"
                return
            challenge.refresh.click(timeout=5000)
            changed = wait_for(
                page,
                lambda: challenge.is_open()
                and (
                    challenge.instruction() != instruction
                    or page.screenshot(clip=challenge.task_area()) != shot
                ),
                10,
                poll,
            )
            if not changed:
                record["outcome"] = "refresh_did_nothing"
                return
            continue

        at, of = challenge.pages()
        entry = {
            "prompt": instruction,
            "page": at,
            "pages": of,
            "examples": challenge.examples(),
            "settled": settled,
            "pictures": found["result"]["pictures"],
            "shown_ms": elapsed(),
        }
        rounds.append(entry)
        if "grid_png" in found:
            (directory / f"grid-{number:02d}.png").write_bytes(base64.b64decode(found["grid_png"]))
        if not args.submit:
            record["outcome"] = "captured"
            return

        asked = monotonic()
        answer = post(
            page,
            "model",
            trial_id,
            {"prompt": instruction, "image": image, "pictures": entry["pictures"]},
        )
        answer.pop("grid_png", None)
        entry["model_ms"] = round((monotonic() - asked) * 1000)
        entry["model"] = answer
        prediction = answer["result"].get("prediction")
        if answer["result"].get("status") != "answered" or prediction.get("type") != "tiles":
            record["outcome"] = "model_error"
            return
        picks = prediction["value"]
        entry["picks"] = picks
        for pick in picks:
            left, top, width, height = entry["pictures"][pick - 1]
            page.mouse.click(
                area["x"] + (left + width / 2) * area["width"],
                area["y"] + (top + height / 2) * area["height"],
            )
        entry["label_before_submit"] = challenge.button_label()
        stale = page.screenshot(clip=area)
        challenge.button.click(timeout=5000)
        entry["answered_ms"] = elapsed()

        def after_answer():
            verdict = verdict_on_page(page)
            if verdict is not None:
                return ("verdict", verdict)
            state = widget_state(page)
            if state.startswith(("error", "closed", "challenge-expired")):
                return ("widget", state)
            if not challenge.is_open():
                return None
            now_at, _ = challenge.pages()
            if at < of and now_at == at + 1:
                return ("next_page", None)
            if at == of and (now_at == 1 and challenge.error.is_visible()):
                return ("rejected", None)
            if at == of and challenge.instruction() != instruction:
                return ("rejected", None)
            return None

        outcome = wait_for(page, after_answer, 20, poll)
        if outcome is None:
            # The last page was answered and the same instruction is still shown: hCaptcha
            # served another challenge of the same kind, which it does after a wrong answer.
            if at == of and challenge.is_open() and challenge.pages()[0] == 1:
                outcome = ("rejected", None)
            else:
                record["outcome"] = "no_response_after_answer"
                return
        kind, value = outcome
        entry["result"] = kind
        if kind == "next_page":
            continue
        stages += 1
        record["stages"] = stages
        if kind == "verdict":
            valid = bool(value["result"].get("valid"))
            record.update(
                outcome="model_solved" if valid else "verification_error",
                hcaptcha=value,
                checkbox_to_verification_ms=elapsed(),
            )
            return
        if kind == "widget":
            record.update(outcome="widget_closed", detail=value)
            return
        rejections += 1
        record["rejections"] = rejections
        if rejections >= max(args.max_stages, 1):
            record["outcome"] = "model_failed"
            return


@contextmanager
def open_browser():
    with (
        sync_playwright() as playwright,
        playwright.chromium.launch(channel="chrome", headless=True) as browser,
    ):
        yield browser


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://hypercaptcha.localhost:8765")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--max-rounds", type=int, default=24, help="The most grid pages answered in one trial."
    )
    parser.add_argument(
        "--max-stages", type=int, default=1,
        help="Keep going after hCaptcha rejects an answer, until it passes the trial or has"
        " rejected this many challenges.",
    )
    parser.add_argument(
        "--max-refreshes", type=int, default=12,
        help="The most tasks of other kinds to pass over in one trial before giving up.",
    )
    parser.add_argument(
        "--device-scale-factor", type=float, default=2.0,
        help="Browser pixel density. hCaptcha draws its canvas at twice the page's size, so 2"
        " captures the pictures at the detail they were drawn with.",
    )
    parser.add_argument("--output", default=Path("reports/hcaptcha"), type=Path)
    parser.add_argument(
        "--submit", action="store_true", help="Ask the configured model and submit its selections."
    )
    parser.add_argument(
        "--fast", action=argparse.BooleanOptionalAction, default=True,
        help="Check the page every 50 milliseconds. --no-fast checks every 200.",
    )
    args = parser.parse_args()
    args.browser = "chrome"
    parts = urlsplit(args.url)
    if (
        parts.scheme != "http"
        or parts.username is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        parser.error("--url must be the URL printed by the lab server.")
    url = args.url.rstrip("/")

    with open_browser() as browser:
        with browser.new_context() as config:
            response = config.request.get(f"{url}/config")
            if not response.ok:
                parser.error(f"The lab returned HTTP {response.status}.")
            settings = response.json()
        if settings.get("captcha") != "hcaptcha":
            parser.error("Start the lab with --captcha hcaptcha.")
        if settings.get("test"):
            parser.error("hCaptcha's test key never shows a challenge. Use your own key.")
        if args.submit and settings.get("provider") != "vllm":
            parser.error("Start the lab with --provider vllm before using --submit.")

        args.output.mkdir(parents=True, exist_ok=True)
        vision_check_id = None
        if args.submit:
            vision_check_id = str(uuid4())
            with browser.new_context() as context:
                page = context.new_page()
                page.goto(f"{url}/config#source={args.browser}", wait_until="commit")
                check = post(page, "vision-check", vision_check_id, {})
            with (args.output / "vision-checks.jsonl").open("a") as log:
                print(json.dumps(check), file=log)
            print(json.dumps(check), flush=True)
            if not check["result"]["passed"]:
                parser.exit(1, "The decision model did not pass the vision check. No challenges were submitted.\n")
        for _ in range(args.trials):
            with browser.new_context(
                locale="en-US",
                viewport={"width": 1280, "height": 900},
                device_scale_factor=args.device_scale_factor,
            ) as context:
                page = context.new_page()
                record = {"trial_id": None, "rounds": []}
                try:
                    run_trial(page, args, url, record)
                except (BrowserTimeout, RuntimeError) as error:
                    record.update(outcome="runner_error", detail=str(error).splitlines()[0])
                if vision_check_id is not None:
                    record["vision_check_id"] = vision_check_id
            record.update(captcha="hcaptcha", browser=args.browser, browser_version=browser.version)
            with (args.output / "trials.jsonl").open("a") as log:
                print(json.dumps(record), file=log)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
