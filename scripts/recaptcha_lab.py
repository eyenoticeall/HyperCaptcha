"""Run, capture, or evaluate Google reCAPTCHA challenges on the local lab in Camoufox."""

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qs, urlsplit

try:
    from ._common import (
        BROWSER,
        MODEL_WAIT_JS,
        SOURCE,
        BrowserTimeout,
        PlaywrightError,
        fetch_lab_config,
        format_model_http_error,
        model_reply_failed,
        non_negative_int,
        open_camoufox,
        positive_float,
        positive_int,
        post,
        requires_vision_check,
        resolve_model_name,
        run_interactive,
        run_smoke,
        run_vision_check,
        supports_vision,
        validate_captcha_config,
        validate_lab_url,
    )
except ImportError:
    from _common import (
        BROWSER,
        MODEL_WAIT_JS,
        SOURCE,
        BrowserTimeout,
        PlaywrightError,
        fetch_lab_config,
        format_model_http_error,
        model_reply_failed,
        non_negative_int,
        open_camoufox,
        positive_float,
        positive_int,
        post,
        requires_vision_check,
        resolve_model_name,
        run_interactive,
        run_smoke,
        run_vision_check,
        supports_vision,
        validate_captcha_config,
        validate_lab_url,
    )


LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# An early answer is used only when no tile's score is this close to the cutoff, because the
# drawn-in tile differs slightly from the browser's rendering and a close score can flip.
EARLY_MARGIN = 0.25


def evaluate_early_answer(answer):
    """Validate an early /model reply. Returns `(drawn_png, problem)`."""
    if not isinstance(answer, dict):
        return None, "no tile answer"
    result = answer.get("result")
    if not isinstance(result, dict):
        return None, "no tile answer"
    prediction = result.get("prediction")
    if (
        result.get("status") != "answered"
        or not isinstance(prediction, dict)
        or prediction.get("type") != "tiles"
    ):
        return None, "no tile answer"
    grid_png = answer.pop("grid_png", None)
    if not isinstance(grid_png, str):
        return None, "missing drawn grid"
    drawn = base64.b64decode(grid_png)
    probs = result.get("tile_probabilities")
    cutoff = answer.get("tile_threshold")
    if (
        isinstance(probs, dict)
        and cutoff is not None
        and any(
            isinstance(score, (int, float)) and abs(score - cutoff) < EARLY_MARGIN
            for score in probs.values()
        )
    ):
        return None, "a score close to the cutoff"
    return drawn, None


def grid_state(grid):
    return grid.locator("td").evaluate_all(
        """cells => cells.map(cell => {
            const images = [...cell.querySelectorAll('img')];
            return {
                sources: images.map(image => image.currentSrc),
                loaded: images.length > 0 && images.every(image => image.complete && image.naturalWidth),
                replacing: cell.classList.contains('rc-imageselect-dynamic-selected')
            };
        })"""
    )


def tile_id_hash(url):
    """A short hash of an image URL's `id` value, so a request can be matched to a tile."""
    values = parse_qs(urlsplit(url).query).get("id")
    return hashlib.sha256(values[0].encode()).hexdigest()[:10] if values else None


class TileWatch:
    """Lists when the widget's own image requests start and finish, in milliseconds since
    `started`, and can save the images it received. It only listens: nothing is requested,
    changed or held back. Paths and parameter names are kept, values are not."""

    def __init__(self, page, started):
        self.log, self.started = [], started
        self.entries, self.unsaved = {}, []
        # Replacement tiles in the order they were requested, and the images received.
        self.tiles, self.bodies = [], {}
        page.on("request", self.begun)
        page.on("requestfinished", self.finished)

    def now(self):
        return round((monotonic() - self.started) * 1000)

    def begun(self, request):
        url = urlsplit(request.url)
        if request.resource_type == "image" and "/recaptcha/" in url.path:
            self.entries[request] = entry = {
                "path": url.path,
                "parameters": sorted(parse_qs(url.query)),
                "id_hash": tile_id_hash(request.url),
                "started_ms": self.now(),
            }
            self.log.append(entry)
            if entry["id_hash"]:
                self.tiles.append(entry)

    def finished(self, request):
        entry = self.entries.get(request)
        if entry is not None:
            entry["finished_ms"] = self.now()
            if entry["id_hash"]:
                self.unsaved.append((entry, request))

    def collect(self, directory=None):
        """Reads the replacement tiles received so far, and writes them to `directory` if
        one is given. Call it between page checks, not from an event handler."""
        while self.unsaved:
            entry, request = self.unsaved.pop(0)
            try:
                body = request.response().body()
            except Exception as error:  # the body can be gone if the reply was redirected
                entry["body_error"] = str(error).splitlines()[0]
                continue
            self.bodies[entry["id_hash"]] = body
            if directory is not None:
                (directory / f"tile-{entry['id_hash']}.img").write_bytes(body)
            entry.update(bytes=len(body), saved_ms=self.now())


def tile_boxes(grid):
    """Each tile's picture area, as fractions of the grid's width and height."""
    return grid.evaluate(
        """grid => {
            const whole = grid.getBoundingClientRect();
            return [...grid.querySelectorAll('td')].map(cell => {
                const picture = cell.querySelector('.rc-image-tile-wrapper');
                if (!picture) return null;
                const box = picture.getBoundingClientRect();
                return [(box.left - whole.left) / whole.width, (box.top - whole.top) / whole.height,
                        box.width / whole.width, box.height / whole.height];
            });
        }"""
    )


def model_without_page(url, trial_id, fields):
    """The request post() makes, sent without the page so that it can run while the page waits."""
    origin = urlsplit(url)._replace(path="", query="", fragment="").geturl()
    request = urllib.request.Request(
        f"{origin}/model",
        data=json.dumps({
            "trial_id": trial_id,
            "source": urlsplit(url).fragment.removeprefix("source=") or SOURCE,
            **fields,
        }).encode(),
        headers={"Origin": origin, "Content-Type": "application/json"},
    )
    with LOCAL.open(request, timeout=125) as response:
        return json.load(response)


def pace(args, usual, fast):
    """Milliseconds to wait between checks: longer with --no-fast."""
    return fast if getattr(args, "fast", True) else usual


def loaded_grid(page, grid, poll=300):
    deadline = monotonic() + 12
    previous = None
    while monotonic() < deadline:
        state = grid_state(grid)
        if len(state) in (9, 16) and all(
            cell["loaded"] and not cell["replacing"] for cell in state
        ):
            if state == previous:
                return state
            previous = state
        else:
            previous = None
        page.wait_for_timeout(poll)
    raise RuntimeError("The Google image grid did not finish loading.")


def wait_for_widget(page, grid, poll=200):
    result = page.locator("#google-result")
    deadline = monotonic() + 25
    while monotonic() < deadline:
        if result.is_visible():
            return json.loads(result.inner_text())
        if grid.is_visible():
            return None
        page.wait_for_timeout(poll)
    raise RuntimeError("Google did not return a token or show a supported image grid.")


def observe(page, trial_id, verdict):
    post(page, "observe", trial_id, {"outcome": verdict})


def run_score(browser, url, settings, *, source=SOURCE):
    """Click 'Request assessment' on v3/v3-enterprise, print Google + model JSON, and exit."""
    with browser.new_context(locale="en-US") as context:
        page = context.new_page()
        print(f"Camoufox browser {browser.version}", flush=True)
        model_http_errors = []
        if settings.get("provider"):
            def on_response(resp):
                if urlsplit(resp.url).path == "/model" and not resp.ok:
                    model_http_errors.append(resp)

            page.on("response", on_response)
        response = page.goto(f"{url}/#source={source}", wait_until="domcontentloaded")
        if response is None or not response.ok:
            raise RuntimeError("The lab did not return a successful response.")
        page.get_by_role("heading", name="Google widget").wait_for()
        page.get_by_role("button", name="Request assessment").click(timeout=25_000)
        result = page.locator("#google-result")
        result.wait_for(state="visible", timeout=35_000)
        record = json.loads(result.inner_text())
        print(json.dumps(record, indent=2), flush=True)
        if settings.get("provider"):
            outcome = page.wait_for_function(MODEL_WAIT_JS, timeout=75_000).json_value()
            if outcome["kind"] != "result":
                http_error = None
                if model_http_errors:
                    resp = model_http_errors[-1]
                    body_error = None
                    try:
                        payload = resp.json()
                        if isinstance(payload, dict):
                            body_error = payload.get("error")
                    except Exception:
                        pass
                    http_error = (resp.status, body_error)
                print(format_model_http_error(http_error, outcome.get("text")), file=sys.stderr)
                raise SystemExit(1)
            model_record = json.loads(outcome["text"])
            print(json.dumps(model_record, indent=2), flush=True)
            if model_reply_failed(model_record):
                raise SystemExit(1)
        if not record.get("result", {}).get("valid"):
            raise SystemExit(1)
        return record


def run_trial(page, args, url, record):
    page.goto(f"{url}/#source={SOURCE}", wait_until="domcontentloaded")
    trial = page.locator("#trial-id")
    trial.wait_for(state="visible")
    page.wait_for_function("document.querySelector('#trial-id').textContent.length > 0")
    trial_id = trial.inner_text()
    record["trial_id"] = trial_id
    frame = page.frame_locator('iframe[src*="/bframe"]')
    grid = frame.locator(".rc-imageselect-target")
    checkbox_started = monotonic()
    watch = None
    log_tiles = getattr(args, "log_tile_network", False)
    early_tiles = getattr(args, "early_tiles", False) and args.submit
    if log_tiles or early_tiles:
        watch = TileWatch(page, checkbox_started)
        if log_tiles:
            record["tile_network"] = watch.log
    page.frame_locator('iframe[title="reCAPTCHA"]').get_by_role("checkbox").click(timeout=35_000)
    result = wait_for_widget(page, grid, pace(args, 200, 50))
    if result is not None:
        outcome = "checkbox_only" if result["result"].get("valid") else "verification_error"
        record.update(outcome=outcome, google=result)
        record["checkbox_to_verification_ms"] = round((monotonic() - checkbox_started) * 1000)
        observe(page, trial_id, "no_image_challenge" if outcome == "checkbox_only" else "abandoned")
        return

    directory = args.output / trial_id
    directory.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(pace(args, 500, 0))
    # With --max-stages the trial goes on after a rejection until Google passes it or has
    # rejected that many times. Google judges some challenges as several grids in a row with
    # one verdict at the end, such as the five grids of a 4 x 4 skip challenge, so answers are
    # numbered in `stages` and verdicts counted in `rejections`. The widget moves between the
    # grids of such a challenge by itself and asks Google only after the last one, so an answer
    # was judged if the widget sent its verify request, or showed an error message. Rounds are
    # limited as well, in case neither is ever seen.
    max_stages = getattr(args, "max_stages", 0) or 0
    stages = rejections = 0
    verify_requests = []
    if max_stages:
        # Listens only: counts the widget's own requests for a verdict.
        page.on("request", lambda request: (
            verify_requests.append(None)
            if urlsplit(request.url).path.endswith("/userverify") else None
        ))
    # The round whose rejection message had not appeared yet when the next grid did.
    awaiting_message = None
    prepared = None
    page_url = page.url
    errors = frame.locator(
        ".rc-imageselect-error-select-more, .rc-imageselect-error-dynamic-more,"
        " .rc-imageselect-error-select-something, .rc-imageselect-incorrect-response"
    )

    def shown_error():
        return next((item for item in errors.all() if item.is_visible()), None)

    def next_challenge(state):
        """Waits for a different grid. Returns whether one appeared."""
        deadline = monotonic() + 12
        while monotonic() < deadline:
            if grid.is_visible() and grid_state(grid) != state:
                return True
            page.wait_for_timeout(pace(args, 250, 50))
        return False

    background = ThreadPoolExecutor(max_workers=1) if early_tiles else None
    try:
        for number in range(1, (max_stages * 8 if max_stages else args.max_rounds) + 1):
            if max_stages and rejections >= max_stages:
                record["outcome"] = "stage_limit"
                observe(page, trial_id, "model_failed")
                return
            # A round answered ahead of time starts when the widget shows the new tiles.
            early, prepared = prepared, None
            if early is None:
                state = loaded_grid(page, grid, pace(args, 300, 100))
                # Milliseconds since the checkbox click at which each step of this round finished.
                timeline = {"grid_loaded": round((monotonic() - checkbox_started) * 1000)}
                if awaiting_message is not None:
                    late = shown_error()
                    if late is not None:
                        awaiting_message["rejection"] = late.inner_text(timeout=5000)
                    awaiting_message = None
            else:
                state = early["state"]
                timeline = {"grid_loaded": early["shown_ms"]}

            def mark(step):
                timeline[step] = round((monotonic() - checkbox_started) * 1000)

            path = directory / f"round-{number:02}.png"
            if early is None:
                prompt = frame.locator(".rc-imageselect-desc-wrapper").inner_text(timeout=5000)
                image = grid.screenshot(path=str(path), animations="disabled", timeout=10_000)
            else:
                prompt, image = early["prompt"], early["image"]
                path.write_bytes(image)
                if log_tiles:
                    # What the widget really shows, to compare with the drawn-in tiles later.
                    grid.screenshot(
                        path=str(directory / f"round-{number:02}-shown.png"),
                        animations="disabled", timeout=10_000,
                    )
            mark("captured")
            image_round = {
                "prompt": prompt,
                "layout": 3 if len(state) == 9 else 4,
                "image": str(path),
                "image_sha256": hashlib.sha256(image).hexdigest(),
                "timeline_ms": timeline,
            }
            if early is not None:
                image_round["early"] = early["timing"]
            record["rounds"].append(image_round)
            if not args.submit:
                record["outcome"] = "captured"
                observe(page, trial_id, "captured")
                return

            reply = early["reply"] if early is not None else post(page, "model", trial_id, {
                "prompt": prompt,
                "layout": image_round["layout"],
                "image": base64.b64encode(image).decode("ascii"),
            })
            mark("model_answered")
            image_round["model"] = reply
            prediction = reply["result"].get("prediction")
            if (
                reply["result"].get("status") != "answered"
                or not isinstance(prediction, dict)
                or prediction.get("type") != "tiles"
            ):
                record["outcome"] = reply["result"].get("status", "model_error")
                observe(page, trial_id, "abandoned")
                return
            if (
                not grid.is_visible()
                or grid_state(grid) != state
                or frame.locator(".rc-imageselect-desc-wrapper").inner_text(timeout=5000) != prompt
            ):
                if max_stages and grid.is_visible():
                    image_round["changed_during_inference"] = True
                    continue
                record["outcome"] = "challenge_changed_during_inference"
                observe(page, trial_id, "abandoned")
                return

            tiles = prediction["value"]
            dynamic = bool(tiles) and "once there are none left" in prompt.lower()
            # The widget fetches each replacement tile when its tile is clicked and shows it a
            # second or more later. The model is asked in that gap, unless --no-early-tiles is given.
            ahead = None
            if dynamic and early_tiles:
                boxes, first_new = tile_boxes(grid), len(watch.tiles)
            for tile in tiles:
                grid.locator("td").nth(tile - 1).click(timeout=5000)
            mark("tiles_clicked")
            if dynamic:
                deadline = monotonic() + 12
                while monotonic() < deadline:
                    new_state = grid_state(grid)
                    if len(new_state) == len(state) and all(
                        new_state[tile - 1]["sources"] != state[tile - 1]["sources"]
                        for tile in tiles
                    ):
                        mark("tiles_replaced")
                        shown = {
                            tile: [tile_id_hash(source) for source in new_state[tile - 1]["sources"]]
                            for tile in tiles
                        }
                        if watch is not None:
                            watch.collect(directory if log_tiles else None)
                        if log_tiles:
                            image_round["replaced_tile_ids"] = shown
                        if early_tiles:
                            # Use the early answer only if the tiles went where it assumed, every
                            # picture is loaded, and no score is close to the cutoff.
                            problem = None
                            if ahead is None:
                                problem = "tiles not received before they were shown"
                            elif any(shown[tile] != [ahead["assumed"][tile]] for tile in tiles):
                                problem = "tiles shown in another order"
                            elif not all(cell["loaded"] and not cell["replacing"] for cell in new_state):
                                problem = "grid still loading"
                            else:
                                try:
                                    answer = ahead["answer"].result(timeout=3)
                                    drawn, problem = evaluate_early_answer(answer)
                                except Exception as error:
                                    detail = str(error).partition("\n")[0] or type(error).__name__
                                    problem = f"early request failed: {detail}"
                            if problem is None:
                                prepared = {
                                    "state": new_state, "prompt": prompt, "image": drawn,
                                    "reply": answer, "shown_ms": timeline["tiles_replaced"],
                                    "timing": ahead["timing"],
                                }
                            else:
                                image_round["early_not_used"] = problem
                        break
                    if watch is not None:
                        watch.collect(directory if log_tiles else None)
                    if early_tiles and ahead is None:
                        new = watch.tiles[first_new:]
                        if len(new) == len(tiles) and all(tile["id_hash"] in watch.bodies for tile in new):
                            # In recorded rounds the widget gave the tiles out in grid order.
                            assumed = {tile: new[index]["id_hash"] for index, tile in enumerate(sorted(tiles))}
                            timing = {"asked_ms": round((monotonic() - checkbox_started) * 1000)}

                            # The server draws each tile into this round's screenshot at its
                            # tile's place and returns the grid it showed the model.
                            def ask(timing=timing, fields={
                                "prompt": prompt, "layout": image_round["layout"],
                                "image": base64.b64encode(image).decode("ascii"),
                                "tiles": [
                                    {"box": boxes[tile - 1],
                                     "image": base64.b64encode(watch.bodies[name]).decode("ascii")}
                                    for tile, name in assumed.items()
                                ],
                            }):
                                answer = model_without_page(page_url, trial_id, fields)
                                timing["answered_ms"] = round((monotonic() - checkbox_started) * 1000)
                                return answer

                            ahead = {
                                "assumed": assumed, "timing": timing,
                                "answer": background.submit(ask),
                            }
                    page.wait_for_timeout(pace(args, 200, 50))
                else:
                    record["outcome"] = "tile_replacement_timeout"
                    observe(page, trial_id, "abandoned")
                    return
                continue

            verify = frame.locator("#recaptcha-verify-button")
            # A message from the previous answer may still be showing. It counts for this answer
            # once it has been seen to clear, or if it is still there after a few seconds.
            stale = shown_error() if max_stages else None
            stale_class = stale.get_attribute("class") if stale is not None else None
            verify_requests_before = len(verify_requests)
            verify.click(timeout=5000)
            mark("verify_clicked")
            stages += 1
            image_round["stage"] = stages
            clicked = monotonic()
            result = page.locator("#google-result")
            deadline = monotonic() + 22
            while monotonic() < deadline:
                if result.is_visible():
                    record["google"] = json.loads(result.inner_text())
                    record["checkbox_to_verification_ms"] = round((monotonic() - checkbox_started) * 1000)
                    verdict = record["google"]["result"]
                    record["outcome"] = (
                        "model_solved" if verdict.get("valid")
                        else "verification_error" if verdict.get("error")
                        else "model_failed"
                    )
                    observed = "abandoned" if record["outcome"] == "verification_error" else record["outcome"]
                    observe(page, trial_id, observed)
                    return
                if grid.is_visible():
                    visible = shown_error()
                    if stale_class is not None:
                        if visible is None or visible.get_attribute("class") != stale_class:
                            stale_class = None
                        elif monotonic() - clicked < 3 and grid_state(grid) == state:
                            visible = None
                        elif grid_state(grid) != state:
                            # The grid changed under the old message: treated as the next grid.
                            image_round["stale_message_kept"] = True
                            visible = None
                        else:
                            stale_class = None
                    hint = (
                        visible is not None
                        and getattr(args, "use_hints", False)
                        and len(image_round.setdefault("hints", [])) < 2
                        and "rc-imageselect-error-select-more" in (visible.get_attribute("class") or "")
                        and grid_state(grid) == state
                    )
                    if hint:
                        # The widget kept the grid and the selection and asked for more tiles.
                        # Add the highest-scoring tile that is not selected yet, then verify again.
                        selected = grid.locator("td").evaluate_all(
                            "cells => cells.map(cell => cell.classList.contains('rc-imageselect-tileselected'))"
                        )
                        scores = reply["result"].get("tile_probabilities") or {}
                        remaining = sorted(
                            (tile for tile in range(1, len(state) + 1) if not selected[tile - 1]),
                            key=lambda tile: scores.get(str(tile), 0.0),
                            reverse=True,
                        )
                        if remaining:
                            tile = remaining[0]
                            image_round["hints"].append({
                                "message": visible.inner_text(timeout=5000),
                                "added": tile,
                                "probability": scores.get(str(tile)),
                            })
                            grid.locator("td").nth(tile - 1).click(timeout=5000)
                            cleared = monotonic() + 3
                            while visible.is_visible() and monotonic() < cleared:
                                page.wait_for_timeout(100)
                            image_round["hints"][-1]["message_cleared"] = not visible.is_visible()
                            verify.click(timeout=5000)
                            deadline = monotonic() + 22
                            page.wait_for_timeout(pace(args, 250, 50))
                            continue
                    if visible is not None and max_stages:
                        image_round["rejection"] = visible.inner_text(timeout=5000)
                        mark("rejected")
                        rejections += 1
                        if rejections >= max_stages:
                            record["widget_message"] = image_round["rejection"]
                            break
                        if "rc-imageselect-incorrect-response" not in (visible.get_attribute("class") or ""):
                            # The widget kept this grid and wants more tiles than the model found,
                            # so ask for a different challenge.
                            frame.locator("#recaptcha-reload-button").click(timeout=5000)
                            image_round["reloaded"] = True
                        if not next_challenge(state):
                            record["outcome"] = "next_challenge_timeout"
                            observe(page, trial_id, "abandoned")
                            return
                        mark("next_grid_shown")
                        break
                    if visible is not None:
                        record["outcome"] = "model_failed"
                        record["widget_message"] = visible.inner_text(timeout=5000)
                        observe(page, trial_id, "model_failed")
                        return
                    if grid_state(grid) != state:
                        mark("next_grid_shown")
                        if len(verify_requests) > verify_requests_before:
                            # Google judged the answer and served another challenge. Its message
                            # tends to appear a moment after the new grid, so it is read later.
                            image_round["judged"] = True
                            rejections += 1
                            awaiting_message = image_round
                        break
                page.wait_for_timeout(pace(args, 250, 50))
            else:
                record["outcome"] = "verification_timeout"
                observe(page, trial_id, "abandoned")
                return

        record["outcome"] = "stage_limit" if max_stages and rejections >= max_stages else "round_limit"
        observe(page, trial_id, "model_failed" if record["outcome"] == "stage_limit" else "abandoned")
    finally:
        if background is not None:
            background.shutdown(wait=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8765",
        help="Local lab server URL (default: http://localhost:8765).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Open a headed Camoufox window to the lab URL and wait until closed.",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Verify the public Google test widget (--google-test-key) once and exit.",
    )
    mode.add_argument(
        "--score",
        action="store_true",
        help="Click 'Request assessment' on v3/v3-enterprise, print Google + model JSON, and exit.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the Camoufox browser window instead of running headless (--interactive is always headed).",
    )
    parser.add_argument(
        "--trials",
        type=positive_int,
        default=None,
        help="Number of automated trials to run (default: 1).",
    )
    parser.add_argument(
        "--max-rounds",
        type=positive_int,
        default=None,
        help="Maximum image rounds per trial when --max-stages is 0 (default: 6).",
    )
    parser.add_argument(
        "--max-stages",
        type=non_negative_int,
        default=None,
        help="Keep going after Google rejects an answer, until it passes the trial or has rejected"
        " this many times. Grids served in a row before one verdict count once. 0 stops at the"
        " first rejection (default: 0).",
    )
    parser.add_argument(
        "--device-scale-factor",
        type=positive_float,
        default=None,
        help="Browser pixel density for automated trials (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for trial screenshots and JSONL logs (default: reports/recaptcha).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Ask the configured model and submit its selections during automated trials.",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Check the page every 50 to 100 ms and read the first grid immediately (default: on;"
        " --no-fast checks every 200 to 300 ms after a half-second pause).",
    )
    parser.add_argument(
        "--early-tiles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="On grids that replace clicked tiles, ask the model about fetched tiles before the"
        " widget reveals them (default: on when submitting; disable with --no-early-tiles).",
    )
    parser.add_argument(
        "--log-tile-network",
        action="store_true",
        help="Record when the widget's own image requests start and finish.",
    )
    parser.add_argument(
        "--use-hints",
        action="store_true",
        help="When the widget asks for more matching tiles, add the next-highest-scoring tile and verify again.",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    url = validate_lab_url(args.url)
    if url is None:
        parser.error("--url must be a localhost or loopback URL printed by the lab server.")

    special_mode = (
        "--interactive" if args.interactive
        else "--smoke" if args.smoke
        else "--score" if args.score
        else None
    )
    if special_mode is not None:
        bad = []
        if args.submit:
            bad.append("--submit")
        if args.trials is not None:
            bad.append("--trials")
        if args.max_rounds is not None:
            bad.append("--max-rounds")
        if args.max_stages is not None:
            bad.append("--max-stages")
        if args.output is not None:
            bad.append("--output")
        if args.fast is not None:
            bad.append("--fast" if args.fast else "--no-fast")
        if args.early_tiles is not None:
            bad.append("--early-tiles" if args.early_tiles else "--no-early-tiles")
        if args.device_scale_factor is not None:
            bad.append("--device-scale-factor")
        if args.log_tile_network:
            bad.append("--log-tile-network")
        if args.use_hints:
            bad.append("--use-hints")
        if bad:
            parser.error(f"{', '.join(bad)} cannot be used with {special_mode}.")

    if args.trials is None:
        args.trials = 1
    if args.max_rounds is None:
        args.max_rounds = 6
    if args.max_stages is None:
        args.max_stages = 0
    if args.device_scale_factor is None:
        args.device_scale_factor = 1.0
    if args.output is None:
        args.output = Path("reports/recaptcha")
    if args.fast is None:
        args.fast = True
    if args.early_tiles is None:
        args.early_tiles = True
    return parser, args, url


def validate_settings(args, settings):
    """Validate `/config` for the requested mode, returning an error message or None."""
    captcha_error = validate_captcha_config(settings, "recaptcha")
    if captcha_error is not None:
        return captcha_error
    if args.interactive:
        return None
    if args.smoke:
        if not settings.get("test") or settings.get("score_based"):
            return "--smoke requires starting the lab with --google-test-key."
        return None
    if args.score:
        if not settings.get("score_based"):
            return "--score requires --mode v3 or --mode v3-enterprise."
        return None
    if settings.get("test"):
        return "Use --smoke to verify the public Google test key, or a live v2 / v2-enterprise checkbox key for trials."
    if settings.get("mode") not in ("v2", "v2-enterprise"):
        return "Use a live v2 or v2-enterprise checkbox key (or --score for v3 / --interactive for manual testing)."
    if args.submit and not supports_vision(settings, recaptcha_compat=True):
        return "Start the lab with a vision provider before using --submit."
    return None


def main(argv=None):
    parser, args, url = parse_args(argv)
    headless = False if args.interactive else not args.headed

    with open_camoufox(headless=headless) as browser:
        try:
            settings = fetch_lab_config(browser, url)
        except RuntimeError as error:
            parser.error(str(error))

        config_error = validate_settings(args, settings)
        if config_error is not None:
            parser.error(config_error)

        if args.interactive:
            run_interactive(browser, url, heading="Google widget")
            return

        if args.smoke:
            record = run_smoke(
                browser,
                url,
                heading="Google widget",
                click_checkbox=lambda page: page.frame_locator('iframe[title="reCAPTCHA"]')
                .get_by_role("checkbox")
                .click(timeout=45_000),
                result_selector="#google-result",
            )
            if not record.get("result", {}).get("valid"):
                raise SystemExit(1)
            return

        if args.score:
            run_score(browser, url, settings)
            return

        args.output.mkdir(parents=True, exist_ok=True)
        vision_check_id = None
        if args.submit and requires_vision_check(settings):
            vision_check_id, check = run_vision_check(browser, url, args.output)
            if not check.get("result", {}).get("passed"):
                parser.exit(1, "The decision model did not pass the vision check. No challenges were submitted.\n")

        for _ in range(args.trials):
            scaled = {} if args.device_scale_factor == 1 else {
                "viewport": {"width": 1280, "height": 800},
                "device_scale_factor": args.device_scale_factor,
            }
            with browser.new_context(locale="en-US", **scaled) as context:
                page = context.new_page()
                record = {"trial_id": None, "rounds": []}
                try:
                    run_trial(page, args, url, record)
                except (BrowserTimeout, PlaywrightError, RuntimeError) as error:
                    detail = (str(error).splitlines() or [type(error).__name__])[0]
                    record.update(outcome="runner_error", detail=detail)
                if vision_check_id is not None:
                    record["vision_check_id"] = vision_check_id
            record.update(mode=settings["mode"], browser=BROWSER, browser_version=browser.version)
            model_name = resolve_model_name(settings, record)
            if model_name is not None:
                record["model_name"] = model_name
            with (args.output / "trials.jsonl").open("a", encoding="utf-8") as log:
                print(json.dumps(record), file=log)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
