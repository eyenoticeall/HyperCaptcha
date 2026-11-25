"""Record raw model scores, skip probabilities, and boxes over a dataset manifest.

Spawns `hypercaptcha serve --record ...` (or connects to an existing loopback server),
runs every grid across N passes with concurrent workers and retries, strips `grid_png`,
and writes resumable records to `recordings.jsonl`. Never reads `.env.*` files.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlsplit

import httpx

from eval.dataset import UNIFORM_3X3_PICTURES
from eval.schema import load_dataset, read_jsonl


def _is_loopback_host(hostname: str | None) -> bool:
    if not hostname:
        return False
    h = hostname.lower()
    return h in ("localhost", "127.0.0.1", "::1") or h.endswith(".localhost")


def _connection_target(url: str) -> tuple[str, dict[str, str]]:
    """Return `(connect_base_url, headers)` using 127.0.0.1 with explicit Host/Origin."""
    parts = urlsplit(url)
    if parts.scheme != "http" or not _is_loopback_host(parts.hostname):
        raise ValueError(f"server URL must be an http:// loopback URL, got {url!r}")
    port = parts.port or 80
    host_header = f"{parts.hostname}:{port}"
    origin = f"http://{host_header}"
    connect_base = f"http://127.0.0.1:{port}"
    return connect_base, {"Host": host_header, "Origin": origin}


def _resolve_binary(binary: str | None) -> str:
    if binary:
        return binary
    for candidate in ("target/release/hypercaptcha", "target/debug/hypercaptcha"):
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
    found = shutil.which("hypercaptcha")
    if found:
        return found
    raise FileNotFoundError(
        "could not find hypercaptcha binary; build with `cargo build --release` or pass --binary"
    )


def _build_serve_command(
    binary: str,
    *,
    service: str,
    record_flag: str,
    model: str | None = None,
    provider: str | None = None,
    model_config: str | None = None,
    base_url: str | None = None,
    grid_max_side_3x3: int | None = None,
    grid_max_side_4x4: int | None = None,
    timeout: int = 60,
    cold_connections: bool = False,
    extra_args: Sequence[str] = (),
) -> list[str]:
    cmd = [binary, "serve", "--port", "0", "--timeout", str(timeout)]
    if service == "hcaptcha":
        cmd.extend(["--captcha", "hcaptcha", "--hcaptcha-test-key"])
    else:
        cmd.extend(["--mode", "v2", "--google-test-key"])
    if model_config:
        cmd.extend(["--model-config", model_config])
    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["--model", model])
    if base_url:
        cmd.extend(["--base-url", base_url])
    if grid_max_side_3x3 is not None:
        cmd.extend(["--grid-max-side-3x3", str(grid_max_side_3x3)])
    if grid_max_side_4x4 is not None:
        cmd.extend(["--grid-max-side-4x4", str(grid_max_side_4x4)])
    if cold_connections:
        cmd.append("--cold-connections")
    if record_flag:
        cmd.append(record_flag)
    cmd.extend(extra_args)
    return cmd


@contextmanager
def _managed_server(
    *,
    server_url: str | None,
    binary: str | None,
    service: str,
    model: str | None,
    provider: str | None,
    model_config: str | None,
    base_url: str | None,
    grid_max_side_3x3: int | None,
    grid_max_side_4x4: int | None,
    timeout: int,
    cold_connections: bool,
    extra_args: Sequence[str],
) -> Iterator[str]:
    if server_url:
        yield server_url.rstrip("/")
        return

    bin_path = _resolve_binary(binary)
    # Try `--record` first (new provider-agnostic flag), falling back to `--record-4x4`
    # if invoked against an older binary that hasn't added the `--record` alias yet.
    flags_to_try = ("--record", "--record-4x4") if service == "recaptcha" else ("--record", "")
    last_err = ""
    proc: subprocess.Popen[str] | None = None
    printed_url = ""

    for flag in flags_to_try:
        cmd = _build_serve_command(
            bin_path,
            service=service,
            record_flag=flag,
            model=model,
            provider=provider,
            model_config=model_config,
            base_url=base_url,
            grid_max_side_3x3=grid_max_side_3x3,
            grid_max_side_4x4=grid_max_side_4x4,
            timeout=timeout,
            cold_connections=cold_connections,
            extra_args=extra_args,
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None and proc.stderr is not None
        line = proc.stdout.readline().strip()
        if line.startswith("Open "):
            printed_url = line.removeprefix("Open ").strip()
            break
        err = proc.stderr.read().strip()
        rc = proc.wait()
        proc = None
        last_err = err or line or f"server exited with code {rc}"
        if flag == "--record" and "unexpected argument '--record'" in last_err:
            continue
        raise RuntimeError(f"hypercaptcha server failed to start: {last_err}")

    if proc is None or not printed_url:
        raise RuntimeError(f"hypercaptcha server failed to start: {last_err}")

    try:
        yield printed_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _resolve_image_path(dataset_path: Path, raw_image: str) -> Path:
    p = Path(raw_image)
    return p if p.is_absolute() else (dataset_path.parent / p).resolve()


def _has_recording_error(result: Any) -> str | None:
    if not isinstance(result, dict):
        return "missing result object"
    if result.get("status") != "answered":
        return str(result.get("error") or result.get("detail") or result.get("status"))
    comp = result.get("vllm_components") or result.get("jev_components") or {}
    if isinstance(comp, dict):
        pres = comp.get("presence")
        if isinstance(pres, dict):
            pres_err = pres.get("error")
            if pres_err:
                return f"presence: {pres_err}"
        boxes = comp.get("boxes")
        if isinstance(boxes, dict):
            box_err = boxes.get("error")
            if box_err:
                return f"boxes: {box_err}"
    return None


def record_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    passes: int = 1,
    workers: int = 2,
    retries: int = 5,
    backoff_seconds: float = 2.0,
    http_timeout: float = 180.0,
    server_url: str | None = None,
    binary: str | None = None,
    service: str | None = None,
    layout: int | None = None,
    model: str | None = None,
    provider: str | None = None,
    model_config: str | None = None,
    base_url: str | None = None,
    grid_max_side_3x3: int | None = None,
    grid_max_side_4x4: int | None = None,
    timeout: int = 60,
    cold_connections: bool = False,
    extra_args: Sequence[str] = (),
    limit: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Send every grid in `dataset_path` across `passes` and append resumable records."""
    if passes < 1:
        raise ValueError("passes must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if retries < 1:
        raise ValueError("retries must be >= 1")

    ds_path = Path(dataset_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = load_dataset(ds_path, layout=layout)
    if not all_items:
        return {
            "total_jobs": 0,
            "skipped_existing": 0,
            "skipped_service": 0,
            "recorded": 0,
            "failed": [],
        }

    inferred_service = service or str(all_items[0].get("service", "recaptcha"))
    items = [it for it in all_items if str(it.get("service", "recaptcha")) == inferred_service]
    skipped_service = len(all_items) - len(items)
    if not items:
        raise ValueError(
            f"no dataset items match service {inferred_service!r} "
            f"(skipped {skipped_service} item(s) from other service(s))"
        )
    if limit is not None:
        items = items[:limit]

    done: set[tuple[str, int]] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            gid = str(row.get("grid_id", row.get("grid", "")))
            pidx = int(row.get("pass_index", 0))
            done.add((gid, pidx))

    jobs: list[tuple[int, dict[str, Any]]] = []
    for p_idx in range(passes):
        for item in items:
            if (str(item["id"]), p_idx) not in done:
                jobs.append((p_idx, item))

    total_jobs = len(items) * passes
    skipped_existing = total_jobs - len(jobs)
    if not jobs:
        return {
            "total_jobs": total_jobs,
            "skipped_existing": skipped_existing,
            "skipped_service": skipped_service,
            "recorded": 0,
            "failed": [],
        }

    lock = threading.Lock()
    recorded = 0
    completed = skipped_existing
    failed: list[dict[str, Any]] = []

    with _managed_server(
        server_url=server_url,
        binary=binary,
        service=inferred_service,
        model=model,
        provider=provider,
        model_config=model_config,
        base_url=base_url,
        grid_max_side_3x3=grid_max_side_3x3,
        grid_max_side_4x4=grid_max_side_4x4,
        timeout=timeout,
        cold_connections=cold_connections,
        extra_args=extra_args,
    ) as url:
        connect_base, headers = _connection_target(url)
        with httpx.Client(timeout=http_timeout, trust_env=False) as client, out_path.open(
            "a", encoding="utf-8"
        ) as out_handle:

            def _run_one(job: tuple[int, dict[str, Any]]) -> None:
                nonlocal recorded, completed
                pass_index, item = job
                gid = str(item["id"])
                item_layout = int(item["layout"])
                img_path = _resolve_image_path(ds_path, str(item["image"]))
                encoded_img = base64.b64encode(img_path.read_bytes()).decode("ascii")

                body: dict[str, Any] = {
                    "trial_id": str(uuid.uuid4()),
                    "source": "browser",
                    "prompt": item["prompt"],
                    "image": encoded_img,
                }
                if inferred_service == "hcaptcha":
                    pictures = item.get("pictures")
                    if pictures is not None:
                        body["pictures"] = pictures
                    elif img_path.name.startswith("grid-"):
                        body["pictures"] = [list(box) for box in UNIFORM_3X3_PICTURES]
                else:
                    body["layout"] = item_layout

                last_error = ""
                result: dict[str, Any] | None = None
                wall_ms = 0

                for attempt in range(retries):
                    t0 = time.monotonic()
                    try:
                        resp = client.post(f"{connect_base}/model", headers=headers, json=body)
                        data = resp.json()
                        wall_ms = round((time.monotonic() - t0) * 1000)
                        if resp.status_code >= 400:
                            last_error = str(data.get("error", f"HTTP {resp.status_code}"))
                            result = None
                        else:
                            data.pop("grid_png", None)
                            result = data.get("result")
                            err = _has_recording_error(result)
                            if err is None:
                                break
                            last_error = err
                    except Exception as exc:  # noqa: BLE001
                        wall_ms = round((time.monotonic() - t0) * 1000)
                        last_error = repr(exc)
                        result = None

                    if attempt + 1 < retries:
                        time.sleep(min(backoff_seconds * (attempt + 1), 15.0))
                else:
                    with lock:
                        failed.append({"grid_id": gid, "pass_index": pass_index, "error": last_error[:240]})
                        completed += 1
                        if progress:
                            print(
                                f"[{completed}/{total_jobs}] pass={pass_index} grid={gid} "
                                f"FAILED: {last_error[:120]} [recorded={recorded} failed={len(failed)}]",
                                file=sys.stderr,
                                flush=True,
                            )
                    return

                assert result is not None
                result.pop("grid_png", None)
                entry = {
                    "grid_id": gid,
                    "pass_index": pass_index,
                    "layout": item_layout,
                    "wall_ms": wall_ms,
                    "result": result,
                }
                with lock:
                    out_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    recorded += 1
                    completed += 1
                    if progress:
                        print(
                            f"[{completed}/{total_jobs}] pass={pass_index} grid={gid} "
                            f"ok ({wall_ms}ms) [recorded={recorded} failed={len(failed)}]",
                            file=sys.stderr,
                            flush=True,
                        )

            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(_run_one, jobs):
                    pass

    return {
        "total_jobs": total_jobs,
        "skipped_existing": skipped_existing,
        "skipped_service": skipped_service,
        "recorded": recorded,
        "failed": failed,
    }


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "record",
        help="Record raw model scores, skip probabilities, and boxes into recordings.jsonl.",
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl.")
    parser.add_argument("-o", "--output", required=True, help="Output recordings.jsonl (resumable).")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes per grid (e.g. 3 for stochastic models).")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent HTTP requests.")
    parser.add_argument("--retries", type=int, default=5, help="Max attempts per grid on transient error.")
    parser.add_argument("--layout", type=int, choices=(3, 4), help="Record only 3x3 or 4x4 grids.")
    parser.add_argument("--limit", type=int, help="Record at most this many grids.")
    parser.add_argument("--server-url", help="Use an already-running loopback lab server URL instead of spawning one.")
    parser.add_argument("--binary", help="Path to the hypercaptcha binary.")
    parser.add_argument(
        "--service",
        "--captcha",
        dest="service",
        choices=("recaptcha", "hcaptcha"),
        help="CAPTCHA service to filter and pass to a spawned server.",
    )
    parser.add_argument("--model", help="Model name or ID passed to `hypercaptcha serve --model`.")
    parser.add_argument("--provider", help="Provider passed to `hypercaptcha serve --provider`.")
    parser.add_argument("--model-config", help="TOML config path passed to `hypercaptcha serve --model-config`.")
    parser.add_argument("--base-url", help="API base URL passed to `hypercaptcha serve --base-url`.")
    parser.add_argument("--grid-max-side-3x3", type=int, help="Passed to `hypercaptcha serve --grid-max-side-3x3`.")
    parser.add_argument("--grid-max-side-4x4", type=int, help="Passed to `hypercaptcha serve --grid-max-side-4x4`.")
    parser.add_argument("--timeout", type=int, default=60, help="Provider request timeout (seconds).")
    parser.add_argument("--cold-connections", action="store_true", help="Disable connection warming on spawned server.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra flag forwarded to `hypercaptcha serve`.")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    stats = record_dataset(
        args.dataset,
        args.output,
        passes=args.passes,
        workers=args.workers,
        retries=args.retries,
        server_url=args.server_url,
        binary=args.binary,
        service=args.service,
        layout=args.layout,
        model=args.model,
        provider=args.provider,
        model_config=args.model_config,
        base_url=args.base_url,
        grid_max_side_3x3=args.grid_max_side_3x3,
        grid_max_side_4x4=args.grid_max_side_4x4,
        timeout=args.timeout,
        cold_connections=args.cold_connections,
        extra_args=args.extra_arg,
        limit=args.limit,
    )
    extra = (
        f", {stats['skipped_service']} other-service skipped"
        if stats.get("skipped_service")
        else ""
    )
    print(
        f"Recorded {stats['recorded']} new rounds to {args.output} "
        f"(skipped {stats['skipped_existing']} existing{extra}, {len(stats['failed'])} failed)."
    )
    if stats["failed"]:
        for item in stats["failed"][:5]:
            print(f"  failed: {item['grid_id']} pass={item['pass_index']}: {item['error']}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record model outputs over dataset.jsonl.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
