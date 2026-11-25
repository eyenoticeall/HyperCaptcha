"""Standalone localhost-only HTTP labeling server and inter-rater repeatability checker."""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

from eval.schema import is_grid_correct, load_dataset, load_labels, parse_prompt

UI_HTML_PATH = Path(__file__).resolve().parent / "label_ui.html"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _resolve_item_image(manifest_dir: Path, image_field: str) -> Path:
    p = Path(image_field)
    if p.is_absolute():
        return p
    return (manifest_dir / p).resolve()


class LabelStore:
    """Thread-safe local manifest + JSONL label store."""

    def __init__(self, dataset_path: str | Path, labels_path: str | Path, layout: int | None = None) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        self.labels_path = Path(labels_path).resolve()
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        raw_items = load_dataset(self.dataset_path, layout=layout)
        self.items: list[dict[str, Any]] = []
        self.item_by_id: dict[str, dict[str, Any]] = {}
        self.image_by_id: dict[str, Path] = {}

        for row in raw_items:
            gid = str(row["id"])
            service = str(row.get("service", "recaptcha"))
            prompt = str(row.get("prompt", ""))
            parsed = parse_prompt(prompt, service)
            item = {
                "id": gid,
                "group_id": str(row.get("group_id", gid)),
                "service": service,
                "layout": int(row["layout"]),
                "prompt": prompt,
                "lead": parsed["lead"],
                "target": parsed["target"],
                "note": parsed["note"],
                "allows_skipping": bool(row.get("allows_skipping", parsed["allows_skipping"])),
                "image_url": f"/api/image/{gid}",
            }
            self.items.append(item)
            self.item_by_id[gid] = item
            self.image_by_id[gid] = _resolve_item_image(self.dataset_path.parent, str(row["image"]))

        self.labels: dict[str, dict[str, Any]] = {}
        if self.labels_path.exists():
            self.labels = load_labels(self.labels_path, include_unclear=True)

    def get_items(self) -> list[dict[str, Any]]:
        return list(self.items)

    def get_labels(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self.labels.items() if k in self.item_by_id}

    def get_image_bytes(self, item_id: str) -> tuple[bytes, str] | None:
        path = self.image_by_id.get(item_id)
        if path is None or not path.is_file():
            return None
        suffix = path.suffix.lower()
        content_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        return path.read_bytes(), content_type

    def save_label(self, payload: dict[str, Any]) -> dict[str, Any]:
        gid = str(payload.get("id", payload.get("grid", ""))).strip()
        if not gid or gid not in self.item_by_id:
            raise ValueError(f"unknown grid id: {gid!r}")
        item = self.item_by_id[gid]
        layout = int(item["layout"])
        max_tile = layout * layout

        raw_target = payload.get("target", payload.get("contains", []))
        raw_sliver = payload.get("sliver", payload.get("barely", []))
        if not isinstance(raw_target, list) or not isinstance(raw_sliver, list):
            raise ValueError("target and sliver must be lists of tile numbers")

        target = sorted({int(t) for t in raw_target})
        sliver = sorted({int(t) for t in raw_sliver})
        for tile in target + sliver:
            if not (1 <= tile <= max_tile):
                raise ValueError(f"tile {tile} out of range 1..{max_tile}")
        overlap = set(target) & set(sliver)
        if overlap:
            raise ValueError(f"tiles cannot be both target and sliver: {sorted(overlap)}")

        record = {
            "id": gid,
            "layout": layout,
            "target": target,
            "sliver": sliver,
            "unclear": bool(payload.get("unclear", False)),
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with self.labels_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self.labels[gid] = record
        return record


def _is_loopback_host_header(host_header: str | None) -> bool:
    """Return True if an HTTP Host header names a loopback host (IPv4, localhost, or [::1])."""
    host = (host_header or "").strip()
    if not host:
        return True
    if any(ch.isspace() for ch in host):
        return False
    try:
        parts = urlsplit(f"http://{host}")
        _ = parts.port
    except ValueError:
        return False
    if parts.username is not None or parts.password is not None or parts.path or parts.query or parts.fragment:
        return False
    hostname = (parts.hostname or "").lower()
    return hostname in LOOPBACK_HOSTS


def create_label_server(
    dataset_path: str | Path,
    labels_path: str | Path,
    *,
    port: int = 8787,
    layout: int | None = None,
) -> ThreadingHTTPServer:
    """Create a localhost-only (`127.0.0.1`) HTTP labeling server."""
    store = LabelStore(dataset_path, labels_path, layout=layout)
    ui_bytes = UI_HTML_PATH.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _check_loopback_host(self) -> bool:
            return _is_loopback_host_header(self.headers.get("Host"))

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if not self._check_loopback_host():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Loopback access only."})
                return
            path = urlsplit(self.path).path
            if path == "/" or path == "/index.html":
                self._send_bytes(HTTPStatus.OK, ui_bytes, "text/html; charset=utf-8")
            elif path == "/api/items":
                self._send_json(HTTPStatus.OK, store.get_items())
            elif path == "/api/labels":
                self._send_json(HTTPStatus.OK, store.get_labels())
            elif path.startswith("/api/image/"):
                item_id = unquote(path.removeprefix("/api/image/"))
                result = store.get_image_bytes(item_id)
                if result is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "Image not found."})
                    return
                body, ctype = result
                self._send_bytes(HTTPStatus.OK, body, ctype)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})

        def do_POST(self) -> None:  # noqa: N802
            if not self._check_loopback_host():
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Loopback access only."})
                return
            path = urlsplit(self.path).path
            if path != "/api/labels":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found."})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request body size."})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                saved = store.save_label(payload)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, saved)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.store = store  # type: ignore[attr-defined]
    return server


# ---------------------------------------------------------------------------
# Same-day / inter-rater repeatability check
# ---------------------------------------------------------------------------


def _cohen_kappa(matrix: list[list[int]]) -> float | None:
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return None
    n = len(matrix)
    po = sum(matrix[i][i] for i in range(n)) / total
    pe = sum(
        (sum(matrix[i]) / total) * (sum(matrix[r][i] for r in range(n)) / total)
        for i in range(n)
    )
    if pe >= 1.0 - 1e-12:
        return 1.0 if po >= 1.0 - 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def compare_labels(
    labels_a_path: str | Path,
    labels_b_path: str | Path,
    *,
    exclude_unclear: bool = True,
    default_layout: int = 4,
) -> dict[str, Any]:
    """Compute grid-level mutual agreement and square-level confusion between two label files."""
    raw_a = load_labels(labels_a_path, include_unclear=True)
    raw_b = load_labels(labels_b_path, include_unclear=True)
    common_all = sorted(set(raw_a) & set(raw_b))

    excluded_unclear = 0
    common: list[str] = []
    for gid in common_all:
        if exclude_unclear and (raw_a[gid]["unclear"] or raw_b[gid]["unclear"]):
            excluded_unclear += 1
            continue
        common.append(gid)

    a_under_b = 0
    b_under_a = 0
    mutual = 0
    exact_target = 0

    states = ("target", "sliver", "empty")
    idx_of = {s: i for i, s in enumerate(states)}
    mat3 = [[0, 0, 0] for _ in range(3)]
    mat2 = [[0, 0], [0, 0]]  # binary: target vs non-target

    total_squares = 0
    for gid in common:
        la, lb = raw_a[gid], raw_b[gid]
        ta, sa = set(la["target"]), set(la["sliver"])
        tb, sb = set(lb["target"]), set(lb["sliver"])

        ok_ab = is_grid_correct(sorted(ta), sorted(tb), sorted(sb))
        ok_ba = is_grid_correct(sorted(tb), sorted(ta), sorted(sa))
        a_under_b += int(ok_ab)
        b_under_a += int(ok_ba)
        mutual += int(ok_ab and ok_ba)
        exact_target += int(ta == tb)

        layout = int(la.get("layout") or lb.get("layout") or default_layout)
        n_tiles = layout * layout
        for t in range(1, n_tiles + 1):
            st_a = "target" if t in ta else "sliver" if t in sa else "empty"
            st_b = "target" if t in tb else "sliver" if t in sb else "empty"
            mat3[idx_of[st_a]][idx_of[st_b]] += 1
            mat2[1 if st_a == "target" else 0][1 if st_b == "target" else 0] += 1
            total_squares += 1

    n = len(common)
    confusion = {
        sa_name: {sb_name: mat3[i][j] for j, sb_name in enumerate(states)}
        for i, sa_name in enumerate(states)
    }
    sq_exact = sum(mat3[i][i] for i in range(3))

    return {
        "common_grids": n,
        "excluded_unclear": excluded_unclear,
        "a_right_under_b": {"count": a_under_b, "total": n, "rate": (a_under_b / n) if n else None},
        "b_right_under_a": {"count": b_under_a, "total": n, "rate": (b_under_a / n) if n else None},
        "mutual_agreement": {"count": mutual, "total": n, "rate": (mutual / n) if n else None},
        "exact_target_match": {"count": exact_target, "total": n, "rate": (exact_target / n) if n else None},
        "total_squares": total_squares,
        "square_exact_agreement": (sq_exact / total_squares) if total_squares else None,
        "cohen_kappa_3class": _cohen_kappa(mat3),
        "cohen_kappa_target": _cohen_kappa(mat2),
        "square_confusion": confusion,
    }


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "label",
        help="Serve the local browser labeling UI or check inter-rater repeatability.",
    )
    parser.add_argument("--dataset", help="Path to dataset.jsonl manifest (required when serving UI).")
    parser.add_argument("--labels", help="Path to labels.jsonl file (required when serving UI).")
    parser.add_argument("--port", type=int, default=8787, help="Localhost port (0 for ephemeral).")
    parser.add_argument("--layout", type=int, choices=(3, 4), help="Filter UI to 3x3 or 4x4 grids.")
    parser.add_argument(
        "--repeat-check",
        nargs=2,
        metavar=("LABELS_A", "LABELS_B"),
        help="Compare two labels.jsonl files for grid-level agreement and square-level confusion.",
    )
    parser.add_argument(
        "--include-unclear",
        action="store_true",
        help="Include grids flagged unclear when running --repeat-check.",
    )
    parser.add_argument("--json", dest="json_out", help="Optional path to write --repeat-check JSON.")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.repeat_check:
        report = compare_labels(
            args.repeat_check[0],
            args.repeat_check[1],
            exclude_unclear=not args.include_unclear,
        )
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        n = report["common_grids"]
        print(f"Repeatability check over {n} common grids (excluded unclear: {report['excluded_unclear']}):")
        if n:
            print(
                f"  A right under B: {report['a_right_under_b']['count']}/{n} "
                f"({report['a_right_under_b']['rate']:.1%})"
            )
            print(
                f"  B right under A: {report['b_right_under_a']['count']}/{n} "
                f"({report['b_right_under_a']['rate']:.1%})"
            )
            print(
                f"  Mutual agreement: {report['mutual_agreement']['count']}/{n} "
                f"({report['mutual_agreement']['rate']:.1%})"
            )
            print(
                f"  Exact target set match: {report['exact_target_match']['count']}/{n} "
                f"({report['exact_target_match']['rate']:.1%})"
            )
            print(
                f"  Square agreement: {report['square_exact_agreement']:.1%} "
                f"(kappa_3class={report['cohen_kappa_3class']:.3f}, "
                f"kappa_target={report['cohen_kappa_target']:.3f})"
            )
            print("  Square confusion (rows=A, cols=B):")
            for st_a, cols in report["square_confusion"].items():
                print(f"    {st_a:7s}: target={cols['target']:4d}  sliver={cols['sliver']:4d}  empty={cols['empty']:4d}")
        return 0

    if not args.dataset or not args.labels:
        raise SystemExit("error: --dataset and --labels are required unless --repeat-check is used")

    server = create_label_server(args.dataset, args.labels, port=args.port, layout=args.layout)
    actual_port = server.server_address[1]
    print(f"Open http://127.0.0.1:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CAPTCHA grid labeling UI and repeatability check.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
