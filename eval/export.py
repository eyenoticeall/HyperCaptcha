"""Export tuned decision-model TOML configs and deterministic `.json.gz` parity fixtures.

Emits TOML strictly conforming to the shared Rust/Python schema contract (validated
via stdlib `tomllib` round-trip) and optionally writes image-free, label-free gzipped
parity fixtures (`mtime=0`) matching `tests/fixtures/*.json.gz`.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.schema import (
    EFFORT_NAMES,
    Efforts,
    EndpointConfig,
    Logistic3x3,
    ModelConfig,
    Plan,
    TuningConfig,
    VALID_CROP_FILTERS,
    VALID_DECISIONS,
    VALID_DIALECTS,
    VALID_IMAGE_FORMATS,
    VALID_PROVIDERS,
    VALID_SERVICES,
    decide_4x4,
    load_dataset,
    parse_recording_entry,
    read_jsonl,
    validate_model_config,
)


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_float(value: float) -> str:
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"cannot serialize non-finite float to TOML: {value}")
    text = repr(f)
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _toml_close_ups(close_ups: Sequence[str | int]) -> str:
    parts = [_toml_str(x) if isinstance(x, str) else str(int(x)) for x in close_ups]
    return "[" + ", ".join(parts) + "]"


def _toml_float_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(_toml_float(v) for v in values) + "]"


def _emit_plan_section(lines: list[str], header: str, plan: Plan) -> None:
    lines.append(f"[{header}]")
    lines.append(f"close_ups = {_toml_close_ups(plan.close_ups)}")
    lines.append(f"select_from = {_toml_float(plan.select_from)}")
    if plan.skip_below is not None:
        lines.append(f"skip_below = {_toml_float(plan.skip_below)}")
    lines.append("")

    if plan.gap is not None:
        lines.append(f"[{header}.gap]")
        lines.append(f"under = {_toml_float(plan.gap.under)}")
        lines.append(f"more_than = {_toml_float(plan.gap.more_than)}")
        lines.append("")

    if plan.boxes is not None:
        b = plan.boxes
        lines.append(f"[{header}.boxes]")
        lines.append(f"exempt_from = {_toml_float(b.exempt_from)}")
        if b.whole is not None:
            lines.append(f"whole = {_toml_float(b.whole)}")
        if b.every is not None:
            lines.append(f"every = {_toml_float(b.every)}")
        if b.windows is not None:
            lines.append(f"windows = {_toml_float(b.windows)}")
        lines.append(f"cover = {_toml_float(b.cover)}")
        lines.append(f"inside = {_toml_float(b.inside)}")
        lines.append("")


def render_model_toml(cfg: ModelConfig) -> str:
    """Serialize a validated `ModelConfig` to canonical TOML matching `rust-core`."""
    validate_model_config(cfg)
    lines: list[str] = [
        f"schema_version = {int(cfg.schema_version)}",
        f"service = {_toml_str(cfg.service)}",
        f"name = {_toml_str(cfg.name)}",
        f"provider = {_toml_str(cfg.provider)}",
        f"model_id = {_toml_str(cfg.model_id)}",
        "",
    ]

    if cfg.endpoint is not None:
        ep = cfg.endpoint
        ep_lines: list[str] = []
        if ep.dialect is not None:
            ep_lines.append(f"dialect = {_toml_str(ep.dialect)}")
        if ep.base_url is not None:
            ep_lines.append(f"base_url = {_toml_str(ep.base_url)}")
        if ep.api_key_env is not None:
            ep_lines.append(f"api_key_env = {_toml_str(ep.api_key_env)}")
        if ep.extra_api_keys_env is not None:
            ep_lines.append(f"extra_api_keys_env = {_toml_str(ep.extra_api_keys_env)}")
        if ep.image_format is not None:
            ep_lines.append(f"image_format = {_toml_str(ep.image_format)}")
        if ep.crop_filter is not None:
            ep_lines.append(f"crop_filter = {_toml_str(ep.crop_filter)}")
        if ep.gzip is not None:
            ep_lines.append(f"gzip = {_toml_bool(ep.gzip)}")
        if ep.max_images is not None:
            ep_lines.append(f"max_images = {int(ep.max_images)}")
        if ep.multi_state_windows is not None:
            ep_lines.append(f"multi_state_windows = {_toml_bool(ep.multi_state_windows)}")
        if ep.supports_every_box is not None:
            ep_lines.append(f"supports_every_box = {_toml_bool(ep.supports_every_box)}")
        if ep.return_meta is not None:
            ep_lines.append(f"return_meta = {_toml_bool(ep.return_meta)}")
        if ep.hedge_on_stall is not None:
            ep_lines.append(f"hedge_on_stall = {_toml_bool(ep.hedge_on_stall)}")
        if ep.warm_connections is not None:
            ep_lines.append(f"warm_connections = {_toml_bool(ep.warm_connections)}")
        if ep.min_patch_tokens is not None:
            ep_lines.append(f"min_patch_tokens = {_toml_bool(ep.min_patch_tokens)}")
        if ep_lines or ep.extra_headers:
            lines.append("[endpoint]")
            lines.extend(ep_lines)
            lines.append("")
        if ep.extra_headers:
            lines.append("[endpoint.extra_headers]")
            for k in sorted(ep.extra_headers):
                lines.append(f"{_toml_str(k)} = {_toml_str(ep.extra_headers[k])}")
            lines.append("")

    if cfg.tuning is not None:
        t = cfg.tuning
        lines.append("[tuning]")
        lines.append(f"default_decision = {_toml_str(t.default_decision)}")
        lines.append("")

        if t.logistic_3x3 is not None:
            lines.append("[tuning.logistic_3x3]")
            lines.append(f"weights = {_toml_float_list(t.logistic_3x3.weights)}")
            lines.append(f"cutoff = {_toml_float(t.logistic_3x3.cutoff)}")
            lines.append("")

        if t.logistic_4x4 is not None:
            lines.append("[tuning.logistic_4x4]")
            lines.append(f"weights = {_toml_float_list(t.logistic_4x4)}")
            lines.append("")

        if t.average_4x4 is not None:
            _emit_plan_section(lines, "tuning.average_4x4", t.average_4x4)

        if t.efforts is not None:
            for tier in EFFORT_NAMES:
                _emit_plan_section(lines, f"tuning.efforts.{tier}", t.efforts.plan(tier))

    text = "\n".join(lines).rstrip() + "\n"
    # Verify lossless round-trip via stdlib tomllib.
    parsed = tomllib.loads(text)
    ModelConfig.from_dict(parsed)
    return text


def _extract_ladder_data(
    sweep_data: Mapping[str, Any] | None,
    fit_data: Mapping[str, Any] | None,
    *,
    service: str = "recaptcha",
    default_decision: str | None = None,
) -> TuningConfig | None:
    logistic_3x3: Logistic3x3 | None = None
    logistic_4x4: tuple[float, float, float, float, float] | None = None
    average_4x4: Plan | None = None
    efforts: Efforts | None = None

    if fit_data is not None:
        b3 = fit_data.get("layout_3x3")
        if isinstance(b3, Mapping) and b3.get("weights"):
            logistic_3x3 = Logistic3x3(
                weights=tuple(float(x) for x in b3["weights"]),  # type: ignore[arg-type]
                cutoff=float(b3.get("cutoff", 0.5)),
            )
        b4 = fit_data.get("layout_4x4")
        if isinstance(b4, Mapping) and isinstance(b4.get("weights"), Mapping):
            w_map = b4["weights"]
            for pref in ("avg", "506", "capture"):
                if pref in w_map and len(w_map[pref]) == 5:
                    logistic_4x4 = tuple(float(x) for x in w_map[pref])  # type: ignore[assignment]
                    break

    if sweep_data is not None:
        raw_w = sweep_data.get("chosen_weights", sweep_data.get("weights"))
        if raw_w is not None:
            if len(raw_w) != 5:
                raise ValueError(f"4x4 weights must have length 5, got {len(raw_w)}")
            logistic_4x4 = tuple(float(x) for x in raw_w)  # type: ignore[assignment]

        raw_efforts = sweep_data.get("efforts", sweep_data.get("levels"))
        if isinstance(raw_efforts, Mapping):
            plans: dict[str, Plan] = {}
            for tier in EFFORT_NAMES:
                if tier not in raw_efforts:
                    raise ValueError(f"ladder JSON missing effort tier {tier!r}")
                entry = raw_efforts[tier]
                plan_dict = entry["plan"] if isinstance(entry, Mapping) and "plan" in entry else entry
                plans[tier] = Plan.from_dict(plan_dict)
            efforts = Efforts(
                low=plans["low"],
                medium=plans["medium"],
                high=plans["high"],
                max=plans["max"],
            )

        if sweep_data.get("average_4x4"):
            average_4x4 = Plan.from_dict(sweep_data["average_4x4"])
        elif efforts is not None:
            # Sensible default average_4x4 plan matching the recorded passes and skip threshold.
            recorded = (sweep_data.get("model_meta") or {}).get("recorded_passes") or []
            avg_passes: tuple[str | int, ...] = (
                ("capture", 506) if ("capture" in recorded and "506" in recorded) else efforts.high.close_ups
            )
            average_4x4 = Plan(
                close_ups=avg_passes,
                select_from=0.5,
                skip_below=efforts.low.skip_below,
            )

    if default_decision is None:
        if service == "hcaptcha":
            resolved_decision = "logistic" if logistic_3x3 is not None else "average"
        else:
            resolved_decision = (
                "logistic" if (logistic_3x3 is not None or logistic_4x4 is not None) else "average"
            )
    else:
        resolved_decision = default_decision

    if service == "hcaptcha" and resolved_decision == "logistic" and logistic_3x3 is None:
        raise ValueError("hcaptcha default_decision='logistic' requires a 3x3 logistic fit (--fit)")

    if logistic_3x3 is None and logistic_4x4 is None and average_4x4 is None and efforts is None:
        if default_decision is not None or service == "hcaptcha":
            return TuningConfig(default_decision=resolved_decision)
        return None
    return TuningConfig(
        default_decision=resolved_decision,
        logistic_3x3=logistic_3x3,
        logistic_4x4=logistic_4x4,
        average_4x4=average_4x4,
        efforts=efforts,
    )


def write_parity_fixture(
    cfg: ModelConfig,
    recordings_path: str | Path,
    output_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    pass_index: int = 0,
) -> dict[str, Any]:
    """Write a deterministic, image-free, label-free `.json.gz` parity fixture (`mtime=0`)."""
    if cfg.tuning is None or cfg.tuning.efforts is None:
        raise ValueError("writing a 4x4 parity fixture requires tuning.efforts")
    if cfg.tuning.default_decision == "logistic" and cfg.tuning.logistic_4x4 is None:
        raise ValueError("logistic parity fixture requires tuning.logistic_4x4 weights")

    if pass_index < 0:
        raise ValueError(f"pass_index must be >= 0, got {pass_index}")

    allowed_ids: set[str] | None = None
    if dataset_path is not None:
        allowed_ids = {str(item["id"]) for item in load_dataset(dataset_path, layout=4)}

    by_grid: dict[str, dict[str, Any]] = {}
    seen_passes: set[int] = set()
    for raw in read_jsonl(recordings_path):
        if int(raw.get("layout", 4)) != 4:
            continue
        parsed = parse_recording_entry(raw, 4)
        if parsed is None:
            continue
        gid = parsed["grid_id"]
        if allowed_ids is not None and gid not in allowed_ids:
            continue
        seen_passes.add(parsed["pass_index"])
        if parsed["pass_index"] == pass_index:
            by_grid[gid] = parsed

    if not by_grid:
        if seen_passes:
            raise ValueError(
                f"no 4x4 recordings found for pass_index={pass_index} "
                f"(available pass_index values: {sorted(seen_passes)})"
            )
        raise ValueError("no 4x4 recordings available to build parity fixture")

    efforts = cfg.tuning.efforts
    weights = cfg.tuning.logistic_4x4
    decision = cfg.tuning.default_decision

    rows: list[dict[str, Any]] = []
    for gid in sorted(by_grid):
        rec = by_grid[gid]
        cap = rec["passes"].get("capture")
        at_506 = rec["passes"].get(506)
        if cap is None and at_506 is not None:
            cap = at_506
        if at_506 is None and cap is not None:
            at_506 = cap
        assert cap is not None and at_506 is not None

        whole_boxes = rec["whole_boxes"] if rec["whole_boxes"] is not None else []
        every_boxes = rec["every_boxes"] if rec["every_boxes"] is not None else []
        window_boxes = rec["window_boxes"] if rec["window_boxes"] is not None else []

        close_map = {"capture": cap, 506: at_506}
        picks = {
            tier: decide_4x4(
                rec["whole"],
                close_map,
                efforts.plan(tier),
                weights=weights,
                decision=decision,
                presence=rec["presence"],
                whole_boxes=whole_boxes,
                every_boxes=every_boxes,
                window_boxes=window_boxes,
            )
            for tier in EFFORT_NAMES
        }
        rows.append(
            {
                "id": gid,
                "presence": rec["presence"],
                "whole": [float(x) for x in rec["whole"].tolist()],
                "captured": [float(x) for x in cap.tolist()],
                "at_506": [float(x) for x in at_506.tolist()],
                "whole_boxes": whole_boxes,
                "every_boxes": every_boxes,
                "window_boxes": window_boxes,
                "picks": picks,
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n]\n"
    with out_path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", fileobj=raw_file, mode="wb", compresslevel=9, mtime=0) as gz:
            gz.write(payload.encode("utf-8"))

    return {"grids": len(rows), "bytes": out_path.stat().st_size, "path": str(out_path)}


def export_model(
    *,
    service: str,
    name: str,
    provider: str,
    model_id: str,
    output_toml: str | Path,
    sweep_path: str | Path | None = None,
    fit_path: str | Path | None = None,
    default_decision: str | None = None,
    endpoint: EndpointConfig | None = None,
    fixture_path: str | Path | None = None,
    recordings_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    pass_index: int = 0,
) -> dict[str, Any]:
    """Export a validated TOML `ModelConfig` and optional deterministic parity fixture."""
    sweep_data = json.loads(Path(sweep_path).read_text(encoding="utf-8")) if sweep_path else None
    fit_data = json.loads(Path(fit_path).read_text(encoding="utf-8")) if fit_path else None
    tuning = _extract_ladder_data(
        sweep_data,
        fit_data,
        service=service,
        default_decision=default_decision,
    )

    cfg = ModelConfig(
        schema_version=1,
        service=service,
        name=name,
        provider=provider,
        model_id=model_id,
        endpoint=endpoint,
        tuning=tuning,
    )
    toml_text = render_model_toml(cfg)

    toml_path = Path(output_toml)
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(toml_text, encoding="utf-8")

    result: dict[str, Any] = {"toml_path": str(toml_path), "toml_bytes": len(toml_text.encode("utf-8"))}
    if fixture_path is not None:
        if recordings_path is None:
            raise ValueError("--fixture requires --recordings")
        result["fixture"] = write_parity_fixture(
            cfg,
            recordings_path,
            fixture_path,
            dataset_path=dataset_path,
            pass_index=pass_index,
        )
    return result


def _parse_extra_headers(items: Sequence[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"--extra-header must be KEY=VALUE with non-empty key and value, got {item!r}")
        headers[key.strip()] = value.strip()
    return headers


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "export",
        help="Export a validated Rust-compatible TOML ModelConfig and optional .json.gz parity fixture.",
    )
    parser.add_argument("--sweep", help="Path to sweep_result.json or hand-edited ladder JSON.")
    parser.add_argument("--fit", help="Optional fit.json supplying logistic_3x3 (and/or fallback logistic_4x4).")
    parser.add_argument("--service", choices=VALID_SERVICES, default="recaptcha", help="Target CAPTCHA service.")
    parser.add_argument("--name", required=True, help="Display/CLI model name, e.g. MyProvider/my-model.")
    parser.add_argument("--provider", choices=VALID_PROVIDERS, required=True, help="Model provider.")
    parser.add_argument("--model-id", required=True, help="Provider wire model ID.")
    parser.add_argument(
        "--default-decision",
        choices=VALID_DECISIONS,
        default=None,
        help="Default 4x4/3x3 decision rule (defaults to 'average' for hcaptcha without a 3x3 fit, otherwise 'logistic' when fitted).",
    )
    parser.add_argument("-o", "--output", required=True, help="Output path for the .toml ModelConfig.")

    # Optional [endpoint] fields (required when --provider custom).
    parser.add_argument("--dialect", choices=VALID_DIALECTS)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--extra-api-keys-env")
    parser.add_argument("--image-format", choices=VALID_IMAGE_FORMATS)
    parser.add_argument("--crop-filter", choices=VALID_CROP_FILTERS)
    parser.add_argument("--gzip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--multi-state-windows", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--supports-every-box", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--return-meta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hedge-on-stall", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warm-connections", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-patch-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extra-header", action="append", default=[], metavar="KEY=VALUE")

    # Optional parity fixture generation.
    parser.add_argument("--fixture", help="Optional output path for deterministic .json.gz parity fixture.")
    parser.add_argument("--recordings", help="Path to recordings.jsonl (required with --fixture).")
    parser.add_argument("--dataset", help="Optional dataset.jsonl to filter fixture grid IDs.")
    parser.add_argument("--pass-index", type=int, default=0, help="Recording pass index for --fixture (default 0).")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    extra_headers = _parse_extra_headers(args.extra_header)
    ep_fields = {
        "dialect": args.dialect,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "extra_api_keys_env": args.extra_api_keys_env,
        "image_format": args.image_format,
        "crop_filter": args.crop_filter,
        "gzip": args.gzip,
        "max_images": args.max_images,
        "multi_state_windows": args.multi_state_windows,
        "supports_every_box": args.supports_every_box,
        "return_meta": args.return_meta,
        "hedge_on_stall": args.hedge_on_stall,
        "warm_connections": args.warm_connections,
        "min_patch_tokens": args.min_patch_tokens,
        "extra_headers": extra_headers,
    }
    has_endpoint = args.provider == "custom" or any(
        v is not None and v != {} for v in ep_fields.values()
    )
    endpoint = EndpointConfig(**ep_fields) if has_endpoint else None

    info = export_model(
        service=args.service,
        name=args.name,
        provider=args.provider,
        model_id=args.model_id,
        output_toml=args.output,
        sweep_path=args.sweep,
        fit_path=args.fit,
        default_decision=args.default_decision,
        endpoint=endpoint,
        fixture_path=args.fixture,
        recordings_path=args.recordings,
        dataset_path=args.dataset,
        pass_index=args.pass_index,
    )
    print(f"Wrote validated TOML ModelConfig to {info['toml_path']} ({info['toml_bytes']} bytes).")
    if "fixture" in info:
        fx = info["fixture"]
        print(f"Wrote deterministic parity fixture to {fx['path']} ({fx['grids']} grids, {fx['bytes']} bytes).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export TOML ModelConfig and parity fixture.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
