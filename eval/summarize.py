"""Summarize live reCAPTCHA or hCaptcha `trials.jsonl` runs with Wilson 95% CIs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import ceil, sqrt
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from eval.schema import allows_skipping_prompt, read_jsonl

WILSON_Z95 = 1.95996398454
TIMELINE_KEYS = ("grid_loaded", "captured", "model_answered", "tiles_clicked", "verify_clicked")


def wilson_proportion(successes: int, attempts: int) -> dict[str, Any]:
    """Compute pass rate and exact Wilson 95% confidence interval."""
    if attempts <= 0:
        return {"passed": int(successes), "total": 0, "rate": None, "wilson95": None}
    rate = successes / attempts
    z2 = WILSON_Z95 * WILSON_Z95
    denom = 1.0 + z2 / attempts
    center = (rate + z2 / (2.0 * attempts)) / denom
    half = WILSON_Z95 * sqrt((rate * (1.0 - rate) + z2 / (4.0 * attempts)) / attempts) / denom
    lower = max(0.0, center - half) if successes > 0 else 0.0
    upper = min(1.0, center + half) if successes < attempts else 1.0
    return {
        "passed": int(successes),
        "total": int(attempts),
        "rate": round(rate, 6),
        "wilson95": [round(lower, 6), round(upper, 6)],
    }


def summarize_distribution(values: Sequence[float | int]) -> dict[str, Any] | None:
    """Return count, median, p90, mean, min, and max for a numeric distribution."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    p90_idx = max(0, ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "median": round(float(median(ordered)), 2),
        "p90": round(float(ordered[p90_idx]), 2),
        "mean": round(float(mean(ordered)), 2),
        "min": round(float(ordered[0]), 2),
        "max": round(float(ordered[-1]), 2),
    }


def _trial_is_verified(trial: Mapping[str, Any]) -> bool:
    for key in ("google", "hcaptcha"):
        verdict = (trial.get(key) or {}).get("result")
        if isinstance(verdict, Mapping) and verdict.get("valid") is True:
            return True
    return trial.get("outcome") in ("model_solved", "checkbox_only")


def _trial_service(trial: Mapping[str, Any]) -> str:
    if trial.get("captcha") == "hcaptcha" or "hcaptcha" in trial:
        return "hcaptcha"
    return "recaptcha"


def _classify_first_round(trial: Mapping[str, Any]) -> str | None:
    rounds = trial.get("rounds") or []
    if not rounds:
        return None
    service = _trial_service(trial)
    first = rounds[0]
    prompt = str(first.get("prompt", ""))
    if service == "hcaptcha":
        pages = int(first.get("pages", 1))
        return "3x3_hcaptcha_multi" if pages > 1 else "3x3_hcaptcha_single"
    layout = int(first.get("layout", 3))
    if layout == 3:
        if "once there are none left" in prompt.lower():
            return "3x3_dynamic"
        return "3x3_static"
    if layout == 4:
        if allows_skipping_prompt(prompt, "recaptcha"):
            return "4x4_skip"
        return "4x4_no_skip"
    return f"{layout}x{layout}"


def _trial_rejections(trial: Mapping[str, Any]) -> int:
    if isinstance(trial.get("rejections"), int):
        return int(trial["rejections"])
    count = 0
    for rnd in trial.get("rounds") or []:
        if rnd.get("rejection") or rnd.get("judged") or rnd.get("result") == "rejected":
            count += 1
    if count == 0 and trial.get("outcome") in ("model_failed", "stage_limit") and trial.get("rounds"):
        count = 1
    return count


def _round_model_latency_ms(rnd: Mapping[str, Any]) -> float | None:
    model = rnd.get("model")
    if isinstance(model, Mapping) and model.get("latency_ms") is not None:
        return float(model["latency_ms"])
    if rnd.get("model_ms") is not None:
        return float(rnd["model_ms"])
    return None


def _round_usage(rnd: Mapping[str, Any]) -> tuple[int | None, int | None]:
    model = rnd.get("model")
    if not isinstance(model, Mapping):
        return (None, None)
    usage = (model.get("result") or {}).get("usage")
    if not isinstance(usage, Mapping):
        return (None, None)
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    return (None if inp is None else int(inp), None if out is None else int(out))


def summarize_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute pass rates, challenge breakdowns, rejections, timelines, and token usage."""
    filtered = [t for t in trials if "kind" not in t or t["kind"] == "trial"]
    n_attempts = len(filtered)
    outcomes = Counter(str(t.get("outcome", "unknown")) for t in filtered)

    verified = [t for t in filtered if _trial_is_verified(t)]
    checkbox_only = [t for t in filtered if t.get("outcome") == "checkbox_only"]
    challenged = [t for t in filtered if t.get("rounds")]
    challenged_solved = [t for t in challenged if _trial_is_verified(t)]

    rejection_counts = [_trial_rejections(t) for t in challenged]
    first_stage_solved = [t for t in challenged_solved if _trial_rejections(t) == 0]
    passed_over_total = sum(len(t.get("passed_over") or []) for t in filtered)

    # Challenge-type and layout rollups (grouped by first round).
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_layout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for t in challenged:
        ctype = _classify_first_round(t)
        if ctype:
            by_type[ctype].append(t)
        service = _trial_service(t)
        side = 3 if service == "hcaptcha" else int(t["rounds"][0].get("layout", 3))
        by_layout[f"{side}x{side}"].append(t)

    def _bucket_stats(bucket: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        wins = [t for t in bucket if _trial_is_verified(t)]
        c2v = [t["checkbox_to_verification_ms"] for t in wins if t.get("checkbox_to_verification_ms") is not None]
        lat = [
            v
            for t in bucket
            for r in t.get("rounds") or []
            if (v := _round_model_latency_ms(r)) is not None
        ]
        rejs = [_trial_rejections(t) for t in bucket]
        return {
            "verified": wilson_proportion(len(wins), len(bucket)),
            "first_stage_pass": wilson_proportion(sum(_trial_rejections(t) == 0 for t in wins), len(bucket)),
            "total_rejections": int(sum(rejs)),
            "checkbox_to_verification_ms": summarize_distribution(c2v),
            "model_latency_ms": summarize_distribution(lat),
        }

    # Timelines across all rounds and solved trials.
    timeline_vals: dict[str, list[float]] = {k: [] for k in TIMELINE_KEYS}
    model_latencies: list[float] = []
    solved_c2v_all: list[float] = []
    solved_c2v_challenged: list[float] = []

    input_tokens_all: list[int] = []
    output_tokens_all: list[int] = []
    input_tokens_by_layout: dict[str, list[int]] = defaultdict(list)
    solved_trial_input_tokens: list[int] = []

    for t in filtered:
        service = _trial_service(t)
        is_ok = _trial_is_verified(t)
        c2v = t.get("checkbox_to_verification_ms")
        if is_ok and c2v is not None:
            solved_c2v_all.append(float(c2v))
            if t.get("rounds"):
                solved_c2v_challenged.append(float(c2v))

        trial_in_tok = 0
        has_trial_tok = False
        for rnd in t.get("rounds") or []:
            tl = rnd.get("timeline_ms") or {}
            if isinstance(tl, Mapping):
                for key in TIMELINE_KEYS:
                    if tl.get(key) is not None:
                        timeline_vals[key].append(float(tl[key]))
            lat = _round_model_latency_ms(rnd)
            if lat is not None:
                model_latencies.append(lat)
            inp, out = _round_usage(rnd)
            if inp is not None:
                input_tokens_all.append(inp)
                side = 3 if service == "hcaptcha" else int(rnd.get("layout", 3))
                input_tokens_by_layout[f"{side}x{side}"].append(inp)
                trial_in_tok += inp
                has_trial_tok = True
            if out is not None:
                output_tokens_all.append(out)
        if is_ok and t.get("rounds") and has_trial_tok:
            solved_trial_input_tokens.append(trial_in_tok)

    timelines = {k: summarize_distribution(v) for k, v in timeline_vals.items()}
    timelines["checkbox_to_verification_ms"] = summarize_distribution(solved_c2v_challenged)
    timelines["checkbox_to_verification_all_verified_ms"] = summarize_distribution(solved_c2v_all)
    timelines["model_latency_ms"] = summarize_distribution(model_latencies)

    tokens = {
        "total_input_tokens": int(sum(input_tokens_all)),
        "total_output_tokens": int(sum(output_tokens_all)),
        "per_round_input": summarize_distribution(input_tokens_all),
        "per_round_output": summarize_distribution(output_tokens_all),
        "per_solved_trial_input": summarize_distribution(solved_trial_input_tokens),
        "by_layout_input": {k: summarize_distribution(v) for k, v in sorted(input_tokens_by_layout.items())},
    }

    return {
        "attempts": n_attempts,
        "outcomes": dict(sorted(outcomes.items())),
        "verified_overall": wilson_proportion(len(verified), n_attempts),
        "verified_challenged": wilson_proportion(len(challenged_solved), len(challenged)),
        "checkbox_only": wilson_proportion(len(checkbox_only), n_attempts),
        "first_stage_pass": wilson_proportion(len(first_stage_solved), len(challenged)),
        "rejections": {
            "total": int(sum(rejection_counts)),
            "trials_with_rejection": int(sum(c > 0 for c in rejection_counts)),
            "distribution": summarize_distribution(rejection_counts),
        },
        "passed_over_tasks": passed_over_total,
        "by_layout": {k: _bucket_stats(v) for k, v in sorted(by_layout.items())},
        "by_challenge_type": {k: _bucket_stats(v) for k, v in sorted(by_type.items())},
        "timelines_ms": timelines,
        "tokens": tokens,
    }


def summarize_files(paths: Sequence[str | Path], *, group_by_condition: bool = False) -> dict[str, Any]:
    """Load one or more `trials.jsonl` files and return a summary report."""
    all_rows: list[dict[str, Any]] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            p = p / "trials.jsonl"
        all_rows.extend(read_jsonl(p))

    trials = [r for r in all_rows if "kind" not in r or r["kind"] == "trial"]
    if group_by_condition:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in trials:
            key = str(t.get("condition") or t.get("model_name") or "default")
            groups[key].append(t)
        return {
            "overall": summarize_trials(trials),
            "by_condition": {k: summarize_trials(v) for k, v in sorted(groups.items())},
        }
    return {"overall": summarize_trials(trials)}


def _fmt_prop(prop: Mapping[str, Any]) -> str:
    if not prop["total"]:
        return "0/0 (—)"
    lo, hi = prop["wilson95"]
    return f"{prop['passed']}/{prop['total']} ({prop['rate']:.1%}, 95% CI {lo:.1%}–{hi:.1%})"


def _fmt_dist(dist: Mapping[str, Any] | None, unit: str = "ms") -> str:
    if not dist:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"med {dist['median']:.0f}{suffix} / p90 {dist['p90']:.0f}{suffix} (n={dist['count']})"


def format_summary_markdown(report: Mapping[str, Any]) -> str:
    """Format a live-trial summary dictionary as readable Markdown."""
    sections: list[tuple[str, Mapping[str, Any]]] = [("Overall", report["overall"])]
    if "by_condition" in report:
        for cond, sub in report["by_condition"].items():
            sections.append((f"Condition: {cond}", sub))

    lines: list[str] = []
    for title, s in sections:
        lines.append(f"## Live Trials Summary — {title}")
        lines.append("")
        lines.append(f"- **Attempts**: {s['attempts']} (outcomes: `{s['outcomes']}`)")
        lines.append(f"- **Verified overall**: {_fmt_prop(s['verified_overall'])}")
        lines.append(f"- **Verified challenged**: {_fmt_prop(s['verified_challenged'])}")
        lines.append(f"- **Checkbox-only**: {_fmt_prop(s['checkbox_only'])}")
        lines.append(f"- **First-stage pass**: {_fmt_prop(s['first_stage_pass'])}")
        lines.append(
            f"- **Rejections**: {s['rejections']['total']} total across "
            f"{s['rejections']['trials_with_rejection']} trials"
        )
        if s["passed_over_tasks"]:
            lines.append(f"- **Non-grid tasks passed over**: {s['passed_over_tasks']}")

        if s["by_challenge_type"]:
            lines.append("")
            lines.append("| Challenge Type | Verified (95% Wilson CI) | Rejections | Click-to-Verify | Model Latency |")
            lines.append("|---|---|---:|---|---|")
            for ctype, b in s["by_challenge_type"].items():
                lines.append(
                    f"| `{ctype}` | {_fmt_prop(b['verified'])} | {b['total_rejections']} | "
                    f"{_fmt_dist(b['checkbox_to_verification_ms'])} | {_fmt_dist(b['model_latency_ms'])} |"
                )

        lines.append("")
        lines.append("### Timelines (ms)")
        lines.append("")
        lines.append("| Milestone | Count | Median (ms) | p90 (ms) | Min (ms) | Max (ms) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key in (*TIMELINE_KEYS, "checkbox_to_verification_ms", "model_latency_ms"):
            d = s["timelines_ms"].get(key)
            if d:
                lines.append(
                    f"| `{key}` | {d['count']} | {d['median']:.0f} | {d['p90']:.0f} | {d['min']:.0f} | {d['max']:.0f} |"
                )

        tok = s["tokens"]
        if tok["total_input_tokens"] or tok["total_output_tokens"]:
            lines.append("")
            lines.append("### Token Usage")
            lines.append("")
            lines.append(
                f"- **Total input tokens**: {tok['total_input_tokens']:,} "
                f"(per round: {_fmt_dist(tok['per_round_input'], '')}; "
                f"per solved trial: {_fmt_dist(tok['per_solved_trial_input'], '')})"
            )
            for layout_k, dist in tok["by_layout_input"].items():
                lines.append(f"  - `{layout_k}` per round: {_fmt_dist(dist, 'tok')}")
            if tok["total_output_tokens"]:
                lines.append(f"- **Total output tokens**: {tok['total_output_tokens']:,}")
        lines.append("")

    return "\n".join(lines).rstrip()


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "summarize",
        help="Summarize live trials.jsonl runs (Wilson 95%% CIs, challenge breakdown, timelines, tokens).",
    )
    parser.add_argument("inputs", nargs="+", help="One or more trials.jsonl files or run directories.")
    parser.add_argument(
        "--group-by-condition",
        action="store_true",
        help="Also emit per-condition/model breakdowns when multiple conditions are present.",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Stdout format.")
    parser.add_argument("--json", dest="json_out", help="Optional path to write JSON summary.")
    parser.add_argument("--markdown", dest="md_out", help="Optional path to write Markdown summary.")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    report = summarize_files(args.inputs, group_by_condition=args.group_by_condition)
    md_text = format_summary_markdown(report)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md_text + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(md_text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize live HyperCaptcha trials.jsonl logs.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
