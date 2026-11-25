"""Standardized ridge logistic regression and grouped K-fold CV for 3x3 and 4x4 grids.

Matches the proven optimization and grouped cross-validation methodology from
`reports/labels/next/boxgrid/liquid/lib.py` and `reports/labels/opt/fit4.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.schema import (
    average_close_up_passes,
    features_3x3,
    features_4x4,
    grid_correct,
    load_dataset,
    load_labels,
    make_group_folds,
    parse_recording_entry,
    read_jsonl,
    sigmoid,
)


def fit_ridge_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = 1.0,
    steps: int = 3000,
    lr: float = 0.5,
) -> np.ndarray:
    """Standardized ridge logistic regression returning raw `[w_0, ..., w_{d-1}, intercept]`."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_samples, n_feats = X_arr.shape
    if n_samples == 0:
        raise ValueError("cannot fit logistic regression on empty feature matrix")
    mu = X_arr.mean(axis=0)
    sd = X_arr.std(axis=0) + 1e-6
    Xs = np.hstack([(X_arr - mu) / sd, np.ones((n_samples, 1), dtype=np.float64)])
    w = np.zeros(n_feats + 1, dtype=np.float64)
    reg = np.ones(n_feats + 1, dtype=np.float64)
    reg[-1] = 0.0
    inv_n = 1.0 / float(n_samples)
    for _ in range(steps):
        p = sigmoid(Xs @ w)
        grad = (Xs.T @ (p - y_arr)) * inv_n + (lam * reg * w) * inv_n
        w -= lr * grad
    raw = w[:-1] / sd
    intercept = float(w[-1] - np.sum(w[:-1] * mu / sd))
    return np.append(raw, intercept)


def fit_4x4_weights(
    W: np.ndarray,
    C: np.ndarray,
    target: np.ndarray,
    sliver: np.ndarray,
    mask: np.ndarray,
    *,
    n_features: int = 4,
    slivers_empty: bool = False,
    lam: float = 1.0,
    steps: int = 3000,
) -> np.ndarray:
    """Fit 4x4 weights on grids in `mask` across all passes; returns 5-element vector."""
    if n_features not in (2, 4):
        raise ValueError(f"n_features must be 2 or 4, got {n_features}")
    F = features_4x4(W, C)[:, mask, :, :n_features]
    keep = np.broadcast_to((~sliver[mask]) | slivers_empty, F.shape[:-1])
    y = np.broadcast_to(target[mask], F.shape[:-1])
    w = fit_ridge_logistic(F[keep], y[keep].astype(np.float64), lam=lam, steps=steps)
    if n_features == 2:
        return np.array([w[0], w[1], 0.0, 0.0, w[2]], dtype=np.float64)
    return w


def fit_3x3_weights(
    W: np.ndarray,
    C: np.ndarray,
    target: np.ndarray,
    sliver: np.ndarray,
    mask: np.ndarray,
    *,
    slivers_empty: bool = False,
    lam: float = 1.0,
    steps: int = 3000,
) -> np.ndarray:
    """Fit 3x3 weights `[whole_lo, crop_lo, intercept]` on grids in `mask` across all passes."""
    F = features_3x3(W, C)[:, mask, :]
    keep = np.broadcast_to((~sliver[mask]) | slivers_empty, F.shape[:-1])
    y = np.broadcast_to(target[mask], F.shape[:-1])
    return fit_ridge_logistic(F[keep], y[keep].astype(np.float64), lam=lam, steps=steps)


def _eval_selection(
    scores: np.ndarray,
    target: np.ndarray,
    sliver: np.ndarray,
    cutoff: float = 0.5,
) -> dict[str, Any]:
    empty = (~target) & (~sliver)
    empty_strict = ~target
    sel = scores >= cutoff
    ok = grid_correct(sel, target, empty)
    ok_strict = grid_correct(sel, target, empty_strict)
    n_grids = target.shape[0]
    right_mean = float(ok.sum(axis=1).mean())
    strict_mean = float(ok_strict.sum(axis=1).mean())
    wrong_picks = float((sel & empty).sum(axis=(1, 2)).mean())
    missed_targets = float((~sel & target).sum(axis=(1, 2)).mean())
    return {
        "cutoff": float(cutoff),
        "right": round(right_mean, 4),
        "total": int(n_grids),
        "rate": round(right_mean / n_grids, 6) if n_grids else 0.0,
        "strict_slivers_right": round(strict_mean, 4),
        "strict_slivers_rate": round(strict_mean / n_grids, 6) if n_grids else 0.0,
        "wrong_picks_per_pass": round(wrong_picks, 4),
        "missed_targets_per_pass": round(missed_targets, 4),
        "per_pass_right": [int(x) for x in ok.sum(axis=1).tolist()],
    }


def _collect_aligned_data(
    dataset_path: str | Path,
    labels_path: str | Path,
    recordings_path: str | Path,
    *,
    layout: int,
    include_unclear: bool = False,
) -> dict[str, Any] | None:
    manifest = load_dataset(dataset_path, layout=layout)
    if not manifest:
        return None
    labels = load_labels(labels_path, include_unclear=include_unclear)
    by_grid: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for raw in read_jsonl(recordings_path):
        rec_layout = int(raw.get("layout", layout))
        if rec_layout != layout:
            continue
        parsed = parse_recording_entry(raw, layout)
        if parsed is not None:
            by_grid[parsed["grid_id"]][parsed["pass_index"]] = parsed

    aligned_items: list[dict[str, Any]] = []
    pass_sets: list[set[int]] = []
    for item in manifest:
        gid = str(item["id"])
        if gid in labels and gid in by_grid:
            aligned_items.append(item)
            pass_sets.append(set(by_grid[gid].keys()))

    if not aligned_items:
        return None

    common_passes = sorted(set.intersection(*pass_sets))
    if not common_passes:
        # Fallback: take pass 0 where available.
        aligned_items = [it for it in aligned_items if 0 in by_grid[str(it["id"])]]
        common_passes = [0]
    if not aligned_items:
        return None

    R = len(common_passes)
    G = len(aligned_items)
    n_tiles = layout * layout

    W = np.zeros((R, G, n_tiles), dtype=np.float64)
    has_capture = all("capture" in by_grid[str(it["id"])][p]["passes"] for it in aligned_items for p in common_passes)
    has_506 = all(506 in by_grid[str(it["id"])][p]["passes"] for it in aligned_items for p in common_passes)
    P_cap = np.zeros((R, G, n_tiles), dtype=np.float64) if has_capture else None
    P_506 = np.zeros((R, G, n_tiles), dtype=np.float64) if has_506 else None

    target = np.zeros((G, n_tiles), dtype=bool)
    sliver = np.zeros((G, n_tiles), dtype=bool)
    ids: list[str] = []
    group_ids: list[str] = []

    for g, item in enumerate(aligned_items):
        gid = str(item["id"])
        ids.append(gid)
        group_ids.append(str(item.get("group_id", gid)))
        lbl = labels[gid]
        for t in lbl["target"]:
            if 1 <= t <= n_tiles:
                target[g, t - 1] = True
        for t in lbl["sliver"]:
            if 1 <= t <= n_tiles:
                sliver[g, t - 1] = True
        for r, p_idx in enumerate(common_passes):
            rec = by_grid[gid][p_idx]
            W[r, g] = rec["whole"]
            if P_cap is not None:
                P_cap[r, g] = rec["passes"]["capture"]
            if P_506 is not None:
                P_506[r, g] = rec["passes"][506]

    return {
        "ids": ids,
        "group_ids": group_ids,
        "passes": common_passes,
        "W": W,
        "P_cap": P_cap,
        "P_506": P_506,
        "target": target,
        "sliver": sliver,
    }


def _fit_3x3_block(
    data: dict[str, Any],
    *,
    folds: int,
    seed: int,
    lam: float,
    slivers_empty: bool,
) -> dict[str, Any]:
    W = data["W"]
    C = data["P_cap"] if data["P_cap"] is not None else data["P_506"]
    assert C is not None
    target, sliver = data["target"], data["sliver"]
    R, G, _ = W.shape
    _, fold_arr, n_groups, k_folds = make_group_folds(data["group_ids"], n_folds=folds, seed=seed)

    F = features_3x3(W, C)
    oof_scores = np.zeros_like(W)
    for f in range(k_folds):
        w_f = fit_3x3_weights(W, C, target, sliver, fold_arr != f, slivers_empty=slivers_empty, lam=lam)
        oof_scores[:, fold_arr == f] = sigmoid(F[:, fold_arr == f] @ w_f[:2] + w_f[2])

    w_full = fit_3x3_weights(W, C, target, sliver, np.ones(G, dtype=bool), slivers_empty=slivers_empty, lam=lam)
    in_scores = sigmoid(F @ w_full[:2] + w_full[2])

    cutoffs = [round(0.30 + 0.05 * i, 2) for i in range(9)]
    best_cutoff = max(
        cutoffs,
        key=lambda c: (_eval_selection(oof_scores, target, sliver, c)["right"], -abs(c - 0.5)),
    )

    return {
        "grids": G,
        "groups": n_groups,
        "passes": R,
        "folds": k_folds,
        "weights": [round(float(x), 6) for x in w_full.tolist()],
        "cutoff": 0.5,
        "best_oof_cutoff": best_cutoff,
        "oof_at_50": _eval_selection(oof_scores, target, sliver, 0.5),
        "oof_at_best_cutoff": _eval_selection(oof_scores, target, sliver, best_cutoff),
        "in_sample_at_50": _eval_selection(in_scores, target, sliver, 0.5),
        "baselines": {
            "whole_only_50": _eval_selection(W, target, sliver, 0.5),
            "crop_only_50": _eval_selection(C, target, sliver, 0.5),
            "average_50_at_50": _eval_selection(0.5 * W + 0.5 * C, target, sliver, 0.5),
            "average_50_at_60": _eval_selection(0.5 * W + 0.5 * C, target, sliver, 0.6),
        },
    }


def _fit_4x4_block(
    data: dict[str, Any],
    *,
    folds: int,
    seed: int,
    lam: float,
    slivers_empty: bool,
) -> dict[str, Any]:
    W = data["W"]
    target, sliver = data["target"], data["sliver"]
    R, G, _ = W.shape
    _, fold_arr, n_groups, k_folds = make_group_folds(data["group_ids"], n_folds=folds, seed=seed)

    modes: dict[str, np.ndarray] = {}
    if data["P_cap"] is not None:
        modes["capture"] = data["P_cap"]
    if data["P_506"] is not None:
        modes["506"] = data["P_506"]
    if data["P_cap"] is not None and data["P_506"] is not None:
        modes["avg"] = average_close_up_passes([data["P_cap"], data["P_506"]])

    mode_results: dict[str, Any] = {}
    weights_out: dict[str, list[float]] = {}

    for mode_name, C in modes.items():
        F = features_4x4(W, C)
        oof_4 = np.zeros_like(W)
        oof_2 = np.zeros_like(W)
        for f in range(k_folds):
            train_mask = fold_arr != f
            test_mask = fold_arr == f
            w4_f = fit_4x4_weights(W, C, target, sliver, train_mask, n_features=4, slivers_empty=slivers_empty, lam=lam)
            w2_f = fit_4x4_weights(W, C, target, sliver, train_mask, n_features=2, slivers_empty=slivers_empty, lam=lam)
            oof_4[:, test_mask] = sigmoid(F[:, test_mask] @ w4_f[:4] + w4_f[4])
            oof_2[:, test_mask] = sigmoid(F[:, test_mask] @ w2_f[:4] + w2_f[4])

        w4_full = fit_4x4_weights(W, C, target, sliver, np.ones(G, dtype=bool), n_features=4, slivers_empty=slivers_empty, lam=lam)
        w2_full = fit_4x4_weights(W, C, target, sliver, np.ones(G, dtype=bool), n_features=2, slivers_empty=slivers_empty, lam=lam)
        in_4 = sigmoid(F @ w4_full[:4] + w4_full[4])
        in_2 = sigmoid(F @ w2_full[:4] + w2_full[4])

        weights_out[mode_name] = [round(float(x), 6) for x in w4_full.tolist()]
        mode_results[mode_name] = {
            "weights_4feat": weights_out[mode_name],
            "weights_2feat": [round(float(x), 6) for x in w2_full.tolist()],
            "oof_4feat": _eval_selection(oof_4, target, sliver, 0.5),
            "in_sample_4feat": _eval_selection(in_4, target, sliver, 0.5),
            "oof_2feat": _eval_selection(oof_2, target, sliver, 0.5),
            "in_sample_2feat": _eval_selection(in_2, target, sliver, 0.5),
            "baselines": {
                "whole_only": _eval_selection(W, target, sliver, 0.5),
                "close_up_only": _eval_selection(C, target, sliver, 0.5),
                "equal_average": _eval_selection(0.5 * W + 0.5 * C, target, sliver, 0.5),
            },
        }

    return {
        "grids": G,
        "groups": n_groups,
        "passes": R,
        "folds": k_folds,
        "weights": weights_out,
        "modes": mode_results,
    }


def fit_dataset(
    dataset_path: str | Path,
    labels_path: str | Path,
    recordings_path: str | Path,
    *,
    layout: int | None = None,
    folds: int = 5,
    seed: int = 7,
    lam: float = 1.0,
    slivers_empty: bool = False,
    include_unclear: bool = False,
) -> dict[str, Any]:
    """Fit 3x3 and/or 4x4 ridge logistic rules with grouped K-fold cross-validation."""
    layouts = (layout,) if layout is not None else (3, 4)
    summary: dict[str, Any] = {
        "folds": folds,
        "seed": seed,
        "lam": lam,
        "slivers_empty": slivers_empty,
    }
    if 3 in layouts:
        d3 = _collect_aligned_data(dataset_path, labels_path, recordings_path, layout=3, include_unclear=include_unclear)
        if d3 is not None:
            summary["layout_3x3"] = _fit_3x3_block(d3, folds=folds, seed=seed, lam=lam, slivers_empty=slivers_empty)
    if 4 in layouts:
        d4 = _collect_aligned_data(dataset_path, labels_path, recordings_path, layout=4, include_unclear=include_unclear)
        if d4 is not None:
            summary["layout_4x4"] = _fit_4x4_block(d4, folds=folds, seed=seed, lam=lam, slivers_empty=slivers_empty)

    if "layout_3x3" not in summary and "layout_4x4" not in summary:
        raise ValueError("no aligned labelled recordings found for requested layout(s)")
    return summary


def format_fit_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    if "layout_3x3" in summary:
        b = summary["layout_3x3"]
        lines.append(
            f"=== 3x3 Logistic Fit ({b['grids']} grids, {b['groups']} groups, "
            f"{b['passes']} pass(es), {b['folds']}-fold grouped CV) ==="
        )
        lines.append(f"  weights [whole_lo, crop_lo, bias]: {b['weights']}")
        oof = b["oof_at_50"]
        ins = b["in_sample_at_50"]
        lines.append(
            f"  OOF @0.50: {oof['right']:.1f}/{oof['total']} ({oof['rate']:.1%}) | "
            f"in-sample: {ins['right']:.1f}/{ins['total']} ({ins['rate']:.1%}) | "
            f"wrong picks: {oof['wrong_picks_per_pass']:.1f}, missed: {oof['missed_targets_per_pass']:.1f}"
        )
        for name, m in b["baselines"].items():
            lines.append(f"  baseline {name:18s}: {m['right']:.1f}/{m['total']} ({m['rate']:.1%})")

    if "layout_4x4" in summary:
        b = summary["layout_4x4"]
        lines.append(
            f"=== 4x4 Logistic Fit ({b['grids']} grids, {b['groups']} groups, "
            f"{b['passes']} pass(es), {b['folds']}-fold grouped CV) ==="
        )
        for mode_name, m in b["modes"].items():
            oof4, ins4 = m["oof_4feat"], m["in_sample_4feat"]
            oof2 = m["oof_2feat"]
            lines.append(f"  [{mode_name}] weights_4feat: {m['weights_4feat']}")
            lines.append(
                f"    4-feat OOF: {oof4['right']:.1f}/{oof4['total']} ({oof4['rate']:.1%}) | "
                f"in-sample: {ins4['right']:.1f}/{ins4['total']} ({ins4['rate']:.1%}) | "
                f"2-feat OOF: {oof2['right']:.1f}/{oof2['total']} ({oof2['rate']:.1%}) | "
                f"wrong={oof4['wrong_picks_per_pass']:.1f}, missed={oof4['missed_targets_per_pass']:.1f}"
            )
            bs = m["baselines"]
            lines.append(
                f"    baselines: whole={bs['whole_only']['right']:.1f}, "
                f"close_up={bs['close_up_only']['right']:.1f}, "
                f"avg50={bs['equal_average']['right']:.1f}"
            )
    return "\n".join(lines)


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "fit",
        help="Fit standardized ridge logistic weights (3x3 and/or 4x4) with grouped K-fold CV.",
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl.")
    parser.add_argument("--labels", required=True, help="Path to labels.jsonl.")
    parser.add_argument("--recordings", required=True, help="Path to recordings.jsonl.")
    parser.add_argument("-o", "--output", help="Optional path to write fit.json.")
    parser.add_argument("--layout", type=int, choices=(3, 4), help="Fit only 3x3 or 4x4 grids.")
    parser.add_argument("--folds", type=int, default=5, help="Number of grouped CV folds (default 5).")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic CV fold RNG seed (default 7).")
    parser.add_argument("--lam", type=float, default=1.0, help="Ridge L2 penalty strength (default 1.0).")
    parser.add_argument("--slivers-empty", action="store_true", help="Treat sliver squares as empty during training.")
    parser.add_argument("--include-unclear", action="store_true", help="Include grids flagged unclear.")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    summary = fit_dataset(
        args.dataset,
        args.labels,
        args.recordings,
        layout=args.layout,
        folds=args.folds,
        seed=args.seed,
        lam=args.lam,
        slivers_empty=args.slivers_empty,
        include_unclear=args.include_unclear,
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(format_fit_report(summary))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit 3x3 and 4x4 ridge logistic weights.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
