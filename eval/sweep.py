"""Vectorized offline 4x4 effort-level parameter sweep with grouped CV and bootstrap CIs.

Performs an exhaustive offline parameter sweep over recorded 4x4 grids:
- Precomputes `boxmax` tensors for recorded box asks (`whole_grid`, `every`, `windows`).
- Sweeps close-up pass modes (`capture`, `506`, `avg`), scoring rules, selection cutoffs,
  skip thresholds, gap rules, exemption thresholds, box confidences, cover, and inside.
- Scores every effort tier (`low`, `medium`, `high`, `max`) with in-sample accuracy,
  grouped K-fold out-of-fold accuracy, leave-one-pass-out cross-pass accuracy & flip rate
  (when `passes >= 2`), simplicity/plateau tie-breaking, token & request estimates, and
  paired grouped bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.fit import fit_4x4_weights
from eval.schema import (
    EFFORT_NAMES,
    LOG_ODDS_3,
    BoxRule,
    GapRule,
    Plan,
    average_close_up_passes,
    decide_4x4,
    features_4x4,
    grid_correct,
    load_dataset,
    load_labels,
    log_odds,
    make_group_folds,
    parse_recording_entry,
    precompute_box_reach,
    read_jsonl,
    sigmoid,
)

DEFAULT_KAPPA = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_EXEMPT = (0.0, 0.60, 0.70, 0.80, 0.90, LOG_ODDS_3, 0.98, 1.0)
DEFAULT_GAP = (None, 0.5, 0.3)
DEFAULT_TAU = (None, 0.3, 0.5, 0.7, 0.9)
DEFAULT_COVER = (0.01, 0.03, 0.05, 0.10)
DEFAULT_INSIDE = (0.0, 0.01, 0.03, 0.10)
DEFAULT_CONF_WHOLE = (0.02, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9)
DEFAULT_CONF_EVERY = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9)
DEFAULT_CONF_WINDOWS = (0.1, 0.3, 0.5, 0.8)

FAST_KAPPA = (0.30, 0.40, 0.50, 0.55, 0.60)
FAST_EXEMPT = (0.0, 0.60, 0.90, LOG_ODDS_3, 1.0)
FAST_GAP = (None, 0.3)
FAST_TAU = (None, 0.5, 0.7)
FAST_COVER = (0.01, 0.05)
FAST_INSIDE = (0.0, 0.01)
FAST_CONF_WHOLE = (0.02, 0.3, 0.5)
FAST_CONF_EVERY = (0.3, 0.5)
FAST_CONF_WINDOWS = (0.3, 0.5)


def _min_int(minimum: int):
    def _parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid int value: {raw!r}") from exc
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
        return value

    return _parse


def grouped_bootstrap_diff(
    group_indices: np.ndarray,
    n_groups: int,
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 4000,
    seed: int = 11,
) -> dict[str, float]:
    """Paired grouped bootstrap 95% CI for `(a - b)` grids right per pass."""
    if isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    d = (a.astype(np.float64) - b.astype(np.float64)).mean(axis=0)
    per_group = np.zeros(n_groups, dtype=np.float64)
    np.add.at(per_group, group_indices, d)
    rng = np.random.default_rng(seed)
    draws = per_group[rng.integers(0, n_groups, size=(n_boot, n_groups))].sum(axis=1)
    return {
        "diff_grids": round(float(d.sum()), 4),
        "ci95_low": round(float(np.percentile(draws, 2.5)), 4),
        "ci95_high": round(float(np.percentile(draws, 97.5)), 4),
    }


def _boxmax(
    reach: list[list[list[list[tuple[float, float, float]]]]],
    cover_grid: Sequence[float],
    inside_grid: Sequence[float],
    R: int,
    G: int,
) -> np.ndarray:
    """Max confidence among boxes reaching each square for every `(cover, inside)` threshold."""
    cov_arr = np.asarray(cover_grid, dtype=np.float64)
    ins_arr = np.asarray(inside_grid, dtype=np.float64)
    out = np.full((len(cover_grid), len(inside_grid), R, G, 16), -1.0, dtype=np.float64)
    for r in range(R):
        for g in range(G):
            for t, entries in enumerate(reach[r][g]):
                for conf, cov, ins in entries:
                    ci = int(np.searchsorted(cov_arr, cov, side="right"))
                    ii = int(np.searchsorted(ins_arr, ins, side="right"))
                    if ci > 0 and ii > 0:
                        out[:ci, :ii, r, g, t] = np.maximum(out[:ci, :ii, r, g, t], conf)
    return out


def _with_skip(
    ok: np.ndarray,
    presence: np.ndarray,
    notarget: np.ndarray,
    tau: float | None,
) -> np.ndarray:
    """Apply skip threshold: grids with `presence < tau` select nothing (right iff no target)."""
    if tau is None:
        return ok
    pres_clean = np.where(np.isnan(presence), np.inf, presence)
    return np.where(pres_clean < tau, notarget, ok)


def _simplicity_penalty(
    *,
    mode: str,
    kappa: float,
    tau: float | None,
    exempt: float,
    gap: float | None,
    cw: float | None,
    ce: float | None,
    cs: float | None,
    cover: float,
    inside: float,
) -> float:
    """Lower is simpler/more standard; breaks ties toward round plateau centers and fewer knobs."""
    penalty = 0.0
    if gap is not None:
        penalty += 1.0
    if inside > 0.0:
        penalty += 1.0
    if cover > 0.01 + 1e-12:
        penalty += 1.0
    penalty += abs(kappa - 0.5)
    if cw is not None:
        penalty += abs(cw - 0.3) * 0.5
        penalty += 0.05  # Prefer cheaper single-question `every` over 9-question `whole` on ties.
    if ce is not None:
        penalty += abs(ce - 0.3) * 0.5
    if cs is not None:
        penalty += abs(cs - 0.5) * 0.5 + 0.1
    if exempt not in (0.0, 0.9, LOG_ODDS_3):
        penalty += 0.1
    if tau is not None and tau not in (0.5, 0.7):
        penalty += abs(tau - 0.7) * 0.1
    if mode == "avg":
        penalty += 0.02
    return penalty


def _mode_to_close_ups(mode: str) -> tuple[str | int, ...]:
    if mode == "capture":
        return ("capture",)
    if mode == "506":
        return (506,)
    if mode == "avg":
        return ("capture", 506)
    raise ValueError(f"unknown close-up mode: {mode!r}")


def _candidate_to_plan(
    *,
    mode: str,
    kappa: float,
    tau: float | None,
    exempt: float,
    gap: float | None,
    cw: float | None,
    ce: float | None,
    cs: float | None,
    cover: float,
    inside: float,
) -> Plan:
    has_any_box = (cw is not None) or (ce is not None) or (cs is not None)
    # exempt == 0.0 means every pick is exempt, so neither gap nor box rule can ever fire.
    box_rule: BoxRule | None = None
    gap_rule: GapRule | None = None
    if exempt > 0.0:
        if has_any_box:
            box_rule = BoxRule(
                exempt_from=min(float(exempt), 1.0),
                whole=cw,
                every=ce,
                windows=cs,
                cover=float(cover),
                inside=float(inside),
            )
        if gap is not None:
            gap_rule = GapRule(under=min(float(exempt), 1.0), more_than=float(gap))
    return Plan(
        close_ups=_mode_to_close_ups(mode),
        select_from=float(kappa),
        skip_below=tau,
        boxes=box_rule,
        gap=gap_rule,
    )


def _load_sweep_data(
    dataset_path: str | Path,
    labels_path: str | Path,
    recordings_path: str | Path,
    *,
    include_unclear: bool = False,
) -> dict[str, Any]:
    manifest = load_dataset(dataset_path, layout=4)
    if not manifest:
        raise ValueError("no 4x4 grids found in dataset.jsonl")
    labels = load_labels(labels_path, include_unclear=include_unclear)

    by_grid: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    resolved_models: list[str] = []
    window_requests_seen: list[int] = []

    for raw in read_jsonl(recordings_path):
        if int(raw.get("layout", 4)) != 4:
            continue
        parsed = parse_recording_entry(raw, 4)
        if parsed is not None:
            by_grid[parsed["grid_id"]][parsed["pass_index"]] = parsed
            if parsed["resolved_model"]:
                resolved_models.append(str(parsed["resolved_model"]))
            if parsed["window_boxes"] is not None:
                window_requests_seen.append(int(parsed["window_requests"]))

    aligned: list[dict[str, Any]] = []
    pass_sets: list[set[int]] = []
    for item in manifest:
        gid = str(item["id"])
        if gid in labels and gid in by_grid:
            aligned.append(item)
            pass_sets.append(set(by_grid[gid].keys()))

    if not aligned:
        raise ValueError("no aligned labelled 4x4 recordings found")

    common_passes = sorted(set.intersection(*pass_sets))
    if not common_passes:
        aligned = [it for it in aligned if 0 in by_grid[str(it["id"])]]
        common_passes = [0]
    if not aligned:
        raise ValueError("no aligned labelled 4x4 recordings with common passes found")

    R = len(common_passes)
    G = len(aligned)
    ids = [str(it["id"]) for it in aligned]
    group_ids = [str(it.get("group_id", it["id"])) for it in aligned]

    W = np.zeros((R, G, 16), dtype=np.float64)
    has_cap = all("capture" in by_grid[gid][p]["passes"] for gid in ids for p in common_passes)
    has_506 = all(506 in by_grid[gid][p]["passes"] for gid in ids for p in common_passes)
    P_cap = np.zeros((R, G, 16), dtype=np.float64) if has_cap else None
    P_506 = np.zeros((R, G, 16), dtype=np.float64) if has_506 else None
    PRES = np.full((R, G), np.nan, dtype=np.float64)

    has_whole_boxes = all(by_grid[gid][p]["whole_boxes"] is not None for gid in ids for p in common_passes)
    has_every_boxes = all(by_grid[gid][p]["every_boxes"] is not None for gid in ids for p in common_passes)
    has_window_boxes = all(by_grid[gid][p]["window_boxes"] is not None for gid in ids for p in common_passes)

    WB_raw: list[list[list[list[float]]]] = [[[] for _ in range(G)] for _ in range(R)]
    EB_raw: list[list[list[list[float]]]] = [[[] for _ in range(G)] for _ in range(R)]
    SW_raw: list[list[list[list[float]]]] = [[[] for _ in range(G)] for _ in range(R)]
    WB_reach: list[list[list[list[tuple[float, float, float]]]]] = [[[] for _ in range(G)] for _ in range(R)]
    EB_reach: list[list[list[list[tuple[float, float, float]]]]] = [[[] for _ in range(G)] for _ in range(R)]
    SW_reach: list[list[list[list[tuple[float, float, float]]]]] = [[[] for _ in range(G)] for _ in range(R)]

    target = np.zeros((G, 16), dtype=bool)
    sliver = np.zeros((G, 16), dtype=bool)
    tok_lists: dict[str, list[int]] = defaultdict(list)

    for g, gid in enumerate(ids):
        lbl = labels[gid]
        for t in lbl["target"]:
            if 1 <= t <= 16:
                target[g, t - 1] = True
        for t in lbl["sliver"]:
            if 1 <= t <= 16:
                sliver[g, t - 1] = True
        for r, p_idx in enumerate(common_passes):
            rec = by_grid[gid][p_idx]
            W[r, g] = rec["whole"]
            if P_cap is not None:
                P_cap[r, g] = rec["passes"]["capture"]
            if P_506 is not None:
                P_506[r, g] = rec["passes"][506]
            if rec["presence"] is not None:
                PRES[r, g] = rec["presence"]
            if has_whole_boxes and rec["whole_boxes"] is not None:
                WB_raw[r][g] = rec["whole_boxes"]
                WB_reach[r][g] = precompute_box_reach(rec["whole_boxes"])
            if has_every_boxes and rec["every_boxes"] is not None:
                EB_raw[r][g] = rec["every_boxes"]
                EB_reach[r][g] = precompute_box_reach(rec["every_boxes"])
            if has_window_boxes and rec["window_boxes"] is not None:
                SW_raw[r][g] = rec["window_boxes"]
                SW_reach[r][g] = precompute_box_reach(rec["window_boxes"])
            for k_tok, val in rec["tokens"].items():
                if val is not None:
                    tok_lists[k_tok].append(val)

    def _med(key: str) -> float:
        vals = tok_lists.get(key)
        return float(np.median(vals)) if vals else 0.0

    # Split recorded `variant` input tokens across single passes if two passes were recorded.
    n_rec_passes = int(has_cap) + int(has_506)
    variant_med = _med("variant")
    per_pass_tok = variant_med / max(n_rec_passes, 1)
    tokens_med = {
        "whole": _med("whole"),
        "presence": _med("presence"),
        "pass_capture": per_pass_tok,
        "pass_506": per_pass_tok,
        "whole_boxes": _med("whole_boxes"),
        "every_boxes": _med("every_boxes"),
        "window_boxes": _med("window_boxes"),
    }

    return {
        "ids": ids,
        "group_ids": group_ids,
        "passes": common_passes,
        "R": R,
        "G": G,
        "W": W,
        "P_cap": P_cap,
        "P_506": P_506,
        "PRES": PRES,
        "has_whole_boxes": has_whole_boxes,
        "has_every_boxes": has_every_boxes,
        "has_window_boxes": has_window_boxes,
        "WB_raw": WB_raw,
        "EB_raw": EB_raw,
        "SW_raw": SW_raw,
        "WB_reach": WB_reach,
        "EB_reach": EB_reach,
        "SW_reach": SW_reach,
        "window_requests": max(window_requests_seen) if window_requests_seen else 1,
        "target": target,
        "sliver": sliver,
        "empty": (~target) & (~sliver),
        "empty_strict": ~target,
        "notarget": ~target.any(axis=1),
        "tokens": tokens_med,
        "resolved_model": resolved_models[0] if resolved_models else None,
    }


def _close_of(D: dict[str, Any], mode: str) -> np.ndarray:
    if mode == "capture":
        assert D["P_cap"] is not None
        return D["P_cap"]
    if mode == "506":
        assert D["P_506"] is not None
        return D["P_506"]
    if mode == "avg":
        assert D["P_cap"] is not None and D["P_506"] is not None
        return average_close_up_passes([D["P_cap"], D["P_506"]])
    raise ValueError(f"unsupported close-up mode: {mode!r}")


def _estimate_plan_cost(D: dict[str, Any], plan: Plan) -> tuple[int, int]:
    """Estimate request count and median input tokens for a 4x4 plan."""
    tok = D["tokens"]
    n_passes = len(plan.close_ups)
    req = 1 + (1 if plan.skip_below is not None else 0) + 4 * n_passes
    t_est = tok["whole"] + (tok["presence"] if plan.skip_below is not None else 0.0)
    for spec in plan.close_ups:
        t_est += tok["pass_capture"] if spec == "capture" else tok["pass_506"]
    if plan.boxes is not None:
        if plan.boxes.whole is not None:
            req += 1
            t_est += tok["whole_boxes"]
        if plan.boxes.every is not None:
            req += 1
            t_est += tok["every_boxes"]
        if plan.boxes.windows is not None:
            req += int(D["window_requests"])
            t_est += tok["window_boxes"]
    return int(req), int(round(t_est))


def _evaluate_plan_exact(
    D: dict[str, Any],
    weights: Sequence[float],
    plan: Plan,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate `plan` via exact Rust-parity `decide_4x4`, returning `(sel, ok)`."""
    R, G = D["R"], D["G"]
    sel = np.zeros((R, G, 16), dtype=bool)
    for r in range(R):
        for g in range(G):
            close_map: dict[Any, np.ndarray] = {}
            if D["P_cap"] is not None:
                close_map["capture"] = D["P_cap"][r, g]
            if D["P_506"] is not None:
                close_map[506] = D["P_506"][r, g]
            pres_val = D["PRES"][r, g]
            pres = None if np.isnan(pres_val) else float(pres_val)
            picks = decide_4x4(
                D["W"][r, g],
                close_map,
                plan,
                weights=weights,
                decision="logistic",
                presence=pres,
                whole_boxes=D["WB_raw"][r][g] if D["has_whole_boxes"] else None,
                every_boxes=D["EB_raw"][r][g] if D["has_every_boxes"] else None,
                window_boxes=D["SW_raw"][r][g] if D["has_window_boxes"] else None,
            )
            for t in picks:
                sel[r, g, t - 1] = True
    ok = grid_correct(sel, D["target"], D["empty"])
    return sel, ok


def _sweep_tier(
    D: dict[str, Any],
    *,
    tier: str,
    modes: Sequence[str],
    weights: np.ndarray,
    fold_mat: np.ndarray,
    k_folds: int,
    kappa_grid: Sequence[float],
    exempt_grid: Sequence[float],
    gap_grid: Sequence[float | None],
    tau_grid: Sequence[float | None],
    cover_grid: Sequence[float],
    inside_grid: Sequence[float],
    conf_whole: Sequence[float],
    conf_every: Sequence[float],
    conf_windows: Sequence[float],
    wb_max: np.ndarray | None,
    eb_max: np.ndarray | None,
    sw_max: np.ndarray | None,
) -> dict[str, Any]:
    """Run vectorized sweep for one effort tier under a fixed 5-weight vector."""
    R, G = D["R"], D["G"]
    target, empty, notarget, pres = D["target"], D["empty"], D["notarget"], D["PRES"]

    K_arr = np.asarray(kappa_grid, dtype=np.float64)[:, None, None, None, None, None]
    E_arr = np.asarray(exempt_grid, dtype=np.float64)[None, :, None, None, None, None]

    # Gap axis: active only for box-free `low`; boxed tiers keep gap disabled (`None`).
    tier_gaps: tuple[float | None, ...] = tuple(gap_grid) if tier == "low" else (None,)
    gap_vals = np.array([9.0 if g is None else float(g) for g in tier_gaps], dtype=np.float64)
    G_arr = gap_vals[None, None, :, None, None, None]

    # Skip thresholds: only include numeric thresholds if any grid actually has `presence`.
    has_presence = bool((~np.isnan(pres)).any())
    active_taus = tuple(tau_grid) if has_presence else (None,)

    candidates_meta: list[dict[str, Any]] = []
    F_blocks: list[np.ndarray] = []
    P_blocks: list[np.ndarray] = []

    for mode in modes:
        C = _close_of(D, mode)
        F4 = features_4x4(D["W"], C)
        S = sigmoid(F4 @ weights[:4] + weights[4])
        gap_diff = C - D["W"]
        # Broadcast base arrays to `(1, 1, 1, R, G, 16)`
        S_b = S[None, None, None, ...]
        gap_b = gap_diff[None, None, None, ...]
        s_ge_k = S_b >= K_arr
        s_ge_e = S_b >= E_arr
        gap_ok = ~(gap_b > G_arr)

        # Build list of `(cw, ce, cs, cover, inside, ok_mask)` for this tier.
        box_specs: list[tuple[float | None, float | None, float | None, float, float, np.ndarray]] = []
        ones_mask = np.ones((R, G, 16), dtype=bool)
        if tier == "low":
            box_specs.append((None, None, None, 0.01, 0.0, ones_mask))
        elif tier in ("medium", "high"):
            if eb_max is not None:
                for ci, cov in enumerate(cover_grid):
                    for ii, ins in enumerate(inside_grid):
                        for ce in conf_every:
                            box_specs.append((None, ce, None, cov, ins, eb_max[ci, ii] >= ce))
            if wb_max is not None:
                for ci, cov in enumerate(cover_grid):
                    for ii, ins in enumerate(inside_grid):
                        for cw in conf_whole:
                            box_specs.append((cw, None, None, cov, ins, wb_max[ci, ii] >= cw))
            if not box_specs:
                box_specs.append((None, None, None, 0.01, 0.0, ones_mask))
        else:  # tier == "max"
            if sw_max is not None and wb_max is not None:
                for ci, cov in enumerate(cover_grid):
                    for ii, ins in enumerate(inside_grid):
                        for cw in conf_whole:
                            a = wb_max[ci, ii] >= cw
                            for cs in conf_windows:
                                box_specs.append((cw, None, cs, cov, ins, a & (sw_max[ci, ii] >= cs)))
            if sw_max is not None and eb_max is not None:
                for ci, cov in enumerate(cover_grid):
                    for ii, ins in enumerate(inside_grid):
                        for ce in conf_every:
                            a = eb_max[ci, ii] >= ce
                            for cs in conf_windows:
                                box_specs.append((None, ce, cs, cov, ins, a & (sw_max[ci, ii] >= cs)))
            if sw_max is not None and wb_max is None and eb_max is None:
                for ci, cov in enumerate(cover_grid):
                    for ii, ins in enumerate(inside_grid):
                        for cs in conf_windows:
                            box_specs.append((None, None, cs, cov, ins, sw_max[ci, ii] >= cs))
            if not box_specs:
                # Fall back to single whole-image boxes or box-free if windows weren't recorded.
                if eb_max is not None:
                    for ci, cov in enumerate(cover_grid):
                        for ii, ins in enumerate(inside_grid):
                            for ce in conf_every:
                                box_specs.append((None, ce, None, cov, ins, eb_max[ci, ii] >= ce))
                if wb_max is not None:
                    for ci, cov in enumerate(cover_grid):
                        for ii, ins in enumerate(inside_grid):
                            for cw in conf_whole:
                                box_specs.append((cw, None, None, cov, ins, wb_max[ci, ii] >= cw))
                if not box_specs:
                    box_specs.append((None, None, None, 0.01, 0.0, ones_mask))

        # Cache inner evaluation by boolean box-mask bytes to avoid redundant work.
        mask_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        n_inner = len(active_taus) * len(kappa_grid) * len(exempt_grid) * len(tier_gaps)

        for cw, ce, cs, cov, ins, ok_mask in box_specs:
            key_bytes = np.packbits(ok_mask).tobytes()
            cached = mask_cache.get(key_bytes)
            if cached is None:
                sel = s_ge_k & (s_ge_e | (ok_mask[None, None, None, ...] & gap_ok))
                ok = grid_correct(sel, target, empty)  # (K, E, Gp, R, G)
                f_taus: list[np.ndarray] = []
                p_taus: list[np.ndarray] = []
                for tau in active_taus:
                    o_t = _with_skip(ok, pres, notarget, tau).astype(np.float32)  # (K, E, Gp, R, G)
                    # Sum over passes R, then project G -> k_folds via fold_mat
                    f_t = (o_t @ fold_mat).sum(axis=-2).astype(np.int32)  # (K, E, Gp, k_folds)
                    p_t = o_t.sum(axis=-1).astype(np.int32)  # (K, E, Gp, R)
                    f_taus.append(f_t)
                    p_taus.append(p_t)
                f_flat = np.stack(f_taus, axis=0).reshape(n_inner, k_folds)
                p_flat = np.stack(p_taus, axis=0).reshape(n_inner, R)
                cached = (f_flat, p_flat)
                mask_cache[key_bytes] = cached

            F_blocks.append(cached[0])
            P_blocks.append(cached[1])
            candidates_meta.append(
                {
                    "mode": mode,
                    "cw": cw,
                    "ce": ce,
                    "cs": cs,
                    "cover": cov,
                    "inside": ins,
                }
            )

    F_all = np.vstack(F_blocks)  # (N_total, k_folds)
    P_all = np.vstack(P_blocks)  # (N_total, R)
    inner_shape = (len(active_taus), len(kappa_grid), len(exempt_grid), len(tier_gaps))
    n_inner = int(np.prod(inner_shape))

    def _decode(flat_idx: int) -> dict[str, Any]:
        block_i, inner_i = divmod(int(flat_idx), n_inner)
        ti, ki, ei, gi = np.unravel_index(inner_i, inner_shape)
        meta = candidates_meta[block_i]
        return {
            "mode": meta["mode"],
            "tau": active_taus[int(ti)],
            "kappa": float(kappa_grid[int(ki)]),
            "exempt": float(exempt_grid[int(ei)]),
            "gap": tier_gaps[int(gi)],
            "cw": meta["cw"],
            "ce": meta["ce"],
            "cs": meta["cs"],
            "cover": float(meta["cover"]),
            "inside": float(meta["inside"]),
        }

    tot = F_all.sum(axis=1)
    top = int(tot.max())
    near_count = int((tot >= top - R).sum())

    # Simplicity tie-breaker among top-scoring candidates (and for fold/pass tie-breaking).
    top_indices = np.flatnonzero(tot == top)
    best_idx = int(min(top_indices, key=lambda idx: _simplicity_penalty(**_decode(int(idx)))))
    best_cfg = _decode(best_idx)
    best_plan = _candidate_to_plan(**best_cfg)

    # Grouped K-fold OOF accuracy: choose on K-1 folds, evaluate on held-out fold.
    oof_sum = 0
    for f in range(k_folds):
        train_tot = tot - F_all[:, f]
        f_top = int(train_tot.max())
        f_cands = np.flatnonzero(train_tot == f_top)
        if len(f_cands) > 1:
            chosen_f = int(min(f_cands, key=lambda idx: _simplicity_penalty(**_decode(int(idx)))))
        else:
            chosen_f = int(f_cands[0])
        oof_sum += int(F_all[chosen_f, f])
    oof_right = oof_sum / float(R)

    # Leave-one-pass-out cross-pass accuracy (when R >= 2).
    cross_pass_right: float | None = None
    if R >= 2:
        p_tot = P_all.sum(axis=1)
        xp_vals: list[int] = []
        for r in range(R):
            tr_p = p_tot - P_all[:, r]
            p_top = int(tr_p.max())
            p_cands = np.flatnonzero(tr_p == p_top)
            chosen_p = int(min(p_cands, key=lambda idx: _simplicity_penalty(**_decode(int(idx)))))
            xp_vals.append(int(P_all[chosen_p, r]))
        cross_pass_right = float(np.mean(xp_vals))

    return {
        "plan": best_plan,
        "in_sample_right": top / float(R),
        "oof_right": oof_right,
        "cross_pass_right": cross_pass_right,
        "near_optimal_count": near_count,
        "candidates_evaluated": int(F_all.shape[0]),
    }


def _evaluate_scoring_diagnostics(
    D: dict[str, Any],
    fold_arr: np.ndarray,
    k_folds: int,
    available_modes: Sequence[str],
    kappa_grid: Sequence[float],
    tau_grid: Sequence[float | None],
) -> list[dict[str, Any]]:
    """Stage-1 diagnostics: average, log-odds blends, and OOF fitted logistic (no boxes)."""
    R, G = D["R"], D["G"]
    target, empty, notarget, pres = D["target"], D["empty"], D["notarget"], D["PRES"]
    has_presence = bool((~np.isnan(pres)).any())
    active_taus = tuple(tau_grid) if has_presence else (None,)

    rules: list[tuple[str, Any]] = [
        ("avg", 0.5),
        ("lo", 0.2),
        ("lo", 0.35),
        ("lo", 0.5),
        ("fit4_oof", None),
    ]
    rows: list[dict[str, Any]] = []
    for mode in available_modes:
        C = _close_of(D, mode)
        for kind, param in rules:
            if kind == "avg":
                S = float(param) * D["W"] + (1.0 - float(param)) * C
                label = f"average (w={param:.2f})"
            elif kind == "lo":
                S = sigmoid(float(param) * log_odds(D["W"]) + (1.0 - float(param)) * log_odds(C))
                label = f"log-odds blend (w={param:.2f})"
            else:
                F4 = features_4x4(D["W"], C)
                S = np.zeros_like(D["W"])
                for f in range(k_folds):
                    wf = fit_4x4_weights(D["W"], C, target, D["sliver"], fold_arr != f, n_features=4)
                    S[:, fold_arr == f] = sigmoid(F4[:, fold_arr == f] @ wf[:4] + wf[4])
                label = "4-feature logistic (OOF)"

            best_score = -1.0
            best_k = 0.5
            best_tau: float | None = None
            for k in kappa_grid:
                ok = grid_correct(S >= k, target, empty)
                for tau in active_taus:
                    val = float(_with_skip(ok, pres, notarget, tau).sum(axis=1).mean())
                    if val > best_score or (val == best_score and abs(k - 0.5) < abs(best_k - 0.5)):
                        best_score = val
                        best_k = float(k)
                        best_tau = tau
            rows.append(
                {
                    "mode": mode,
                    "rule": label,
                    "best_cutoff": best_k,
                    "best_skip_below": best_tau,
                    "right": round(best_score, 4),
                    "rate": round(best_score / G, 6),
                }
            )
    rows.sort(key=lambda r: -r["right"])
    return rows


def sweep_4x4(
    dataset_path: str | Path,
    labels_path: str | Path,
    recordings_path: str | Path,
    *,
    fit_path: str | Path | None = None,
    fixed_weights: Sequence[float] | None = None,
    folds: int = 5,
    seed: int = 7,
    n_boot: int = 4000,
    fast: bool = False,
    collapse_non_improving: bool = True,
    include_unclear: bool = False,
) -> dict[str, Any]:
    """Run the full 4x4 effort-tier sweep and return structured results + markdown table."""
    if isinstance(folds, bool) or folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    if isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    D = _load_sweep_data(dataset_path, labels_path, recordings_path, include_unclear=include_unclear)
    R, G = D["R"], D["G"]
    gi, fold_arr, n_groups, k_folds = make_group_folds(D["group_ids"], n_folds=folds, seed=seed)

    fold_mat = np.zeros((G, k_folds), dtype=np.float32)
    fold_mat[np.arange(G), fold_arr] = 1.0

    single_modes: list[str] = []
    if D["P_cap"] is not None:
        single_modes.append("capture")
    if D["P_506"] is not None:
        single_modes.append("506")
    available_modes = list(single_modes)
    if D["P_cap"] is not None and D["P_506"] is not None:
        available_modes.append("avg")
    if not available_modes:
        raise ValueError("recordings contain no 4x4 close-up passes")

    kappa_grid = FAST_KAPPA if fast else DEFAULT_KAPPA
    exempt_grid = FAST_EXEMPT if fast else DEFAULT_EXEMPT
    gap_grid = FAST_GAP if fast else DEFAULT_GAP
    tau_grid = FAST_TAU if fast else DEFAULT_TAU
    cover_grid = FAST_COVER if fast else DEFAULT_COVER
    inside_grid = FAST_INSIDE if fast else DEFAULT_INSIDE
    conf_whole = FAST_CONF_WHOLE if fast else DEFAULT_CONF_WHOLE
    conf_every = FAST_CONF_EVERY if fast else DEFAULT_CONF_EVERY
    conf_windows = FAST_CONF_WINDOWS if fast else DEFAULT_CONF_WINDOWS

    # Precompute boxmax tensors once for all available box sources.
    wb_max = _boxmax(D["WB_reach"], cover_grid, inside_grid, R, G) if D["has_whole_boxes"] else None
    eb_max = _boxmax(D["EB_reach"], cover_grid, inside_grid, R, G) if D["has_every_boxes"] else None
    sw_max = _boxmax(D["SW_reach"], cover_grid, inside_grid, R, G) if D["has_window_boxes"] else None

    # Candidate 5-weight vectors: either user-supplied, loaded from fit.json, or fitted now.
    weights_by_mode: dict[str, list[float]] = {}
    if fixed_weights is not None:
        if len(fixed_weights) != 5:
            raise ValueError(f"--weights expects 5 floats, got {len(fixed_weights)}")
        candidate_vectors: dict[str, np.ndarray] = {"custom": np.asarray(fixed_weights, dtype=np.float64)}
        weights_by_mode["custom"] = [round(float(x), 6) for x in fixed_weights]
    else:
        loaded_fit_weights: dict[str, list[float]] = {}
        if fit_path is not None:
            fit_data = json.loads(Path(fit_path).read_text(encoding="utf-8"))
            loaded_fit_weights = (fit_data.get("layout_4x4") or {}).get("weights") or {}
        candidate_vectors = {}
        all_mask = np.ones(G, dtype=bool)
        for m in available_modes:
            if m in loaded_fit_weights and len(loaded_fit_weights[m]) == 5:
                vec = np.asarray(loaded_fit_weights[m], dtype=np.float64)
            else:
                vec = fit_4x4_weights(D["W"], _close_of(D, m), D["target"], D["sliver"], all_mask, n_features=4)
            vec_r = np.round(vec, 6)
            candidate_vectors[m] = vec_r
            weights_by_mode[m] = [float(x) for x in vec_r.tolist()]

    # Stage 1: score-rule diagnostics (no boxes).
    stage1_diagnostics = _evaluate_scoring_diagnostics(D, fold_arr, k_folds, available_modes, kappa_grid, tau_grid)

    # Naive baseline: equal probability average of whole + first available close-up at 0.5, no skip/rules.
    base_close = _close_of(D, "avg" if "avg" in available_modes else single_modes[0])
    base_sel = (0.5 * D["W"] + 0.5 * base_close) >= 0.5
    base_ok = grid_correct(base_sel, D["target"], D["empty"])
    base_strict = grid_correct(base_sel, D["target"], D["empty_strict"])

    tier_modes = {
        "low": tuple(single_modes),
        "medium": tuple(single_modes),
        "high": tuple(["avg"] if "avg" in available_modes else single_modes),
        "max": tuple(["avg"] if "avg" in available_modes else single_modes),
    }

    # Sweep all 4 effort tiers for each candidate weight vector, then choose the best single vector.
    by_vector: dict[str, dict[str, dict[str, Any]]] = {}
    for v_name, vec in candidate_vectors.items():
        tier_res: dict[str, dict[str, Any]] = {}
        for tier in EFFORT_NAMES:
            tier_res[tier] = _sweep_tier(
                D,
                tier=tier,
                modes=tier_modes[tier],
                weights=vec,
                fold_mat=fold_mat,
                k_folds=k_folds,
                kappa_grid=kappa_grid,
                exempt_grid=exempt_grid,
                gap_grid=gap_grid,
                tau_grid=tau_grid,
                cover_grid=cover_grid,
                inside_grid=inside_grid,
                conf_whole=conf_whole,
                conf_every=conf_every,
                conf_windows=conf_windows,
                wb_max=wb_max,
                eb_max=eb_max,
                sw_max=sw_max,
            )
        by_vector[v_name] = tier_res

    def _vector_score(v_name: str) -> tuple[float, float]:
        res = by_vector[v_name]
        oof_total = sum(res[t]["oof_right"] for t in EFFORT_NAMES)
        in_total = sum(res[t]["in_sample_right"] for t in EFFORT_NAMES)
        return (oof_total, in_total)

    chosen_vec_source = max(by_vector.keys(), key=_vector_score)
    chosen_weights = [float(x) for x in candidate_vectors[chosen_vec_source].tolist()]
    chosen_tiers_raw = by_vector[chosen_vec_source]

    # Build final effort ladder, optionally collapsing a higher tier onto a cheaper lower tier
    # when the extra requests do not improve either OOF or in-sample accuracy.
    final_plans: dict[str, Plan] = {}
    efforts_out: dict[str, dict[str, Any]] = {}
    sel_by_tier: dict[str, np.ndarray] = {}
    ok_by_tier: dict[str, np.ndarray] = {}

    prev_tier: str | None = None
    for tier in EFFORT_NAMES:
        raw = chosen_tiers_raw[tier]
        plan = raw["plan"]
        inherited_from: str | None = None

        if collapse_non_improving and prev_tier is not None:
            prev_raw = chosen_tiers_raw[prev_tier]
            prev_plan = final_plans[prev_tier]
            _, prev_tok = _estimate_plan_cost(D, prev_plan)
            _, cur_tok = _estimate_plan_cost(D, plan)
            if (
                raw["oof_right"] <= prev_raw["oof_right"]
                and raw["in_sample_right"] <= prev_raw["in_sample_right"]
                and cur_tok >= prev_tok
            ):
                plan = prev_plan
                inherited_from = prev_tier

        final_plans[tier] = plan
        sel, ok = _evaluate_plan_exact(D, chosen_weights, plan)
        sel_by_tier[tier] = sel
        ok_by_tier[tier] = ok

        # Cross-check vectorized vs exact Rust-parity evaluation when not inherited.
        in_sample_exact = float(ok.sum(axis=1).mean())
        if inherited_from is None:
            assert abs(in_sample_exact - raw["in_sample_right"]) < 1e-6, (
                f"vectorized sweep ({raw['in_sample_right']}) diverged from decide_4x4 ({in_sample_exact}) on {tier}"
            )

        strict_ok = grid_correct(sel, D["target"], D["empty_strict"])
        wrong_picks = float((sel & D["empty"]).sum(axis=(1, 2)).mean())
        missed_targets = float((~sel & D["target"]).sum(axis=(1, 2)).mean())
        est_req, est_tok = _estimate_plan_cost(D, plan)

        pass_flips: float | None = None
        if R >= 2:
            flips = [
                int((sel[r1] != sel[r2]).any(axis=-1).sum())
                for r1, r2 in itertools.combinations(range(R), 2)
            ]
            pass_flips = round(float(np.mean(flips)), 2)

        source_rec = chosen_tiers_raw[inherited_from] if inherited_from else raw
        efforts_out[tier] = {
            "plan": plan.to_dict(),
            "inherited_from": inherited_from,
            "metrics": {
                "in_sample_right": round(in_sample_exact, 4),
                "in_sample_rate": round(in_sample_exact / G, 6),
                "oof_right": round(float(source_rec["oof_right"]), 4),
                "oof_rate": round(float(source_rec["oof_right"]) / G, 6),
                "cross_pass_right": None
                if source_rec["cross_pass_right"] is None
                else round(float(source_rec["cross_pass_right"]), 4),
                "pass_flip_grids": pass_flips,
                "per_pass_right": [int(x) for x in ok.sum(axis=1).tolist()],
                "strict_slivers_right": round(float(strict_ok.sum(axis=1).mean()), 4),
                "wrong_picks_per_pass": round(wrong_picks, 4),
                "missed_targets_per_pass": round(missed_targets, 4),
                "near_optimal_count": int(source_rec["near_optimal_count"]),
                "candidates_evaluated": int(source_rec["candidates_evaluated"]),
                "estimated_requests": est_req,
                "estimated_tokens": est_tok,
            },
        }
        prev_tier = tier

    # Paired grouped bootstrap 95% CIs: low vs baseline, and adjacent tiers.
    comparisons: dict[str, Any] = {}
    low_vs_base = grouped_bootstrap_diff(gi, n_groups, ok_by_tier["low"], base_ok, n_boot=n_boot)
    low_vs_base["answers_differ_per_pass"] = round(
        float((sel_by_tier["low"] != base_sel).any(axis=-1).sum(axis=1).mean()), 2
    )
    comparisons["low_minus_baseline"] = low_vs_base

    for lower, upper in zip(EFFORT_NAMES, EFFORT_NAMES[1:]):
        diff = grouped_bootstrap_diff(gi, n_groups, ok_by_tier[upper], ok_by_tier[lower], n_boot=n_boot)
        diff["answers_differ_per_pass"] = round(
            float((sel_by_tier[upper] != sel_by_tier[lower]).any(axis=-1).sum(axis=1).mean()), 2
        )
        comparisons[f"{upper}_minus_{lower}"] = diff

    result: dict[str, Any] = {
        "model_meta": {
            "resolved_model": D["resolved_model"],
            "recorded_passes": available_modes,
            "recorded_boxes": {
                "whole_grid": D["has_whole_boxes"],
                "every": D["has_every_boxes"],
                "windows": D["has_window_boxes"],
            },
        },
        "dataset_stats": {
            "grids": G,
            "groups": n_groups,
            "passes": R,
            "folds": k_folds,
            "skip_asked_grids": int((~np.isnan(D["PRES"][0])).sum()),
        },
        "weights_by_mode": weights_by_mode,
        "chosen_weights_source": chosen_vec_source,
        "chosen_weights": chosen_weights,
        "baseline": {
            "description": "equal average (50/50) at cutoff 0.50, no skip, no rules",
            "right": round(float(base_ok.sum(axis=1).mean()), 4),
            "total": G,
            "rate": round(float(base_ok.sum(axis=1).mean()) / G, 6),
            "strict_slivers_right": round(float(base_strict.sum(axis=1).mean()), 4),
        },
        "stage1_diagnostics": stage1_diagnostics[:10],
        "efforts": efforts_out,
        "comparisons": comparisons,
    }
    result["markdown_summary"] = format_sweep_markdown(result)
    return result


def format_sweep_markdown(result: dict[str, Any]) -> str:
    """Format sweep results into a clean, publication-ready Markdown summary table."""
    ds = result["dataset_stats"]
    G = ds["grids"]
    R = ds["passes"]
    lines = [
        f"## 4x4 Effort-Level Sweep ({G} grids, {ds['groups']} groups, {R} pass(es), {ds['folds']}-fold grouped CV)",
        "",
        f"- **Chosen 4x4 logistic weights** (`{result['chosen_weights_source']}`): `{result['chosen_weights']}`",
        f"- **Naive 50/50 average baseline**: {result['baseline']['right']:.1f}/{G} ({result['baseline']['rate']:.1%})",
        "",
        "| Tier | Reqs | Est. Tokens | In-Sample | OOF (Grouped CV) | Cross-Pass | Pass Flips | Strict Slivers | Plan Summary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for tier in EFFORT_NAMES:
        entry = result["efforts"][tier]
        m = entry["metrics"]
        p = entry["plan"]
        xp = "—" if m["cross_pass_right"] is None else f"{m['cross_pass_right']:.1f}"
        flips = "—" if m["pass_flip_grids"] is None else f"{m['pass_flip_grids']:.1f}"
        desc_parts = [f"passes={p['close_ups']}", f"cutoff={p['select_from']:.2f}"]
        if p.get("skip_below") is not None:
            desc_parts.append(f"skip<{p['skip_below']:.2f}")
        if p.get("gap"):
            desc_parts.append(f"gap({p['gap']['under']:.2f},{p['gap']['more_than']:.2f})")
        if p.get("boxes"):
            b = p["boxes"]
            asks = [f"{k}>={b[k]}" for k in ("whole", "every", "windows") if b.get(k) is not None]
            desc_parts.append(f"boxes({','.join(asks)}, exempt={b['exempt_from']:.3f})")
        if entry.get("inherited_from"):
            desc_parts.append(f"= {entry['inherited_from']}")
        lines.append(
            f"| {tier} | {m['estimated_requests']} | {m['estimated_tokens']:,} | "
            f"{m['in_sample_right']:.1f} ({m['in_sample_rate']:.1%}) | "
            f"{m['oof_right']:.1f} ({m['oof_rate']:.1%}) | {xp} | {flips} | "
            f"{m['strict_slivers_right']:.1f} | {'; '.join(desc_parts)} |"
        )
    lines.append("")
    lines.append("### Paired Grouped Bootstrap 95% CIs (grids right / pass)")
    lines.append("")
    for key, comp in result["comparisons"].items():
        lines.append(
            f"- **{key}**: `{comp['diff_grids']:+.1f}` "
            f"(95% CI `{comp['ci95_low']:+.1f}` to `{comp['ci95_high']:+.1f}`; "
            f"picks differ on {comp['answers_differ_per_pass']:.1f} grids/pass)"
        )
    return "\n".join(lines)


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "sweep",
        help="Run vectorized 4x4 effort-level parameter sweep with grouped CV and bootstrap CIs.",
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset.jsonl.")
    parser.add_argument("--labels", required=True, help="Path to labels.jsonl.")
    parser.add_argument("--recordings", required=True, help="Path to recordings.jsonl.")
    parser.add_argument("-o", "--output", required=True, help="Path to write sweep_result.json.")
    parser.add_argument("--fit", help="Optional fit.json supplying pre-fitted 4x4 weights.")
    parser.add_argument("--weights", help="Optional comma-separated 5 fixed 4x4 logistic weights.")
    parser.add_argument("--folds", type=_min_int(2), default=5, help="Grouped CV fold count (>= 2, default 5).")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic CV seed (default 7).")
    parser.add_argument("--boot", type=_min_int(1), default=4000, help="Bootstrap resamples for 95%% CIs (>= 1, default 4000).")
    parser.add_argument("--fast", action="store_true", help="Use a coarser parameter grid for quick smoke sweeps.")
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Do not collapse higher effort tiers onto cheaper tiers when OOF/in-sample do not improve.",
    )
    parser.add_argument("--include-unclear", action="store_true", help="Include grids flagged unclear.")
    parser.add_argument("--markdown", help="Optional path to write the Markdown summary table.")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    fixed = [float(x.strip()) for x in args.weights.split(",")] if args.weights else None
    result = sweep_4x4(
        args.dataset,
        args.labels,
        args.recordings,
        fit_path=args.fit,
        fixed_weights=fixed,
        folds=args.folds,
        seed=args.seed,
        n_boot=args.boot,
        fast=args.fast,
        collapse_non_improving=not args.no_collapse,
        include_unclear=args.include_unclear,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(result["markdown_summary"] + "\n", encoding="utf-8")
    print(result["markdown_summary"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vectorized 4x4 effort-level sweep.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
