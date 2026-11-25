"""Build a portable, deduplicated `dataset.jsonl` manifest from lab run directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from eval.schema import allows_skipping_prompt, read_jsonl

# Uniform 3x3 tile boxes `[left, top, width, height]` for pre-composed gapless hCaptcha grids.
UNIFORM_3X3_PICTURES: list[list[float]] = [
    [col / 3.0, row / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    for row in range(3)
    for col in range(3)
]


def _find_trial_files(inputs: Sequence[str | Path]) -> list[tuple[Path, Path, str]]:
    """Return `(trials_jsonl_path, run_dir, source_run_name)` for each input."""
    found: list[tuple[Path, Path, str]] = []
    for raw in inputs:
        p = Path(raw).resolve()
        if p.is_file():
            found.append((p, p.parent, p.parent.name))
        elif p.is_dir():
            direct = p / "trials.jsonl"
            if direct.is_file():
                found.append((direct, p, p.name))
            else:
                nested = sorted(p.rglob("trials.jsonl"))
                if not nested:
                    raise FileNotFoundError(f"no trials.jsonl found under {p}")
                for tf in nested:
                    found.append((tf, tf.parent, tf.parent.name))
        else:
            raise FileNotFoundError(f"input path does not exist: {raw}")
    return found


def _resolve_round_image(
    rnd: dict[str, Any],
    run_dir: Path,
    trial_id: str | None,
    round_idx: int,
    service: str,
) -> tuple[Path | None, bool]:
    """Resolve a round's PNG image path and whether it is a pre-composed hCaptcha 3x3 grid."""
    raw_image = rnd.get("image")
    if raw_image:
        cand = Path(raw_image)
        if cand.is_file():
            resolved = cand.resolve()
            return resolved, (service == "hcaptcha" and resolved.name.startswith("grid-"))
        for base in (run_dir, run_dir / (trial_id or "")):
            alt = base / cand.name
            if alt.is_file():
                resolved = alt.resolve()
                return resolved, (service == "hcaptcha" and resolved.name.startswith("grid-"))
            rel = base / cand
            if rel.is_file():
                resolved = rel.resolve()
                return resolved, (service == "hcaptcha" and resolved.name.startswith("grid-"))
        return None, False

    # hcaptcha_lab.py writes grid-NN.png / task-NN.png inside run_dir / trial_id.
    if trial_id:
        tdir = run_dir / trial_id
        if tdir.is_dir():
            if service == "hcaptcha":
                grids = sorted(tdir.glob("grid-*.png"))
                if round_idx < len(grids):
                    return grids[round_idx].resolve(), True
                tasks = sorted(tdir.glob("task-*.png"))
                if round_idx < len(tasks):
                    return tasks[round_idx].resolve(), False
            else:
                cand = tdir / f"round-{round_idx + 1:02d}.png"
                if cand.is_file():
                    return cand.resolve(), False
    return None, False


def _format_image_path(image_path: Path, output_dir: Path, *, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(image_path.resolve())
    try:
        return Path(os.path.relpath(image_path.resolve(), output_dir.resolve())).as_posix()
    except ValueError:
        return str(image_path.resolve())


def build_dataset(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    layout: int | None = None,
    service: str | None = None,
    absolute_paths: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Scan run directories, deduplicate grids by image SHA-256, and write `dataset.jsonl`."""
    out_path = Path(output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trial_files = _find_trial_files(inputs)

    seen_hashes: set[str] = set()
    manifest: list[dict[str, Any]] = []
    stats = {
        "trial_files": len(trial_files),
        "trials_scanned": 0,
        "rounds_scanned": 0,
        "duplicates_skipped": 0,
        "missing_images": 0,
        "filtered_out": 0,
        "written": 0,
        "groups": 0,
    }

    for trials_path, run_dir, source_run in trial_files:
        rows = read_jsonl(trials_path)
        for trial_idx, trial in enumerate(rows):
            # Support both bare trial dicts (chrome_lab/hcaptcha_lab) and tagged logs (`kind: "trial"`).
            if "kind" in trial and trial["kind"] != "trial":
                continue
            rounds = trial.get("rounds")
            if not isinstance(rounds, list):
                continue
            stats["trials_scanned"] += 1

            trial_service = "hcaptcha" if trial.get("captcha") == "hcaptcha" else "recaptcha"
            if service is not None and trial_service != service:
                stats["filtered_out"] += len(rounds)
                continue

            raw_trial_id = trial.get("trial_id")
            group_id = str(raw_trial_id) if raw_trial_id else f"{source_run}:trial-{trial_idx + 1}"

            for round_idx, rnd in enumerate(rounds):
                stats["rounds_scanned"] += 1
                rnd_layout = int(rnd.get("layout", 3 if trial_service == "hcaptcha" else 4))
                if layout is not None and rnd_layout != layout:
                    stats["filtered_out"] += 1
                    continue

                prompt = str(rnd.get("prompt", "")).strip()
                if not prompt:
                    stats["filtered_out"] += 1
                    continue

                img_path, is_precomposed_grid = _resolve_round_image(
                    rnd, run_dir, raw_trial_id, round_idx, trial_service
                )
                if img_path is None:
                    stats["missing_images"] += 1
                    continue

                sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
                if sha in seen_hashes:
                    stats["duplicates_skipped"] += 1
                    continue
                seen_hashes.add(sha)

                entry: dict[str, Any] = {
                    "id": f"{trial_service[:3]}-{rnd_layout}x{rnd_layout}-{sha[:12]}",
                    "group_id": group_id,
                    "service": trial_service,
                    "layout": rnd_layout,
                    "prompt": prompt,
                    "image": _format_image_path(img_path, out_path.parent, absolute_paths=absolute_paths),
                    "image_sha256": sha,
                    "allows_skipping": allows_skipping_prompt(prompt, trial_service),
                    "source_run": source_run,
                }
                if trial_service == "hcaptcha":
                    if is_precomposed_grid:
                        entry["pictures"] = [list(box) for box in UNIFORM_3X3_PICTURES]
                    elif isinstance(rnd.get("pictures"), list) and len(rnd["pictures"]) == 9:
                        entry["pictures"] = rnd["pictures"]
                manifest.append(entry)
                if limit is not None and len(manifest) >= limit:
                    break
            if limit is not None and len(manifest) >= limit:
                break
        if limit is not None and len(manifest) >= limit:
            break

    with out_path.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["written"] = len(manifest)
    stats["groups"] = len({e["group_id"] for e in manifest})
    return stats


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "dataset",
        help="Build a deduplicated dataset.jsonl manifest from lab run directories.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more chrome_lab / hcaptcha_lab run directories or trials.jsonl files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to write dataset.jsonl.",
    )
    parser.add_argument(
        "--layout",
        type=int,
        choices=(3, 4),
        help="Include only 3x3 or 4x4 grids.",
    )
    parser.add_argument(
        "--service",
        choices=("recaptcha", "hcaptcha"),
        help="Include only grids from one CAPTCHA service.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute image paths instead of paths relative to dataset.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after writing this many unique grids.",
    )
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    stats = build_dataset(
        args.inputs,
        args.output,
        layout=args.layout,
        service=args.service,
        absolute_paths=args.absolute_paths,
        limit=args.limit,
    )
    print(
        f"Wrote {stats['written']} unique grids across {stats['groups']} groups to {args.output} "
        f"(scanned {stats['rounds_scanned']} rounds from {stats['trials_scanned']} trials; "
        f"skipped {stats['duplicates_skipped']} duplicates, {stats['missing_images']} missing images, "
        f"{stats['filtered_out']} filtered)."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a portable dataset.jsonl manifest.")
    sub = parser.add_subparsers(dest="_cmd")
    p = add_subparser(sub)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
