"""Unified CLI for HyperCaptcha offline evaluation (`hypercaptcha-eval` / `python -m eval`)."""

from __future__ import annotations

import argparse
from typing import Sequence

from eval import dataset, export, fit, label_server, record, summarize, sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypercaptcha-eval",
        description="Offline dataset, labeling, recording, fitting, sweep, export, and summary toolkit for HyperCaptcha.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    dataset.add_subparser(subparsers)
    label_server.add_subparser(subparsers)
    record.add_subparser(subparsers)
    fit.add_subparser(subparsers)
    sweep.add_subparser(subparsers)
    export.add_subparser(subparsers)
    summarize.add_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
