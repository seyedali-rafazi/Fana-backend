"""Command-line entry point used by the nightly job.

    python -m app.cli analyze --input /data/input.csv --output /data/out.json \
        --now 2026-10-26T10:30:00Z
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze, parse_now


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.cli", description="Bazarcheh support triage")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("analyze", help="analyze a support export CSV")
    run.add_argument("--input", required=True, type=Path, help="path to the CSV export")
    run.add_argument("--output", required=True, type=Path, help="where to write out.json")
    run.add_argument("--now", default=None, help="reference time, ISO-8601 UTC")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "analyze":
        report = analyze(args.input, now=parse_now(args.now))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"wrote {args.output}: {report['total_unique_messages']} unique messages, "
            f"{len(report['urgent_message_ids'])} urgent"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
