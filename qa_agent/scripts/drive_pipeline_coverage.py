#!/usr/bin/env python3
"""Deterministic pipeline-builder coverage (resolves preset ids at runtime)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_agent.scripts.drive_coverage import drive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-coverage-pct", type=float, default=75.0)
    parser.add_argument("--max-steps", type=int, default=35)
    args = parser.parse_args()
    result = asyncio.run(
        drive(
            profile="pipelines",
            mobile=False,
            min_coverage_pct=args.min_coverage_pct,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
