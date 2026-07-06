#!/usr/bin/env python3
"""Launch four focused mobile cursor workers (replaces noisy mobile_ux_worker heuristics)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().parent / "mobile_cursor_worker.py"
PYTHON = ROOT / ".venv" / "bin" / "python3"
DEFAULT_SLICES = (0, 1, 2, 3)


def _run_worker(args: tuple[int, int]) -> dict:
    slice_id, max_steps = args
    cmd = [
        str(PYTHON if PYTHON.is_file() else sys.executable),
        str(WORKER),
        "--slice-id",
        str(slice_id),
        "--max-steps",
        str(max_steps),
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    elapsed = round(time.time() - started, 1)
    payload: dict = {"slice_id": slice_id, "elapsed_s": elapsed, "exit_code": proc.returncode}
    if proc.stdout.strip():
        try:
            payload["result"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload["stdout"] = proc.stdout[-4000:]
    if proc.returncode != 0:
        payload["stderr"] = proc.stderr[-4000:]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slices",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SLICES),
        help="Comma-separated slice ids 0–3 (default: 0,1,2,3)",
    )
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    slice_ids = [int(part.strip()) for part in args.slices.split(",") if part.strip()]
    jobs = [(slice_id, args.max_steps) for slice_id in slice_ids]

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_run_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.get("slice_id", 0))
    total_issues = sum((r.get("result") or {}).get("issues_reported", 0) for r in results)
    summary = {
        "mode": "mobile_cursor_slices",
        "slices": slice_ids,
        "completed": sum(1 for r in results if r.get("exit_code") == 0),
        "failed": [r for r in results if r.get("exit_code") != 0],
        "total_issues_reported": total_issues,
        "runs": [
            {
                "slice_id": r.get("slice_id"),
                "slice_name": (r.get("result") or {}).get("slice_name"),
                "run_id": (r.get("result") or {}).get("run_id"),
                "report_path": (r.get("result") or {}).get("report_path"),
                "issues_reported": (r.get("result") or {}).get("issues_reported"),
                "overall_ux_score": (r.get("result") or {}).get("overall_ux_score"),
                "paths": (r.get("result") or {}).get("paths"),
                "elapsed_s": r.get("elapsed_s"),
            }
            for r in results
        ],
    }
    out = json.dumps(summary, indent=2, ensure_ascii=False)
    print(out)
    default_out = ROOT / "qa_agent" / "runs" / "parallel_mobile_cursor_summary.json"
    output_path = Path(args.output) if args.output else default_out
    output_path.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
