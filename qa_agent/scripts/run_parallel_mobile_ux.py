#!/usr/bin/env python3
"""Launch N parallel mobile UX workers (separate processes, separate browsers)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().parent / "mobile_ux_worker.py"
PYTHON = ROOT / ".venv" / "bin" / "python3"


def _run_worker(args: tuple[int, int, float, int]) -> dict:
    worker_id, workers, min_coverage_pct, max_steps = args
    cmd = [
        str(PYTHON if PYTHON.is_file() else sys.executable),
        str(WORKER),
        "--worker-id",
        str(worker_id),
        "--workers",
        str(workers),
        "--min-coverage-pct",
        str(min_coverage_pct),
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
    payload: dict = {"worker_id": worker_id, "elapsed_s": elapsed, "exit_code": proc.returncode}
    if proc.stdout.strip():
        try:
            payload["result"] = json.loads(proc.stdout)
        except json.JSONParseError:
            payload["stdout"] = proc.stdout[-4000:]
    if proc.returncode != 0:
        payload["stderr"] = proc.stderr[-4000:]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--min-coverage-pct", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=55)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    jobs = [
        (worker_id, args.workers, args.min_coverage_pct, args.max_steps)
        for worker_id in range(args.workers)
    ]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.get("worker_id", 0))
    summary = {
        "workers": args.workers,
        "completed": sum(1 for r in results if r.get("exit_code") == 0),
        "failed": [r for r in results if r.get("exit_code") != 0],
        "runs": [
            {
                "worker_id": r.get("worker_id"),
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
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
