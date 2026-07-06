from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_RUNS_ROOT = Path(__file__).resolve().parent / "runs"
_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


def runs_root() -> Path:
    return _RUNS_ROOT


def resolve_run_dir(run_id: str) -> Path | None:
    if not _RUN_ID_PATTERN.match(run_id):
        return None
    run_dir = _RUNS_ROOT / run_id
    if run_dir.is_dir():
        return run_dir
    return None


def list_disk_runs(*, limit: int = 20) -> list[dict[str, Any]]:
    if not _RUNS_ROOT.is_dir():
        return []

    candidates: list[tuple[str, Path]] = []
    for entry in _RUNS_ROOT.iterdir():
        if not entry.is_dir() or not _RUN_ID_PATTERN.match(entry.name):
            continue
        if (entry / "events.jsonl").exists() or (entry / "report.md").exists():
            candidates.append((entry.name, entry))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [summarize_disk_run(run_id, run_dir) for run_id, run_dir in candidates[:limit]]


def summarize_disk_run(run_id: str, run_dir: Path) -> dict[str, Any]:
    metrics = load_metrics(run_dir)
    events_summary = _summarize_events(run_dir / "events.jsonl")
    status = events_summary.get("status", "unknown")
    if status == "unknown" and (run_dir / "report.md").exists():
        status = "completed"

    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "source": "disk",
        "run_dir": str(run_dir),
        "started_at": events_summary.get("started_at"),
        "finished_at": events_summary.get("finished_at"),
        "current_step": events_summary.get("current_step", 0),
        "issues_count": events_summary.get("issues_count", 0),
        "summary": events_summary.get("summary"),
        "error": events_summary.get("error"),
        "exploration_profile": events_summary.get("exploration_profile"),
        "report_path": str(run_dir / "report.md") if (run_dir / "report.md").exists() else None,
        "metrics_path": str(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None,
        "events_path": str(run_dir / "events.jsonl") if (run_dir / "events.jsonl").exists() else None,
    }
    if metrics:
        payload["overall_ux_score"] = metrics.get("overall_score")
        payload["key_route_coverage"] = metrics.get("key_route_coverage")
        payload["exploration_profile"] = metrics.get(
            "exploration_profile", payload.get("exploration_profile")
        )
    if events_summary.get("goal"):
        payload["goal"] = events_summary["goal"]
    if events_summary.get("max_steps"):
        payload["max_steps"] = events_summary["max_steps"]
    return payload


def _summarize_events(events_path: Path) -> dict[str, Any]:
    if not events_path.exists():
        return {}

    summary: dict[str, Any] = {
        "current_step": 0,
        "issues_count": 0,
        "status": "unknown",
    }
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("event")
        data = event.get("data") or {}
        step = event.get("step")
        if isinstance(step, int):
            summary["current_step"] = max(summary["current_step"], step)

        if event_type == "run_start":
            summary["status"] = "running"
            summary["started_at"] = event.get("ts")
            summary["goal"] = data.get("goal")
            summary["max_steps"] = data.get("max_steps")
            summary["exploration_profile"] = data.get("exploration_profile")
        elif event_type == "issue_reported":
            summary["issues_count"] = int(summary.get("issues_count", 0)) + 1
        elif event_type == "run_complete":
            summary["status"] = "completed"
            summary["finished_at"] = event.get("ts")
            summary["summary"] = data.get("summary")
            summary["issues_count"] = data.get("issues_count", summary.get("issues_count", 0))
        elif event_type == "run_failed":
            summary["status"] = "failed"
            summary["finished_at"] = event.get("ts")
            summary["error"] = data.get("error")
        elif event_type == "run_finish_requested" and summary.get("status") == "running":
            summary["summary"] = data.get("summary")

    return summary


def load_metrics(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_report(run_dir: Path) -> str | None:
    report_path = run_dir / "report.md"
    if not report_path.exists():
        return None
    return report_path.read_text(encoding="utf-8")


def load_events(run_dir: Path) -> str | None:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return None
    return events_path.read_text(encoding="utf-8")


def list_issues_from_events(run_dir: Path) -> list[dict[str, Any]]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []

    issues: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "issue_reported":
            continue
        data = event.get("data") or {}
        issues.append(
            {
                "index": len(issues) + 1,
                "step": event.get("step"),
                "url": event.get("url"),
                "severity": data.get("severity"),
                "title": data.get("title"),
            }
        )
    return issues


def resolve_screenshot_path(run_dir: Path, name: str) -> Path | None:
    if ".." in name or name.startswith("/"):
        return None
    path = run_dir / "screenshots" / name
    if path.is_file():
        return path
    return None


def resolve_issue_screenshot(run_dir: Path, issue_number: int) -> Path | None:
    if issue_number < 1:
        return None
    return resolve_screenshot_path(run_dir, f"issue-{issue_number:02d}.png")
