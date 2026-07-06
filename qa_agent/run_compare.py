from __future__ import annotations

from pathlib import Path
from typing import Any

from qa_agent.run_store import list_issues_from_events, load_metrics, resolve_run_dir


def compare_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
    dir_a = resolve_run_dir(run_id_a)
    dir_b = resolve_run_dir(run_id_b)
    if dir_a is None:
        return {"error": f"Unknown run_id: {run_id_a}"}
    if dir_b is None:
        return {"error": f"Unknown run_id: {run_id_b}"}

    metrics_a = load_metrics(dir_a) or {}
    metrics_b = load_metrics(dir_b) or {}
    issues_a = list_issues_from_events(dir_a)
    issues_b = list_issues_from_events(dir_b)

    titles_a = {item["title"] for item in issues_a if item.get("title")}
    titles_b = {item["title"] for item in issues_b if item.get("title")}

    return {
        "run_a": run_id_a,
        "run_b": run_id_b,
        "score_delta": _delta(metrics_b.get("overall_score"), metrics_a.get("overall_score")),
        "coverage_delta": _delta(
            metrics_b.get("key_route_coverage"), metrics_a.get("key_route_coverage")
        ),
        "run_a_summary": {
            "overall_score": metrics_a.get("overall_score"),
            "key_route_coverage": metrics_a.get("key_route_coverage"),
            "issues_count": metrics_a.get("issues_by_severity") or len(issues_a),
            "exploration_profile": metrics_a.get("exploration_profile"),
            "missing_areas": metrics_a.get("missing_areas", []),
        },
        "run_b_summary": {
            "overall_score": metrics_b.get("overall_score"),
            "key_route_coverage": metrics_b.get("key_route_coverage"),
            "issues_count": metrics_b.get("issues_by_severity") or len(issues_b),
            "exploration_profile": metrics_b.get("exploration_profile"),
            "missing_areas": metrics_b.get("missing_areas", []),
        },
        "new_issues_in_b": sorted(titles_b - titles_a),
        "resolved_issues_in_b": sorted(titles_a - titles_b),
        "coverage_improved": _coverage_gained(metrics_a, metrics_b),
        "coverage_regressed": _coverage_lost(metrics_a, metrics_b),
    }


def _delta(new: Any, old: Any) -> float | None:
    if new is None or old is None:
        return None
    try:
        return round(float(new) - float(old), 1)
    except (TypeError, ValueError):
        return None


def _coverage_gained(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> list[str]:
    visited_a = set(metrics_a.get("visited_areas") or [])
    visited_b = set(metrics_b.get("visited_areas") or [])
    return sorted(visited_b - visited_a)


def _coverage_lost(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> list[str]:
    visited_a = set(metrics_a.get("visited_areas") or [])
    visited_b = set(metrics_b.get("visited_areas") or [])
    return sorted(visited_a - visited_b)
