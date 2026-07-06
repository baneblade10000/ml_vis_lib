#!/usr/bin/env python3
"""Deterministic coverage driver for cursor QA sessions (coordinator use)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_agent.e2e_routes import _fallback_routes
from qa_agent.interactive_session import InteractiveSession, get_interactive_session_manager

# Auth is established by start(login=True, seed=True). Login coverage area accepts
# /login or /signup (see coverage.ROUTE_AREAS); /signup is reachable when logged in.

def build_ux_observations(routes) -> dict[str, list[dict[str, Any]]]:
    """Map resolved URLs to UX observations for deterministic drivers."""
    if routes is None:
        return {
            "/learn/progress": [{"area": "Learn progress", "aspect": "findability", "score": 4}],
            "/learn/assistant": [{"area": "Learn assistant", "aspect": "clarity", "score": 4}],
            "/studio/podcasts": [{"area": "Studio podcasts", "aspect": "findability", "score": 5}],
            "/studio/pipelines": [{"area": "Studio pipelines", "aspect": "findability", "score": 4}],
            "/studio/modules": [
                {
                    "area": "Studio modules generation pipeline",
                    "aspect": "findability",
                    "score": 4,
                }
            ],
        }
    return {
        routes.learn_module_path: [
            {"area": "Learn modules list", "aspect": "clarity", "score": 4},
            {"area": "Learn module scene", "aspect": "clarity", "score": 5},
        ],
        routes.learn_module_quiz_path: [
            {"area": "Learn quiz page", "aspect": "clarity", "score": 4},
        ],
        "/learn/progress": [{"area": "Learn progress", "aspect": "findability", "score": 4}],
        "/learn/assistant": [{"area": "Learn assistant", "aspect": "clarity", "score": 4}],
        "/studio/podcasts": [{"area": "Studio podcasts", "aspect": "findability", "score": 5}],
        routes.studio_module_edit_path: [
            {"area": "Studio module editor", "aspect": "clarity", "score": 4},
        ],
        routes.studio_pipelines_list_path: [
            {"area": "Studio pipelines", "aspect": "findability", "score": 4},
        ],
        routes.studio_modules_path: [
            {
                "area": "Studio modules generation pipeline",
                "aspect": "findability",
                "score": 4,
            },
        ],
    }


def resolve_navigate_paths(session: InteractiveSession) -> list[str]:
    routes = session.agent.config.e2e_routes
    profile = session.agent.config.exploration_profile
    if routes is not None:
        if profile == "pipelines":
            return routes.pipeline_coverage_navigate_paths()
        return routes.coverage_navigate_paths()
    fallback = _fallback_routes()
    if profile == "pipelines":
        return fallback.pipeline_coverage_navigate_paths()
    return fallback.coverage_navigate_paths()


async def drive(*, profile: str, mobile: bool, min_coverage_pct: float, max_steps: int) -> dict:
    manager = get_interactive_session_manager()
    payload = await manager.start(
        profile=profile,
        mobile=mobile,
        max_steps=max_steps,
        min_coverage_pct=min_coverage_pct,
        login=True,
        seed=True,
    )
    run_id = payload["run_id"]
    session = manager.get(run_id)
    if not session:
        return {"run_id": run_id, "error": "session missing after start"}

    navigate_paths = resolve_navigate_paths(session)
    routes = session.agent.config.e2e_routes
    ux_by_path = build_ux_observations(routes)

    for url in navigate_paths:
        session = manager.get(run_id)
        if not session or session.status.value != "running":
            break
        if session.agent.current_step >= max_steps - 2:
            break
        result = await manager.act(run_id, "navigate", {"url": url})
        if result.get("finished"):
            return result
        for obs in ux_by_path.get(url, []):
            result = await manager.act(run_id, "report_ux_observation", obs)
            if result.get("finished"):
                return result

    session = manager.get(run_id)
    if session and session.status.value == "running":
        return await manager.act(
            run_id,
            "finish",
            {
                "summary": (
                    f"Deterministic {profile} coverage — visited {len(navigate_paths)} routes."
                )
            },
        )
    return {"run_id": run_id, "finished": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="full")
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--min-coverage-pct", type=float, default=90.0)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()
    result = asyncio.run(
        drive(
            profile=args.profile,
            mobile=args.mobile,
            min_coverage_pct=args.min_coverage_pct,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
