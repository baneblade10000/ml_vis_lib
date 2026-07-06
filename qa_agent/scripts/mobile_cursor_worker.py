#!/usr/bin/env python3
"""Focused mobile UX worker for one cursor slice (no generic touch-target heuristics)."""

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

from qa_agent.interactive_session import get_interactive_session_manager
from qa_agent.mobile_cursor_slices import get_mobile_cursor_slice, resolve_slice_paths
from qa_agent.mobile_targeted_checks import TARGETED_MOBILE_CHECKS, findings_to_issues
from qa_agent.scripts.drive_coverage import build_ux_observations


async def _page_eval(session, script: str) -> Any:
    return await session.browser.page.evaluate(script)


def _area_label(url: str) -> str:
    if "/quiz" in url:
        return "Learn quiz mobile"
    if "/learn/modules/" in url:
        return "Learn module scene mobile"
    if url.startswith("/learn/"):
        return f"Learn {url.split('/learn/')[-1].split('?')[0]} mobile"
    if "/studio/pipelines/" in url and url != "/studio/pipelines":
        return "Studio pipeline editor mobile"
    if url.startswith("/studio/"):
        return f"Studio {url.split('/studio/')[-1].split('?')[0]} mobile"
    return f"Mobile {url}"


async def run(*, slice_id: int, max_steps: int | None = None) -> dict:
    slice_cfg = get_mobile_cursor_slice(slice_id)
    step_budget = max_steps or slice_cfg.max_steps

    manager = get_interactive_session_manager()
    payload = await manager.start(
        profile=slice_cfg.profile,
        mobile=True,
        goal=slice_cfg.goal,
        max_steps=step_budget,
        min_coverage_pct=0.0,
        login=True,
        seed=True,
    )
    run_id = payload["run_id"]
    session = manager.get(run_id)
    if not session:
        return {"slice_id": slice_id, "run_id": run_id, "error": "session missing"}

    routes = session.agent.config.e2e_routes
    if routes is None:
        return {"slice_id": slice_id, "run_id": run_id, "error": "e2e routes missing"}

    paths = resolve_slice_paths(routes, slice_id)
    ux_by_path = build_ux_observations(routes)
    issues_reported = 0
    reported_titles: set[str] = set()

    async def act(tool: str, args: dict | None = None) -> dict:
        return await manager.act(run_id, tool, args or {})

    for url in paths:
        session = manager.get(run_id)
        if not session or session.status.value != "running":
            break
        if session.agent.current_step >= step_budget - 4:
            break

        result = await act("navigate", {"url": url})
        if result.get("finished"):
            break

        await act("scroll", {"direction": "down", "amount": 900})
        check = await _page_eval(session, TARGETED_MOBILE_CHECKS)
        area = _area_label(url)

        for issue in findings_to_issues(check.get("findings") or [], url):
            title = issue["title"]
            if title in reported_titles:
                continue
            reported_titles.add(title)
            await act("report_issue", issue)
            issues_reported += 1

        await act(
            "report_ux_observation",
            {
                "area": area,
                "aspect": "efficiency",
                "score": 4,
                "notes": f"Mobile slice {slice_cfg.name} @ {check.get('viewport')}px",
            },
        )

        for obs in ux_by_path.get(url, []):
            mobile_obs = {**obs, "area": obs.get("area", area) + " (mobile)"}
            await act("report_ux_observation", mobile_obs)

        await act("scroll", {"direction": "up", "amount": 400})

    finish = await act(
        "finish",
        {
            "summary": (
                f"Mobile cursor slice {slice_id} ({slice_cfg.name}): visited {len(paths)} routes, "
                f"reported {issues_reported} targeted layout issues."
            )
        },
    )
    return {
        "slice_id": slice_id,
        "slice_name": slice_cfg.name,
        "run_id": run_id,
        "paths": paths,
        "issues_reported": issues_reported,
        "finished": finish.get("finished"),
        "report_path": finish.get("report_path"),
        "overall_ux_score": finish.get("overall_ux_score"),
        "summary": finish.get("summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-id", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()
    result = asyncio.run(
        run(
            slice_id=args.slice_id,
            max_steps=args.max_steps or None,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
