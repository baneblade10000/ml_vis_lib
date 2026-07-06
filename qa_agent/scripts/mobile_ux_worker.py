#!/usr/bin/env python3
"""Mobile UX worker — one slice of the full platform route map (parallel coordinator).

Deprecated for issue detection: generic sub-44px heuristics are noisy.
Prefer `mobile_cursor_worker.py` + `run_parallel_mobile_cursor.py` (4 focused slices).
"""

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
from qa_agent.scripts.drive_coverage import build_ux_observations, resolve_navigate_paths


async def _page_eval(session, script: str) -> Any:
    return await session.browser.page.evaluate(script)


MOBILE_CHECKS = """
() => {
  const issues = [];
  const vw = window.innerWidth;
  const buttons = Array.from(document.querySelectorAll('button, a[href], [role="button"]'));
  let smallTargets = 0;
  for (const el of buttons.slice(0, 80)) {
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) continue;
    if (r.width < 44 || r.height < 44) smallTargets += 1;
  }
  if (smallTargets >= 3) {
    issues.push(`many sub-44px touch targets (${smallTargets} sampled)`);
  }
  const doc = document.documentElement;
  if (doc.scrollWidth > vw + 8) {
    issues.push('horizontal overflow detected');
  }
  const bottomNav = document.querySelector('nav') || document.querySelector('[class*="bottom"]');
  if (bottomNav) {
    const rect = bottomNav.getBoundingClientRect();
    if (rect.bottom >= window.innerHeight - 2 && rect.height > 40) {
      issues.push('fixed bottom chrome may reduce content area');
    }
  }
  return { viewport: vw, issues };
}
"""


def _mobile_ux_obs(url: str, area: str, check: dict) -> list[dict[str, Any]]:
    obs: list[dict[str, Any]] = [
        {"area": area, "aspect": "efficiency", "score": 4, "notes": f"Mobile viewport {check.get('viewport')}px"},
        {"area": area, "aspect": "findability", "score": 4},
    ]
    issues = check.get("issues") or []
    if "horizontal overflow" in " ".join(issues):
        obs.append({"area": area, "aspect": "clarity", "score": 2, "notes": "; ".join(issues)})
    elif issues:
        obs.append({"area": area, "aspect": "efficiency", "score": 3, "notes": "; ".join(issues)})
    return obs


def _area_label(url: str) -> str:
    if "/learn/modules/" in url and "/quiz" in url:
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


async def run(*, worker_id: int, workers: int, min_coverage_pct: float, max_steps: int) -> dict:
    manager = get_interactive_session_manager()
    payload = await manager.start(
        profile="mobile_full",
        mobile=True,
        max_steps=max_steps,
        min_coverage_pct=min_coverage_pct,
        login=True,
        seed=True,
    )
    run_id = payload["run_id"]
    session = manager.get(run_id)
    if not session:
        return {"worker_id": worker_id, "run_id": run_id, "error": "session missing"}

    all_paths = resolve_navigate_paths(session)
    my_paths = [path for index, path in enumerate(all_paths) if index % workers == worker_id]
    routes = session.agent.config.e2e_routes
    ux_by_path = build_ux_observations(routes)
    issues_reported = 0

    async def act(tool: str, args: dict | None = None) -> dict:
        return await manager.act(run_id, tool, args or {})

    for url in my_paths:
        session = manager.get(run_id)
        if not session or session.status.value != "running":
            break
        if session.agent.current_step >= max_steps - 3:
            break

        result = await act("navigate", {"url": url})
        if result.get("finished"):
            break

        await act("scroll", {"direction": "down", "amount": 700})
        check = await _page_eval(session, MOBILE_CHECKS)
        area = _area_label(url)
        for obs in _mobile_ux_obs(url, area, check):
            await act("report_ux_observation", obs)

        check_issues = check.get("issues") or []
        if check_issues:
            await act(
                "report_issue",
                {
                    "severity": "minor" if "overflow" not in " ".join(check_issues) else "major",
                    "title": f"Mobile layout friction on {url}",
                    "description": "; ".join(check_issues),
                    "repro_steps": [
                        f"Open {url} on 390px viewport",
                        "Scroll page and inspect touch targets / overflow",
                    ],
                },
            )
            issues_reported += 1

        for obs in ux_by_path.get(url, []):
            mobile_obs = {**obs, "area": obs.get("area", area) + " (mobile)"}
            await act("report_ux_observation", mobile_obs)

        await act("scroll", {"direction": "up", "amount": 300})

    finish = await act(
        "finish",
        {
            "summary": (
                f"Mobile UX worker {worker_id + 1}/{workers}: visited {len(my_paths)} routes, "
                f"reported {issues_reported} mobile layout issues."
            )
        },
    )
    return {
        "worker_id": worker_id,
        "run_id": run_id,
        "paths": my_paths,
        "issues_reported": issues_reported,
        "finished": finish.get("finished"),
        "report_path": finish.get("report_path"),
        "overall_ux_score": finish.get("overall_ux_score"),
        "summary": finish.get("summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--min-coverage-pct", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=55)
    args = parser.parse_args()
    result = asyncio.run(
        run(
            worker_id=args.worker_id,
            workers=args.workers,
            min_coverage_pct=args.min_coverage_pct,
            max_steps=args.max_steps,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
