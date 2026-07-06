#!/usr/bin/env python3
"""Smoke-test pipeline QA tools (testids, page context, preset switch)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_agent.interactive_session import get_interactive_session_manager


async def smoke() -> dict:
    manager = get_interactive_session_manager()
    start = await manager.start(profile="pipelines", max_steps=20, min_coverage_pct=0, login=True, seed=True)
    run_id = start["run_id"]
    routes = start.get("system_prompt", "")

    async def act(tool: str, args: dict | None = None) -> dict:
        return await manager.act(run_id, tool, args or {})

    checks: list[dict] = []

    # Modules — generation pipeline select
    r = await act("navigate", {"url": "/studio/modules"})
    digest = r.get("dom_digest", "")
    checks.append(
        {
            "step": "modules generation select",
            "ok": "generation-pipeline-select" in digest or "testid=generation-pipeline-select" in digest,
            "url": r.get("url"),
        }
    )

    # Default preset editor — save/run buttons in page context
    default_path = None
    for line in routes.splitlines():
        if "Default preset editor:" in line:
            default_path = line.split(":", 1)[1].strip()
            break
    if not default_path:
        default_path = "/studio/pipelines"

    r = await act("navigate", {"url": default_path})
    digest = r.get("dom_digest", "")
    checks.append(
        {
            "step": "default editor toolbar",
            "ok": "pipeline-save-btn" in digest and "pipeline-run-btn" in digest,
            "url": r.get("url"),
        }
    )

    # Add a node from palette (budget tweak enables save without full palette click)
    if "pipeline-palette-add-" in digest:
        testid = next(
            line.split("testid=")[1].split()[0]
            for line in digest.splitlines()
            if "pipeline-palette-add-" in line and "testid=" in line
        )
        await act("click_testid", {"testid": testid})
        r = await act("wait_for", {"seconds": 0.5})
        digest = r.get("dom_digest", "")
    checks.append(
        {
            "step": "save enabled after change",
            "ok": "pipeline-save-btn: disabled=false" in digest,
            "url": r.get("url"),
        }
    )

    # Switch preset — unsaved dialog
    r = await act("click_testid", {"testid": "pipeline-preset-fast"})
    digest = r.get("dom_digest", "")
    checks.append(
        {
            "step": "unsaved switch dialog",
            "ok": "unsaved-changes dialog may be open" in digest,
            "url": r.get("url"),
        }
    )
    await act("click_testid", {"testid": "unsaved-switch-discard"})

    r = await act("finish", {"summary": "Pipeline tools smoke test complete."})
    return {"run_id": run_id, "checks": checks, "finished": r.get("finished"), "report_path": r.get("report_path")}


def main() -> None:
    result = asyncio.run(smoke())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    failed = [c for c in result["checks"] if not c["ok"]]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
