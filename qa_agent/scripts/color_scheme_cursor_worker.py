#!/usr/bin/env python3
"""Screenshot sweep driver for one color-scheme cursor slice (0–4)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_agent.color_scheme_cursor_slices import get_color_scheme_cursor_slice, resolve_slice_paths
from qa_agent.interactive_session import get_interactive_session_manager


async def run(*, slice_id: int, max_steps: int = 45, finish: bool = False) -> dict:
    slice_cfg = get_color_scheme_cursor_slice(slice_id)
    manager = get_interactive_session_manager()
    payload = await manager.start(
        goal=slice_cfg.goal,
        max_steps=max_steps,
        profile=slice_cfg.profile,
        login=True,
        seed=True,
    )
    run_id = payload["run_id"]
    session = manager.get(run_id)
    if not session or session.agent.config.e2e_routes is None:
        return {"slice_id": slice_id, "error": "session or routes missing", "run_id": run_id}

    routes = resolve_slice_paths(session.agent.config.e2e_routes, slice_id)
    cfg = session.agent.config
    captures: list[dict] = []

    async def act(tool: str, args: dict | None = None) -> dict:
        return await manager.act(run_id, tool, args or {})

    async def ensure_logged_in() -> None:
        if "/login" in session.browser.page.url:
            await session.browser.login(cfg.test_email, cfg.test_password)

    async def goto(path: str) -> dict:
        r = await act("navigate", {"url": path})
        await ensure_logged_in()
        if "/login" in session.browser.page.url:
            r = await act("navigate", {"url": path})
        await act("wait_for", {"seconds": 0.8})
        return r

    for scheme in slice_cfg.schemes:
        for mode in ("light", "dark"):
            r = await act("set_appearance", {"color_scheme": scheme, "theme_mode": mode})
            if r.get("finished"):
                break
            for path in routes:
                if session.agent.current_step >= max_steps - 2:
                    break
                r = await goto(path)
                if r.get("finished"):
                    break
                step = r.get("step", 0)
                captures.append(
                    {
                        "scheme": scheme,
                        "mode": mode,
                        "path": path,
                        "step": step,
                        "url": r.get("url"),
                        "screenshot": r.get("screenshot_resource"),
                    }
                )

    status: dict = {"finished": False}
    if finish:
        status = await act(
            "finish",
            {
                "summary": (
                    f"Color scheme slice {slice_id} ({slice_cfg.name}): "
                    f"{len(captures)} captures across {list(slice_cfg.schemes)}."
                )
            },
        )

    return {
        "slice_id": slice_id,
        "slice_name": slice_cfg.name,
        "run_id": run_id,
        "schemes": list(slice_cfg.schemes),
        "paths": routes,
        "captures": len(captures),
        "finished": status.get("finished"),
        "report_path": status.get("report_path"),
        "issues_count": status.get("issues_count"),
        "overall_ux_score": status.get("overall_ux_score"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-id", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        run(slice_id=args.slice_id, max_steps=args.max_steps, finish=args.finish)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
