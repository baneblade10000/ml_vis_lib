#!/usr/bin/env python3
"""Full pipeline QA checklist (cursor-driven simulation via qa_cursor_act tools)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_agent.interactive_session import get_interactive_session_manager


def _parse_route(system_prompt: str, label: str) -> str | None:
    for line in system_prompt.splitlines():
        if line.strip().startswith(f"- {label}:"):
            return line.split(":", 1)[1].strip()
    return None


def _find_testid_in_digest(digest: str, prefix: str) -> str | None:
    for line in digest.splitlines():
        if prefix in line and "testid=" in line:
            return line.split("testid=")[1].split()[0]
    return None


def _preset_duplicate_counts(options: list[dict[str, Any]]) -> tuple[int, int]:
    default = 0
    fast = 0
    for option in options:
        label = (option.get("label") or "").lower()
        if "preset" not in label:
            continue
        if "default" in label and "module" in label:
            default += 1
        elif "fast" in label and "module" in label:
            fast += 1
    return default, fast


async def _page_eval(session, script: str) -> Any:
    return await session.browser.page.evaluate(script)


async def run_checklist() -> dict:
    manager = get_interactive_session_manager()
    start = await manager.start(
        profile="pipelines",
        max_steps=40,
        min_coverage_pct=75,
        login=True,
        seed=True,
    )
    run_id = start["run_id"]
    session = manager.get(run_id)
    assert session is not None
    prompt = start.get("system_prompt", "")

    default_path = _parse_route(prompt, "Default preset editor") or "/studio/pipelines"
    fast_path = _parse_route(prompt, "Fast preset editor") or default_path
    material_path = _parse_route(prompt, "Material detail") or "/studio/materials"

    async def act(tool: str, args: dict | None = None) -> dict:
        return await manager.act(run_id, tool, args or {})

    checks: list[dict[str, Any]] = []

    async def check(name: str, ok: bool, **extra: Any) -> None:
        checks.append({"step": name, "ok": ok, **extra})

    # 1) Modules — generation pipeline select
    r = await act("navigate", {"url": "/studio/modules"})
    digest = r.get("dom_digest", "")
    await check(
        "generation-pipeline-select visible",
        "testid=generation-pipeline-select" in digest or "generation-pipeline-select" in digest,
        url=r.get("url"),
    )
    options = await _page_eval(
        session,
        """() => {
            const sel = document.querySelector('[data-testid="generation-pipeline-select"]');
            if (!sel) return [];
            return Array.from(sel.options).map(o => ({ value: o.value, label: o.textContent?.trim() }));
        }""",
    )
    await check("generation presets in select", len(options or []) >= 2, options=options)
    default_dupes, fast_dupes = _preset_duplicate_counts(options or [])
    if default_dupes > 1 or fast_dupes > 1:
        await act(
            "report_issue",
            {
                "severity": "minor",
                "title": "Duplicate module pipeline presets in generation select",
                "description": (
                    f"The generation pipeline picker lists {default_dupes} Default and "
                    f"{fast_dupes} Fast preset entries. Seeded presets should appear once "
                    "each; duplicates confuse authors and break stable QA preset ids."
                ),
                "repro_steps": [
                    "Open /studio/modules",
                    "Expand the generation pipeline select",
                    "Count duplicate Default/Fast preset labels",
                ],
            },
        )
        await check("duplicate presets reported", True, default=default_dupes, fast=fast_dupes)
    else:
        await check("no duplicate presets", True)
    if options and len(options) >= 2:
        alt = options[1]["value"]
        r = await act("select_testid", {"testid": "generation-pipeline-select", "value": alt})
        selected = await _page_eval(
            session,
            '() => document.querySelector(\'[data-testid="generation-pipeline-select"]\')?.value',
        )
        await check("generation select changed", selected == alt, selected=selected)
    await act(
        "report_ux_observation",
        {"area": "Modules generation pipeline", "aspect": "findability", "score": 4},
    )

    # 2) Default preset editor — palette, inspector, run panel, save
    r = await act("navigate", {"url": default_path})
    digest = r.get("dom_digest", "")
    await check(
        "editor save/run in page context",
        "pipeline-save-btn" in digest and "pipeline-run-btn" in digest,
        url=r.get("url"),
    )

    palette_testid = _find_testid_in_digest(digest, "pipeline-palette-add-")
    if palette_testid:
        await act("click_testid", {"testid": palette_testid})
        r = await act("wait_for", {"seconds": 0.5})
    digest = r.get("dom_digest", "")

    node_count = await _page_eval(session, "() => document.querySelectorAll('.react-flow__node').length")
    await check("node added to canvas", (node_count or 0) >= 1, nodes=node_count)

    clicked_node = await _page_eval(
        session,
        """() => {
            const node = document.querySelector('.react-flow__node');
            if (!node) return false;
            node.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            return true;
        }""",
    )
    await act("wait_for", {"seconds": 0.4})
    inspector_visible = await _page_eval(
        session,
        """() => {
            const panel = document.querySelector('#pipeline-inspector');
            if (!panel) return false;
            const text = panel.textContent || '';
            return text.length > 20;
        }""",
    )
    await check("inspector opens on node click", bool(clicked_node and inspector_visible))

    expanded_before = await _page_eval(
        session,
        """() => document.querySelector('[data-testid="pipeline-run-panel-toggle"]')?.getAttribute('aria-expanded')""",
    )
    r = await act("click_testid", {"testid": "pipeline-run-panel-toggle"})
    digest = r.get("dom_digest", "")
    expanded_after = await _page_eval(
        session,
        """() => document.querySelector('[data-testid="pipeline-run-panel-toggle"]')?.getAttribute('aria-expanded')""",
    )
    await check(
        "run panel toggle via testid",
        expanded_before != expanded_after
        or (
            "pipeline-run-panel-toggle: aria-expanded=false" in digest
            if expanded_before == "true"
            else "pipeline-run-panel-toggle: aria-expanded=true" in digest
        ),
        before=expanded_before,
        after=expanded_after,
    )
    digest = r.get("dom_digest", "")
    await check("save enabled after edit", "pipeline-save-btn: disabled=false" in digest)

    # 3) Unsaved switch Default → Fast
    r = await act("click_testid", {"testid": "pipeline-preset-fast"})
    digest = r.get("dom_digest", "")
    await check("unsaved dialog on preset switch", "unsaved-changes dialog may be open" in digest)
    await act("click_testid", {"testid": "unsaved-switch-discard"})
    r = await act("wait_for", {"seconds": 0.5})
    url = r.get("url", "")
    fast_ids = {o["value"] for o in (options or []) if "Fast" in (o.get("label") or "")}
    await check(
        "switched to fast preset",
        any(fid in url for fid in fast_ids) if fast_ids else fast_path.split("/")[-1] in url,
        url=url,
        fast_ids=sorted(fast_ids),
    )

    # 4) Materials — generate modal open + cancel
    r = await act("navigate", {"url": "/studio/materials"})
    await check("materials list", "/studio/materials" in (r.get("url") or ""), url=r.get("url"))
    r = await act("navigate", {"url": material_path})
    digest = r.get("dom_digest", "")
    await check(
        "generate-course-btn in digest",
        "testid=generate-course-btn" in digest or "generate-course-btn" in digest,
        url=r.get("url"),
    )
    r = await act("click_testid", {"testid": "generate-course-btn"})
    await act("wait_for", {"seconds": 0.5})
    modal_open = await _page_eval(
        session,
        "() => Boolean(document.querySelector('[data-testid=\"generate-course-modal\"]'))",
    )
    modal_ok = False
    if modal_open:
        r = await act("click_testid", {"testid": "generate-course-cancel-btn"})
        await act("wait_for", {"seconds": 0.3})
        modal_ok = not await _page_eval(
            session,
            "() => Boolean(document.querySelector('[data-testid=\"generate-course-modal\"]'))",
        )
    await check(
        "generate modal open and cancel via testid",
        modal_open and modal_ok,
        modal_open=modal_open,
    )

    await act(
        "report_ux_observation",
        {"area": "Pipeline editor", "aspect": "clarity", "score": 4},
    )

    finish = await act(
        "finish",
        {
            "summary": (
                "Pipeline checklist complete: generation select, editor palette/inspector/run panel, "
                "unsaved preset switch, materials generate modal cancel."
            )
        },
    )

    return {
        "run_id": run_id,
        "checks": checks,
        "failed": [c for c in checks if not c["ok"]],
        "finished": finish.get("finished"),
        "coverage_pct": finish.get("coverage_pct"),
        "report_path": finish.get("report_path"),
        "summary": finish.get("summary"),
    }


def main() -> None:
    result = asyncio.run(run_checklist())
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if result.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
