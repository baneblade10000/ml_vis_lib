from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from qa_agent.coverage import (
    PROFILE_GOALS,
    PROFILE_LABELS,
    profile_checklist,
    list_profiles,
)
from qa_agent.cursor_sampling_llm import client_supports_sampling
from qa_agent.dataset_cursor_slices import (
    get_dataset_cursor_slice,
    list_dataset_cursor_slices,
)
from qa_agent.mobile_cursor_slices import get_mobile_cursor_slice, list_mobile_cursor_slices
from qa_agent.interactive_session import get_interactive_session_manager
from qa_agent.run_compare import compare_runs
from qa_agent.run_manager import get_run_manager
from qa_agent.run_store import load_events, load_metrics, load_report, resolve_run_dir

logger = logging.getLogger("qa_agent.mcp")

mcp = FastMCP(
    "ml-vis-qa-agent",
    instructions=(
        "QA browser exploration for the ml-vis playground (Decision Boundary demo). "
        "Use Cursor chat-driven mode: qa_cursor_start then loop qa_cursor_act. "
        "You are the vision model — after every MCP response, Read screenshot_path before "
        "choosing the next action. Trust pixels over dom_digest. "
        "Poll qa_get_run_status for past runs; read report/metrics when done."
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _require_sampling(ctx: Context) -> tuple[Any, str]:
    """Only for autonomous runs when Cursor adds MCP sampling support."""
    session = ctx.session
    if not client_supports_sampling(session):
        raise ValueError(
            "Autonomous qa_start_exploration requires MCP sampling (not yet supported in Cursor). "
            "Use qa_cursor_start + qa_cursor_act instead."
        )
    return session, ctx.request_id


@mcp.tool()
async def qa_cursor_start(
    goal: str | None = None,
    max_steps: int = 45,
    headless: bool = True,
    login: bool = False,
    seed: bool = False,
    base_url: str | None = None,
    api_base_url: str | None = None,
    test_email: str | None = None,
    test_password: str | None = None,
    mobile: bool = False,
    profile: str | None = "full",
    min_coverage_pct: float = 0.0,
) -> str:
    """Start stepwise QA. Read screenshot_path with Read tool before each qa_cursor_act."""
    manager = get_interactive_session_manager()
    payload = await manager.start(
        goal=goal,
        max_steps=max_steps,
        headless=headless,
        login=login,
        seed=seed,
        base_url=base_url,
        api_base_url=api_base_url,
        test_email=test_email,
        test_password=test_password,
        mobile=mobile,
        profile=profile,
        min_coverage_pct=min_coverage_pct,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_cursor_act(
    run_id: str,
    tool: str,
    args: dict[str, Any] | None = None,
) -> str:
    """Execute one QA action; returns next observation with screenshot_path."""
    manager = get_interactive_session_manager()
    payload = await manager.act(run_id, tool, args)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_cursor_abort(run_id: str) -> str:
    """Abort a cursor-driven session and close the browser."""
    payload = await get_interactive_session_manager().abort(run_id)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_start_exploration(
    goal: str | None = None,
    max_steps: int = 45,
    headless: bool = True,
    login: bool = False,
    seed: bool = False,
    base_url: str | None = None,
    api_base_url: str | None = None,
    test_email: str | None = None,
    test_password: str | None = None,
    mobile: bool = True,
    profile: str | None = None,
    min_coverage_pct: float = 0.0,
    elicit_config: bool = False,
    ctx: Context | None = None,
) -> str:
    """Start background autonomous QA (disabled in Cursor — use qa_cursor_start)."""
    _ = (
        goal,
        max_steps,
        headless,
        login,
        seed,
        base_url,
        api_base_url,
        test_email,
        test_password,
        mobile,
        profile,
        min_coverage_pct,
        elicit_config,
        ctx,
    )
    return json.dumps(
        {
            "error": "Autonomous background runs need MCP sampling (Cursor does not support it yet).",
            "use_instead": "qa_cursor_start",
            "hint": (
                "Loop: qa_cursor_start → Read screenshot_path → qa_cursor_act. "
                "You (Cursor chat) are the vision model."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
async def qa_list_profiles() -> str:
    """List QA exploration profiles (full, training, i18n, mobile_full) and their coverage scope."""
    return json.dumps({"profiles": list_profiles()}, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_list_mobile_cursor_slices() -> str:
    """List three focused mobile QA slices for parallel cursor-driven audits."""
    return json.dumps({"slices": list_mobile_cursor_slices()}, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_cursor_start_mobile_slice(
    slice_id: int,
    max_steps: int = 35,
    headless: bool = True,
    login: bool = False,
    seed: bool = False,
    base_url: str | None = None,
    api_base_url: str | None = None,
    min_coverage_pct: float = 0.0,
) -> str:
    """Start a cursor-driven mobile session for one slice (0–2)."""
    slice_cfg = get_mobile_cursor_slice(slice_id)
    manager = get_interactive_session_manager()
    payload = await manager.start(
        goal=slice_cfg.goal,
        max_steps=max_steps,
        headless=headless,
        login=login,
        seed=seed,
        base_url=base_url,
        api_base_url=api_base_url,
        mobile=True,
        profile=slice_cfg.profile,
        min_coverage_pct=min_coverage_pct,
    )
    payload["mobile_cursor_slice"] = {
        "id": slice_cfg.id,
        "name": slice_cfg.name,
        "label": slice_cfg.label,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_list_dataset_slices() -> str:
    """List dataset/activation audit slices for parallel cursor-driven QA."""
    return json.dumps({"slices": list_dataset_cursor_slices()}, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_cursor_start_dataset_slice(
    slice_id: int,
    max_steps: int = 35,
    headless: bool = True,
    login: bool = False,
    seed: bool = False,
    base_url: str | None = None,
    api_base_url: str | None = None,
    min_coverage_pct: float = 0.0,
) -> str:
    """Start a cursor-driven session for one dataset/activation audit slice (0–2)."""
    _ = (api_base_url, login, seed)
    slice_cfg = get_dataset_cursor_slice(slice_id)
    manager = get_interactive_session_manager()
    payload = await manager.start(
        goal=slice_cfg.goal,
        max_steps=max_steps,
        headless=headless,
        base_url=base_url,
        mobile=False,
        profile=slice_cfg.profile,
        min_coverage_pct=min_coverage_pct,
    )
    payload["dataset_cursor_slice"] = {
        "id": slice_cfg.id,
        "name": slice_cfg.name,
        "label": slice_cfg.label,
        "datasets": list(slice_cfg.datasets),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_get_run_status(run_id: str) -> str:
    """Get status for a QA exploration run (pending/running/completed/failed)."""
    status = get_run_manager().get_status(run_id)
    if not status:
        return json.dumps({"error": f"Unknown run_id: {run_id}"}, ensure_ascii=False)
    return json.dumps(status, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_list_runs(limit: int = 20, include_disk: bool = True) -> str:
    """List recent QA runs (in-memory session + disk history when include_disk=true)."""
    runs = get_run_manager().list_runs(limit=limit, include_disk=include_disk)
    return json.dumps({"runs": runs}, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_get_report(run_id: str) -> str:
    """Return markdown report for a completed QA run."""
    manager = get_run_manager()
    report = manager.get_report(run_id)
    if report is None:
        status = manager.get_status(run_id)
        if not status:
            return json.dumps({"error": f"Unknown run_id: {run_id}"}, ensure_ascii=False)
        return json.dumps(
            {
                "error": "Report not ready",
                "status": status.get("status"),
                "current_step": status.get("current_step"),
            },
            ensure_ascii=False,
            indent=2,
        )
    return report


@mcp.tool()
async def qa_get_metrics(run_id: str) -> str:
    """Return UX metrics JSON for a completed QA run."""
    manager = get_run_manager()
    metrics = manager.get_metrics(run_id)
    if metrics is None:
        status = manager.get_status(run_id)
        if not status:
            return json.dumps({"error": f"Unknown run_id: {run_id}"}, ensure_ascii=False)
        return json.dumps(
            {
                "error": "Metrics not ready",
                "status": status.get("status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    return json.dumps(metrics, ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_compare_runs(run_id_a: str, run_id_b: str) -> str:
    """Compare UX scores, coverage, and issues between two runs (A=baseline, B=newer)."""
    return json.dumps(compare_runs(run_id_a, run_id_b), ensure_ascii=False, indent=2)


@mcp.tool()
async def qa_get_issue_screenshot(run_id: str, issue_number: int = 1) -> str:
    """Return base64 PNG for an issue screenshot (issue-01.png, issue-02.png, …)."""
    path = get_run_manager().get_issue_screenshot_path(run_id, issue_number)
    if path is None:
        return json.dumps(
            {"error": f"Issue screenshot not found for run {run_id} #{issue_number}"},
            ensure_ascii=False,
        )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return json.dumps(
        {
            "run_id": run_id,
            "issue_number": issue_number,
            "filename": path.name,
            "mime_type": "image/png",
            "data_base64": encoded,
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def qa_cancel_run(run_id: str) -> str:
    """Cancel a running QA exploration."""
    cancelled = await get_run_manager().cancel_run(run_id)
    return json.dumps({"run_id": run_id, "cancelled": cancelled}, ensure_ascii=False)


@mcp.resource("qa://runs/{run_id}/events")
def qa_run_events(run_id: str) -> str:
    """JSONL event log for a run."""
    run_dir = resolve_run_dir(run_id)
    if not run_dir:
        return json.dumps({"error": f"Unknown run_id: {run_id}"})
    content = load_events(run_dir)
    if content is None:
        return json.dumps({"error": "events.jsonl not found", "run_id": run_id})
    return content


@mcp.resource("qa://runs/{run_id}/report")
def qa_run_report_resource(run_id: str) -> str:
    """Markdown report for a run."""
    run_dir = resolve_run_dir(run_id)
    if not run_dir:
        return json.dumps({"error": f"Unknown run_id: {run_id}"})
    content = load_report(run_dir)
    if content is None:
        return json.dumps({"error": "report.md not ready", "run_id": run_id})
    return content


@mcp.resource("qa://runs/{run_id}/metrics")
def qa_run_metrics_resource(run_id: str) -> str:
    """UX metrics JSON for a run."""
    run_dir = resolve_run_dir(run_id)
    if not run_dir:
        return json.dumps({"error": f"Unknown run_id: {run_id}"})
    metrics = load_metrics(run_dir)
    if metrics is None:
        return json.dumps({"error": "metrics.json not ready", "run_id": run_id})
    return json.dumps(metrics, ensure_ascii=False, indent=2)


@mcp.resource("qa://runs/{run_id}/screenshots/{filename}")
def qa_run_screenshot_resource(run_id: str, filename: str) -> bytes:
    """PNG screenshot from a run (e.g. step-01.png, issue-01.png)."""
    run_dir = resolve_run_dir(run_id)
    if not run_dir:
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    path = run_dir / "screenshots" / filename
    if not path.is_file() or ".." in filename:
        raise FileNotFoundError(f"Screenshot not found: {filename}")
    return path.read_bytes()


def _register_profile_prompts() -> None:
    for profile_name in PROFILE_GOALS:

        def _make(name: str = profile_name) -> None:
            @mcp.prompt(name=f"qa-profile-{name}")
            def profile_prompt() -> str:
                label = PROFILE_LABELS[name]  # type: ignore[index]
                return (
                    f"# QA profile: {label}\n\n"
                    f"{PROFILE_GOALS[name]}\n\n"  # type: ignore[index]
                    f"{profile_checklist(name)}\n"  # type: ignore[arg-type]
                )

        _make()


_register_profile_prompts()


@mcp.prompt(name="qa-mobile-cursor-parallel")
def qa_mobile_cursor_parallel_prompt() -> str:
    return """# Mobile QA — 3 parallel cursor slices (ml-vis playground)

Use focused sessions instead of one long mobile_full run.

## Slices (call `qa_list_mobile_cursor_slices`)

| id | scope |
|----|--------|
| 0 | Header, locale switcher, canvas visibility |
| 1 | Control rows: selects, sliders, training buttons |
| 2 | Training flow: Play/Step, epoch metrics, replay scrubber |

## Per slice

1. `qa_cursor_start_mobile_slice(slice_id=N, max_steps=30)`
2. Read `screenshot_path` after each step — you are the vision model.
3. Loop `qa_cursor_act` — exercise controls at `/`, scroll as needed.
"""


@mcp.prompt(name="qa-dataset-parallel")
def qa_dataset_parallel_prompt() -> str:
    return """# Dataset / activation audit — 3 parallel cursor slices

Exercise Decision Boundary datasets and activations with Cursor vision on every screenshot.

## Slices (call `qa_list_dataset_slices`)

| id | focus |
|----|--------|
| 0 | circles, xor |
| 1 | spiral, gaussian (+ noise slider) |
| 2 | tanh, relu, sigmoid on circles |

## Per slice

1. `qa_cursor_start_dataset_slice(slice_id=N, max_steps=35)`
2. Read `screenshot_path` each step; trust pixels over DOM.
3. `report_issue` for broken training or blank canvas; `report_ux_observation` on chart clarity.
"""


@mcp.prompt(name="qa-cursor-driver")
def qa_cursor_driver_prompt() -> str:
    return """# Cursor-driven QA exploration (ml-vis playground)

You are the QA agent AND the vision model. MCP controls the browser; you Read screenshots.

## Loop

1. `qa_cursor_start(profile="full", mobile=false, max_steps=45, min_coverage_pct=50)`
2. **Read `screenshot_path`** from the response (Read tool — mandatory every step).
3. Analyze pixels; check `heuristic_digest` for blank-canvas warnings.
4. Pick **one** tool from `tools` and call `qa_cursor_act(run_id, tool, args)`.
5. Repeat until `finished=true`.
6. Read `qa_get_report(run_id)`.

## Rules

- NEVER call `qa_cursor_act` without reading the current step screenshot first.
- Trust screenshot pixels over `dom_digest` when they disagree.
- Single-page app — use element `ref` values from `dom_digest`.
- Call `report_issue` for bugs, `report_ux_observation` on key screens.
- Do not call `finish` until `coverage_pct` >= `min_coverage_pct` (if set).
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Starting QA agent MCP server (repo=%s)", _repo_root())
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
