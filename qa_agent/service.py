from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from qa_agent.agent import ExplorationAgent
from qa_agent.browser import BrowserSession
from qa_agent.config import RunConfig
from qa_agent.llm import ActionChooser
from qa_agent.cursor_sampling_llm import CursorSamplingLLM

logger = logging.getLogger("qa_agent.service")


@dataclass
class RunResult:
    run_id: str
    report_path: Path
    metrics_path: Path
    events_path: Path
    summary: str
    steps_taken: int
    issues_count: int
    metrics: object


ProgressCallback = Callable[[int, int, dict[str, Any] | None], None]


def create_action_chooser(config: RunConfig) -> ActionChooser:
    if config.llm_backend != "cursor_sampling":
        raise ValueError(
            "QA agent only supports cursor_sampling LLM backend (start runs via MCP)"
        )
    if config.sampling_session is None:
        raise ValueError("cursor_sampling backend requires an MCP session")
    return CursorSamplingLLM(
        config.sampling_session,
        related_request_id=config.sampling_request_id,
        timeout=config.llm_timeout,
    )


async def execute_run(
    config: RunConfig,
    *,
    run_id: str,
    on_progress: ProgressCallback | None = None,
) -> RunResult:
    browser = BrowserSession(
        config.base_url,
        headless=config.headless,
        viewport_mode=config.viewport_mode,
        console_errors=config.console_errors,
        network_errors=config.network_errors,
    )
    llm = create_action_chooser(config)
    agent = ExplorationAgent(config, browser, llm, run_id=run_id, on_progress=on_progress)

    await browser.start()
    try:
        report_path = await agent.run()
    finally:
        await browser.close()

    metrics_path = report_path.parent / "metrics.json"
    events_path = report_path.parent / "events.jsonl"
    return RunResult(
        run_id=run_id,
        report_path=report_path,
        metrics_path=metrics_path,
        events_path=events_path,
        summary=agent.summary,
        steps_taken=len(agent.trace),
        issues_count=len(agent.issues),
        metrics=agent.metrics,
    )
