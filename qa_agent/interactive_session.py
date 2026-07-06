from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from qa_agent.agent import ExplorationAgent, MaxStepsReached, observation_payload
from qa_agent.browser import BrowserSession
from qa_agent.config import build_interactive_run_config
from qa_agent.e2e_routes import resolve_e2e_routes
from qa_agent.tools import TOOL_DEFINITIONS
from qa_agent.vision import analyze_screenshot_heuristic

logger = logging.getLogger("qa_agent.interactive_session")

_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


class SessionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class InteractiveSession:
    run_id: str
    agent: ExplorationAgent
    browser: BrowserSession
    status: SessionStatus = SessionStatus.RUNNING
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str = ""


def _tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": item["function"]["name"],
            "description": item["function"].get("description", ""),
            "parameters": item["function"]["parameters"],
        }
        for item in TOOL_DEFINITIONS
    ]


def _session_response(session: InteractiveSession, payload: dict[str, Any]) -> dict[str, Any]:
    agent = session.agent
    return {
        "run_id": session.run_id,
        "status": session.status.value,
        "goal": agent.config.goal,
        "exploration_profile": agent.config.exploration_profile,
        "viewport_mode": agent.config.viewport_mode,
        "system_prompt": agent.system_prompt,
        "tools": _tool_catalog(),
        "vision_backend": "cursor_chat",
        "next_action": (
            "Read screenshot_path with the Read tool, analyze pixels, "
            "then call qa_cursor_act with one tool."
        ),
        **payload,
    }


def _heuristic_digest(session: InteractiveSession, observation: Any) -> dict[str, Any]:
    dom_digest = session.agent._format_observation(observation)
    digest = analyze_screenshot_heuristic(
        base64.b64decode(observation.screenshot_b64),
        dom_digest=dom_digest,
    )
    return digest.to_dict()


class InteractiveSessionManager:
    """Stepwise QA — Playwright captures screenshots; Cursor chat agent is the vision model."""

    def __init__(self) -> None:
        self._sessions: dict[str, InteractiveSession] = {}

    def get(self, run_id: str) -> InteractiveSession | None:
        return self._sessions.get(run_id)

    async def start(
        self,
        *,
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
        profile: str | None = None,
        min_coverage_pct: float = 0.0,
    ) -> dict[str, Any]:
        _ = (api_base_url, test_email, test_password, login, seed)
        config = build_interactive_run_config(
            base_url=base_url,
            goal=goal,
            max_steps=max_steps,
            headless=headless,
            viewport_mode="mobile" if mobile else "desktop",
            exploration_profile=profile,
            min_coverage_pct=min_coverage_pct,
        )
        run_id = _new_run_id()
        if not config.output_dir:
            config.output_dir = _RUNS_ROOT / run_id

        browser = BrowserSession(
            config.base_url,
            headless=config.headless,
            viewport_mode=config.viewport_mode,
            console_errors=config.console_errors,
            network_errors=config.network_errors,
        )
        agent = ExplorationAgent(config, browser, llm=None, run_id=run_id)
        session = InteractiveSession(run_id=run_id, agent=agent, browser=browser)
        self._sessions[run_id] = session

        try:
            config.e2e_routes = await resolve_e2e_routes()
            await browser.start()
            await agent.initialize()
            observation = await agent.observe_step()
            heuristic_digest = _heuristic_digest(session, observation)
        except Exception as exc:
            session.status = SessionStatus.FAILED
            session.error = str(exc)
            await browser.close()
            self._sessions.pop(run_id, None)
            raise

        return _session_response(
            session,
            observation_payload(
                agent,
                observation,
                step=agent.current_step,
                heuristic_digest=heuristic_digest,
            ),
        )

    async def act(
        self,
        run_id: str,
        tool: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(run_id)
        agent = session.agent
        tool_args = args or {}

        try:
            result = await agent.act_step(tool, tool_args)
        except Exception as exc:
            session.status = SessionStatus.FAILED
            session.error = str(exc)
            await session.browser.close()
            raise

        if agent.finished:
            report_path = await agent.finalize()
            session.status = SessionStatus.COMPLETED
            await session.browser.close()
            metrics = agent.metrics.to_dict() if agent.metrics else {}
            return _session_response(
                session,
                {
                    "finished": True,
                    "tool": tool,
                    "tool_args": tool_args,
                    "tool_result": result,
                    "summary": agent.summary,
                    "report_path": str(report_path),
                    "overall_ux_score": metrics.get("overall_score"),
                    "issues_count": len(agent.issues),
                    "steps_taken": len(agent.trace),
                },
            )

        try:
            observation = await agent.observe_step()
            heuristic_digest = _heuristic_digest(session, observation)
        except MaxStepsReached:
            agent.summary = (
                agent.summary
                or f"Stopped at max_steps ({agent.config.max_steps}) without finish."
            )
            report_path = await agent.finalize()
            session.status = SessionStatus.COMPLETED
            await session.browser.close()
            metrics = agent.metrics.to_dict() if agent.metrics else {}
            return _session_response(
                session,
                {
                    "finished": True,
                    "tool": tool,
                    "tool_args": tool_args,
                    "tool_result": result,
                    "summary": agent.summary,
                    "report_path": str(report_path),
                    "overall_ux_score": metrics.get("overall_score"),
                    "issues_count": len(agent.issues),
                    "steps_taken": len(agent.trace),
                    "reason": "max_steps_reached",
                },
            )

        payload = observation_payload(
            agent,
            observation,
            step=agent.current_step,
            heuristic_digest=heuristic_digest,
        )
        payload.update(
            {
                "finished": False,
                "tool": tool,
                "tool_args": tool_args,
                "tool_result": result,
            }
        )
        return _session_response(session, payload)

    async def abort(self, run_id: str) -> dict[str, Any]:
        session = self._sessions.pop(run_id, None)
        if not session:
            return {"run_id": run_id, "aborted": False, "error": "Unknown session"}
        session.status = SessionStatus.ABORTED
        await session.browser.close()
        return {"run_id": run_id, "aborted": True, "status": session.status.value}

    def _require_session(self, run_id: str) -> InteractiveSession:
        session = self._sessions.get(run_id)
        if not session or session.status != SessionStatus.RUNNING:
            raise ValueError(f"No active cursor session for run_id={run_id}")
        return session


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


_manager: InteractiveSessionManager | None = None


def get_interactive_session_manager() -> InteractiveSessionManager:
    global _manager
    if _manager is None:
        _manager = InteractiveSessionManager()
    return _manager
