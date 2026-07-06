from __future__ import annotations

import asyncio

import pytest

from qa_agent.config import RunConfig, build_interactive_run_config
from qa_agent.interactive_session import InteractiveSessionManager


def test_build_interactive_run_config_uses_cursor_chat() -> None:
    config = build_interactive_run_config(exploration_profile="full")
    assert config.llm_backend == "cursor_chat"
    assert config.sampling_session is None


def test_interactive_act_unknown_session() -> None:
    manager = InteractiveSessionManager()

    async def _run() -> None:
        with pytest.raises(ValueError, match="No active cursor session"):
            await manager.act("nonexistent-run-id", "click", {"ref": "e1"})

    asyncio.run(_run())


def test_agent_run_requires_llm() -> None:
    from qa_agent.agent import ExplorationAgent
    from qa_agent.browser import BrowserSession

    config = RunConfig(
        base_url="http://localhost:5173",
        goal="test",
    )
    browser = BrowserSession(config.base_url)
    agent = ExplorationAgent(config, browser, llm=None)

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="Autonomous run requires an LLM"):
            await agent.run()

    asyncio.run(_run())
