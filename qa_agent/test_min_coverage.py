from __future__ import annotations

from qa_agent.agent import _live_coverage_pct
from qa_agent.config import RunConfig
from qa_agent.models import AgentTraceStep


class _StubBrowser:
    console_errors: list[str] = []
    network_errors: list[str] = []


def test_finish_coverage_gate_blocks_early_finish() -> None:
    run_config = RunConfig(
        base_url="http://localhost:5173",
        goal="test",
        exploration_profile="i18n",
        min_coverage_pct=80.0,
    )

    class _Agent:
        run_id = "test"
        config = run_config
        trace = [
            AgentTraceStep(
                step=1,
                tool="navigate",
                args={"url": "/"},
                result={"url": "http://localhost:5173/"},
                url="http://localhost:5173/",
            )
        ]
        issues = []
        ux_observations = []
        browser = _StubBrowser()

    coverage, missing = _live_coverage_pct(_Agent())  # type: ignore[arg-type]
    assert coverage < 80.0
    assert missing
