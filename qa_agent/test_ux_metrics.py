from __future__ import annotations

from qa_agent.models import AgentTraceStep, Issue, UxObservation
from qa_agent.ux_metrics import compute_ux_metrics


def test_compute_ux_metrics_scores_clean_run() -> None:
    trace = [
        AgentTraceStep(
            step=1,
            tool="navigate",
            args={"url": "/"},
            result={"url": "http://localhost:5173/"},
            url="http://localhost:5173/",
            duration_ms=120,
        ),
        AgentTraceStep(
            step=2,
            tool="select_option",
            args={"ref": "e2", "value": "xor"},
            result={"url": "http://localhost:5173/"},
            url="http://localhost:5173/",
            duration_ms=80,
        ),
    ]
    metrics = compute_ux_metrics(
        run_id="test-run",
        trace=trace,
        issues=[],
        ux_observations=[
            UxObservation(area="canvas", aspect="clarity", score=4, url="/", step=1),
        ],
        console_errors=[],
        network_errors=[],
        exploration_profile="training",
    )

    assert metrics.overall_score >= 70
    assert metrics.ux_observation_score == 75.0
    assert metrics.steps_failed == 0
    assert metrics.observations_by_aspect["clarity"] == 4.0
    assert metrics.area_coverage["dataset_controls"] is True


def test_compute_ux_metrics_penalizes_errors() -> None:
    trace = [
        AgentTraceStep(
            step=1,
            tool="click",
            args={"ref": "e1"},
            result={"success": False, "error": "timeout"},
            url="http://localhost:5173/",
            duration_ms=10000,
        ),
    ]
    issues = [
        Issue(
            severity="major",
            title="Play button broken",
            description="Training does not start",
            url="/",
            step=1,
        )
    ]
    metrics = compute_ux_metrics(
        run_id="test-run",
        trace=trace,
        issues=issues,
        ux_observations=[],
        console_errors=["TypeError: x is undefined"],
        network_errors=[],
        exploration_profile="training",
    )

    assert metrics.overall_score < 90
    assert metrics.stability_score < 85
    assert metrics.steps_failed == 1
    assert metrics.issues_by_severity["major"] == 1
    assert any("console" in item.lower() for item in metrics.recommendations)


def test_play_click_counts_toward_training_coverage() -> None:
    trace = [
        AgentTraceStep(
            step=1,
            tool="click",
            args={"ref": "e5"},
            result={"clicked": "e5", "name": "Play", "url": "http://localhost:5173/"},
            url="http://localhost:5173/",
            duration_ms=100,
        ),
    ]
    metrics = compute_ux_metrics(
        run_id="test-run",
        trace=trace,
        issues=[],
        ux_observations=[],
        console_errors=[],
        network_errors=[],
        exploration_profile="training",
    )
    assert metrics.area_coverage["training_controls"] is True


def test_full_playground_interaction_coverage() -> None:
    base = "http://localhost:5173/"
    trace = [
        AgentTraceStep(1, "navigate", {"url": "/"}, {"url": base}, base, 100),
        AgentTraceStep(
            2,
            "select_option",
            {"ref": "e1", "value": "ru"},
            {"url": base},
            base,
            80,
        ),
        AgentTraceStep(
            3,
            "select_option",
            {"ref": "e2", "value": "xor"},
            {"url": base},
            base,
            80,
        ),
        AgentTraceStep(
            4,
            "select_option",
            {"ref": "e3", "value": "relu"},
            {"url": base},
            base,
            80,
        ),
        AgentTraceStep(5, "fill", {"ref": "e4", "text": "8,8"}, {"url": base}, base, 80),
        AgentTraceStep(
            6,
            "click",
            {"ref": "e5"},
            {"clicked": "e5", "name": "Play", "url": base},
            base,
            80,
        ),
        AgentTraceStep(
            7,
            "click",
            {"ref": "e6"},
            {"clicked": "e6", "name": "Step +1", "url": base},
            base,
            80,
        ),
    ]
    metrics = compute_ux_metrics(
        run_id="playground-full",
        trace=trace,
        issues=[],
        ux_observations=[
            UxObservation(area="decision boundary canvas", aspect="clarity", score=5, url="/", step=6),
        ],
        console_errors=[],
        network_errors=[],
        exploration_profile="full",
    )

    assert metrics.key_route_coverage >= 60
    assert metrics.area_coverage["locale"] is True
    assert metrics.area_coverage["dataset_controls"] is True
    assert metrics.area_coverage["training_controls"] is True
    assert metrics.area_coverage["canvas_interaction"] is True
