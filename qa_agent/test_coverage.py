from __future__ import annotations

from qa_agent.coverage import (
    classify_path,
    classify_trace_step,
    classify_ux_observation,
    match_route_area,
    resolve_profile,
)
from qa_agent.coverage import ROUTE_AREAS
from qa_agent.models import AgentTraceStep, UxObservation


def _area(key: str):
    return next(area for area in ROUTE_AREAS if area.key == key)


def test_classify_playground_root() -> None:
    assert "playground_load" in classify_path("/")


def test_classify_trace_dataset_select() -> None:
    step = AgentTraceStep(
        step=1,
        tool="select_option",
        args={"ref": "e2", "value": "xor"},
        result={"url": "http://localhost:5173/"},
        url="http://localhost:5173/",
    )
    assert "dataset_controls" in classify_trace_step(step)


def test_classify_trace_locale_select() -> None:
    step = AgentTraceStep(
        step=1,
        tool="select_option",
        args={"ref": "e1", "value": "ru"},
        result={"url": "http://localhost:5173/"},
        url="http://localhost:5173/",
    )
    assert "locale" in classify_trace_step(step)


def test_classify_trace_play_click() -> None:
    step = AgentTraceStep(
        step=1,
        tool="click",
        args={"ref": "e8"},
        result={"clicked": "e8", "name": "Play", "url": "http://localhost:5173/"},
        url="http://localhost:5173/",
    )
    assert "training_controls" in classify_trace_step(step)


def test_classify_ux_observation_canvas() -> None:
    obs = UxObservation(area="decision boundary canvas", aspect="clarity", score=4)
    assert "canvas_interaction" in classify_ux_observation(obs)


def test_resolve_profile_defaults() -> None:
    assert resolve_profile(None, viewport_mode="desktop") == "full"
    assert resolve_profile(None, viewport_mode="mobile") == "mobile_full"
    assert resolve_profile("training", viewport_mode="mobile") == "training"
    assert resolve_profile("i18n", viewport_mode="desktop") == "i18n"


def test_playground_load_matches_root_only() -> None:
    area = _area("playground_load")
    assert match_route_area("/", area)
    assert not match_route_area("/storybook", area)
