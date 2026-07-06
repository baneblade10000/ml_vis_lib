from __future__ import annotations

from qa_agent.e2e_routes import E2eRoutes
from qa_agent.mobile_cursor_slices import (
    MOBILE_CURSOR_SLICES,
    get_mobile_cursor_slice,
    list_mobile_cursor_slices,
    resolve_slice_paths,
)


def test_mobile_cursor_slices_count() -> None:
    assert len(MOBILE_CURSOR_SLICES) == 3
    assert len(list_mobile_cursor_slices()) == 3


def test_resolve_slice_paths_playground() -> None:
    paths = resolve_slice_paths(E2eRoutes(), 0)
    assert paths == ["/"]


def test_get_mobile_cursor_slice() -> None:
    slice_cfg = get_mobile_cursor_slice(1)
    assert slice_cfg.name == "mobile_controls"
    assert slice_cfg.profile == "mobile_full"
