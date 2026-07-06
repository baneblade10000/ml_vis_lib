from __future__ import annotations

from qa_agent.dataset_cursor_slices import (
    DATASET_CURSOR_SLICES,
    get_dataset_cursor_slice,
    list_dataset_cursor_slices,
)


def test_dataset_cursor_slices_count() -> None:
    assert len(DATASET_CURSOR_SLICES) == 3
    assert len(list_dataset_cursor_slices()) == 3


def test_get_dataset_cursor_slice() -> None:
    slice_cfg = get_dataset_cursor_slice(1)
    assert slice_cfg.name == "datasets_hard"
    assert "spiral" in slice_cfg.datasets
