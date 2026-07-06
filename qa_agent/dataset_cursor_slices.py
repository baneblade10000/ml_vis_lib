"""Dataset-focused QA slices for ml-vis playground (cursor-driven)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DatasetCursorSliceId = Literal[0, 1, 2]


@dataclass(frozen=True)
class DatasetCursorSlice:
    id: DatasetCursorSliceId
    name: str
    label: str
    profile: Literal["training", "full"]
    goal: str
    datasets: tuple[str, ...]
    max_steps: int = 35


DATASET_CURSOR_SLICES: tuple[DatasetCursorSlice, ...] = (
    DatasetCursorSlice(
        id=0,
        name="datasets_easy",
        label="Datasets · circles & xor",
        profile="training",
        datasets=("circles", "xor"),
        goal=(
            "Dataset QA: select circles and xor, reset/train for each, and verify the "
            "decision boundary canvas updates without console errors."
        ),
    ),
    DatasetCursorSlice(
        id=1,
        name="datasets_hard",
        label="Datasets · spiral & gaussian",
        profile="training",
        datasets=("spiral", "gaussian"),
        goal=(
            "Dataset QA: select spiral and gaussian, tweak noise slider, reset/train, "
            "and confirm validation accuracy moves and the canvas renders distinct boundaries."
        ),
    ),
    DatasetCursorSlice(
        id=2,
        name="activations",
        label="Activations · tanh / relu / sigmoid",
        profile="training",
        datasets=("circles",),
        goal=(
            "Activation QA: on circles dataset, switch tanh, relu, and sigmoid; "
            "reset/train briefly for each and compare boundary smoothness in screenshots."
        ),
    ),
)


def list_dataset_cursor_slices() -> list[dict[str, str | int | list[str]]]:
    return [
        {
            "id": slice_cfg.id,
            "name": slice_cfg.name,
            "label": slice_cfg.label,
            "profile": slice_cfg.profile,
            "datasets": list(slice_cfg.datasets),
            "max_steps": slice_cfg.max_steps,
        }
        for slice_cfg in DATASET_CURSOR_SLICES
    ]


def get_dataset_cursor_slice(slice_id: int) -> DatasetCursorSlice:
    for slice_cfg in DATASET_CURSOR_SLICES:
        if slice_cfg.id == slice_id:
            return slice_cfg
    raise ValueError(f"Unknown dataset cursor slice id: {slice_id}")
