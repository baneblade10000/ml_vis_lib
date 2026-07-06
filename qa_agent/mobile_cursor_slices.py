"""Focused mobile QA slices for ml-vis playground (cursor-driven)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MobileCursorSliceId = Literal[0, 1, 2]

ANTI_NOISE_RULES = (
    "Mobile UX audit at 390px viewport for ml-vis playground. "
    "DO NOT report cosmetic spacing differences that still leave controls usable. "
    "DO report: canvas clipped or missing, controls overlapping, horizontal overflow, "
    "sliders unreachable, and locale switcher hidden off-screen."
)


@dataclass(frozen=True)
class MobileCursorSlice:
    id: MobileCursorSliceId
    name: str
    label: str
    profile: Literal["mobile_full", "i18n", "training"]
    goal: str
    max_steps: int = 30


MOBILE_CURSOR_SLICES: tuple[MobileCursorSlice, ...] = (
    MobileCursorSlice(
        id=0,
        name="mobile_shell",
        label="Mobile · header & layout",
        profile="mobile_full",
        goal=(
            f"{ANTI_NOISE_RULES} "
            "Focus: app header, locale switcher, theme section title, and overall layout at /. "
            "Scroll the page and verify the canvas remains visible above the fold or after one scroll."
        ),
    ),
    MobileCursorSlice(
        id=1,
        name="mobile_controls",
        label="Mobile · control rows",
        profile="mobile_full",
        goal=(
            f"{ANTI_NOISE_RULES} "
            "Focus: dataset/activation selects, hidden layers input, noise/lr/epochs sliders, "
            "and Play / Step / Reset buttons. Check wrapping and tap targets."
        ),
    ),
    MobileCursorSlice(
        id=2,
        name="mobile_training",
        label="Mobile · training & replay",
        profile="training",
        goal=(
            f"{ANTI_NOISE_RULES} "
            "Focus: run Play or Step +1, confirm epoch metric updates, scrub replay frames, "
            "and verify the decision boundary canvas redraws."
        ),
    ),
)


def list_mobile_cursor_slices() -> list[dict[str, str | int]]:
    return [
        {
            "id": slice_cfg.id,
            "name": slice_cfg.name,
            "label": slice_cfg.label,
            "profile": slice_cfg.profile,
            "max_steps": slice_cfg.max_steps,
        }
        for slice_cfg in MOBILE_CURSOR_SLICES
    ]


def get_mobile_cursor_slice(slice_id: int) -> MobileCursorSlice:
    for slice_cfg in MOBILE_CURSOR_SLICES:
        if slice_cfg.id == slice_id:
            return slice_cfg
    raise ValueError(f"Unknown mobile cursor slice id: {slice_id}")


def resolve_slice_paths(_routes: object, slice_id: int) -> list[str]:
    _ = slice_id
    return ["/"]
