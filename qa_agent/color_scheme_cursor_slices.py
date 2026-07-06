"""Five focused color-scheme QA slices for cursor-driven visual design audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qa_agent.e2e_routes import E2eRoutes

ColorSchemeCursorSliceId = Literal[0, 1, 2, 3, 4]

COLOR_SCHEME_AUDIT_RULES = (
    "Visual design / color scheme audit on desktop viewport (~1280px). "
    "For EACH assigned color scheme, inspect pages in BOTH light AND dark theme using set_appearance. "
    "DO report: hardcoded colors bypassing tokens, illegible text/background contrast, muddy or clashing "
    "palette mixes, components using raw Tailwind grays/blues instead of semantic tokens, broken status "
    "badge colors, invisible or overly loud hover/focus/selection states, Learn vs Studio looking like "
    "different apps under the same scheme, neon/ethereal/sepia palettes looking broken, "
    "cheap, or unintentionally dirty. "
    "DO NOT report: sub-44px touch targets, DesktopOnlyGate, "
    "or subjective taste without clear visual evidence in the screenshot."
)


@dataclass(frozen=True)
class ColorSchemeCursorSlice:
    id: ColorSchemeCursorSliceId
    name: str
    label: str
    profile: Literal["learn", "studio", "full"]
    schemes: tuple[str, ...]
    goal: str
    max_steps: int = 45


def _learn_modules_list_path(routes: E2eRoutes) -> str:
    module_id = routes.learn_module_path.rstrip("/").split("/")[-1]
    if module_id.startswith("6a43b709"):
        return "/learn/modules?course=6a43b709c9232acc2cdbf5e2"
    return "/learn/modules"


COLOR_SCHEME_CURSOR_SLICES: tuple[ColorSchemeCursorSlice, ...] = (
    ColorSchemeCursorSlice(
        id=0,
        name="learn_reader_palettes",
        label="Learn · reader (study, classic)",
        profile="learn",
        schemes=("study", "classic"),
        goal=(
            f"{COLOR_SCHEME_AUDIT_RULES} "
            "Schemes: study, classic. "
            "Routes: Learn modules list, module reader scenes, inline quiz. "
            "Check reader chrome, quiz options, progress badges, bottom/footer bars, callouts."
        ),
    ),
    ColorSchemeCursorSlice(
        id=1,
        name="learn_shell_palettes",
        label="Learn · shell (sepia, ethereal)",
        profile="learn",
        schemes=("sepia", "ethereal"),
        goal=(
            f"{COLOR_SCHEME_AUDIT_RULES} "
            "Schemes: sepia, ethereal. "
            "Routes: /learn/progress, /learn/podcasts, /learn/assistant, /learn/settings. "
            "Check charts, stats cards, assistant bubbles, settings palette picker itself."
        ),
    ),
    ColorSchemeCursorSlice(
        id=2,
        name="studio_content_palettes",
        label="Studio · content (neon, ethereal)",
        profile="studio",
        schemes=("neon", "ethereal"),
        goal=(
            f"{COLOR_SCHEME_AUDIT_RULES} "
            "Schemes: neon, ethereal. "
            "Routes: /studio/materials list, material detail, /studio/modules list. "
            "Check list toolbars, status badges, card actions, borders on dark-vivid palettes."
        ),
    ),
    ColorSchemeCursorSlice(
        id=3,
        name="studio_tools_palettes",
        label="Studio · tools (study, sepia)",
        profile="studio",
        schemes=("study", "sepia"),
        goal=(
            f"{COLOR_SCHEME_AUDIT_RULES} "
            "Schemes: study, sepia. "
            "Routes: pipelines list, pipeline editor, /studio/assistant, /studio/settings. "
            "Check warm palettes in dense Studio UI: forms, tabs, graph nodes, side panels."
        ),
    ),
    ColorSchemeCursorSlice(
        id=4,
        name="cross_app_palettes",
        label="Cross-app (classic, neon)",
        profile="full",
        schemes=("classic", "neon"),
        goal=(
            f"{COLOR_SCHEME_AUDIT_RULES} "
            "Schemes: classic, neon. "
            "Routes: /login, Learn modules list, Studio materials list — same scheme across shells. "
            "Verify palette persists and looks coherent when switching Learn ↔ Studio."
        ),
    ),
)


def get_color_scheme_cursor_slice(slice_id: int) -> ColorSchemeCursorSlice:
    for item in COLOR_SCHEME_CURSOR_SLICES:
        if item.id == slice_id:
            return item
    raise ValueError(f"Unknown color scheme cursor slice id: {slice_id} (expected 0–4)")


def resolve_slice_paths(routes: E2eRoutes, slice_id: int) -> list[str]:
    modules_list = _learn_modules_list_path(routes)
    by_id: dict[int, list[str]] = {
        0: [
            modules_list,
            routes.learn_module_path,
            routes.learn_module_quiz_path,
        ],
        1: [
            "/learn/progress",
            "/learn/podcasts",
            "/learn/assistant",
            "/learn/settings",
        ],
        2: [
            "/studio/materials",
            routes.studio_material_detail_path,
            routes.studio_modules_path,
        ],
        3: [
            routes.studio_pipelines_list_path,
            routes.studio_pipeline_default_path,
            "/studio/assistant",
            "/studio/settings",
        ],
        4: [
            "/login",
            modules_list,
            "/studio/materials",
        ],
    }
    if slice_id not in by_id:
        raise ValueError(f"Unknown color scheme cursor slice id: {slice_id}")
    return by_id[slice_id]


def list_color_scheme_cursor_slices() -> list[dict[str, str | int | list[str]]]:
    return [
        {
            "id": item.id,
            "name": item.name,
            "label": item.label,
            "profile": item.profile,
            "schemes": list(item.schemes),
            "max_steps": item.max_steps,
            "goal_preview": item.goal[:180] + "…",
        }
        for item in COLOR_SCHEME_CURSOR_SLICES
    ]
