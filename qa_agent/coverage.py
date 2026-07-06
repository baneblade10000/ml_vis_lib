from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:
    from qa_agent.models import AgentTraceStep, UxObservation

ExplorationProfile = Literal["full", "playground", "training", "i18n", "mobile_full"]

DATASETS = frozenset({"circles", "xor", "spiral", "gaussian"})
ACTIVATIONS = frozenset({"tanh", "relu", "sigmoid"})
LOCALES = frozenset({"en", "ru"})


@dataclass(frozen=True)
class RouteArea:
    key: str
    label: str
    prefixes: tuple[str, ...]


ROUTE_AREAS: tuple[RouteArea, ...] = (
    RouteArea("playground_load", "Playground · initial load", ("/",)),
    RouteArea("locale", "Playground · locale switcher (en/ru)", ("/",)),
    RouteArea("dataset_controls", "Playground · dataset picker", ("/",)),
    RouteArea("activation_controls", "Playground · activation picker", ("/",)),
    RouteArea("hyperparams", "Playground · hyperparameters", ("/",)),
    RouteArea("training_controls", "Playground · Play / Step / Reset", ("/",)),
    RouteArea("replay_scrubber", "Playground · replay frame scrubber", ("/",)),
    RouteArea("canvas_interaction", "Playground · decision boundary canvas", ("/",)),
)

PROFILE_AREAS: dict[ExplorationProfile, tuple[str, ...]] = {
    "full": tuple(area.key for area in ROUTE_AREAS),
    "playground": tuple(area.key for area in ROUTE_AREAS),
    "mobile_full": tuple(area.key for area in ROUTE_AREAS),
    "training": (
        "playground_load",
        "dataset_controls",
        "activation_controls",
        "hyperparams",
        "training_controls",
        "replay_scrubber",
        "canvas_interaction",
    ),
    "i18n": ("playground_load", "locale"),
}

PROFILE_GOALS: dict[ExplorationProfile, str] = {
    "full": (
        "Full playground QA for ml-vis: exercise the Decision Boundary playground at /. "
        "Switch locale (en/ru), try each dataset (circles, xor, spiral, gaussian), "
        "change activation (tanh, relu, sigmoid), adjust hidden layers / noise / "
        "learning rate / epochs, run Play and Step +1 training, use Reset & train, "
        "scrub replay frames, and verify the canvas updates without console errors. "
        "Report bugs and rate key screens with report_ux_observation."
    ),
    "playground": (
        "Full playground QA for ml-vis: exercise the Decision Boundary playground at /. "
        "Switch locale (en/ru), try each dataset, change activation, adjust hyperparameters, "
        "run training controls, scrub replay frames, and verify the canvas updates."
    ),
    "mobile_full": (
        "Mobile playground QA (390px): same Decision Boundary controls on a narrow viewport. "
        "Check control rows wrap cleanly, sliders remain usable, canvas stays visible, "
        "and locale switcher is reachable. Focus on tap targets, overflow, and layout."
    ),
    "training": (
        "Training-flow QA: pick at least two datasets and two activations, tweak "
        "hyperparameters, run Play until epochs advance, Step +1, Reset & train after "
        "config changes, and scrub replay frames. Verify metrics (epoch, val acc) update "
        "and the boundary visualization animates."
    ),
    "i18n": (
        "Locale QA: switch between English and Russian via the header locale select. "
        "Verify app title/description and control labels update; report missing or "
        "mixed-language strings."
    ),
}

PROFILE_LABELS: dict[ExplorationProfile, str] = {
    "full": "Full playground",
    "playground": "Full playground",
    "training": "Training & replay flow",
    "i18n": "Locale / i18n",
    "mobile_full": "Mobile playground",
}


def normalize_path(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return path


def match_route_area(path: str, area: RouteArea) -> bool:
    if area.key == "playground_load":
        return path == "/"
    return False


def classify_path(path: str) -> set[str]:
    if normalize_path(path) == "/":
        return {"playground_load"}
    return set()


def _click_label(step: AgentTraceStep) -> str:
    for key in ("name", "label", "text"):
        value = step.result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def classify_trace_step(step: AgentTraceStep) -> set[str]:
    matched: set[str] = set()
    route = normalize_path(str(step.result.get("url") or step.url or ""))
    if route == "/":
        matched.add("playground_load")

    args = step.args or {}
    tool = step.tool

    if tool == "select_option":
        value = str(args.get("value", "")).lower()
        if value in LOCALES:
            matched.add("locale")
        elif value in DATASETS:
            matched.add("dataset_controls")
        elif value in ACTIVATIONS:
            matched.add("activation_controls")

    if tool == "fill":
        matched.add("hyperparams")

    if tool == "press_key" and str(args.get("key", "")).lower() in {"space", " "}:
        matched.add("training_controls")

    if tool == "click":
        label = _click_label(step)
        if any(token in label for token in ("play", "pause", "step", "reset")):
            matched.add("training_controls")
        if "replay" in label or "frame" in label:
            matched.add("replay_scrubber")

    if tool in {"scroll", "wait_for"} and step.result.get("canvas_epoch") is not None:
        matched.add("canvas_interaction")

    return matched


def classify_ux_observation(observation: UxObservation) -> set[str]:
    area = observation.area.lower()
    matched: set[str] = set()
    if any(token in area for token in ("locale", "i18n", "language", "язык")):
        matched.add("locale")
    if "dataset" in area:
        matched.add("dataset_controls")
    if "activation" in area:
        matched.add("activation_controls")
    if any(token in area for token in ("hyper", "slider", "noise", "learning", "epoch")):
        matched.add("hyperparams")
    if any(token in area for token in ("train", "play", "step", "reset")):
        matched.add("training_controls")
    if "replay" in area or "scrub" in area:
        matched.add("replay_scrubber")
    if any(token in area for token in ("canvas", "chart", "boundary", "plot")):
        matched.add("canvas_interaction")
    if "playground" in area or "header" in area:
        matched.add("playground_load")
    return matched


def resolve_profile(
    profile: str | None,
    *,
    viewport_mode: str = "desktop",
) -> ExplorationProfile:
    if profile in PROFILE_AREAS:
        return profile  # type: ignore[return-value]
    if viewport_mode == "mobile":
        return "mobile_full"
    return "full"


def goal_for_profile(profile: ExplorationProfile) -> str:
    return PROFILE_GOALS[profile]


def required_areas_for_profile(profile: ExplorationProfile) -> tuple[str, ...]:
    return PROFILE_AREAS[profile]


def profile_checklist(profile: ExplorationProfile) -> str:
    lines = ["Coverage checklist (exercise as many interactions as possible):"]
    area_by_key = {area.key: area for area in ROUTE_AREAS}
    hints = {
        "locale": "select en and ru in the header locale dropdown",
        "dataset_controls": "select circles, xor, spiral, gaussian",
        "activation_controls": "select tanh, relu, sigmoid",
        "hyperparams": "adjust hidden layers text, noise / lr / epochs sliders",
        "training_controls": "Play, Pause (Space), Step +1, Reset & train",
        "replay_scrubber": "move replay frame slider after training",
        "canvas_interaction": "observe boundary updates; rate canvas clarity",
        "playground_load": "load / and verify header + chart render",
    }
    for key in PROFILE_AREAS[profile]:
        area = area_by_key[key]
        hint = hints.get(key, "")
        suffix = f" — {hint}" if hint else ""
        lines.append(f"- {area.label}{suffix}")
    return "\n".join(lines)


def list_profiles() -> list[dict[str, str]]:
    return [
        {
            "profile": name,
            "label": PROFILE_LABELS[name],
            "goal_preview": PROFILE_GOALS[name][:160] + "…",
        }
        for name in PROFILE_AREAS
    ]
