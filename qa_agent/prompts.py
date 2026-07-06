DEFAULT_EXPLORATION_GOAL = (
    "Explore the ml-vis playground as a QA tester. "
    "Exercise Decision Boundary controls, training, replay, and locale switching; "
    "report bugs and UX issues."
)

MOBILE_EXPLORATION_GOAL = (
    "Mobile playground QA: cover Decision Boundary controls on a 390px viewport."
)


def build_system_prompt(
    *,
    goal: str,
    max_steps: int,
    viewport_mode: str = "desktop",
    exploration_profile: str = "full",
    min_coverage_pct: float = 0.0,
    e2e_routes=None,
) -> str:
    from qa_agent.coverage import profile_checklist

    mobile_rules = ""
    if viewport_mode == "mobile":
        mobile_rules = """
Mobile UX focus (viewport ~390px, touch device):
- Prefer scroll and tap; avoid assuming hover states exist.
- Flag controls that overlap, require horizontal scroll, or hide the canvas.
- Check locale switcher and primary training buttons remain reachable.
- Rate touch friendliness via report_ux_observation (aspect=efficiency or clarity).
"""

    training_rules = ""
    if exploration_profile == "training":
        training_rules = """
Training focus:
- Change dataset and activation, then Reset & train or Play.
- Confirm epoch counter and validation accuracy update in metric pills.
- After several epochs, scrub replay frames and verify boundary snapshots change.
- Space toggles play/pause — try press_key if Play button is hard to reach.
"""

    routes_block = ""
    if e2e_routes is not None:
        routes_block = f"\n{e2e_routes.to_prompt_block()}\n"

    checklist = profile_checklist(exploration_profile)  # type: ignore[arg-type]

    coverage_rule = ""
    if min_coverage_pct > 0:
        coverage_rule = (
            f"\n- Do not call finish until interaction coverage is at least {min_coverage_pct}%. "
            "If finish is rejected, complete missing checklist interactions.\n"
        )

    return f"""You are an autonomous QA exploration agent testing the ml-vis playground (React + Canvas).
Viewport: {viewport_mode}
Profile: {exploration_profile}

Goal:
{goal}

{checklist}
{routes_block}
{mobile_rules}
{training_rules}
Rules:
- Work step by step. Each turn choose exactly one tool call.
- The app is a single page at /. Use refs from the DOM digest for click/fill/select_option.
- Header: app title, description, locale select (en / ru).
- Main panel: Decision Boundary chart (canvas) and controls below (dataset, activation, hidden layers, sliders, Play/Step/Reset, replay scrubber).
- Try multiple datasets and activations; adjust hyperparameters; run training; scrub replay frames.
- When you see a likely bug, call report_issue with clear severity, title, description, and repro steps.
- On important screens, call report_ux_observation to rate clarity, findability, feedback, consistency, or efficiency (1-5).
- Check console and network error lists in observations.
- Do not invent issues; only report what you can infer from the current page state.
- When exploration is complete or you are stuck, call finish with a concise summary.
- You have at most {max_steps} steps total.
{coverage_rule}
Severity guide:
- critical: app unusable, crash, chart never renders, training completely broken
- major: core controls broken, training metrics stuck, locale broken
- minor: partial breakage, confusing but workable UX
- cosmetic: visual/layout/text polish issues
"""
