from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from qa_agent.coverage import (
    ExplorationProfile,
    ROUTE_AREAS,
    classify_path,
    classify_trace_step,
    classify_ux_observation,
    required_areas_for_profile,
    resolve_profile,
)
from qa_agent.models import AgentTraceStep, Issue, UxObservation

SEVERITY_WEIGHTS = {
    "critical": 30,
    "major": 15,
    "minor": 5,
    "cosmetic": 2,
}

UX_ASPECTS = ("clarity", "findability", "feedback", "consistency", "efficiency")


@dataclass
class UxMetrics:
    run_id: str
    overall_score: float
    stability_score: float
    navigation_efficiency: float
    error_health: float
    issue_burden: float
    ux_observation_score: float | None
    steps_total: int
    steps_failed: int
    unique_routes: int
    redundant_navigations: int
    back_navigations: int
    key_route_coverage: float
    console_error_count: int
    network_error_count: int
    exploration_profile: str = "full"
    area_coverage: dict[str, bool] = field(default_factory=dict)
    visited_areas: list[str] = field(default_factory=list)
    missing_areas: list[str] = field(default_factory=list)
    issues_by_severity: dict[str, int] = field(default_factory=dict)
    observations_by_aspect: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _route_path(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return path


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


# High unique-route coverage implies hub revisits during area sweeps, not aimless loops.
_HIGH_COVERAGE_RATIO = 0.7
_REVISIT_PENALTY_CAP = 8.0


def _count_consecutive_duplicate_navigates(navigate_urls: list[str]) -> int:
    return sum(
        1
        for i in range(1, len(navigate_urls))
        if navigate_urls[i] and navigate_urls[i] == navigate_urls[i - 1]
    )


def _navigation_efficiency_penalty(
    *,
    steps_total: int,
    unique_routes: int,
    navigate_urls: list[str],
    steps_failed: int,
    back_navigations: int,
) -> tuple[float, int]:
    """Return (efficiency_penalty, redundant_navigations for reporting)."""
    route_revisits = max(0, len(navigate_urls) - len(set(navigate_urls)))
    consecutive_redundant = _count_consecutive_duplicate_navigates(navigate_urls)
    coverage_ratio = unique_routes / steps_total if steps_total else 0.0

    if coverage_ratio >= _HIGH_COVERAGE_RATIO:
        non_consecutive_revisits = max(0, route_revisits - consecutive_redundant)
        redundant_penalty = consecutive_redundant * 6 + min(
            non_consecutive_revisits * 2,
            _REVISIT_PENALTY_CAP,
        )
        redundant_navigations = consecutive_redundant
    else:
        redundant_penalty = route_revisits * 6
        redundant_navigations = route_revisits

    efficiency_penalty = (
        redundant_penalty
        + back_navigations * 4
        + steps_failed * 5
        + max(0, steps_total - unique_routes * 3) * 2
    )
    return efficiency_penalty, redundant_navigations


def compute_ux_metrics(
    *,
    run_id: str,
    trace: list[AgentTraceStep],
    issues: list[Issue],
    ux_observations: list[UxObservation],
    console_errors: list[str],
    network_errors: list[str],
    exploration_profile: ExplorationProfile | str | None = None,
    viewport_mode: str = "desktop",
) -> UxMetrics:
    profile = resolve_profile(exploration_profile, viewport_mode=viewport_mode)
    required_areas = required_areas_for_profile(profile)
    area_labels = {area.key: area.label for area in ROUTE_AREAS}
    steps_total = len(trace)
    steps_failed = sum(1 for step in trace if step.result.get("success") is False)

    routes: list[str] = []
    visited_area_keys: set[str] = set()
    for step in trace:
        route = _route_path(str(step.result.get("url") or step.url or ""))
        if route:
            routes.append(route)
            visited_area_keys.update(classify_path(route))
        visited_area_keys.update(classify_trace_step(step))

    for observation in ux_observations:
        visited_area_keys.update(classify_ux_observation(observation))

    unique_routes = len(set(routes))

    navigate_urls = [
        _route_path(str(step.result.get("url") or step.url or ""))
        for step in trace
        if step.tool == "navigate"
    ]

    back_navigations = sum(1 for step in trace if step.tool == "go_back")

    area_coverage = {key: key in visited_area_keys for key in required_areas}
    visited_areas = [area_labels[key] for key in required_areas if key in visited_area_keys]
    missing_areas = [area_labels[key] for key in required_areas if key not in visited_area_keys]
    covered_count = sum(1 for key in required_areas if key in visited_area_keys)
    key_route_coverage = _clamp(
        100.0 * covered_count / len(required_areas) if required_areas else 0.0
    )

    issues_by_severity: dict[str, int] = {}
    issue_penalty = 0.0
    for issue in issues:
        issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1
        issue_penalty += SEVERITY_WEIGHTS.get(issue.severity, 5)

    issue_burden = _clamp(100.0 - issue_penalty)

    console_error_count = len(console_errors)
    network_error_count = len(network_errors)
    technical_penalty = console_error_count * 4 + network_error_count * 8
    error_health = _clamp(100.0 - technical_penalty)

    stability_penalty = (
        steps_failed * 8
        + console_error_count * 3
        + network_error_count * 6
        + issues_by_severity.get("critical", 0) * 25
        + issues_by_severity.get("major", 0) * 12
    )
    stability_score = _clamp(100.0 - stability_penalty)

    if steps_total == 0:
        navigation_efficiency = 0.0
        redundant_navigations = 0
    else:
        efficiency_penalty, redundant_navigations = _navigation_efficiency_penalty(
            steps_total=steps_total,
            unique_routes=unique_routes,
            navigate_urls=navigate_urls,
            steps_failed=steps_failed,
            back_navigations=back_navigations,
        )
        navigation_efficiency = _clamp(100.0 - efficiency_penalty)

    observations_by_aspect: dict[str, list[int]] = {aspect: [] for aspect in UX_ASPECTS}
    for observation in ux_observations:
        observations_by_aspect.setdefault(observation.aspect, []).append(observation.score)

    aspect_averages = {
        aspect: round(sum(scores) / len(scores), 2)
        for aspect, scores in observations_by_aspect.items()
        if scores
    }
    if aspect_averages:
        avg_rating = sum(aspect_averages.values()) / len(aspect_averages)
        ux_observation_score = round(_clamp((avg_rating - 1) / 4 * 100), 1)
    else:
        ux_observation_score = None

    if ux_observation_score is None:
        overall_score = round(
            stability_score * 0.30
            + navigation_efficiency * 0.25
            + error_health * 0.25
            + issue_burden * 0.20,
            1,
        )
    else:
        overall_score = round(
            stability_score * 0.25
            + navigation_efficiency * 0.20
            + error_health * 0.20
            + issue_burden * 0.20
            + ux_observation_score * 0.15,
            1,
        )

    recommendations = _build_recommendations(
        steps_failed=steps_failed,
        console_error_count=console_error_count,
        network_error_count=network_error_count,
        redundant_navigations=redundant_navigations,
        back_navigations=back_navigations,
        key_route_coverage=key_route_coverage,
        missing_areas=missing_areas,
        issues_by_severity=issues_by_severity,
        aspect_averages=aspect_averages,
    )

    return UxMetrics(
        run_id=run_id,
        overall_score=overall_score,
        stability_score=round(stability_score, 1),
        navigation_efficiency=round(navigation_efficiency, 1),
        error_health=round(error_health, 1),
        issue_burden=round(issue_burden, 1),
        ux_observation_score=ux_observation_score,
        steps_total=steps_total,
        steps_failed=steps_failed,
        unique_routes=unique_routes,
        redundant_navigations=redundant_navigations,
        back_navigations=back_navigations,
        key_route_coverage=round(key_route_coverage, 1),
        exploration_profile=profile,
        area_coverage=area_coverage,
        visited_areas=visited_areas,
        missing_areas=missing_areas,
        console_error_count=console_error_count,
        network_error_count=network_error_count,
        issues_by_severity=issues_by_severity,
        observations_by_aspect=aspect_averages,
        recommendations=recommendations,
    )


def _build_recommendations(
    *,
    steps_failed: int,
    console_error_count: int,
    network_error_count: int,
    redundant_navigations: int,
    back_navigations: int,
    key_route_coverage: float,
    missing_areas: list[str],
    issues_by_severity: dict[str, int],
    aspect_averages: dict[str, float],
) -> list[str]:
    recommendations: list[str] = []

    if issues_by_severity.get("critical"):
        recommendations.append("Fix critical issues before release; they block core flows.")
    if network_error_count:
        recommendations.append("Investigate failing API requests surfaced during exploration.")
    if console_error_count:
        recommendations.append("Resolve frontend console errors to improve stability perception.")
    if steps_failed:
        recommendations.append("Review broken interactions where agent actions failed.")
    if redundant_navigations > 2:
        recommendations.append("Reduce navigation loops; improve wayfinding and primary CTAs.")
    if back_navigations > 1:
        recommendations.append("Users needed back navigation often; check dead ends and unclear paths.")
    if key_route_coverage < 60:
        if missing_areas:
            preview = ", ".join(missing_areas[:4])
            suffix = "…" if len(missing_areas) > 4 else ""
            recommendations.append(
                f"Increase route coverage; areas not visited: {preview}{suffix}."
            )
        else:
            recommendations.append("Increase coverage of key product areas in the exploration profile.")

    for aspect, average in aspect_averages.items():
        if average <= 2.5:
            recommendations.append(f"Improve {aspect}: agent rated it {average}/5.")

    if not recommendations:
        recommendations.append("No major UX regressions detected in this run.")

    return recommendations
