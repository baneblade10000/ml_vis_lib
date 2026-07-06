from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qa_agent.models import AgentTraceStep, Issue, UxObservation
from qa_agent.ux_metrics import UxMetrics


def write_report(
    *,
    run_dir: Path,
    run_id: str,
    goal: str,
    summary: str,
    issues: list[Issue],
    ux_observations: list[UxObservation],
    trace: list[AgentTraceStep],
    console_errors: list[str],
    network_errors: list[str],
    token_usage: dict[str, int],
    metrics: UxMetrics,
) -> Path:
    report_path = run_dir / "report.md"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# QA Exploration Report",
        "",
        f"- Generated: {generated_at}",
        f"- Run ID: `{run_id}`",
        f"- Run directory: `{run_dir}`",
        f"- Steps taken: {len(trace)}",
        f"- Issues found: {len(issues)}",
        f"- UX observations: {len(ux_observations)}",
        f"- Overall UX score: **{metrics.overall_score}/100**",
        "",
        "## Goal",
        "",
        goal,
        "",
        "## Summary",
        "",
        summary,
        "",
        "## UX metrics",
        "",
        f"- Overall: **{metrics.overall_score}/100**",
        f"- Stability: {metrics.stability_score}/100",
        f"- Navigation efficiency: {metrics.navigation_efficiency}/100",
        f"- Error health: {metrics.error_health}/100",
        f"- Issue burden: {metrics.issue_burden}/100",
    ]
    if metrics.ux_observation_score is not None:
        lines.append(f"- Agent UX ratings: {metrics.ux_observation_score}/100")
    lines.extend(
        [
            f"- Key route coverage: {metrics.key_route_coverage}%",
            f"- Profile: `{metrics.exploration_profile}`",
            f"- Unique routes visited: {metrics.unique_routes}",
            f"- Redundant navigations: {metrics.redundant_navigations}",
            "",
        ]
    )
    if metrics.visited_areas or metrics.missing_areas:
        lines.extend(["### Route coverage by area", ""])
        for area in metrics.visited_areas:
            lines.append(f"- ✓ {area}")
        for area in metrics.missing_areas:
            lines.append(f"- ✗ {area}")
        lines.append("")
    lines.extend(
        [
            "### Recommendations",
            "",
        ]
    )
    for recommendation in metrics.recommendations:
        lines.append(f"- {recommendation}")
    lines.append("")

    if metrics.observations_by_aspect:
        lines.extend(["### UX ratings by aspect", ""])
        for aspect, average in metrics.observations_by_aspect.items():
            lines.append(f"- {aspect}: {average}/5")
        lines.append("")

    lines.extend(
        [
            "## Token usage",
            "",
            f"- Prompt tokens: {token_usage.get('prompt_tokens', 0)}",
            f"- Completion tokens: {token_usage.get('completion_tokens', 0)}",
            "",
        ]
    )

    if issues:
        lines.extend(["## Findings", ""])
        for index, issue in enumerate(issues, start=1):
            lines.extend(
                [
                    f"### {index}. [{issue.severity.upper()}] {issue.title}",
                    "",
                    f"- URL: {issue.url or '(unknown)'}",
                    f"- Step: {issue.step}",
                    "",
                    issue.description,
                    "",
                ]
            )
            if issue.repro_steps:
                lines.append("Reproduction steps:")
                for step_index, step in enumerate(issue.repro_steps, start=1):
                    lines.append(f"{step_index}. {step}")
                lines.append("")
            if issue.screenshot_path:
                lines.append(
                    f"![Issue {index}](screenshots/{issue.screenshot_path})"
                )
                lines.append("")
    else:
        lines.extend(["## Findings", "", "No issues were reported by the agent.", ""])

    if ux_observations:
        lines.extend(["## UX observations", ""])
        for index, observation in enumerate(ux_observations, start=1):
            lines.extend(
                [
                    f"### {index}. {observation.area} ({observation.aspect})",
                    "",
                    f"- Score: {observation.score}/5",
                    f"- URL: {observation.url or '(unknown)'}",
                    f"- Step: {observation.step}",
                    "",
                    observation.notes or "(no notes)",
                    "",
                ]
            )

    if console_errors:
        lines.extend(["## Console errors", ""])
        for entry in console_errors:
            lines.append(f"- {entry}")
        lines.append("")

    if network_errors:
        lines.extend(["## Network errors", ""])
        for entry in network_errors:
            lines.append(f"- {entry}")
        lines.append("")

    lines.extend(["## Action trace", ""])
    for step in trace:
        lines.extend(
            [
                f"### Step {step.step}",
                "",
                f"- URL: {step.url}",
                f"- Tool: `{step.tool}`",
                f"- Args: `{step.args}`",
                f"- Result: `{step.result}`",
                f"- Duration: {step.duration_ms} ms",
                f"- Screenshot: `screenshots/step-{step.step:02d}.png`",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_metrics_json(path: Path, metrics: object) -> None:
    payload = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
