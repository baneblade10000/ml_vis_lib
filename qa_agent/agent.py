from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from qa_agent.browser import BrowserSession, Observation, format_tool_result
from qa_agent.config import RunConfig
from qa_agent.events import RunEventLogger
from qa_agent.llm import ActionChooser
from qa_agent.models import AgentTraceStep, Issue, UxObservation
from qa_agent.prompts import build_system_prompt
from qa_agent.report import write_report
from qa_agent.report import write_metrics_json
from qa_agent.tools import BROWSER_TOOLS
from qa_agent.ux_metrics import UxMetrics, compute_ux_metrics

logger = logging.getLogger("qa_agent.agent")

ProgressCallback = Callable[..., None | Awaitable[None]]


class MaxStepsReached(Exception):
    """Raised when observe_step exceeds config.max_steps."""


def observation_payload(
    agent: "ExplorationAgent",
    observation: Observation,
    *,
    step: int,
    heuristic_digest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage_pct, missing_areas = _live_coverage_pct(agent)
    screenshot_name = f"step-{step:02d}.png"
    screenshot_path = agent.run_dir / "screenshots" / screenshot_name
    payload: dict[str, Any] = {
        "step": step,
        "max_steps": agent.config.max_steps,
        "url": observation.url,
        "title": observation.title,
        "dom_digest": agent._format_observation(observation),
        "console_errors": observation.console_errors,
        "network_errors": observation.network_errors,
        "screenshot_resource": f"qa://runs/{agent.run_id}/screenshots/{screenshot_name}",
        "screenshot_path": str(screenshot_path),
        "coverage_pct": coverage_pct,
        "missing_areas": missing_areas,
        "issues_count": len(agent.issues),
        "ux_observations_count": len(agent.ux_observations),
    }
    if heuristic_digest is not None:
        payload["heuristic_digest"] = heuristic_digest
    return payload


def _live_coverage_pct(agent: "ExplorationAgent") -> tuple[float, list[str]]:
    metrics = compute_ux_metrics(
        run_id=agent.run_id,
        trace=agent.trace,
        issues=agent.issues,
        ux_observations=agent.ux_observations,
        console_errors=agent.browser.console_errors,
        network_errors=agent.browser.network_errors,
        exploration_profile=agent.config.exploration_profile,
        viewport_mode=agent.config.viewport_mode,
    )
    return metrics.key_route_coverage, metrics.missing_areas


class ExplorationAgent:
    def __init__(
        self,
        config: RunConfig,
        browser: BrowserSession,
        llm: ActionChooser | None = None,
        *,
        run_id: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.browser = browser
        self.llm = llm
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.on_progress = on_progress
        self.issues: list[Issue] = []
        self.ux_observations: list[UxObservation] = []
        self.trace: list[AgentTraceStep] = []
        self.messages: list[dict[str, Any]] = []
        self.summary = ""
        self.finished = False
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.metrics: UxMetrics | None = None
        self._events: RunEventLogger | None = None
        self._run_dir: Path | None = None
        self._system_prompt = ""
        self._current_step = 0
        self._last_observation: Observation | None = None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def run_dir(self) -> Path:
        if self._run_dir is None:
            raise RuntimeError("Agent not initialized; call initialize() first")
        return self._run_dir

    async def initialize(self) -> None:
        self._run_dir = self._prepare_run_dir()
        self._events = RunEventLogger(self._run_dir, self.run_id)
        self._system_prompt = build_system_prompt(
            goal=self.config.goal,
            max_steps=self.config.max_steps,
            viewport_mode=self.config.viewport_mode,
            exploration_profile=self.config.exploration_profile,
            min_coverage_pct=self.config.min_coverage_pct,
            e2e_routes=self.config.e2e_routes,
        )
        self._events.emit(
            "run_start",
            data={
                "goal": self.config.goal,
                "max_steps": self.config.max_steps,
                "base_url": self.config.base_url,
                "viewport_mode": self.config.viewport_mode,
                "exploration_profile": self.config.exploration_profile,
                "mode": self.config.llm_backend,
            },
        )
        await self.browser.navigate("/")

    async def observe_step(self) -> Observation:
        if self._run_dir is None or self._events is None:
            raise RuntimeError("Agent not initialized; call initialize() first")
        if self.finished:
            raise RuntimeError("Run already finished")

        self._current_step += 1
        if self._current_step > self.config.max_steps:
            raise MaxStepsReached(
                f"Reached max_steps ({self.config.max_steps}); call finish or abort"
            )

        step = self._current_step
        if self.on_progress:
            result = self.on_progress(
                step,
                len(self.issues),
                {"url": None, "tool": None, "phase": "observe"},
            )
            if asyncio.iscoroutine(result):
                await result

        observation = await self.browser.observe()
        await self._save_step_screenshot(self._run_dir, step, observation)
        self._last_observation = observation
        self._events.emit("step_observe", step=step, url=observation.url)
        return observation

    async def act_step(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        observation: Observation | None = None,
    ) -> dict[str, Any]:
        if self._run_dir is None or self._events is None:
            raise RuntimeError("Agent not initialized; call initialize() first")

        obs = observation or self._last_observation
        if obs is None:
            raise RuntimeError("No observation for this step; call observe_step() first")

        step = self._current_step
        step_started = time.perf_counter()
        self._events.emit(
            "llm_choice",
            step=step,
            tool=tool_name,
            url=obs.url,
            data={"args": tool_args, "usage": {"prompt_tokens": 0, "completion_tokens": 0}},
        )

        result = await self._execute_tool(tool_name, tool_args, obs, step, self._run_dir)
        duration_ms = int((time.perf_counter() - step_started) * 1000)
        self.trace.append(
            AgentTraceStep(
                step=step,
                tool=tool_name,
                args=tool_args,
                result=result,
                url=obs.url,
                duration_ms=duration_ms,
            )
        )
        self._events.emit(
            "step_complete",
            step=step,
            tool=tool_name,
            url=obs.url,
            duration_ms=duration_ms,
            data={"result": result},
        )
        if self.on_progress:
            progress_result = self.on_progress(
                step,
                len(self.issues),
                {
                    "url": obs.url,
                    "tool": tool_name,
                    "phase": "complete",
                    "duration_ms": duration_ms,
                },
            )
            if asyncio.iscoroutine(progress_result):
                await progress_result

        self.messages.append(
            {"role": "assistant", "content": f"Called {tool_name}({tool_args})"}
        )
        self.messages.append(
            {"role": "user", "content": f"Tool result:\n{format_tool_result(result)}"}
        )
        return result

    async def finalize(self) -> Path:
        if self._run_dir is None or self._events is None:
            raise RuntimeError("Agent not initialized; call initialize() first")

        if not self.summary:
            self.summary = "Exploration stopped after reaching the step limit."

        self.metrics = compute_ux_metrics(
            run_id=self.run_id,
            trace=self.trace,
            issues=self.issues,
            ux_observations=self.ux_observations,
            console_errors=self.browser.console_errors,
            network_errors=self.browser.network_errors,
            exploration_profile=self.config.exploration_profile,
            viewport_mode=self.config.viewport_mode,
        )
        write_metrics_json(self._run_dir / "metrics.json", self.metrics)

        report_path = write_report(
            run_dir=self._run_dir,
            run_id=self.run_id,
            goal=self.config.goal,
            summary=self.summary,
            issues=self.issues,
            ux_observations=self.ux_observations,
            trace=self.trace,
            console_errors=self.browser.console_errors,
            network_errors=self.browser.network_errors,
            token_usage={
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
            },
            metrics=self.metrics,
        )
        self._events.emit(
            "run_complete",
            data={
                "summary": self.summary,
                "issues_count": len(self.issues),
                "steps_taken": len(self.trace),
                "overall_ux_score": self.metrics.overall_score,
                "report_path": str(report_path),
            },
        )
        logger.info(
            "Run complete run_id=%s report=%s ux_score=%s",
            self.run_id,
            report_path,
            self.metrics.overall_score,
            extra={"run_id": self.run_id, "event": "run_complete"},
        )
        return report_path

    async def run(self) -> Path:
        if self.llm is None:
            raise RuntimeError("Autonomous run requires an LLM; use observe_step/act_step for cursor_chat")

        await self.initialize()
        for step in range(1, self.config.max_steps + 1):
            observation = await self.observe_step()
            if self.finished:
                break

            try:
                tool_name, tool_args, usage = await self.llm.choose_action(
                    system_prompt=self._system_prompt,
                    messages=self.messages,
                    screenshot_b64=observation.screenshot_b64,
                    dom_digest=self._format_observation(observation),
                    step=step,
                    max_steps=self.config.max_steps,
                )
            except Exception as exc:
                error = str(exc)
                logger.error(
                    "LLM choose_action failed at step=%s: %s",
                    step,
                    error,
                    extra={
                        "run_id": self.run_id,
                        "step": step,
                        "event": "llm_error",
                    },
                )
                self._events.emit(
                    "llm_error",
                    step=step,
                    url=observation.url,
                    data={"error": error, "error_type": type(exc).__name__},
                )
                self._events.emit(
                    "run_failed",
                    step=step,
                    data={"error": error, "failed_at_step": step},
                )
                raise
            self.total_prompt_tokens += usage["prompt_tokens"]
            self.total_completion_tokens += usage["completion_tokens"]

            await self.act_step(tool_name, tool_args, observation=observation)
            logger.info(
                "step=%s tool=%s url=%s",
                step,
                tool_name,
                observation.url,
                extra={
                    "run_id": self.run_id,
                    "step": step,
                    "tool": tool_name,
                    "event": "step_complete",
                },
            )

            if self.finished:
                break

        return await self.finalize()

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        observation: Observation,
        step: int,
        run_dir: Path,
    ) -> dict[str, Any]:
        if tool_name == "report_ux_observation":
            observation_record = UxObservation(
                area=tool_args.get("area", "unknown"),
                aspect=tool_args.get("aspect", "clarity"),
                score=int(tool_args.get("score", 3)),
                notes=tool_args.get("notes", ""),
                url=observation.url,
                step=step,
            )
            self.ux_observations.append(observation_record)
            self._events.emit(
                "ux_observation",
                step=step,
                url=observation.url,
                data={
                    "area": observation_record.area,
                    "aspect": observation_record.aspect,
                    "score": observation_record.score,
                },
            )
            return {
                "recorded": True,
                "ux_observation_count": len(self.ux_observations),
            }

        if tool_name == "report_issue":
            screenshot_path = await self._save_issue_screenshot(run_dir, len(self.issues) + 1)
            issue = Issue(
                severity=tool_args.get("severity", "minor"),
                title=tool_args.get("title", "Untitled issue"),
                description=tool_args.get("description", ""),
                repro_steps=list(tool_args.get("repro_steps") or []),
                screenshot_path=screenshot_path.name,
                url=observation.url,
                step=step,
            )
            self.issues.append(issue)
            self._events.emit(
                "issue_reported",
                step=step,
                url=observation.url,
                data={
                    "severity": issue.severity,
                    "title": issue.title,
                },
            )
            return {"recorded": True, "issue_count": len(self.issues)}

        if tool_name == "finish":
            min_coverage = self.config.min_coverage_pct
            if min_coverage > 0:
                coverage_pct, missing_areas = _live_coverage_pct(self)
                if coverage_pct < min_coverage:
                    preview = ", ".join(missing_areas[:4])
                    suffix = "…" if len(missing_areas) > 4 else ""
                    return {
                        "finished": False,
                        "error": (
                            f"Coverage {coverage_pct}% is below minimum {min_coverage}%. "
                            f"Visit missing areas before finish: {preview}{suffix}"
                        ),
                        "coverage_pct": coverage_pct,
                        "missing_areas": missing_areas,
                    }
            self.summary = tool_args.get("summary", "Exploration finished.")
            self.finished = True
            self._events.emit("run_finish_requested", step=step, data={"summary": self.summary})
            return {"finished": True, "summary": self.summary}

        if tool_name not in BROWSER_TOOLS:
            return {"success": False, "error": f"Unknown tool '{tool_name}'"}

        try:
            if tool_name == "navigate":
                return await self.browser.navigate(tool_args["url"])
            if tool_name == "click":
                return await self.browser.click(tool_args["ref"])
            if tool_name == "click_testid":
                return await self.browser.click_testid(tool_args["testid"])
            if tool_name == "fill":
                return await self.browser.fill(tool_args["ref"], tool_args["text"])
            if tool_name == "select_option":
                return await self.browser.select_option(tool_args["ref"], tool_args["value"])
            if tool_name == "select_testid":
                return await self.browser.select_testid(tool_args["testid"], tool_args["value"])
            if tool_name == "press_key":
                return await self.browser.press_key(tool_args["key"])
            if tool_name == "scroll":
                return await self.browser.scroll(
                    direction=tool_args.get("direction", "down"),
                    amount=int(tool_args.get("amount", 600)),
                )
            if tool_name == "go_back":
                return await self.browser.go_back()
            if tool_name == "wait_for":
                return await self.browser.wait_for(float(tool_args.get("seconds", 2.0)))
            if tool_name == "set_appearance":
                return await self.browser.set_appearance(
                    theme_mode=tool_args.get("theme_mode"),
                    color_scheme=tool_args.get("color_scheme"),
                )
        except Exception as exc:
            logger.warning(
                "Tool %s failed: %s",
                tool_name,
                exc,
                extra={"run_id": self.run_id, "step": step, "tool": tool_name, "event": "tool_error"},
            )
            self._events.emit(
                "tool_error",
                step=step,
                tool=tool_name,
                url=observation.url,
                data={"error": str(exc)},
            )
            return {"success": False, "error": str(exc)}

        return {"success": False, "error": f"Unhandled tool '{tool_name}'"}

    def _format_observation(self, observation: Observation) -> str:
        lines = [
            observation.dom_digest,
            "",
            "Recent console errors:",
        ]
        lines.extend(observation.console_errors or ["(none)"])
        lines.append("")
        lines.append("Recent network errors:")
        lines.extend(observation.network_errors or ["(none)"])
        return "\n".join(lines)

    def _prepare_run_dir(self) -> Path:
        if self.config.output_dir:
            run_dir = self.config.output_dir
        else:
            run_dir = Path(__file__).resolve().parent / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "screenshots").mkdir(exist_ok=True)
        return run_dir

    async def _save_step_screenshot(self, run_dir: Path, step: int, observation: Observation) -> None:
        path = run_dir / "screenshots" / f"step-{step:02d}.png"
        path.write_bytes(await self.browser.screenshot_png())

    async def _save_issue_screenshot(self, run_dir: Path, issue_number: int) -> Path:
        path = run_dir / "screenshots" / f"issue-{issue_number:02d}.png"
        path.write_bytes(await self.browser.screenshot_png())
        return path
