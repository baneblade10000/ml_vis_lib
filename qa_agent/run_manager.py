from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from qa_agent.config import RunConfig, ViewportMode, build_mcp_run_config
from qa_agent.coverage import ExplorationProfile
from qa_agent.run_store import (
    list_disk_runs,
    load_metrics,
    load_report,
    resolve_issue_screenshot,
    resolve_run_dir,
    resolve_screenshot_path,
    summarize_disk_run,
)
from qa_agent.service import execute_run
from qa_agent.ux_metrics import UxMetrics

logger = logging.getLogger("qa_agent.run_manager")

_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunRecord:
    run_id: str
    status: RunStatus
    goal: str
    max_steps: int
    started_at: str
    exploration_profile: str = "full"
    run_dir: Path | None = None
    report_path: Path | None = None
    metrics_path: Path | None = None
    events_path: Path | None = None
    current_step: int = 0
    issues_count: int = 0
    error: str = ""
    finished_at: str = ""
    summary: str = ""
    metrics: UxMetrics | None = None
    _task: asyncio.Task | None = field(default=None, repr=False, compare=False)


ProgressHook = Callable[..., None | asyncio.Future | Any]


class RunManager:
    """In-memory registry of exploration runs for CLI and MCP control."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def list_runs(self, *, limit: int = 20, include_disk: bool = True) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}

        if include_disk:
            for item in list_disk_runs(limit=limit * 2):
                merged[item["run_id"]] = item

        for record in self._runs.values():
            merged[record.run_id] = self._serialize(record)

        records = sorted(
            merged.values(),
            key=lambda item: item.get("started_at") or item.get("run_id", ""),
            reverse=True,
        )
        return records[:limit]

    def resolve_run(self, run_id: str) -> RunRecord | None:
        record = self._runs.get(run_id)
        if record:
            return record
        run_dir = resolve_run_dir(run_id)
        if not run_dir:
            return None
        summary = summarize_disk_run(run_id, run_dir)
        status_value = summary.get("status", "unknown")
        try:
            status = RunStatus(status_value)
        except ValueError:
            status = RunStatus.COMPLETED if summary.get("report_path") else RunStatus.FAILED

        return RunRecord(
            run_id=run_id,
            status=status,
            goal=summary.get("goal") or "",
            max_steps=int(summary.get("max_steps") or 0),
            exploration_profile=summary.get("exploration_profile") or "full",
            started_at=summary.get("started_at") or "",
            run_dir=run_dir,
            report_path=Path(summary["report_path"]) if summary.get("report_path") else None,
            metrics_path=Path(summary["metrics_path"]) if summary.get("metrics_path") else None,
            events_path=Path(summary["events_path"]) if summary.get("events_path") else None,
            current_step=int(summary.get("current_step") or 0),
            issues_count=int(summary.get("issues_count") or 0),
            error=summary.get("error") or "",
            finished_at=summary.get("finished_at") or "",
            summary=summary.get("summary") or "",
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.resolve_run(run_id)

    def get_status(self, run_id: str) -> dict[str, Any] | None:
        record = self.resolve_run(run_id)
        if not record:
            return None
        return self._serialize(record)

    def get_report(self, run_id: str) -> str | None:
        record = self.resolve_run(run_id)
        if not record:
            return None
        if record.report_path and record.report_path.exists():
            return record.report_path.read_text(encoding="utf-8")
        if record.run_dir:
            return load_report(record.run_dir)
        return None

    def get_metrics(self, run_id: str) -> dict[str, Any] | None:
        record = self.resolve_run(run_id)
        if not record:
            return None
        if record.metrics:
            return record.metrics.to_dict()
        if record.metrics_path and record.metrics_path.exists():
            return json.loads(record.metrics_path.read_text(encoding="utf-8"))
        if record.run_dir:
            return load_metrics(record.run_dir)
        return None

    def get_issue_screenshot_path(self, run_id: str, issue_number: int) -> Path | None:
        record = self.resolve_run(run_id)
        if not record or not record.run_dir:
            return None
        return resolve_issue_screenshot(record.run_dir, issue_number)

    def get_screenshot_path(self, run_id: str, filename: str) -> Path | None:
        record = self.resolve_run(run_id)
        if not record or not record.run_dir:
            return None
        return resolve_screenshot_path(record.run_dir, filename)

    async def start_run(
        self,
        *,
        base_url: str | None = None,
        api_base_url: str | None = None,
        goal: str | None = None,
        max_steps: int = 45,
        headless: bool = True,
        login: bool = True,
        seed: bool = True,
        output_dir: Path | None = None,
        test_email: str | None = None,
        test_password: str | None = None,
        run_id: str | None = None,
        sampling_session: Any | None = None,
        sampling_request_id: str | None = None,
        viewport_mode: ViewportMode = "desktop",
        exploration_profile: ExplorationProfile | str | None = None,
        min_coverage_pct: float = 0.0,
        progress_hook: ProgressHook | None = None,
    ) -> str:
        if sampling_session is None:
            raise ValueError("QA exploration requires an MCP Cursor sampling session")
        config = build_mcp_run_config(
            base_url=base_url,
            api_base_url=api_base_url,
            goal=goal,
            max_steps=max_steps,
            headless=headless,
            login=login,
            seed=seed,
            output_dir=output_dir,
            test_email=test_email,
            test_password=test_password,
            sampling_session=sampling_session,
            sampling_request_id=sampling_request_id,
            viewport_mode=viewport_mode,
            exploration_profile=exploration_profile,
            min_coverage_pct=min_coverage_pct,
        )
        resolved_run_id = run_id or self._new_run_id()
        if not config.output_dir:
            config.output_dir = _RUNS_ROOT / resolved_run_id

        record = RunRecord(
            run_id=resolved_run_id,
            status=RunStatus.PENDING,
            goal=config.goal,
            max_steps=config.max_steps,
            exploration_profile=config.exploration_profile,
            started_at=datetime.now(timezone.utc).isoformat(),
            run_dir=config.output_dir,
        )
        self._runs[resolved_run_id] = record
        record._task = asyncio.create_task(
            self._execute(resolved_run_id, config, progress_hook=progress_hook)
        )
        return resolved_run_id

    async def cancel_run(self, run_id: str) -> bool:
        record = self._runs.get(run_id)
        if not record or not record._task:
            return False
        if record.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return False
        record._task.cancel()
        record.status = RunStatus.CANCELLED
        record.finished_at = datetime.now(timezone.utc).isoformat()
        return True

    async def _execute(
        self,
        run_id: str,
        config: RunConfig,
        *,
        progress_hook: ProgressHook | None = None,
    ) -> None:
        record = self._runs[run_id]
        record.status = RunStatus.RUNNING

        def on_progress(step: int, issues_count: int, meta: dict[str, Any] | None = None) -> None:
            record.current_step = step
            record.issues_count = issues_count
            if not progress_hook:
                return
            try:
                result = progress_hook(step, issues_count, meta or {})
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as exc:
                logger.warning("Progress hook failed for run %s: %s", run_id, exc)

        try:
            result = await execute_run(config, run_id=run_id, on_progress=on_progress)
        except asyncio.CancelledError:
            record.status = RunStatus.CANCELLED
            record.finished_at = datetime.now(timezone.utc).isoformat()
            raise
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            record.status = RunStatus.FAILED
            record.error = str(exc)
            record.finished_at = datetime.now(timezone.utc).isoformat()
            _append_run_failed_event(record, str(exc))
            return

        record.status = RunStatus.COMPLETED
        record.report_path = result.report_path
        record.metrics_path = result.metrics_path
        record.events_path = result.events_path
        record.summary = result.summary
        record.current_step = result.steps_taken
        record.issues_count = result.issues_count
        record.metrics = result.metrics
        record.finished_at = datetime.now(timezone.utc).isoformat()

    def _serialize(self, record: RunRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "run_id": record.run_id,
            "status": record.status.value,
            "goal": record.goal,
            "max_steps": record.max_steps,
            "exploration_profile": record.exploration_profile,
            "current_step": record.current_step,
            "issues_count": record.issues_count,
            "started_at": record.started_at,
            "finished_at": record.finished_at or None,
            "summary": record.summary or None,
            "error": record.error or None,
            "run_dir": str(record.run_dir) if record.run_dir else None,
            "report_path": str(record.report_path) if record.report_path else None,
            "metrics_path": str(record.metrics_path) if record.metrics_path else None,
            "events_path": str(record.events_path) if record.events_path else None,
        }
        if record.metrics:
            payload["overall_ux_score"] = record.metrics.overall_score
        return payload

    @staticmethod
    def _new_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _append_run_failed_event(record: RunRecord, error: str) -> None:
    """Append run_failed to events.jsonl if not already present (backup for early crashes)."""
    if not record.run_dir:
        return
    events_path = record.run_dir / "events.jsonl"
    if not events_path.exists():
        return
    try:
        existing = events_path.read_text(encoding="utf-8")
        if "run_failed" in existing:
            return
    except OSError:
        return
    payload = {
        "event": "run_failed",
        "run_id": record.run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "data": {"error": error},
    }
    try:
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not append run_failed event for %s: %s", record.run_id, exc)


_manager: RunManager | None = None


def get_run_manager() -> RunManager:
    global _manager
    if _manager is None:
        _manager = RunManager()
    return _manager
