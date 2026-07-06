from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunEvent:
    event: str
    run_id: str
    ts: str = field(default_factory=utc_now_iso)
    step: int | None = None
    tool: str | None = None
    url: str | None = None
    duration_ms: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["data"]:
            del payload["data"]
        return {key: value for key, value in payload.items() if value is not None}


class RunEventLogger:
    """Append-only JSONL event log per exploration run."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_id = run_id
        self._path = run_dir / "events.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: str, **fields: Any) -> None:
        data = fields.pop("data", {})
        record = RunEvent(
            event=event,
            run_id=self.run_id,
            data=data if isinstance(data, dict) else {},
            **fields,
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
