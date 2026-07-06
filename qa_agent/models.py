from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Issue:
    severity: str
    title: str
    description: str
    repro_steps: list[str] = field(default_factory=list)
    screenshot_path: str = ""
    url: str = ""
    step: int = 0


@dataclass
class UxObservation:
    area: str
    aspect: str
    score: int
    notes: str = ""
    url: str = ""
    step: int = 0


@dataclass
class AgentTraceStep:
    step: int
    tool: str
    args: dict[str, Any]
    result: dict[str, Any]
    url: str
    duration_ms: int = 0
