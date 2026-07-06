from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from qa_agent.coverage import ExplorationProfile, goal_for_profile, resolve_profile
from qa_agent.e2e_routes import E2eRoutes


_REPO_ROOT = Path(__file__).resolve().parent.parent

LlmBackend = Literal["cursor_chat", "cursor_sampling"]
ViewportMode = Literal["desktop", "mobile"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_REQUEST_TIMEOUT: float = 120.0

    E2E_BASE_URL: str = "http://localhost:5173"
    E2E_STORYBOOK_URL: str = "http://localhost:6006"


@dataclass
class RunConfig:
    base_url: str
    goal: str
    max_steps: int = 45
    headless: bool = True
    login: bool = False
    seed: bool = False
    output_dir: Path | None = None
    llm_timeout: float = 120.0
    llm_backend: LlmBackend = "cursor_chat"
    sampling_session: Any | None = None
    sampling_request_id: str | None = None
    viewport_mode: ViewportMode = "desktop"
    exploration_profile: ExplorationProfile = "full"
    min_coverage_pct: float = 0.0
    e2e_routes: E2eRoutes | None = None
    console_errors: list[str] = field(default_factory=list)
    network_errors: list[str] = field(default_factory=list)
    # Legacy fields kept for call-site compatibility (unused by ml-vis playground).
    api_base_url: str = ""
    test_email: str = ""
    test_password: str = ""


def load_settings() -> Settings:
    load_dotenv(_REPO_ROOT / ".env")
    return Settings()


def _base_run_config(
    *,
    base_url: str | None,
    goal: str | None,
    max_steps: int,
    headless: bool,
    login: bool,
    seed: bool,
    output_dir: Path | None,
    sampling_session: Any,
    sampling_request_id: str | None,
    viewport_mode: ViewportMode,
    exploration_profile: ExplorationProfile | str | None,
    min_coverage_pct: float,
) -> RunConfig:
    settings = load_settings()
    profile = resolve_profile(exploration_profile, viewport_mode=viewport_mode)
    if sampling_session is None:
        raise ValueError("QA runs require a Cursor MCP sampling session")

    return RunConfig(
        base_url=base_url or settings.E2E_BASE_URL,
        goal=goal or goal_for_profile(profile),
        max_steps=max_steps,
        headless=headless,
        login=login,
        seed=seed,
        output_dir=output_dir,
        llm_backend="cursor_sampling",
        sampling_session=sampling_session,
        sampling_request_id=sampling_request_id,
        llm_timeout=settings.LLM_REQUEST_TIMEOUT,
        viewport_mode=viewport_mode,
        exploration_profile=profile,
        min_coverage_pct=min_coverage_pct,
    )


def build_mcp_run_config(
    *,
    base_url: str | None = None,
    goal: str | None = None,
    max_steps: int = 45,
    headless: bool = True,
    login: bool = False,
    seed: bool = False,
    output_dir: Path | None = None,
    sampling_session: Any | None = None,
    sampling_request_id: str | None = None,
    viewport_mode: ViewportMode = "desktop",
    exploration_profile: ExplorationProfile | str | None = None,
    min_coverage_pct: float = 0.0,
    **_: Any,
) -> RunConfig:
    """Run config for MCP-triggered explorations using Cursor sampling."""
    return _base_run_config(
        base_url=base_url,
        goal=goal,
        max_steps=max_steps,
        headless=headless,
        login=login,
        seed=seed,
        output_dir=output_dir,
        sampling_session=sampling_session,
        sampling_request_id=sampling_request_id,
        viewport_mode=viewport_mode,
        exploration_profile=exploration_profile,
        min_coverage_pct=min_coverage_pct,
    )


def build_interactive_run_config(
    *,
    base_url: str | None = None,
    goal: str | None = None,
    max_steps: int = 45,
    headless: bool = True,
    login: bool = False,
    seed: bool = False,
    output_dir: Path | None = None,
    viewport_mode: ViewportMode = "desktop",
    exploration_profile: ExplorationProfile | str | None = None,
    min_coverage_pct: float = 0.0,
    **_: Any,
) -> RunConfig:
    """Run config for stepwise sessions — Cursor chat agent is the vision model."""
    settings = load_settings()
    profile = resolve_profile(exploration_profile, viewport_mode=viewport_mode)

    return RunConfig(
        base_url=base_url or settings.E2E_BASE_URL,
        goal=goal or goal_for_profile(profile),
        max_steps=max_steps,
        headless=headless,
        login=login,
        seed=seed,
        output_dir=output_dir,
        llm_backend="cursor_chat",
        llm_timeout=settings.LLM_REQUEST_TIMEOUT,
        viewport_mode=viewport_mode,
        exploration_profile=profile,
        min_coverage_pct=min_coverage_pct,
    )
