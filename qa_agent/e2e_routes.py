"""Playground URLs for QA (single-page ml-vis app)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class E2eRoutes:
    playground_path: str = "/"

    def to_prompt_block(self) -> str:
        return "\n".join(
            [
                "ml-vis playground routes:",
                f"- Decision Boundary demo: {self.playground_path}",
                "- Storybook (optional, separate dev server): http://localhost:6006",
            ]
        )

    def coverage_navigate_paths(self) -> list[str]:
        return [self.playground_path]


async def resolve_e2e_routes(
    api_base_url: str | None = None,
    email: str | None = None,
    password: str | None = None,
    *,
    seed_payload: dict | None = None,
) -> E2eRoutes:
    """No backend seeding — playground is a static SPA."""
    _ = (api_base_url, email, password, seed_payload)
    return E2eRoutes()
