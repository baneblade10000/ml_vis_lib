from __future__ import annotations

import asyncio

from qa_agent.e2e_routes import E2eRoutes, resolve_e2e_routes


def test_default_routes_playground() -> None:
    routes = E2eRoutes()
    assert routes.playground_path == "/"
    assert routes.coverage_navigate_paths() == ["/"]


def test_prompt_block_mentions_playground() -> None:
    block = E2eRoutes().to_prompt_block()
    assert "Decision Boundary demo" in block
    assert "localhost:6006" in block


def test_resolve_e2e_routes_no_backend() -> None:
    routes = asyncio.run(resolve_e2e_routes())
    assert routes.playground_path == "/"
