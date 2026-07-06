from __future__ import annotations

from qa_agent.e2e_routes import E2eRoutes


def test_build_paths_playground_only() -> None:
    routes = E2eRoutes()
    paths = routes.coverage_navigate_paths()
    assert paths == ["/"]
