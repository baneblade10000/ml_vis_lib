from __future__ import annotations

from typing import Any, Protocol


class ActionChooser(Protocol):
    async def choose_action(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        screenshot_b64: str,
        dom_digest: str,
        step: int,
        max_steps: int,
    ) -> tuple[str, dict[str, Any], dict[str, int]]: ...
