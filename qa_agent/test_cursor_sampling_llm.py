from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import mcp.types as types
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from qa_agent.cursor_sampling_llm import (
    CursorSamplingLLM,
    SamplingTimeoutError,
    _parse_json_tool_call,
)


def test_parse_json_tool_call() -> None:
    tool_name, tool_args = _parse_json_tool_call(
        '{"tool": "click", "args": {"ref": "e1"}}'
    )
    assert tool_name == "click"
    assert tool_args == {"ref": "e1"}


def _mock_session(*, supports_tools: bool = False) -> MagicMock:
    session = MagicMock()
    session.check_client_capability.return_value = supports_tools
    return session


@pytest.mark.anyio
async def test_json_text_mode_on_success() -> None:
    session = _mock_session(supports_tools=False)
    session.create_message = AsyncMock(
        return_value=types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(
                type="text",
                text='{"tool": "finish", "args": {"summary": "done"}}',
            ),
            model="test-model",
        )
    )
    llm = CursorSamplingLLM(session, timeout=5.0)
    tool_name, tool_args, usage = await llm.choose_action(
        system_prompt="test",
        messages=[],
        screenshot_b64="abc",
        dom_digest="dom",
        step=1,
        max_steps=5,
    )
    assert tool_name == "finish"
    assert tool_args == {"summary": "done"}
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}
    session.create_message.assert_awaited_once()


@pytest.mark.anyio
async def test_tools_mode_falls_back_to_json_text() -> None:
    session = _mock_session(supports_tools=True)

    async def create_message(
        messages: list[Any],
        *,
        max_tokens: int,
        system_prompt: str | None = None,
        temperature: float | None = None,
        tools: list[types.Tool] | None = None,
        tool_choice: types.ToolChoice | None = None,
        related_request_id: str | None = None,
    ) -> types.CreateMessageResult | types.CreateMessageResultWithTools:
        if tools is not None:
            raise McpError(
                ErrorData(code=-32602, message="Client does not support sampling tools capability")
            )
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(
                type="text",
                text='{"tool": "wait_for", "args": {"seconds": 1}}',
            ),
            model="test-model",
        )

    session.create_message = AsyncMock(side_effect=create_message)
    llm = CursorSamplingLLM(session, timeout=5.0)
    tool_name, tool_args, _ = await llm.choose_action(
        system_prompt="test",
        messages=[],
        screenshot_b64="abc",
        dom_digest="dom",
        step=1,
        max_steps=5,
    )
    assert tool_name == "wait_for"
    assert tool_args == {"seconds": 1}
    assert session.create_message.await_count == 2


@pytest.mark.anyio
async def test_sampling_timeout_fails_fast() -> None:
    session = _mock_session(supports_tools=False)

    async def slow_create_message(*args: Any, **kwargs: Any) -> types.CreateMessageResult:
        await asyncio.sleep(10)
        raise AssertionError("should have timed out")

    session.create_message = AsyncMock(side_effect=slow_create_message)
    llm = CursorSamplingLLM(session, timeout=0.1)
    with pytest.raises(RuntimeError, match="Cursor sampling failed after all modes"):
        await llm.choose_action(
            system_prompt="test",
            messages=[],
            screenshot_b64="abc",
            dom_digest="dom",
            step=1,
            max_steps=5,
        )


@pytest.mark.anyio
async def test_create_message_timeout_raises_sampling_timeout() -> None:
    session = _mock_session(supports_tools=False)

    async def slow_create_message(*args: Any, **kwargs: Any) -> types.CreateMessageResult:
        await asyncio.sleep(10)
        raise AssertionError("should have timed out")

    session.create_message = AsyncMock(side_effect=slow_create_message)
    llm = CursorSamplingLLM(session, timeout=0.1)
    with pytest.raises(SamplingTimeoutError):
        await llm._create_message([], system_prompt="test")
