from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TYPE_CHECKING

import mcp.types as types
from json_repair import repair_json
from mcp.shared.exceptions import McpError

from qa_agent.tools import TOOL_DEFINITIONS

if TYPE_CHECKING:
    from mcp.server.session import ServerSession

logger = logging.getLogger("qa_agent.cursor_sampling_llm")

_DEFAULT_TIMEOUT = 120.0

_JSON_TOOL_SUFFIX = """
Respond with ONLY a JSON object (no markdown fences), shape:
{{"tool": "<tool_name>", "args": {{<parameters>}}}}

Available tools:
{tools_json}
"""


def to_mcp_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=definition["function"]["name"],
            description=definition["function"].get("description"),
            inputSchema=definition["function"]["parameters"],
        )
        for definition in TOOL_DEFINITIONS
    ]


def _tools_catalog_json() -> str:
    catalog = [
        {
            "name": item["function"]["name"],
            "description": item["function"].get("description", ""),
            "parameters": item["function"]["parameters"],
        }
        for item in TOOL_DEFINITIONS
    ]
    return json.dumps(catalog, ensure_ascii=False, indent=2)


def _supports_sampling_tools(session: ServerSession) -> bool:
    return session.check_client_capability(
        types.ClientCapabilities(
            sampling=types.SamplingCapability(tools=types.SamplingToolsCapability()),
        )
    )


def client_supports_sampling(session: ServerSession) -> bool:
    """True if the MCP client advertises sampling/createMessage support."""
    return session.check_client_capability(
        types.ClientCapabilities(sampling=types.SamplingCapability()),
    )


def _to_sampling_messages(messages: list[dict[str, Any]]) -> list[types.SamplingMessage]:
    converted: list[types.SamplingMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        converted.append(
            types.SamplingMessage(
                role=role,
                content=types.TextContent(type="text", text=content),
            )
        )
    return converted


def _extract_tool_use(content: Any) -> types.ToolUseContent:
    if isinstance(content, types.ToolUseContent):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, types.ToolUseContent):
                return block
    raise RuntimeError("Cursor sampling response did not include a tool call")


def _extract_text(content: Any) -> str:
    if isinstance(content, types.TextContent):
        return content.text
    if isinstance(content, list):
        parts = [block.text for block in content if isinstance(block, types.TextContent)]
        if parts:
            return "\n".join(parts)
    raise RuntimeError("Cursor sampling response did not include text content")


def _parse_json_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    payload = json.loads(repair_json(cleaned))
    tool_name = payload.get("tool") or payload.get("name")
    tool_args = payload.get("args") or payload.get("arguments") or {}
    if not tool_name:
        raise RuntimeError(f"JSON tool response missing tool name: {text[:200]}")
    if not isinstance(tool_args, dict):
        raise RuntimeError(f"JSON tool args must be an object: {text[:200]}")
    return str(tool_name), tool_args


class SamplingTimeoutError(TimeoutError):
    """Raised when MCP sampling/createMessage does not respond in time."""


class CursorSamplingLLM:
    """Use the MCP client's LLM (Cursor) via sampling/createMessage."""

    def __init__(
        self,
        session: ServerSession,
        *,
        related_request_id: str | None = None,
        max_tokens: int = 4096,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._session = session
        self._related_request_id = related_request_id
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._mcp_tools = to_mcp_tools()
        self._use_tools = _supports_sampling_tools(session)
        logger.info(
            "Cursor sampling mode: %s (timeout=%ss)",
            "tools" if self._use_tools else "json_text",
            self._timeout,
        )

    async def choose_action(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        screenshot_b64: str,
        dom_digest: str,
        step: int,
        max_steps: int,
    ) -> tuple[str, dict[str, Any], dict[str, int]]:
        user_text = (
            f"Step {step}/{max_steps}\n\n"
            f"DOM digest:\n{dom_digest}\n\n"
            "Decide the next browser action or report an issue/finish."
        )
        sampling_messages = _to_sampling_messages(messages)
        sampling_messages.append(
            types.SamplingMessage(
                role="user",
                content=[
                    types.TextContent(type="text", text=user_text),
                    types.ImageContent(
                        type="image",
                        data=screenshot_b64,
                        mimeType="image/png",
                    ),
                ],
            )
        )

        modes: list[tuple[str, bool]] = []
        if self._use_tools:
            modes.append(("tools", True))
        modes.append(("json_text", False))

        errors: list[str] = []
        for mode_name, use_tools in modes:
            try:
                return await self._choose_with_mode(
                    use_tools=use_tools,
                    mode_name=mode_name,
                    system_prompt=system_prompt,
                    sampling_messages=sampling_messages,
                )
            except Exception as exc:
                message = f"{mode_name}: {exc}"
                errors.append(message)
                logger.warning("Cursor sampling failed (%s)", message)

        raise RuntimeError(
            "Cursor sampling failed after all modes: " + "; ".join(errors)
        )

    async def _choose_with_mode(
        self,
        *,
        use_tools: bool,
        mode_name: str,
        system_prompt: str,
        sampling_messages: list[types.SamplingMessage],
    ) -> tuple[str, dict[str, Any], dict[str, int]]:
        if use_tools:
            result = await self._create_message(
                sampling_messages,
                system_prompt=system_prompt,
                tools=self._mcp_tools,
                tool_choice=types.ToolChoice(mode="required"),
            )
            tool_use = _extract_tool_use(result.content)
            tool_name = tool_use.name
            tool_args = dict(tool_use.input or {})
        else:
            json_system = system_prompt + _JSON_TOOL_SUFFIX.format(
                tools_json=_tools_catalog_json(),
            )
            result = await self._create_message(
                sampling_messages,
                system_prompt=json_system,
            )
            tool_name, tool_args = _parse_json_tool_call(_extract_text(result.content))

        logger.info(
            "Cursor sampling chose tool=%s args=%s model=%s mode=%s",
            tool_name,
            tool_args,
            result.model,
            mode_name,
        )
        return tool_name, tool_args, {"prompt_tokens": 0, "completion_tokens": 0}

    async def describe_screenshot(
        self,
        *,
        screenshot_b64: str,
        dom_digest: str,
        goal: str,
        max_tokens: int = 450,
    ) -> str:
        """Ask Cursor's built-in vision model to describe a QA screenshot."""
        prompt = (
            f"QA goal: {goal}\n\n"
            "Describe what is ACTUALLY visible in this screenshot. Be concrete:\n"
            "- Is the main canvas/graph empty or populated?\n"
            "- Are side panels (dataset, output, loss chart) showing content or blank shells?\n"
            "- Any obvious layout bugs?\n\n"
            f"DOM digest (may disagree with pixels — trust the screenshot):\n{dom_digest[:3500]}"
        )
        sampling_messages = [
            types.SamplingMessage(
                role="user",
                content=[
                    types.TextContent(type="text", text=prompt),
                    types.ImageContent(
                        type="image",
                        data=screenshot_b64,
                        mimeType="image/png",
                    ),
                ],
            )
        ]
        result = await self._create_message(
            sampling_messages,
            system_prompt=(
                "You are a QA vision assistant. Describe only what you see in the screenshot. "
                "Flag empty canvases, missing charts, and DOM/screenshot mismatches."
            ),
            max_tokens=max_tokens,
        )
        text = _extract_text(result.content).strip()
        if not text:
            raise RuntimeError("Cursor vision returned empty description")
        return text

    async def _create_message(
        self,
        sampling_messages: list[types.SamplingMessage],
        *,
        system_prompt: str,
        tools: list[types.Tool] | None = None,
        tool_choice: types.ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> types.CreateMessageResult | types.CreateMessageResultWithTools:
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "system_prompt": system_prompt,
            "temperature": 0.2,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self._related_request_id is not None:
            kwargs["related_request_id"] = self._related_request_id

        try:
            return await asyncio.wait_for(
                self._session.create_message(sampling_messages, **kwargs),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise SamplingTimeoutError(
                f"MCP sampling/createMessage timed out after {self._timeout}s"
            ) from exc
        except McpError as exc:
            raise RuntimeError(
                "Cursor MCP sampling is unavailable (enable MCP sampling in Cursor, "
                "or retry after MCP reload). "
                f"Details: {exc}"
            ) from exc
