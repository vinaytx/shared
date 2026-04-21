"""Debug logging for the agent: captures inbound messages, LLM prompts/responses, tool calls, and outbound messages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.providers.base import LLMResponse, ToolCallRequest

_SEP = "=" * 80
_DIV = "-" * 80


class DebugLog:
    """Writes human-readable debug entries to ~/.nanobot/logs/debug_YYYY-MM-DD.log."""

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def log_inbound(self, session_key: str, channel: str, sender: str, content: str) -> None:
        """Log a message received from a user or agent."""
        header = f"Session: {session_key} | Channel: {channel} | Sender: {sender}"
        self._write("INBOUND MESSAGE", header, content)

    def log_prompt(
        self,
        session_key: str,
        model: str,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Log the raw prompt (messages list) sent to the LLM."""
        header = (
            f"Session: {session_key} | Model: {model} | Iteration: {iteration}"
            f" | Messages: {len(messages)}"
        )
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")

            label = f"[{role}]"
            if role == "TOOL" and name:
                label = f"[TOOL result: {name} (id={tool_call_id})]"

            content_str = _render_content(content)

            if tool_calls:
                tc_json = json.dumps(tool_calls, indent=2, ensure_ascii=False)
                content_str = (content_str + f"\n[tool_calls]\n{tc_json}").strip()

            parts.append(f"{label}\n{content_str}")

        body = ("\n\n" + _DIV + "\n").join(parts)

        if tools:
            tool_names = ", ".join(t.get("function", {}).get("name", "?") for t in tools)
            body += f"\n\n[Available tools: {tool_names}]"

        self._write("LLM PROMPT", header, body)

    def log_response(
        self,
        session_key: str,
        model: str,
        iteration: int,
        response: LLMResponse,
    ) -> None:
        """Log the raw response received from the LLM."""
        header = (
            f"Session: {session_key} | Model: {model} | Iteration: {iteration}"
            f" | Finish: {response.finish_reason}"
        )
        parts: list[str] = []

        if response.reasoning_content:
            parts.append(f"[reasoning]\n{response.reasoning_content}")

        if response.content:
            parts.append(f"[content]\n{response.content}")

        if response.tool_calls:
            tcs = [
                {"name": tc.name, "id": tc.id, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]
            parts.append(f"[tool_calls]\n{json.dumps(tcs, indent=2, ensure_ascii=False)}")

        if response.usage:
            parts.append(f"[usage] {response.usage}")

        body = ("\n\n" + _DIV + "\n").join(parts) if parts else "(empty response)"
        self._write("LLM RESPONSE", header, body)

    def log_tool_call(
        self,
        session_key: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        """Log a tool call and its result."""
        header = f"Session: {session_key} | Tool: {tool_name}"
        args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
        result_str = result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False, default=str)
        body = f"[arguments]\n{args_str}\n\n{_DIV}\n[result]\n{result_str}"
        self._write("TOOL CALL", header, body)

    def log_outbound(self, session_key: str, channel: str, content: str) -> None:
        """Log the final message sent to the user."""
        header = f"Session: {session_key} | Channel: {channel}"
        self._write("OUTBOUND MESSAGE", header, content)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_file(self) -> Path:
        date = datetime.now().strftime("%Y-%m-%d")
        return self._logs_dir / f"debug_{date}.log"

    def _write(self, event: str, header: str, body: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = f"\n{_SEP}\n[{ts}] {event}\n{header}\n{_DIV}\n{body}\n{_SEP}\n"
        try:
            with open(self._log_file(), "a", encoding="utf-8") as fh:
                fh.write(entry)
        except OSError:
            pass  # never crash the agent due to logging


def _render_content(content: Any) -> str:
    """Convert message content (str or list of blocks) to a readable string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    parts.append("[IMAGE (base64)]")
                else:
                    parts.append(f"[IMAGE: {url}]")
            elif btype == "tool_result":
                tc_content = _render_content(block.get("content", ""))
                parts.append(f"[tool_result id={block.get('tool_use_id')}]\n{tc_content}")
            elif btype == "tool_use":
                args = json.dumps(block.get("input", {}), ensure_ascii=False)
                parts.append(f"[tool_use name={block.get('name')} id={block.get('id')}]\n{args}")
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)
