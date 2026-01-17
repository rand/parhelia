"""Headless Claude Code execution.

Implements:
- [SPEC-02.10] Session Identification
- [SPEC-02.13] Headless Execution
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def generate_session_id(task_id: str, prefix: str = "ph") -> str:
    """Generate a unique session identifier.

    Implements [SPEC-02.10].

    Args:
        task_id: User-provided task identifier.
        prefix: Session ID prefix (default: 'ph' for Parhelia).

    Returns:
        Session ID in format: {prefix}-{task-id}-{timestamp}
    """
    # Sanitize task_id for tmux compatibility
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", task_id.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")

    # Generate timestamp + short UUID for guaranteed uniqueness
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]

    return f"{prefix}-{sanitized}-{timestamp}-{unique_suffix}"


def build_headless_command(
    prompt: str,
    max_turns: int = 50,
    allowed_tools: list[str] | None = None,
    working_dir: str | None = None,
) -> list[str]:
    """Build command for headless Claude Code execution.

    Implements [SPEC-02.13].

    Args:
        prompt: The task prompt for Claude.
        max_turns: Maximum conversation turns (default: 50).
        allowed_tools: Optional list of allowed tools.
        working_dir: Working directory (handled externally by tmux).

    Returns:
        Command list suitable for subprocess execution.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--max-turns",
        str(max_turns),
    ]

    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])

    return cmd


@dataclass
class OutputEvent:
    """Parsed event from Claude Code JSONL output.

    Implements [SPEC-02.13].
    """

    type: str
    raw: str = ""

    # For assistant messages
    content: str = ""

    # For tool use
    tool_name: str = ""
    tool_use_id: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)

    # For tool result
    # tool_use_id also used here

    # For result
    result: str = ""
    cost_usd: float = 0.0

    # For error
    error_message: str = ""

    # For needs_human
    reason: str = ""
    context: str = ""


class HeadlessOutputParser:
    """Parse JSONL output from headless Claude Code.

    Implements [SPEC-02.13].
    """

    def parse_line(self, line: str) -> OutputEvent:
        """Parse a single line of JSONL output.

        Args:
            line: A line of JSONL output.

        Returns:
            Parsed OutputEvent.
        """
        line = line.strip()
        if not line:
            return OutputEvent(type="empty", raw=line)

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return OutputEvent(type="unknown", raw=line)

        event_type = data.get("type", "unknown")

        if event_type == "assistant":
            content = self._extract_content(data.get("message", {}))
            return OutputEvent(type="assistant", raw=line, content=content)

        elif event_type == "tool_use":
            return OutputEvent(
                type="tool_use",
                raw=line,
                tool_name=data.get("tool", ""),
                tool_use_id=data.get("id", ""),
                tool_input=data.get("input", {}),
            )

        elif event_type == "tool_result":
            content = data.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            return OutputEvent(
                type="tool_result",
                raw=line,
                tool_use_id=data.get("tool_use_id", ""),
                content=str(content),
            )

        elif event_type == "result":
            return OutputEvent(
                type="result",
                raw=line,
                result=data.get("result", ""),
                cost_usd=data.get("cost_usd", 0.0),
            )

        elif event_type == "error":
            error = data.get("error", {})
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            return OutputEvent(type="error", raw=line, error_message=message)

        elif event_type == "needs_human":
            return OutputEvent(
                type="needs_human",
                raw=line,
                reason=data.get("reason", ""),
                context=data.get("context", ""),
            )

        else:
            return OutputEvent(type=event_type, raw=line)

    def _extract_content(self, message: dict) -> str:
        """Extract text content from assistant message."""
        content = message.get("content", [])
        if isinstance(content, str):
            return content

        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts)


@dataclass
class ExecutionResult:
    """Result from headless execution.

    Implements [SPEC-02.13].
    """

    success: bool
    result: str = ""
    cost_usd: float = 0.0
    error_message: str = ""
    events: list[OutputEvent] = field(default_factory=list)
    needs_human: bool = False
    needs_human_reason: str = ""


class HeadlessExecutor:
    """Execute Claude Code in headless mode.

    Implements [SPEC-02.13].
    """

    def __init__(self):
        """Initialize the executor."""
        self.parser = HeadlessOutputParser()

    async def run(
        self,
        prompt: str,
        working_dir: str,
        max_turns: int = 50,
        allowed_tools: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Run Claude Code headlessly.

        Args:
            prompt: The task prompt.
            working_dir: Working directory for execution.
            max_turns: Maximum conversation turns.
            allowed_tools: Optional list of allowed tools.
            env: Optional environment variables.

        Returns:
            ExecutionResult with outcome and collected events.
        """
        cmd = build_headless_command(
            prompt=prompt,
            max_turns=max_turns,
            allowed_tools=allowed_tools,
        )

        # Prepare environment
        import os

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Run the process
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            env=process_env,
        )

        events: list[OutputEvent] = []
        result_event: OutputEvent | None = None
        error_event: OutputEvent | None = None
        needs_human_event: OutputEvent | None = None

        # Read output line by line
        if proc.stdout:
            async for line_bytes in proc.stdout:
                line = line_bytes.decode("utf-8", errors="replace")
                event = self.parser.parse_line(line)
                events.append(event)

                if event.type == "result":
                    result_event = event
                elif event.type == "error":
                    error_event = event
                elif event.type == "needs_human":
                    needs_human_event = event

        exit_code = await proc.wait()

        # Build result
        if error_event:
            return ExecutionResult(
                success=False,
                error_message=error_event.error_message,
                events=events,
            )

        if needs_human_event:
            return ExecutionResult(
                success=False,
                needs_human=True,
                needs_human_reason=needs_human_event.reason,
                events=events,
            )

        if result_event:
            return ExecutionResult(
                success=True,
                result=result_event.result,
                cost_usd=result_event.cost_usd,
                events=events,
            )

        # No explicit result - check exit code
        return ExecutionResult(
            success=exit_code == 0,
            error_message=f"Process exited with code {exit_code}" if exit_code != 0 else "",
            events=events,
        )
