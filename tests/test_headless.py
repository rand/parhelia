"""Tests for headless Claude Code execution.

@trace SPEC-02.10 - Session Identification
@trace SPEC-02.13 - Headless Execution
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSessionIdentification:
    """Tests for session ID generation - SPEC-02.10."""

    def test_generate_session_id_format(self):
        """@trace SPEC-02.10 - Session ID MUST follow format {prefix}-{task-id}-{timestamp}-{uuid}."""
        from parhelia.headless import generate_session_id

        session_id = generate_session_id("fix-auth")

        assert session_id.startswith("ph-")
        assert "fix-auth" in session_id
        # Check format: ph-fix-auth-YYYYMMDDTHHMMSS-uuid
        parts = session_id.split("-")
        assert len(parts) >= 4
        # Timestamp part contains 'T'
        assert any("T" in part for part in parts)

    def test_generate_session_id_with_custom_prefix(self):
        """@trace SPEC-02.10 - Session ID prefix SHOULD be configurable."""
        from parhelia.headless import generate_session_id

        session_id = generate_session_id("my-task", prefix="custom")

        assert session_id.startswith("custom-")
        assert "my-task" in session_id

    def test_generate_session_id_unique(self):
        """@trace SPEC-02.10 - Session IDs MUST be unique."""
        from parhelia.headless import generate_session_id

        ids = [generate_session_id("task") for _ in range(10)]

        assert len(set(ids)) == 10  # All unique

    def test_generate_session_id_sanitizes_task_id(self):
        """@trace SPEC-02.10 - Task ID MUST be sanitized for tmux compatibility."""
        from parhelia.headless import generate_session_id

        session_id = generate_session_id("Fix Auth Bug!")

        # Should not contain spaces or special chars
        assert " " not in session_id
        assert "!" not in session_id


class TestHeadlessCommandBuilder:
    """Tests for headless command building - SPEC-02.13."""

    def test_build_headless_command_basic(self):
        """@trace SPEC-02.13 - Headless command MUST use --output-format stream-json."""
        from parhelia.headless import build_headless_command

        cmd = build_headless_command(prompt="Fix the bug")

        assert "claude" in cmd
        assert "-p" in cmd
        assert "Fix the bug" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd

    def test_build_headless_command_with_max_turns(self):
        """@trace SPEC-02.13 - Headless command SHOULD support max_turns."""
        from parhelia.headless import build_headless_command

        cmd = build_headless_command(prompt="Do something", max_turns=25)

        assert "--max-turns" in cmd
        assert "25" in cmd

    def test_build_headless_command_with_allowed_tools(self):
        """@trace SPEC-02.13 - Headless command SHOULD support allowed tools."""
        from parhelia.headless import build_headless_command

        cmd = build_headless_command(
            prompt="Edit file", allowed_tools=["Read", "Edit", "Bash"]
        )

        assert "--allowedTools" in cmd
        assert "Read,Edit,Bash" in cmd

    def test_build_headless_command_with_working_dir(self):
        """@trace SPEC-02.13 - Headless command SHOULD support working directory."""
        from parhelia.headless import build_headless_command

        cmd = build_headless_command(prompt="Do work", working_dir="/vol/workspace")

        # Working dir is handled by tmux session, not claude command
        # But we may want to track it
        assert "claude" in cmd

    def test_build_headless_command_default_max_turns(self):
        """@trace SPEC-02.13 - Default max_turns SHOULD be 50."""
        from parhelia.headless import build_headless_command

        cmd = build_headless_command(prompt="Task")

        assert "--max-turns" in cmd
        assert "50" in cmd


class TestJSONLParsing:
    """Tests for JSONL output parsing - SPEC-02.13."""

    def test_parse_assistant_message(self):
        """@trace SPEC-02.13 - Parser MUST handle assistant messages."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "I'll fix that bug."}]
                },
            }
        )

        event = parser.parse_line(line)

        assert event.type == "assistant"
        assert "fix that bug" in event.content

    def test_parse_tool_use(self):
        """@trace SPEC-02.13 - Parser MUST handle tool use events."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = json.dumps(
            {
                "type": "tool_use",
                "tool": "Read",
                "input": {"file_path": "/path/to/file.py"},
            }
        )

        event = parser.parse_line(line)

        assert event.type == "tool_use"
        assert event.tool_name == "Read"
        assert event.tool_input["file_path"] == "/path/to/file.py"

    def test_parse_tool_result(self):
        """@trace SPEC-02.13 - Parser MUST handle tool result events."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = json.dumps(
            {
                "type": "tool_result",
                "tool_use_id": "tool_123",
                "content": "File contents here",
            }
        )

        event = parser.parse_line(line)

        assert event.type == "tool_result"
        assert event.tool_use_id == "tool_123"
        assert "File contents" in event.content

    def test_parse_result_event(self):
        """@trace SPEC-02.13 - Parser MUST handle result events."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = json.dumps(
            {
                "type": "result",
                "result": "Task completed successfully",
                "cost_usd": 0.45,
            }
        )

        event = parser.parse_line(line)

        assert event.type == "result"
        assert event.result == "Task completed successfully"
        assert event.cost_usd == 0.45

    def test_parse_error_event(self):
        """@trace SPEC-02.13 - Parser MUST handle error events."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = json.dumps({"type": "error", "error": {"message": "API rate limit"}})

        event = parser.parse_line(line)

        assert event.type == "error"
        assert "rate limit" in event.error_message

    def test_parse_invalid_json(self):
        """@trace SPEC-02.13 - Parser MUST handle invalid JSON gracefully."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = "not valid json {"

        event = parser.parse_line(line)

        assert event.type == "unknown"
        assert event.raw == line

    def test_parse_needs_human(self):
        """@trace SPEC-02.13 - Parser MUST detect needs_human signals."""
        from parhelia.headless import HeadlessOutputParser

        parser = HeadlessOutputParser()
        line = json.dumps(
            {
                "type": "needs_human",
                "reason": "Permission denied",
                "context": "Need approval for database drop",
            }
        )

        event = parser.parse_line(line)

        assert event.type == "needs_human"
        assert event.reason == "Permission denied"


class TestHeadlessExecutor:
    """Tests for HeadlessExecutor class - SPEC-02.13."""

    @pytest.fixture
    def executor(self):
        """Create HeadlessExecutor instance."""
        from parhelia.headless import HeadlessExecutor

        return HeadlessExecutor()

    @pytest.mark.asyncio
    async def test_executor_runs_command(self, executor):
        """@trace SPEC-02.13 - Executor MUST run claude command."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.__aiter__ = lambda self: aiter([])
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc

            result = await executor.run(prompt="Fix bug", working_dir="/tmp")

            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert "claude" in call_args[0]

    @pytest.mark.asyncio
    async def test_executor_collects_results(self, executor):
        """@trace SPEC-02.13 - Executor MUST collect and return results."""
        output_lines = [
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Working on it"}]}}),
            json.dumps({"type": "result", "result": "Done", "cost_usd": 0.10}),
        ]

        async def aiter_lines():
            for line in output_lines:
                yield (line + "\n").encode()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.__aiter__ = lambda self: aiter_lines()
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc

            result = await executor.run(prompt="Task", working_dir="/tmp")

            assert result.success is True
            assert result.result == "Done"
            assert result.cost_usd == 0.10


async def aiter(items):
    """Helper to create async iterator."""
    for item in items:
        yield item
