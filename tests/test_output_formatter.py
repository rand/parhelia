"""Tests for output formatter."""

from __future__ import annotations

import json

import pytest

from parhelia.output_formatter import (
    ErrorCode,
    NextAction,
    OutputFormatter,
    OutputMetadata,
    ProgressEvent,
    format_budget_exceeded,
    format_budget_status,
    format_session_not_found,
    format_task_submitted,
)


class TestOutputFormatter:
    """Tests for OutputFormatter class."""

    def test_success_json_mode(self):
        """Test success response in JSON mode."""
        formatter = OutputFormatter(json_mode=True)
        result = formatter.success(
            data={"task_id": "task-123"},
            message="Task created",
        )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["data"]["task_id"] == "task-123"
        assert "metadata" in parsed
        assert "next_actions" in parsed

    def test_success_human_mode(self):
        """Test success response in human mode."""
        formatter = OutputFormatter(json_mode=False)
        result = formatter.success(
            data={"task_id": "task-123", "status": "running"},
            message="Task created",
        )

        assert "\u2713 Task created" in result
        assert "task_id: task-123" in result
        assert "status: running" in result

    def test_success_with_next_actions(self):
        """Test success response with next actions."""
        formatter = OutputFormatter(json_mode=True)
        result = formatter.success(
            data={"task_id": "task-123"},
            next_actions=[
                NextAction(
                    action="attach",
                    description="Attach to session",
                    command="parhelia attach task-123",
                ),
            ],
        )

        parsed = json.loads(result)
        assert len(parsed["next_actions"]) == 1
        assert parsed["next_actions"][0]["action"] == "attach"
        assert parsed["next_actions"][0]["command"] == "parhelia attach task-123"

    def test_error_json_mode(self):
        """Test error response in JSON mode."""
        formatter = OutputFormatter(json_mode=True)
        result = formatter.error(
            code=ErrorCode.BUDGET_EXCEEDED,
            message="Budget exceeded",
            details={"remaining_usd": 2.50},
        )

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert parsed["error"]["code"] == "BUDGET_EXCEEDED"
        assert parsed["error"]["message"] == "Budget exceeded"
        assert parsed["error"]["details"]["remaining_usd"] == 2.50
        assert len(parsed["suggestions"]) > 0

    def test_error_human_mode(self):
        """Test error response in human mode."""
        formatter = OutputFormatter(json_mode=False)
        result = formatter.error(
            code=ErrorCode.SESSION_NOT_FOUND,
            message="Session not found: task-xyz",
        )

        assert "\u2717 Error: Session not found: task-xyz" in result
        assert "Suggestions:" in result

    def test_error_with_custom_suggestions(self):
        """Test error response with custom suggestions."""
        formatter = OutputFormatter(json_mode=True)
        result = formatter.error(
            code=ErrorCode.INTERNAL_ERROR,
            message="Something went wrong",
            suggestions=["Try again", "Contact support"],
        )

        parsed = json.loads(result)
        assert parsed["suggestions"] == ["Try again", "Contact support"]

    def test_progress_event(self):
        """Test progress event formatting."""
        formatter = OutputFormatter(json_mode=True)
        result = formatter.progress(
            phase="creating_sandbox",
            percent=30,
            message="Provisioning container...",
        )

        parsed = json.loads(result)
        assert parsed["type"] == "progress"
        assert parsed["phase"] == "creating_sandbox"
        assert parsed["percent"] == 30
        assert parsed["message"] == "Provisioning container..."

    def test_success_with_nested_data(self):
        """Test success response with nested data structures."""
        formatter = OutputFormatter(json_mode=False)
        result = formatter.success(
            data={
                "task_id": "task-123",
                "worker": {"id": "worker-456", "status": "running"},
                "tags": ["gpu", "ml", "training"],
            },
        )

        assert "worker:" in result
        assert "id: worker-456" in result
        assert "tags:" in result

    def test_success_with_metadata(self):
        """Test success response with metadata."""
        formatter = OutputFormatter(json_mode=True)
        metadata = OutputMetadata(duration_ms=1234, cost_usd=0.05)
        result = formatter.success(
            data={"status": "completed"},
            metadata=metadata,
        )

        parsed = json.loads(result)
        assert parsed["metadata"]["duration_ms"] == 1234
        assert parsed["metadata"]["cost_usd"] == 0.05
        assert "timestamp" in parsed["metadata"]


class TestConvenienceFunctions:
    """Tests for convenience formatting functions."""

    def test_format_task_submitted_json(self):
        """Test task submitted formatting in JSON mode."""
        result = format_task_submitted(
            task_id="task-abc123",
            worker_id="worker-def456",
            session_id="ph-task-abc123",
            json_mode=True,
        )

        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["data"]["task_id"] == "task-abc123"
        assert parsed["data"]["worker_id"] == "worker-def456"
        assert len(parsed["next_actions"]) == 3  # status, attach, logs

    def test_format_task_submitted_human(self):
        """Test task submitted formatting in human mode."""
        result = format_task_submitted(
            task_id="task-abc123",
            json_mode=False,
        )

        assert "Task submitted successfully" in result
        assert "task-abc123" in result

    def test_format_budget_status_json(self):
        """Test budget status formatting in JSON mode."""
        result = format_budget_status(
            ceiling_usd=10.0,
            used_usd=2.50,
            remaining_usd=7.50,
            usage_percent=25.0,
            task_count=5,
            json_mode=True,
        )

        parsed = json.loads(result)
        assert parsed["data"]["ceiling_usd"] == 10.0
        assert parsed["data"]["used_usd"] == 2.50
        assert parsed["data"]["usage_percent"] == 25.0

    def test_format_budget_status_with_warning(self):
        """Test budget status includes increase action when usage high."""
        result = format_budget_status(
            ceiling_usd=10.0,
            used_usd=8.50,
            remaining_usd=1.50,
            usage_percent=85.0,
            task_count=10,
            json_mode=True,
        )

        parsed = json.loads(result)
        actions = [a["action"] for a in parsed["next_actions"]]
        assert "increase" in actions

    def test_format_budget_exceeded_json(self):
        """Test budget exceeded error formatting."""
        result = format_budget_exceeded(
            requested_usd=5.0,
            remaining_usd=2.50,
            ceiling_usd=10.0,
            json_mode=True,
        )

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert parsed["error"]["code"] == "BUDGET_EXCEEDED"
        assert parsed["error"]["details"]["requested_usd"] == 5.0
        assert len(parsed["suggestions"]) > 0

    def test_format_session_not_found_json(self):
        """Test session not found error formatting."""
        result = format_session_not_found(
            session_id="task-xyz",
            json_mode=True,
        )

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert parsed["error"]["code"] == "SESSION_NOT_FOUND"
        assert "task-xyz" in parsed["error"]["message"]


class TestProgressEvent:
    """Tests for ProgressEvent class."""

    def test_to_dict(self):
        """Test ProgressEvent to_dict method."""
        event = ProgressEvent(
            phase="initializing",
            percent=50,
            message="Starting Claude Code...",
        )

        result = event.to_dict()
        assert result["type"] == "progress"
        assert result["phase"] == "initializing"
        assert result["percent"] == 50

    def test_to_json(self):
        """Test ProgressEvent to_json method."""
        event = ProgressEvent(
            phase="running",
            percent=75,
            message="Executing task...",
        )

        result = event.to_json()
        parsed = json.loads(result)
        assert parsed["phase"] == "running"


class TestNextAction:
    """Tests for NextAction class."""

    def test_to_dict(self):
        """Test NextAction to_dict method."""
        action = NextAction(
            action="attach",
            description="Attach to session",
            command="parhelia attach task-123",
        )

        result = action.to_dict()
        assert result["action"] == "attach"
        assert result["description"] == "Attach to session"
        assert result["command"] == "parhelia attach task-123"


class TestErrorSuggestions:
    """Tests for error code suggestions."""

    def test_default_suggestions_for_error_codes(self):
        """Test that all error codes have default suggestions."""
        formatter = OutputFormatter(json_mode=True)

        for code in ErrorCode:
            result = formatter.error(code=code, message="Test error")
            parsed = json.loads(result)
            assert len(parsed["suggestions"]) > 0, f"No suggestions for {code}"
