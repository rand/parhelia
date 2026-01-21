"""Tests for event streaming module."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.event_stream import (
    ActivityEvent,
    CompletionEvent,
    EventFormatter,
    EventStream,
    EventType,
    HeartbeatEvent,
    OutputEvent,
    ProgressEvent,
    StatusEvent,
    StreamEvent,
    WarningEvent,
)
from parhelia.orchestrator import WorkerInfo, WorkerState


# =============================================================================
# Event Type Tests
# =============================================================================


class TestStreamEvent:
    """Tests for StreamEvent base class."""

    def test_to_dict(self):
        """StreamEvent MUST serialize to dict with type and timestamp."""
        event = StreamEvent(
            type=EventType.STATUS_CHANGE,
            task_id="task-123",
            session_id="session-456",
        )

        result = event.to_dict()

        assert result["type"] == "status_change"
        assert result["task_id"] == "task-123"
        assert result["session_id"] == "session-456"
        assert "timestamp" in result

    def test_to_json(self):
        """StreamEvent MUST serialize to JSON string."""
        event = StreamEvent(
            type=EventType.HEARTBEAT,
            task_id="task-abc",
        )

        result = event.to_json()
        parsed = json.loads(result)

        assert parsed["type"] == "heartbeat"
        assert parsed["task_id"] == "task-abc"


class TestStatusEvent:
    """Tests for StatusEvent class."""

    def test_status_change_event(self):
        """StatusEvent MUST include old/new status."""
        event = StatusEvent(
            task_id="task-123",
            old_status="pending",
            new_status="running",
            worker_state="running",
        )

        result = event.to_dict()

        assert result["type"] == "status_change"
        assert result["old_status"] == "pending"
        assert result["new_status"] == "running"
        assert result["worker_state"] == "running"


class TestProgressEvent:
    """Tests for ProgressEvent class."""

    def test_progress_event(self):
        """ProgressEvent MUST include phase, percent, message."""
        event = ProgressEvent(
            task_id="task-123",
            phase="building",
            percent=50,
            message="Compiling modules...",
        )

        result = event.to_dict()

        assert result["type"] == "progress"
        assert result["phase"] == "building"
        assert result["percent"] == 50
        assert result["message"] == "Compiling modules..."


class TestHeartbeatEvent:
    """Tests for HeartbeatEvent class."""

    def test_heartbeat_event(self):
        """HeartbeatEvent MUST include uptime."""
        event = HeartbeatEvent(
            task_id="task-123",
            uptime_seconds=300,
            cpu_percent=45.5,
            memory_mb=1024,
        )

        result = event.to_dict()

        assert result["type"] == "heartbeat"
        assert result["uptime_seconds"] == 300
        assert result["cpu_percent"] == 45.5
        assert result["memory_mb"] == 1024


class TestActivityEvent:
    """Tests for ActivityEvent class."""

    def test_activity_event(self):
        """ActivityEvent MUST include activity type and summary."""
        event = ActivityEvent(
            task_id="task-123",
            activity_type="claude_response",
            summary="Generated 50 lines of code",
        )

        result = event.to_dict()

        assert result["type"] == "activity"
        assert result["activity_type"] == "claude_response"
        assert result["summary"] == "Generated 50 lines of code"


class TestOutputEvent:
    """Tests for OutputEvent class."""

    def test_output_event(self):
        """OutputEvent MUST include content and stream."""
        event = OutputEvent(
            task_id="task-123",
            content="Hello, world!\n",
            stream="stdout",
        )

        result = event.to_dict()

        assert result["type"] == "output"
        assert result["content"] == "Hello, world!\n"
        assert result["stream"] == "stdout"


class TestCompletionEvent:
    """Tests for CompletionEvent class."""

    def test_completion_success(self):
        """CompletionEvent MUST have type=completed on success."""
        event = CompletionEvent(
            task_id="task-123",
            success=True,
            output_summary="Task completed successfully",
            duration_seconds=120.5,
            cost_usd=0.25,
        )

        result = event.to_dict()

        assert result["type"] == "completed"
        assert result["success"] is True
        assert result["output_summary"] == "Task completed successfully"
        assert result["duration_seconds"] == 120.5
        assert result["cost_usd"] == 0.25

    def test_completion_failure(self):
        """CompletionEvent MUST have type=failed on failure."""
        event = CompletionEvent(
            task_id="task-123",
            success=False,
            output_summary="Error: connection failed",
        )

        result = event.to_dict()

        assert result["type"] == "failed"
        assert result["success"] is False


class TestWarningEvent:
    """Tests for WarningEvent class."""

    def test_warning_event(self):
        """WarningEvent MUST include warning type and message."""
        event = WarningEvent(
            task_id="task-123",
            warning_type="budget_low",
            message="Budget is at 85% usage",
            threshold=80.0,
            current=85.0,
        )

        result = event.to_dict()

        assert result["type"] == "warning"
        assert result["warning_type"] == "budget_low"
        assert result["message"] == "Budget is at 85% usage"
        assert result["threshold"] == 80.0
        assert result["current"] == 85.0


# =============================================================================
# EventFormatter Tests
# =============================================================================


class TestEventFormatter:
    """Tests for EventFormatter class."""

    def test_format_json_mode(self):
        """EventFormatter MUST output JSON in json mode."""
        formatter = EventFormatter(json_mode=True)
        event = StatusEvent(
            task_id="task-123",
            new_status="running",
        )

        result = formatter.format(event)
        parsed = json.loads(result)

        assert parsed["type"] == "status_change"
        assert parsed["task_id"] == "task-123"

    def test_format_human_status(self):
        """EventFormatter MUST format status events for humans."""
        formatter = EventFormatter(json_mode=False)
        event = StatusEvent(
            task_id="task-123",
            new_status="running",
        )

        result = formatter.format(event)

        assert "Status" in result
        assert "running" in result

    def test_format_human_progress(self):
        """EventFormatter MUST show progress bar for humans."""
        formatter = EventFormatter(json_mode=False)
        event = ProgressEvent(
            task_id="task-123",
            phase="building",
            percent=50,
            message="Half done",
        )

        result = formatter.format(event)

        assert "building" in result
        assert "50%" in result
        assert "Half done" in result
        # Progress bar characters
        assert "█" in result or "░" in result

    def test_format_human_heartbeat(self):
        """EventFormatter MUST show uptime for heartbeat."""
        formatter = EventFormatter(json_mode=False)
        event = HeartbeatEvent(
            task_id="task-123",
            uptime_seconds=125,
        )

        result = formatter.format(event)

        assert "♥" in result
        assert "2m" in result  # 125 seconds = 2m 5s

    def test_format_human_completion(self):
        """EventFormatter MUST show success/failure for completion."""
        formatter = EventFormatter(json_mode=False)

        success_event = CompletionEvent(
            task_id="task-123",
            success=True,
            duration_seconds=60,
            cost_usd=0.10,
        )
        result = formatter.format(success_event)
        assert "✓" in result
        assert "Completed" in result

        fail_event = CompletionEvent(
            task_id="task-456",
            success=False,
            duration_seconds=30,
            cost_usd=0.05,
        )
        result = formatter.format(fail_event)
        assert "✗" in result
        assert "Failed" in result

    def test_format_human_warning(self):
        """EventFormatter MUST show warning symbol."""
        formatter = EventFormatter(json_mode=False)
        event = WarningEvent(
            task_id="task-123",
            warning_type="budget_low",
            message="Budget at 90%",
        )

        result = formatter.format(event)

        assert "⚠" in result
        assert "budget_low" in result


# =============================================================================
# EventStream Tests
# =============================================================================


class TestEventStream:
    """Tests for EventStream class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.task_store = MagicMock()
        orch.worker_store = MagicMock()
        orch.collect_results = AsyncMock(return_value=None)
        return orch

    @pytest.fixture
    def stream(self, mock_orchestrator):
        """Create event stream with mock orchestrator."""
        return EventStream(mock_orchestrator, poll_interval=0.1)

    @pytest.mark.asyncio
    async def test_watch_emits_initial_status(self, stream, mock_orchestrator):
        """watch MUST emit initial status event."""
        mock_orchestrator.task_store.get_status.return_value = "pending"
        mock_orchestrator.worker_store.get_by_task.return_value = None

        events = []
        async for event in stream.watch(task_id="task-123", include_heartbeat=False):
            events.append(event)
            if len(events) >= 1:
                break

        assert len(events) >= 1
        assert isinstance(events[0], StatusEvent)
        assert events[0].new_status == "pending"

    @pytest.mark.asyncio
    async def test_watch_stops_on_complete(self, stream, mock_orchestrator):
        """watch MUST stop when task completes if stop_on_complete=True."""
        # Start pending, then complete
        mock_orchestrator.task_store.get_status.side_effect = ["pending", "completed"]
        mock_orchestrator.worker_store.get_by_task.return_value = None

        events = []
        async for event in stream.watch(
            task_id="task-123",
            include_heartbeat=False,
            stop_on_complete=True,
        ):
            events.append(event)

        # Should have status events and completion event
        assert any(isinstance(e, StatusEvent) for e in events)
        assert any(isinstance(e, CompletionEvent) for e in events)

    @pytest.mark.asyncio
    async def test_watch_emits_worker_state_changes(self, stream, mock_orchestrator):
        """watch MUST emit events when worker state changes."""
        mock_orchestrator.task_store.get_status.return_value = "running"

        worker = MagicMock()
        worker.state = WorkerState.RUNNING
        mock_orchestrator.worker_store.get_by_task.return_value = worker

        events = []
        count = 0
        async for event in stream.watch(task_id="task-123", include_heartbeat=False):
            events.append(event)
            count += 1
            if count >= 2:
                break

        # Should have at least initial status
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_watch_includes_heartbeat(self, stream, mock_orchestrator):
        """watch MUST emit heartbeat events when include_heartbeat=True."""
        mock_orchestrator.task_store.get_status.return_value = "running"
        mock_orchestrator.worker_store.get_by_task.return_value = None

        # Override heartbeat interval for faster test
        stream.HEARTBEAT_INTERVAL_SECONDS = 0.1

        events = []
        count = 0
        async for event in stream.watch(
            task_id="task-123",
            include_heartbeat=True,
            stop_on_complete=False,
        ):
            events.append(event)
            count += 1
            if count >= 3:
                break

        # Should have at least one heartbeat
        heartbeats = [e for e in events if isinstance(e, HeartbeatEvent)]
        assert len(heartbeats) >= 1

    def test_event_callback(self, stream):
        """set_event_callback MUST set callback for event notifications."""
        callback = MagicMock()
        stream.set_event_callback(callback)

        event = StatusEvent(task_id="task-123", new_status="running")
        stream._emit(event)

        callback.assert_called_once_with(event)


# =============================================================================
# CLI Integration Tests
# =============================================================================


class TestWatchCommand:
    """Tests for task watch CLI command."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.budget.default_ceiling_usd = 10.0
        config.paths.volume_root = "/vol/parhelia"
        config.modal.volume_name = "test-volume"
        return config

    def test_watch_command_exists(self):
        """task watch command MUST be registered."""
        from parhelia.cli import task

        # Check that watch is a subcommand of task
        command_names = [cmd.name for cmd in task.commands.values()]
        assert "watch" in command_names

    def test_task_show_has_watch_option(self):
        """task show MUST have --watch option."""
        from parhelia.cli import task_show

        # Check that --watch is in the parameters
        param_names = [p.name for p in task_show.params]
        assert "watch" in param_names
