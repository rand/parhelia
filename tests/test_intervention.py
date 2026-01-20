"""Tests for Parhelia human intervention signaling.

Tests intervention detection, notification, and management per SPEC-02.17.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from parhelia.intervention import (
    InterventionAction,
    InterventionConfig,
    InterventionManager,
    InterventionReason,
    InterventionRequest,
    InterventionState,
    NotificationDispatcher,
    OutputMonitor,
    TimeoutMonitor,
    cli_notification_handler,
)


# =============================================================================
# InterventionRequest Tests
# =============================================================================


class TestInterventionRequest:
    """Tests for InterventionRequest data class."""

    def test_request_creation(self):
        """InterventionRequest MUST initialize with required fields."""
        request = InterventionRequest(
            id="int-001",
            session_id="session-1",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Need help with database operation",
        )

        assert request.id == "int-001"
        assert request.session_id == "session-1"
        assert request.reason == InterventionReason.CLAUDE_REQUESTED
        assert request.state == InterventionState.PENDING

    def test_request_timestamps(self):
        """InterventionRequest MUST track timestamps."""
        before = datetime.now()
        request = InterventionRequest(
            id="int-001",
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Session idle",
        )
        after = datetime.now()

        assert before <= request.created_at <= after
        assert request.acknowledged_at is None
        assert request.resolved_at is None

    def test_request_with_actions(self):
        """InterventionRequest MUST support actions."""
        actions = [
            InterventionAction(label="Attach", url="parhelia://attach/session-1"),
            InterventionAction(label="Dismiss", url="parhelia://dismiss/int-001"),
        ]

        request = InterventionRequest(
            id="int-001",
            session_id="session-1",
            reason=InterventionReason.PERMISSION,
            context="Need approval",
            actions=actions,
        )

        assert len(request.actions) == 2
        assert request.actions[0].label == "Attach"


# =============================================================================
# InterventionConfig Tests
# =============================================================================


class TestInterventionConfig:
    """Tests for InterventionConfig class."""

    def test_default_config(self):
        """InterventionConfig MUST have sensible defaults."""
        config = InterventionConfig()

        assert config.timeout_minutes == 10.0
        assert config.check_interval_seconds == 60.0
        assert config.auto_checkpoint_on_intervention is True

    def test_custom_config(self):
        """InterventionConfig MUST accept custom values."""
        config = InterventionConfig(
            timeout_minutes=5.0,
            check_interval_seconds=30.0,
        )

        assert config.timeout_minutes == 5.0
        assert config.check_interval_seconds == 30.0


# =============================================================================
# InterventionManager Tests
# =============================================================================


class TestInterventionManager:
    """Tests for InterventionManager class."""

    @pytest.fixture
    def manager(self):
        """Create an intervention manager for testing."""
        return InterventionManager()

    @pytest.mark.asyncio
    async def test_create_intervention(self, manager):
        """create_intervention MUST create request."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Need help",
        )

        assert request.session_id == "session-1"
        assert request.reason == InterventionReason.CLAUDE_REQUESTED
        assert request.state == InterventionState.PENDING

    @pytest.mark.asyncio
    async def test_create_intervention_with_suggested_action(self, manager):
        """create_intervention MUST include suggested action."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.PERMISSION,
            context="Need database access",
            suggested_action="Grant read permission",
        )

        assert request.suggested_action == "Grant read permission"

    @pytest.mark.asyncio
    async def test_create_intervention_generates_id(self, manager):
        """create_intervention MUST generate unique IDs."""
        request1 = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )
        request2 = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Still idle",
        )

        assert request1.id != request2.id

    @pytest.mark.asyncio
    async def test_create_intervention_includes_default_actions(self, manager):
        """create_intervention MUST include default actions."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Need help",
        )

        labels = [a.label for a in request.actions]
        assert "Attach" in labels
        assert "Dismiss" in labels

    @pytest.mark.asyncio
    async def test_acknowledge(self, manager):
        """acknowledge MUST update request state."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )

        updated = await manager.acknowledge(request.id)

        assert updated.state == InterventionState.ACKNOWLEDGED
        assert updated.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_not_found(self, manager):
        """acknowledge MUST raise for unknown request."""
        with pytest.raises(ValueError, match="not found"):
            await manager.acknowledge("nonexistent")

    @pytest.mark.asyncio
    async def test_resolve(self, manager):
        """resolve MUST update request state."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Need help",
        )

        updated = await manager.resolve(
            request.id,
            response="Issue fixed",
            responded_by="user@example.com",
        )

        assert updated.state == InterventionState.RESOLVED
        assert updated.resolved_at is not None
        assert updated.response == "Issue fixed"
        assert updated.responded_by == "user@example.com"

    @pytest.mark.asyncio
    async def test_dismiss(self, manager):
        """dismiss MUST update request state."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )

        updated = await manager.dismiss(request.id)

        assert updated.state == InterventionState.DISMISSED
        assert updated.resolved_at is not None

    @pytest.mark.asyncio
    async def test_get_request(self, manager):
        """get_request MUST return existing request."""
        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.ERROR,
            context="Error occurred",
        )

        result = await manager.get_request(request.id)

        assert result is request

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, manager):
        """get_request MUST return None for unknown."""
        result = await manager.get_request("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_requests(self, manager):
        """get_session_requests MUST return session's requests."""
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle 1",
        )
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle 2",
        )
        await manager.create_intervention(
            session_id="session-2",
            reason=InterventionReason.TIMEOUT,
            context="Other session",
        )

        requests = await manager.get_session_requests("session-1")

        assert len(requests) == 2

    @pytest.mark.asyncio
    async def test_get_session_requests_with_state_filter(self, manager):
        """get_session_requests MUST filter by state."""
        request1 = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle 1",
        )
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle 2",
        )

        await manager.resolve(request1.id)

        pending = await manager.get_session_requests(
            "session-1",
            state=InterventionState.PENDING,
        )

        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, manager):
        """get_pending_requests MUST return all pending."""
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )
        request2 = await manager.create_intervention(
            session_id="session-2",
            reason=InterventionReason.ERROR,
            context="Error",
        )

        await manager.resolve(request2.id)

        pending = await manager.get_pending_requests()

        assert len(pending) == 1
        assert pending[0].session_id == "session-1"

    @pytest.mark.asyncio
    async def test_has_pending_intervention(self, manager):
        """has_pending_intervention MUST return True when pending."""
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )

        assert await manager.has_pending_intervention("session-1") is True
        assert await manager.has_pending_intervention("session-2") is False

    @pytest.mark.asyncio
    async def test_on_intervention_callback(self, manager):
        """on_intervention callback MUST be called."""
        callback_called = False
        callback_request = None

        async def on_intervention(request):
            nonlocal callback_called, callback_request
            callback_called = True
            callback_request = request

        manager.on_intervention(on_intervention)

        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Need help",
        )

        assert callback_called is True
        assert callback_request is request

    @pytest.mark.asyncio
    async def test_on_resolved_callback(self, manager):
        """on_resolved callback MUST be called."""
        callback_called = False

        async def on_resolved(request):
            nonlocal callback_called
            callback_called = True

        manager.on_resolved(on_resolved)

        request = await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )
        await manager.resolve(request.id)

        assert callback_called is True

    @pytest.mark.asyncio
    async def test_notification_handler(self, manager):
        """notification_handler MUST be called on create."""
        handler_called = False
        handler_title = None

        async def handler(title, body, actions):
            nonlocal handler_called, handler_title
            handler_called = True
            handler_title = title

        manager.set_notification_handler(handler)

        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Need help",
        )

        assert handler_called is True
        assert "session-1" in handler_title


# =============================================================================
# OutputMonitor Tests
# =============================================================================


class TestOutputMonitor:
    """Tests for OutputMonitor class."""

    @pytest.fixture
    def manager(self):
        """Create an intervention manager for testing."""
        return InterventionManager()

    @pytest.fixture
    def monitor(self, manager):
        """Create an output monitor for testing."""
        return OutputMonitor(manager)

    @pytest.mark.asyncio
    async def test_process_line_needs_human(self, monitor):
        """process_line MUST detect needs_human signal."""
        line = json.dumps({
            "type": "needs_human",
            "reason": "Permission denied",
            "context": "Cannot access database",
            "suggested_action": "Grant permission",
        })

        request = await monitor.process_line("session-1", line)

        assert request is not None
        assert request.reason == InterventionReason.CLAUDE_REQUESTED
        assert "database" in request.context

    @pytest.mark.asyncio
    async def test_process_line_needs_human_no_spaces(self, monitor):
        """process_line MUST detect needs_human without spaces."""
        line = '{"type":"needs_human","context":"Help needed"}'

        request = await monitor.process_line("session-1", line)

        assert request is not None

    @pytest.mark.asyncio
    async def test_process_line_no_signal(self, monitor):
        """process_line MUST return None for normal output."""
        line = '{"type": "message", "content": "Hello"}'

        request = await monitor.process_line("session-1", line)

        assert request is None

    @pytest.mark.asyncio
    async def test_process_line_invalid_json(self, monitor):
        """process_line MUST handle invalid JSON gracefully."""
        line = "This is not JSON but contains \"type\": \"needs_human\""

        request = await monitor.process_line("session-1", line)

        assert request is None

    @pytest.mark.asyncio
    async def test_process_line_extracts_metadata(self, monitor):
        """process_line MUST extract metadata from event."""
        line = json.dumps({
            "type": "needs_human",
            "reason": "Approval needed",
            "context": "Destructive operation",
            "suggested_action": "Review and approve",
        })

        request = await monitor.process_line("session-1", line)

        assert request.metadata.get("reason_detail") == "Approval needed"
        assert request.suggested_action == "Review and approve"


# =============================================================================
# TimeoutMonitor Tests
# =============================================================================


class TestTimeoutMonitor:
    """Tests for TimeoutMonitor class."""

    @pytest.fixture
    def manager(self):
        """Create an intervention manager for testing."""
        return InterventionManager()

    @pytest.fixture
    def monitor(self, manager):
        """Create a timeout monitor for testing."""
        config = InterventionConfig(timeout_minutes=1.0)
        return TimeoutMonitor(manager, config)

    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor):
        """start_monitoring MUST add session."""
        await monitor.start_monitoring("session-1")

        assert "session-1" in monitor._monitoring

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """stop_monitoring MUST remove session."""
        await monitor.start_monitoring("session-1")
        await monitor.stop_monitoring("session-1")

        assert "session-1" not in monitor._monitoring

    @pytest.mark.asyncio
    async def test_record_activity(self, monitor):
        """record_activity MUST update last activity."""
        await monitor.start_monitoring("session-1")
        old_time = monitor._monitoring["session-1"]

        await asyncio.sleep(0.01)
        await monitor.record_activity("session-1")

        new_time = monitor._monitoring["session-1"]
        assert new_time > old_time

    @pytest.mark.asyncio
    async def test_check_timeouts_no_timeout(self, monitor):
        """check_timeouts MUST return empty when not timed out."""
        await monitor.start_monitoring("session-1")

        interventions = await monitor.check_timeouts()

        assert len(interventions) == 0

    @pytest.mark.asyncio
    async def test_check_timeouts_with_timeout(self, monitor):
        """check_timeouts MUST create intervention when timed out."""
        await monitor.start_monitoring("session-1")
        # Set last activity to past timeout
        monitor._monitoring["session-1"] = datetime.now() - timedelta(minutes=5)

        interventions = await monitor.check_timeouts()

        assert len(interventions) == 1
        assert interventions[0].reason == InterventionReason.TIMEOUT

    @pytest.mark.asyncio
    async def test_check_timeouts_no_duplicate(self, monitor):
        """check_timeouts MUST not create duplicate interventions."""
        await monitor.start_monitoring("session-1")
        monitor._monitoring["session-1"] = datetime.now() - timedelta(minutes=5)

        # First check creates intervention
        interventions1 = await monitor.check_timeouts()
        assert len(interventions1) == 1

        # Second check should not create another (already pending)
        interventions2 = await monitor.check_timeouts()
        assert len(interventions2) == 0

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """start/stop MUST manage monitoring loop."""
        await monitor.start()
        assert monitor._running is True
        assert monitor._task is not None

        await monitor.stop()
        assert monitor._running is False


# =============================================================================
# NotificationDispatcher Tests
# =============================================================================


class TestNotificationDispatcher:
    """Tests for NotificationDispatcher class."""

    @pytest.fixture
    def dispatcher(self):
        """Create a notification dispatcher for testing."""
        return NotificationDispatcher()

    @pytest.mark.asyncio
    async def test_register_handler(self, dispatcher):
        """register_handler MUST add handler."""
        async def handler(title, body, actions):
            pass

        dispatcher.register_handler("cli", handler)

        assert "cli" in dispatcher._handlers

    @pytest.mark.asyncio
    async def test_send_to_channel(self, dispatcher):
        """send MUST call channel handler."""
        handler_called = False

        async def handler(title, body, actions):
            nonlocal handler_called
            handler_called = True

        dispatcher.register_handler("cli", handler)

        await dispatcher.send("Title", "Body", [])

        assert handler_called is True

    @pytest.mark.asyncio
    async def test_send_to_specific_channels(self, dispatcher):
        """send MUST call only specified channels."""
        cli_called = False
        webhook_called = False

        async def cli_handler(title, body, actions):
            nonlocal cli_called
            cli_called = True

        async def webhook_handler(title, body, actions):
            nonlocal webhook_called
            webhook_called = True

        dispatcher.register_handler("cli", cli_handler)
        dispatcher.register_handler("webhook", webhook_handler)

        await dispatcher.send("Title", "Body", [], channels=["cli"])

        assert cli_called is True
        assert webhook_called is False

    @pytest.mark.asyncio
    async def test_send_handles_errors(self, dispatcher):
        """send MUST continue on handler errors."""
        async def failing_handler(title, body, actions):
            raise RuntimeError("Handler failed")

        success_called = False

        async def success_handler(title, body, actions):
            nonlocal success_called
            success_called = True

        dispatcher.register_handler("failing", failing_handler)
        dispatcher.register_handler("success", success_handler)

        # Should not raise
        await dispatcher.send("Title", "Body", [])

        assert success_called is True


# =============================================================================
# CLI Notification Handler Tests
# =============================================================================


class TestCLINotificationHandler:
    """Tests for cli_notification_handler function."""

    @pytest.mark.asyncio
    async def test_cli_handler_prints(self, capsys):
        """cli_notification_handler MUST print notification."""
        actions = [
            InterventionAction(label="Attach", url="parhelia://attach/session-1"),
        ]

        await cli_notification_handler(
            title="Test Title",
            body="Test body content",
            actions=actions,
        )

        captured = capsys.readouterr()
        assert "Test Title" in captured.out
        assert "Test body content" in captured.out
        assert "Attach" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


class TestInterventionIntegration:
    """Integration tests for intervention system."""

    @pytest.mark.asyncio
    async def test_full_intervention_flow(self):
        """Full intervention flow MUST work correctly."""
        manager = InterventionManager()
        monitor = OutputMonitor(manager)

        # Simulate Claude output with needs_human signal
        line = json.dumps({
            "type": "needs_human",
            "reason": "Permission denied",
            "context": "Cannot drop database table",
            "suggested_action": "Review and approve",
        })

        # Detect intervention
        request = await monitor.process_line("session-1", line)
        assert request is not None
        assert request.state == InterventionState.PENDING

        # Acknowledge
        await manager.acknowledge(request.id)
        assert request.state == InterventionState.ACKNOWLEDGED

        # Resolve
        await manager.resolve(request.id, response="Approved")
        assert request.state == InterventionState.RESOLVED

    @pytest.mark.asyncio
    async def test_timeout_intervention_flow(self):
        """Timeout intervention flow MUST work correctly."""
        manager = InterventionManager()
        config = InterventionConfig(timeout_minutes=0.01)  # Very short for test
        timeout_monitor = TimeoutMonitor(manager, config)

        # Start monitoring
        await timeout_monitor.start_monitoring("session-1")

        # Simulate time passing
        timeout_monitor._monitoring["session-1"] = datetime.now() - timedelta(minutes=1)

        # Check for timeouts
        interventions = await timeout_monitor.check_timeouts()

        assert len(interventions) == 1
        assert interventions[0].reason == InterventionReason.TIMEOUT

    @pytest.mark.asyncio
    async def test_multiple_sessions_interventions(self):
        """Manager MUST handle multiple sessions."""
        manager = InterventionManager()

        # Create interventions for multiple sessions
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.TIMEOUT,
            context="Idle",
        )
        await manager.create_intervention(
            session_id="session-2",
            reason=InterventionReason.CLAUDE_REQUESTED,
            context="Help",
        )
        await manager.create_intervention(
            session_id="session-1",
            reason=InterventionReason.ERROR,
            context="Error",
        )

        session1_requests = await manager.get_session_requests("session-1")
        session2_requests = await manager.get_session_requests("session-2")

        assert len(session1_requests) == 2
        assert len(session2_requests) == 1
