"""Tests for heartbeat monitoring.

Tests [SPEC-03.12] Heartbeat Monitoring.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.heartbeat import (
    HeartbeatEvent,
    HeartbeatInfo,
    HeartbeatMonitor,
    HeartbeatSender,
    HeartbeatState,
)


# =============================================================================
# HeartbeatInfo Tests
# =============================================================================


class TestHeartbeatInfo:
    """Tests for HeartbeatInfo dataclass."""

    def test_creation(self):
        """HeartbeatInfo MUST track session heartbeat status."""
        info = HeartbeatInfo(
            session_id="session-1",
            last_heartbeat=datetime.now(),
        )

        assert info.session_id == "session-1"
        assert info.missed_count == 0
        assert info.state == HeartbeatState.HEALTHY
        assert info.is_healthy

    def test_time_since_heartbeat(self):
        """HeartbeatInfo MUST calculate time since last heartbeat."""
        past = datetime.now() - timedelta(seconds=60)
        info = HeartbeatInfo(
            session_id="session-1",
            last_heartbeat=past,
        )

        time_since = info.time_since_heartbeat

        assert time_since.total_seconds() >= 60

    def test_is_healthy_states(self):
        """is_healthy MUST return True only for HEALTHY state."""
        info = HeartbeatInfo(
            session_id="session-1",
            last_heartbeat=datetime.now(),
        )

        assert info.is_healthy

        info.state = HeartbeatState.WARNING
        assert not info.is_healthy

        info.state = HeartbeatState.CRITICAL
        assert not info.is_healthy

        info.state = HeartbeatState.DEAD
        assert not info.is_healthy


# =============================================================================
# HeartbeatEvent Tests
# =============================================================================


class TestHeartbeatEvent:
    """Tests for HeartbeatEvent dataclass."""

    def test_creation(self):
        """HeartbeatEvent MUST capture state transitions."""
        event = HeartbeatEvent(
            session_id="session-1",
            previous_state=HeartbeatState.HEALTHY,
            new_state=HeartbeatState.WARNING,
            missed_count=1,
        )

        assert event.session_id == "session-1"
        assert event.previous_state == HeartbeatState.HEALTHY
        assert event.new_state == HeartbeatState.WARNING
        assert event.missed_count == 1
        assert event.timestamp is not None


# =============================================================================
# HeartbeatMonitor Tests
# =============================================================================


class TestHeartbeatMonitor:
    """Tests for HeartbeatMonitor class."""

    def test_initialization_defaults(self):
        """HeartbeatMonitor MUST use default interval and threshold."""
        monitor = HeartbeatMonitor()

        assert monitor.interval == 30.0
        assert monitor.missed_threshold == 3

    def test_initialization_custom(self):
        """HeartbeatMonitor MUST accept custom configuration."""
        monitor = HeartbeatMonitor(
            interval=10.0,
            missed_threshold=5,
        )

        assert monitor.interval == 10.0
        assert monitor.missed_threshold == 5

    def test_register_session(self):
        """register_session MUST add session to monitoring."""
        monitor = HeartbeatMonitor()

        info = monitor.register_session("session-1", {"env": "test"})

        assert info.session_id == "session-1"
        assert info.metadata["env"] == "test"
        assert monitor.get_session_info("session-1") is not None

    def test_unregister_session(self):
        """unregister_session MUST remove session from monitoring."""
        monitor = HeartbeatMonitor()
        monitor.register_session("session-1")

        removed = monitor.unregister_session("session-1")

        assert removed is not None
        assert removed.session_id == "session-1"
        assert monitor.get_session_info("session-1") is None

    def test_unregister_nonexistent(self):
        """unregister_session MUST return None for unknown session."""
        monitor = HeartbeatMonitor()

        result = monitor.unregister_session("unknown")

        assert result is None

    def test_record_heartbeat(self):
        """record_heartbeat MUST update session state."""
        monitor = HeartbeatMonitor()
        monitor.register_session("session-1")

        # Wait a bit to ensure time passes
        info = monitor.record_heartbeat("session-1", {"status": "running"})

        assert info is not None
        assert info.missed_count == 0
        assert info.state == HeartbeatState.HEALTHY
        assert info.metadata["status"] == "running"

    def test_record_heartbeat_unknown_session(self):
        """record_heartbeat MUST return None for unknown session."""
        monitor = HeartbeatMonitor()

        result = monitor.record_heartbeat("unknown")

        assert result is None

    def test_get_all_sessions(self):
        """get_all_sessions MUST return all monitored sessions."""
        monitor = HeartbeatMonitor()
        monitor.register_session("session-1")
        monitor.register_session("session-2")
        monitor.register_session("session-3")

        sessions = monitor.get_all_sessions()

        assert len(sessions) == 3
        session_ids = {s.session_id for s in sessions}
        assert session_ids == {"session-1", "session-2", "session-3"}

    def test_get_unhealthy_sessions(self):
        """get_unhealthy_sessions MUST filter unhealthy sessions."""
        monitor = HeartbeatMonitor()
        info1 = monitor.register_session("session-1")
        info2 = monitor.register_session("session-2")
        monitor.register_session("session-3")

        info1.state = HeartbeatState.WARNING
        info2.state = HeartbeatState.CRITICAL

        unhealthy = monitor.get_unhealthy_sessions()

        assert len(unhealthy) == 2
        session_ids = {s.session_id for s in unhealthy}
        assert session_ids == {"session-1", "session-2"}

    @pytest.mark.asyncio
    async def test_check_heartbeats_detects_missed(self):
        """check_heartbeats MUST detect missed heartbeats."""
        monitor = HeartbeatMonitor(interval=1.0, missed_threshold=2)
        monitor.register_session("session-1")

        # Simulate time passing (manually set last_heartbeat in the past)
        info = monitor.get_session_info("session-1")
        info.last_heartbeat = datetime.now() - timedelta(seconds=3)

        events = await monitor.check_heartbeats()

        # Should detect missed heartbeats
        assert len(events) >= 1
        assert info.state in [HeartbeatState.WARNING, HeartbeatState.CRITICAL]

    @pytest.mark.asyncio
    async def test_check_heartbeats_state_transitions(self):
        """check_heartbeats MUST transition through states."""
        monitor = HeartbeatMonitor(interval=1.0, missed_threshold=2)
        monitor.register_session("session-1")
        info = monitor.get_session_info("session-1")

        # Miss 1 heartbeat -> WARNING
        info.last_heartbeat = datetime.now() - timedelta(seconds=1.5)
        await monitor.check_heartbeats()
        assert info.state == HeartbeatState.WARNING

        # Miss threshold heartbeats -> CRITICAL
        info.last_heartbeat = datetime.now() - timedelta(seconds=3.0)
        await monitor.check_heartbeats()
        assert info.state == HeartbeatState.CRITICAL

        # Miss more -> DEAD
        info.last_heartbeat = datetime.now() - timedelta(seconds=5.0)
        await monitor.check_heartbeats()
        assert info.state == HeartbeatState.DEAD

    @pytest.mark.asyncio
    async def test_callbacks_called_on_state_change(self):
        """HeartbeatMonitor MUST call callbacks on state changes."""
        warning_callback = AsyncMock()
        critical_callback = AsyncMock()
        dead_callback = AsyncMock()

        monitor = HeartbeatMonitor(
            interval=1.0,
            missed_threshold=2,
            on_warning=warning_callback,
            on_critical=critical_callback,
            on_dead=dead_callback,
        )
        monitor.register_session("session-1")
        info = monitor.get_session_info("session-1")

        # Trigger WARNING
        info.last_heartbeat = datetime.now() - timedelta(seconds=1.5)
        await monitor.check_heartbeats()
        warning_callback.assert_called_once()

        # Trigger CRITICAL
        info.last_heartbeat = datetime.now() - timedelta(seconds=3.0)
        await monitor.check_heartbeats()
        critical_callback.assert_called_once()

        # Trigger DEAD
        info.last_heartbeat = datetime.now() - timedelta(seconds=5.0)
        await monitor.check_heartbeats()
        dead_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovery_callback(self):
        """HeartbeatMonitor MUST call recovery callback on recovery."""
        recovery_callback = AsyncMock()

        monitor = HeartbeatMonitor(
            interval=1.0,
            missed_threshold=2,
            on_recovery=recovery_callback,
        )
        monitor.register_session("session-1")
        info = monitor.get_session_info("session-1")

        # Set to WARNING state
        info.state = HeartbeatState.WARNING

        # Record heartbeat (triggers recovery)
        monitor.record_heartbeat("session-1")

        # Give callback time to execute
        await asyncio.sleep(0.1)

        recovery_callback.assert_called_once()
        assert info.state == HeartbeatState.HEALTHY

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """HeartbeatMonitor MUST start and stop background task."""
        monitor = HeartbeatMonitor(interval=0.1)

        assert not monitor.is_running

        await monitor.start()
        assert monitor.is_running

        await monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_monitor_loop_checks_periodically(self):
        """HeartbeatMonitor MUST check heartbeats periodically when running."""
        monitor = HeartbeatMonitor(interval=0.1)
        monitor.register_session("session-1")
        info = monitor.get_session_info("session-1")

        # Set old heartbeat
        info.last_heartbeat = datetime.now() - timedelta(seconds=1)

        await monitor.start()
        await asyncio.sleep(0.25)  # Wait for a couple checks
        await monitor.stop()

        # State should have changed due to missed heartbeats
        assert info.state != HeartbeatState.HEALTHY


# =============================================================================
# HeartbeatSender Tests
# =============================================================================


class TestHeartbeatSender:
    """Tests for HeartbeatSender class."""

    def test_initialization(self):
        """HeartbeatSender MUST initialize with session ID."""
        sender = HeartbeatSender(session_id="session-1", interval=10.0)

        assert sender.session_id == "session-1"
        assert sender.interval == 10.0

    def test_set_metadata(self):
        """HeartbeatSender MUST track metadata."""
        sender = HeartbeatSender(session_id="session-1")

        sender.set_metadata("status", "running")
        sender.set_metadata("progress", 50)

        assert sender._metadata["status"] == "running"
        assert sender._metadata["progress"] == 50

    @pytest.mark.asyncio
    async def test_send_heartbeat_with_custom_func(self):
        """HeartbeatSender MUST use custom send function."""
        send_mock = AsyncMock()
        sender = HeartbeatSender(
            session_id="session-1",
            send_func=send_mock,
        )
        sender.set_metadata("status", "active")

        await sender.send_heartbeat()

        send_mock.assert_called_once()
        call_args = send_mock.call_args
        assert call_args[0][0] == "session-1"
        assert "status" in call_args[0][1]
        assert "timestamp" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """HeartbeatSender MUST start and stop background task."""
        send_mock = AsyncMock()
        sender = HeartbeatSender(
            session_id="session-1",
            interval=0.1,
            send_func=send_mock,
        )

        assert not sender.is_running

        await sender.start()
        assert sender.is_running

        await asyncio.sleep(0.25)
        await sender.stop()

        assert not sender.is_running
        # Should have sent multiple heartbeats
        assert send_mock.call_count >= 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestHeartbeatIntegration:
    """Integration tests for heartbeat system."""

    @pytest.mark.asyncio
    async def test_sender_monitor_integration(self):
        """HeartbeatSender and HeartbeatMonitor MUST work together."""
        monitor = HeartbeatMonitor(interval=0.1, missed_threshold=2)
        monitor.register_session("session-1")

        # Create sender that records heartbeat to monitor
        async def send_to_monitor(session_id: str, metadata: dict):
            monitor.record_heartbeat(session_id, metadata)

        sender = HeartbeatSender(
            session_id="session-1",
            interval=0.1,
            send_func=send_to_monitor,
        )

        # Start both
        await monitor.start()
        await sender.start()

        # Let them run
        await asyncio.sleep(0.3)

        # Check state is healthy
        info = monitor.get_session_info("session-1")
        assert info.state == HeartbeatState.HEALTHY

        # Stop sender, monitor should detect missed heartbeats
        await sender.stop()
        await asyncio.sleep(0.4)

        # State should degrade
        assert info.state != HeartbeatState.HEALTHY

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_failure_detection_workflow(self):
        """Full failure detection workflow MUST trigger callbacks."""
        events_received = []

        async def on_warning(event: HeartbeatEvent):
            events_received.append(("warning", event))

        async def on_critical(event: HeartbeatEvent):
            events_received.append(("critical", event))

        monitor = HeartbeatMonitor(
            interval=0.1,
            missed_threshold=2,
            on_warning=on_warning,
            on_critical=on_critical,
        )
        monitor.register_session("session-1")

        # Simulate container failure by not sending heartbeats
        await monitor.start()
        await asyncio.sleep(0.5)  # Let monitor run and detect failures
        await monitor.stop()

        # Should have received warning and critical events
        assert len(events_received) >= 2
        event_types = [e[0] for e in events_received]
        assert "warning" in event_types
        assert "critical" in event_types


# =============================================================================
# HeartbeatState Tests
# =============================================================================


class TestHeartbeatState:
    """Tests for HeartbeatState enum."""

    def test_state_values(self):
        """HeartbeatState MUST define all states."""
        assert HeartbeatState.HEALTHY.value == "healthy"
        assert HeartbeatState.WARNING.value == "warning"
        assert HeartbeatState.CRITICAL.value == "critical"
        assert HeartbeatState.DEAD.value == "dead"
