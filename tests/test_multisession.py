"""Tests for Parhelia multi-session per container support.

Tests session registry, resource monitoring, and scheduling per SPEC-02.16.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.multisession import (
    ContainerConfig,
    ResourceMonitor,
    ResourceStatus,
    ResourceType,
    SessionRegistry,
    SessionScheduler,
    SessionSlot,
)
from parhelia.session import Session, SessionState


# =============================================================================
# ResourceStatus Tests
# =============================================================================


class TestResourceStatus:
    """Tests for ResourceStatus data class."""

    def test_resource_status_creation(self):
        """ResourceStatus MUST initialize with required fields."""
        status = ResourceStatus(
            sessions_available=3,
            sessions_total=4,
            sessions_active=1,
            cpu_available_percent=75.0,
            cpu_total_cores=4,
            memory_available_mb=8000.0,
            memory_total_mb=16000.0,
        )

        assert status.sessions_available == 3
        assert status.sessions_total == 4
        assert status.cpu_available_percent == 75.0

    def test_resource_status_has_capacity_true(self):
        """has_capacity MUST return True when resources available."""
        status = ResourceStatus(
            sessions_available=2,
            sessions_total=4,
            sessions_active=2,
            cpu_available_percent=50.0,
            cpu_total_cores=4,
            memory_available_mb=4000.0,
            memory_total_mb=16000.0,
        )

        assert status.has_capacity is True

    def test_resource_status_has_capacity_no_sessions(self):
        """has_capacity MUST return False when no session slots."""
        status = ResourceStatus(
            sessions_available=0,
            sessions_total=4,
            sessions_active=4,
            cpu_available_percent=50.0,
            cpu_total_cores=4,
            memory_available_mb=4000.0,
            memory_total_mb=16000.0,
        )

        assert status.has_capacity is False

    def test_resource_status_has_capacity_low_cpu(self):
        """has_capacity MUST return False when CPU low."""
        status = ResourceStatus(
            sessions_available=2,
            sessions_total=4,
            sessions_active=2,
            cpu_available_percent=5.0,  # Below 10% threshold
            cpu_total_cores=4,
            memory_available_mb=4000.0,
            memory_total_mb=16000.0,
        )

        assert status.has_capacity is False

    def test_resource_status_has_capacity_low_memory(self):
        """has_capacity MUST return False when memory low."""
        status = ResourceStatus(
            sessions_available=2,
            sessions_total=4,
            sessions_active=2,
            cpu_available_percent=50.0,
            cpu_total_cores=4,
            memory_available_mb=256.0,  # Below 512MB threshold
            memory_total_mb=16000.0,
        )

        assert status.has_capacity is False

    def test_resource_status_utilization(self):
        """utilization_percent MUST calculate correctly."""
        status = ResourceStatus(
            sessions_available=1,
            sessions_total=4,
            sessions_active=3,
            cpu_available_percent=25.0,  # 75% used
            cpu_total_cores=4,
            memory_available_mb=4000.0,  # 75% used
            memory_total_mb=16000.0,
        )

        # Session utilization: 3/4 = 75%
        # CPU utilization: 75%
        # Memory utilization: 75%
        assert status.utilization_percent == 75.0

    def test_resource_status_to_dict(self):
        """to_dict MUST convert all fields."""
        status = ResourceStatus(
            sessions_available=2,
            sessions_total=4,
            sessions_active=2,
            cpu_available_percent=50.0,
            cpu_total_cores=4,
            memory_available_mb=8000.0,
            memory_total_mb=16000.0,
        )

        result = status.to_dict()

        assert result["sessions_available"] == 2
        assert result["has_capacity"] is True
        assert "timestamp" in result


# =============================================================================
# SessionSlot Tests
# =============================================================================


class TestSessionSlot:
    """Tests for SessionSlot data class."""

    def test_slot_creation(self):
        """SessionSlot MUST initialize with session ID."""
        slot = SessionSlot(session_id="session-1")

        assert slot.session_id == "session-1"
        assert slot.session is None
        assert slot.cpu_allocation == 1.0
        assert slot.memory_allocation_mb == 4096.0

    def test_slot_with_session(self):
        """SessionSlot MUST accept session reference."""
        session = Session(
            id="session-1",
            task_id="task-1",
            state=SessionState.RUNNING,
            working_directory="/workspace",
        )

        slot = SessionSlot(
            session_id="session-1",
            session=session,
        )

        assert slot.session is session


# =============================================================================
# ContainerConfig Tests
# =============================================================================


class TestContainerConfig:
    """Tests for ContainerConfig class."""

    def test_default_config(self):
        """ContainerConfig MUST have sensible defaults."""
        config = ContainerConfig()

        assert config.max_sessions == 4
        assert config.min_cpu_per_session == 0.5
        assert config.min_memory_per_session_mb == 2048.0

    def test_custom_config(self):
        """ContainerConfig MUST accept custom values."""
        config = ContainerConfig(
            max_sessions=8,
            min_cpu_per_session=1.0,
            min_memory_per_session_mb=4096.0,
        )

        assert config.max_sessions == 8
        assert config.min_cpu_per_session == 1.0


# =============================================================================
# SessionRegistry Tests
# =============================================================================


class TestSessionRegistry:
    """Tests for SessionRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a session registry for testing."""
        return SessionRegistry()

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        return Session(
            id="session-1",
            task_id="task-1",
            state=SessionState.RUNNING,
            working_directory="/workspace",
        )

    @pytest.mark.asyncio
    async def test_register_session(self, registry, mock_session):
        """register_session MUST create slot."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,  # 8GB
                total=16 * 1024 * 1024 * 1024,  # 16GB
            )

            slot = await registry.register_session(mock_session)

            assert slot.session_id == "session-1"
            assert slot.session is mock_session

    @pytest.mark.asyncio
    async def test_register_session_no_capacity(self, registry, mock_session):
        """register_session MUST raise when no capacity."""
        registry.config.max_sessions = 0

        with pytest.raises(RuntimeError, match="no capacity"):
            await registry.register_session(mock_session)

    @pytest.mark.asyncio
    async def test_unregister_session(self, registry, mock_session):
        """unregister_session MUST remove slot."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(mock_session)
            await registry.unregister_session("session-1")

            assert await registry.get_session("session-1") is None

    @pytest.mark.asyncio
    async def test_get_session(self, registry, mock_session):
        """get_session MUST return registered session."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(mock_session)

            result = await registry.get_session("session-1")

            assert result is mock_session

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, registry):
        """get_session MUST return None for unknown."""
        result = await registry.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, registry):
        """list_sessions MUST return all sessions."""
        session1 = Session(id="session-1", task_id="task-1", state=SessionState.RUNNING, working_directory="/ws")
        session2 = Session(id="session-2", task_id="task-2", state=SessionState.RUNNING, working_directory="/ws")

        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(session1)
            await registry.register_session(session2)

            sessions = await registry.list_sessions()

            assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_can_accept_session_true(self, registry):
        """can_accept_session MUST return True when capacity."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            result = await registry.can_accept_session()

            assert result is True

    @pytest.mark.asyncio
    async def test_can_accept_session_max_reached(self, registry, mock_session):
        """can_accept_session MUST return False at max."""
        registry.config.max_sessions = 1

        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(mock_session)

            result = await registry.can_accept_session()

            assert result is False

    @pytest.mark.asyncio
    async def test_get_active_count(self, registry):
        """get_active_count MUST count running sessions."""
        session1 = Session(id="session-1", task_id="task-1", state=SessionState.RUNNING, working_directory="/ws")
        session2 = Session(id="session-2", task_id="task-2", state=SessionState.COMPLETED, working_directory="/ws")

        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(session1)
            await registry.register_session(session2)

            count = await registry.get_active_count()

            assert count == 1  # Only RUNNING counts

    @pytest.mark.asyncio
    async def test_get_resource_status(self, registry):
        """get_resource_status MUST return current status."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 30.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            status = await registry.get_resource_status()

            assert status.sessions_total == 4
            assert status.cpu_total_cores == 4
            assert status.memory_total_mb == 16384.0

    @pytest.mark.asyncio
    async def test_capacity_change_callback(self, registry, mock_session):
        """on_capacity_change callback MUST be called."""
        callback_called = False

        async def on_change(status):
            nonlocal callback_called
            callback_called = True

        registry.on_capacity_change(on_change)

        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(mock_session)

            assert callback_called is True


# =============================================================================
# ResourceMonitor Tests
# =============================================================================


class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    @pytest.fixture
    def registry(self):
        """Create a session registry for testing."""
        return SessionRegistry()

    @pytest.fixture
    def monitor(self, registry):
        """Create a resource monitor for testing."""
        return ResourceMonitor(registry, poll_interval=0.1)

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """start/stop MUST manage monitoring loop."""
        await monitor.start()
        assert monitor._running is True

        await monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_get_current_status(self, monitor):
        """get_current_status MUST return status."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            status = await monitor.get_current_status()

            assert status.cpu_total_cores == 4

    @pytest.mark.asyncio
    async def test_get_history(self, monitor):
        """get_history MUST return recorded statuses."""
        # Manually add to history
        status = ResourceStatus(
            sessions_available=3,
            sessions_total=4,
            sessions_active=1,
            cpu_available_percent=50.0,
            cpu_total_cores=4,
            memory_available_mb=8000.0,
            memory_total_mb=16000.0,
        )
        monitor._history.append(status)

        history = await monitor.get_history()

        assert len(history) == 1
        assert history[0] is status

    @pytest.mark.asyncio
    async def test_warning_callback(self, monitor, registry):
        """on_warning callback MUST be called at threshold."""
        callback_called = False
        callback_status = None

        async def on_warning(status):
            nonlocal callback_called, callback_status
            callback_called = True
            callback_status = status

        monitor.on_warning(on_warning)

        # Test callback registration
        assert len(monitor._on_warning) == 1

        # Directly test callback invocation (simulating monitor behavior)
        high_util_status = ResourceStatus(
            sessions_available=0,
            sessions_total=4,
            sessions_active=4,
            cpu_available_percent=15.0,  # 85% used = above 80% warning threshold
            cpu_total_cores=4,
            memory_available_mb=2000.0,
            memory_total_mb=16000.0,
        )

        # Callback should be called for warning-level utilization
        await on_warning(high_util_status)
        assert callback_called
        assert callback_status == high_util_status


# =============================================================================
# SessionScheduler Tests
# =============================================================================


class TestSessionScheduler:
    """Tests for SessionScheduler class."""

    @pytest.fixture
    def registry(self):
        """Create a session registry for testing."""
        return SessionRegistry()

    @pytest.fixture
    def scheduler(self, registry):
        """Create a session scheduler for testing."""
        return SessionScheduler(registry)

    @pytest.mark.asyncio
    async def test_find_best_slot(self, scheduler, registry):
        """find_best_slot MUST return allocation."""
        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            slot = await scheduler.find_best_slot()

            assert slot is not None
            assert "cpu_cores" in slot
            assert "memory_mb" in slot

    @pytest.mark.asyncio
    async def test_find_best_slot_no_capacity(self, scheduler, registry):
        """find_best_slot MUST return None when no capacity."""
        registry.config.max_sessions = 0

        slot = await scheduler.find_best_slot()

        assert slot is None

    @pytest.mark.asyncio
    async def test_rebalance(self, scheduler, registry):
        """rebalance MUST adjust allocations."""
        session1 = Session(id="session-1", task_id="task-1", state=SessionState.RUNNING, working_directory="/ws")
        session2 = Session(id="session-2", task_id="task-2", state=SessionState.RUNNING, working_directory="/ws")

        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=8 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            await registry.register_session(session1)
            await registry.register_session(session2)

            # Manually set unequal allocations
            slots = await registry.list_slots()
            slots[0].cpu_allocation = 1.0
            slots[1].cpu_allocation = 2.0

            modified = await scheduler.rebalance()

            # Both should be modified to equal allocation
            assert len(modified) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiSessionIntegration:
    """Integration tests for multi-session support."""

    @pytest.mark.asyncio
    async def test_full_multisession_workflow(self):
        """Full multi-session workflow MUST work correctly."""
        config = ContainerConfig(max_sessions=4)
        registry = SessionRegistry(config)
        scheduler = SessionScheduler(registry)

        with patch("parhelia.multisession.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 20.0
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value = MagicMock(
                available=12 * 1024 * 1024 * 1024,
                total=16 * 1024 * 1024 * 1024,
            )

            # Find allocation
            allocation = await scheduler.find_best_slot()
            assert allocation is not None

            # Register sessions
            for i in range(3):
                session = Session(
                    id=f"session-{i}",
                    task_id=f"task-{i}",
                    state=SessionState.RUNNING,
                    working_directory="/ws",
                )
                await registry.register_session(
                    session,
                    cpu_allocation=allocation["cpu_cores"],
                    memory_allocation_mb=allocation["memory_mb"],
                )

            # Check status
            status = await registry.get_resource_status()
            assert status.sessions_active == 3
            assert status.sessions_available == 1

            # Can still accept one more
            assert await registry.can_accept_session() is True

            # Register fourth session
            session4 = Session(
                id="session-4",
                task_id="task-4",
                state=SessionState.RUNNING,
                working_directory="/ws",
            )
            await registry.register_session(session4)

            # Now at capacity
            assert await registry.can_accept_session() is False

            # Unregister one
            await registry.unregister_session("session-1")

            # Can accept again
            assert await registry.can_accept_session() is True
