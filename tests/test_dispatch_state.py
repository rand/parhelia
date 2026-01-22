"""Tests for dispatch and state integration (Wave 2: Container Lifecycle).

Tests the integration between dispatch.py, heartbeat.py, and state.py
for container lifecycle management.

@trace SPEC-21.10 - Container Registry Schema
@trace SPEC-21.11 - Events Table Schema
@trace SPEC-21.12 - Heartbeat History Schema
"""

from __future__ import annotations

import pytest

from parhelia.dispatch import (
    DispatchMode,
    TaskDispatcher,
)
from parhelia.heartbeat import HeartbeatMonitor
from parhelia.orchestrator import (
    Task,
    TaskRequirements,
    TaskType,
    WorkerState,
)
from parhelia.persistence import PersistentOrchestrator
from parhelia.state import (
    Container,
    ContainerState,
    EventType,
    HealthStatus,
    Heartbeat,
    StateStore,
)


# =============================================================================
# Container Creation on Dispatch Tests
# =============================================================================


class TestContainerCreationOnDispatch:
    """Tests for container record creation during dispatch."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        """Create temp database path."""
        return tmp_path / "test.db"

    @pytest.fixture
    def orchestrator(self, tmp_db):
        """Create orchestrator with temp database."""
        return PersistentOrchestrator(db_path=tmp_db)

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store with temp database."""
        return StateStore(db_path=tmp_path / "state.db")

    @pytest.fixture
    def dispatcher(self, orchestrator, state_store):
        """Create dispatcher with state store in dry-run mode."""
        return TaskDispatcher(
            orchestrator,
            skip_modal=True,
            state_store=state_store,
        )

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return Task(
            id="task-state-123",
            prompt="Test container lifecycle",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

    @pytest.mark.asyncio
    async def test_container_created_on_dispatch(
        self, dispatcher, orchestrator, state_store, sample_task
    ):
        """@trace SPEC-21.10 - Container MUST be created on dispatch."""
        await orchestrator.submit_task(sample_task)
        result = await dispatcher.dispatch(sample_task)

        assert result.success is True

        # Verify container was created
        containers = state_store.get_containers_for_task(sample_task.id)
        assert len(containers) == 1

        container = containers[0]
        assert container.task_id == sample_task.id
        assert container.modal_sandbox_id.startswith("dry-run-")
        assert container.state == ContainerState.CREATED

    @pytest.mark.asyncio
    async def test_container_linked_to_worker(
        self, dispatcher, orchestrator, state_store, sample_task
    ):
        """@trace SPEC-21.13 - Container MUST be linked to worker."""
        await orchestrator.submit_task(sample_task)
        result = await dispatcher.dispatch(sample_task)

        # Get the worker
        worker = orchestrator.get_worker(result.worker_id)
        assert worker is not None
        assert worker.container_id is not None

        # Verify container exists
        container = state_store.get_container(worker.container_id)
        assert container is not None
        assert container.worker_id == result.worker_id

    @pytest.mark.asyncio
    async def test_container_created_event_emitted(
        self, dispatcher, orchestrator, state_store, sample_task
    ):
        """@trace SPEC-21.11 - Container creation event MUST be emitted."""
        await orchestrator.submit_task(sample_task)
        await dispatcher.dispatch(sample_task)

        # Check events
        events = state_store.get_events(
            task_id=sample_task.id,
            event_type=EventType.CONTAINER_CREATED,
        )
        assert len(events) == 1
        assert events[0].task_id == sample_task.id
        assert "dry-run-" in events[0].message

    @pytest.mark.asyncio
    async def test_dispatch_without_state_store(self, orchestrator, sample_task):
        """Dispatch MUST work without state store (backwards compatible)."""
        dispatcher = TaskDispatcher(orchestrator, skip_modal=True)
        await orchestrator.submit_task(sample_task)

        result = await dispatcher.dispatch(sample_task)

        assert result.success is True
        # Worker should not have container_id
        worker = orchestrator.get_worker(result.worker_id)
        assert worker.container_id is None


# =============================================================================
# Heartbeat Persistence Tests
# =============================================================================


class TestHeartbeatPersistence:
    """Tests for heartbeat persistence via StateStore."""

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store with temp database."""
        return StateStore(db_path=tmp_path / "state.db")

    @pytest.fixture
    def monitor(self, state_store):
        """Create heartbeat monitor with state store."""
        return HeartbeatMonitor(
            interval=1.0,
            state_store=state_store,
        )

    @pytest.fixture
    def container(self, state_store):
        """Create a test container."""
        container = Container.create(
            modal_sandbox_id="test-sandbox-hb",
            task_id="task-hb-123",
            worker_id="worker-hb-123",
        )
        state_store.create_container(container)
        return container

    def test_heartbeat_persisted_to_state_store(
        self, monitor, state_store, container
    ):
        """@trace SPEC-21.12 - Heartbeats MUST be persisted to StateStore."""
        # Register session with container_id
        monitor.register_session(
            session_id="session-1",
            container_id=container.id,
        )

        # Record heartbeat with metrics
        monitor.record_heartbeat(
            session_id="session-1",
            metadata={
                "cpu_percent": 45.0,
                "memory_percent": 60.0,
                "tmux_active": True,
            },
        )

        # Verify heartbeat was persisted
        heartbeats = state_store.get_heartbeat_history(container.id, limit=10)
        assert len(heartbeats) == 1

        hb = heartbeats[0]
        assert hb.container_id == container.id
        assert hb.cpu_percent == 45.0
        assert hb.memory_percent == 60.0
        assert hb.tmux_active is True

    def test_heartbeat_updates_container_health(
        self, monitor, state_store, container
    ):
        """@trace SPEC-21.12 - Heartbeat MUST update container health status."""
        # Set container to unhealthy initially
        state_store.update_container_health(container.id, HealthStatus.UNHEALTHY)

        # Register and record heartbeat
        monitor.register_session(
            session_id="session-2",
            container_id=container.id,
        )
        monitor.record_heartbeat(session_id="session-2")

        # Container should now be healthy
        updated_container = state_store.get_container(container.id)
        assert updated_container.health_status == HealthStatus.HEALTHY

    def test_heartbeat_updates_last_heartbeat_timestamp(
        self, monitor, state_store, container
    ):
        """@trace SPEC-21.12 - Heartbeat MUST update last_heartbeat_at."""
        monitor.register_session(
            session_id="session-3",
            container_id=container.id,
        )

        # Initially no heartbeat
        assert container.last_heartbeat_at is None

        # Record heartbeat
        monitor.record_heartbeat(session_id="session-3")

        # Check timestamp updated
        updated_container = state_store.get_container(container.id)
        assert updated_container.last_heartbeat_at is not None

    def test_heartbeat_without_container_id(self, monitor, state_store):
        """Heartbeat without container_id MUST NOT fail."""
        # Register session without container_id
        monitor.register_session(session_id="session-no-container")

        # Should not raise
        info = monitor.record_heartbeat(session_id="session-no-container")
        assert info is not None

    def test_unregister_session_cleans_up_mapping(self, monitor, container):
        """Unregister MUST clean up container mapping."""
        monitor.register_session(
            session_id="session-cleanup",
            container_id=container.id,
        )
        assert "session-cleanup" in monitor._session_to_container

        monitor.unregister_session("session-cleanup")
        assert "session-cleanup" not in monitor._session_to_container


# =============================================================================
# Container State Update Tests
# =============================================================================


class TestContainerStateUpdates:
    """Tests for container state updates on worker state changes."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        """Create temp database path."""
        return tmp_path / "test.db"

    @pytest.fixture
    def orchestrator(self, tmp_db):
        """Create orchestrator with temp database."""
        return PersistentOrchestrator(db_path=tmp_db)

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store with temp database."""
        return StateStore(db_path=tmp_path / "state.db")

    @pytest.fixture
    def dispatcher(self, orchestrator, state_store):
        """Create dispatcher with state store in dry-run mode."""
        return TaskDispatcher(
            orchestrator,
            skip_modal=True,
            state_store=state_store,
        )

    def test_update_container_state_method(
        self, dispatcher, orchestrator, state_store
    ):
        """@trace SPEC-21.10 - _update_container_state MUST update container."""
        # Create container and worker manually
        container = Container.create(
            modal_sandbox_id="test-sandbox-state",
            task_id="task-state-update",
            worker_id="worker-state-update",
        )
        state_store.create_container(container)

        # Create and register worker
        from parhelia.orchestrator import WorkerInfo

        worker = WorkerInfo(
            id="worker-state-update",
            task_id="task-state-update",
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
            container_id=container.id,
        )
        orchestrator.worker_store.save(worker)

        # Update container state via dispatcher
        dispatcher._update_container_state(
            worker_id="worker-state-update",
            new_state=ContainerState.TERMINATED,
            exit_code=0,
            reason="Test termination",
        )

        # Verify container state updated
        updated = state_store.get_container(container.id)
        assert updated.state == ContainerState.TERMINATED
        assert updated.terminated_at is not None

    def test_update_container_state_emits_event(
        self, dispatcher, orchestrator, state_store
    ):
        """@trace SPEC-21.11 - State change MUST emit event."""
        # Create container and worker
        container = Container.create(
            modal_sandbox_id="test-sandbox-event",
            task_id="task-event-test",
            worker_id="worker-event-test",
        )
        state_store.create_container(container)

        from parhelia.orchestrator import WorkerInfo

        worker = WorkerInfo(
            id="worker-event-test",
            task_id="task-event-test",
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
            container_id=container.id,
        )
        orchestrator.worker_store.save(worker)

        # Update state
        dispatcher._update_container_state(
            worker_id="worker-event-test",
            new_state=ContainerState.TERMINATED,
            exit_code=1,
            reason="Task failed",
        )

        # Check for termination event
        events = state_store.get_events(
            container_id=container.id,
            event_type=EventType.CONTAINER_TERMINATED,
        )
        assert len(events) >= 1

    def test_update_container_state_without_state_store(self, orchestrator):
        """Update MUST be no-op without state store."""
        dispatcher = TaskDispatcher(orchestrator, skip_modal=True)

        # Should not raise
        dispatcher._update_container_state(
            worker_id="nonexistent",
            new_state=ContainerState.TERMINATED,
        )

    def test_update_container_state_without_container_id(
        self, dispatcher, orchestrator
    ):
        """Update MUST handle worker without container_id."""
        from parhelia.orchestrator import WorkerInfo

        # Worker without container_id
        worker = WorkerInfo(
            id="worker-no-container",
            task_id="task-no-container",
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
        )
        orchestrator.worker_store.save(worker)

        # Should not raise
        dispatcher._update_container_state(
            worker_id="worker-no-container",
            new_state=ContainerState.TERMINATED,
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullContainerLifecycle:
    """Integration tests for full container lifecycle."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        """Create temp database path."""
        return tmp_path / "test.db"

    @pytest.fixture
    def orchestrator(self, tmp_db):
        """Create orchestrator with temp database."""
        return PersistentOrchestrator(db_path=tmp_db)

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store with temp database."""
        return StateStore(db_path=tmp_path / "state.db")

    @pytest.fixture
    def dispatcher(self, orchestrator, state_store):
        """Create dispatcher with state store in dry-run mode."""
        return TaskDispatcher(
            orchestrator,
            skip_modal=True,
            state_store=state_store,
        )

    @pytest.fixture
    def monitor(self, state_store):
        """Create heartbeat monitor with state store."""
        return HeartbeatMonitor(
            interval=1.0,
            state_store=state_store,
        )

    @pytest.mark.asyncio
    async def test_full_lifecycle_dispatch_to_heartbeat(
        self, dispatcher, orchestrator, state_store, monitor
    ):
        """Test full lifecycle: dispatch -> container -> heartbeat."""
        task = Task(
            id="task-full-lifecycle",
            prompt="Full lifecycle test",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )
        await orchestrator.submit_task(task)

        # Dispatch creates container
        result = await dispatcher.dispatch(task)
        assert result.success

        # Get container
        containers = state_store.get_containers_for_task(task.id)
        assert len(containers) == 1
        container = containers[0]

        # Register session with container for heartbeats
        monitor.register_session(
            session_id="session-full",
            container_id=container.id,
        )

        # Record heartbeat
        monitor.record_heartbeat(
            session_id="session-full",
            metadata={"cpu_percent": 25.0, "claude_responsive": True},
        )

        # Verify heartbeat persisted
        heartbeats = state_store.get_heartbeat_history(container.id)
        assert len(heartbeats) == 1
        assert heartbeats[0].cpu_percent == 25.0

        # Verify container health updated
        updated = state_store.get_container(container.id)
        assert updated.last_heartbeat_at is not None

    @pytest.mark.asyncio
    async def test_container_stats_after_dispatch(
        self, dispatcher, orchestrator, state_store
    ):
        """Test container stats after dispatching tasks."""
        tasks = [
            Task(
                id=f"task-stats-{i}",
                prompt=f"Stats test {i}",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            )
            for i in range(3)
        ]

        for task in tasks:
            await orchestrator.submit_task(task)
            await dispatcher.dispatch(task)

        # Check stats
        stats = state_store.get_container_stats()
        assert stats.total == 3
        assert stats.by_state.get("created", 0) == 3
