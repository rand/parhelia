"""Tests for Parhelia control plane state management.

Tests SQLite-backed storage for containers, events, and heartbeats.

Implements tests for:
- [SPEC-21.10] Container Registry
- [SPEC-21.11] Events Table
- [SPEC-21.12] Heartbeat History
- [SPEC-21.30] StateStore Interface
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from parhelia.state import (
    Container,
    ContainerState,
    ContainerStats,
    ContainerStore,
    Event,
    EventStore,
    EventType,
    HealthStatus,
    Heartbeat,
    HeartbeatStore,
    StateStore,
)


# =============================================================================
# ContainerStore Tests
# =============================================================================


class TestContainerStore:
    """Tests for ContainerStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a ContainerStore with temp database."""
        return ContainerStore(tmp_path / "test.db")

    @pytest.fixture
    def sample_container(self):
        """Create a sample container."""
        return Container.create(
            modal_sandbox_id="sb-xyz789",
            task_id="task-abc123",
            worker_id="worker-def456",
            region="us-east",
            cpu_cores=4,
            memory_mb=8192,
        )

    def test_save_and_get(self, store, sample_container):
        """ContainerStore MUST save and retrieve containers."""
        store.save(sample_container)

        retrieved = store.get(sample_container.id)
        assert retrieved is not None
        assert retrieved.id == sample_container.id
        assert retrieved.modal_sandbox_id == sample_container.modal_sandbox_id
        assert retrieved.task_id == sample_container.task_id

    def test_get_nonexistent(self, store):
        """ContainerStore MUST return None for missing containers."""
        result = store.get("nonexistent-container")
        assert result is None

    def test_get_by_modal_id(self, store, sample_container):
        """ContainerStore MUST retrieve by Modal sandbox ID."""
        store.save(sample_container)

        retrieved = store.get_by_modal_id(sample_container.modal_sandbox_id)
        assert retrieved is not None
        assert retrieved.id == sample_container.id

    def test_get_by_task(self, store, sample_container):
        """ContainerStore MUST retrieve containers by task ID."""
        store.save(sample_container)

        # Create a second container for same task
        container2 = Container.create(
            modal_sandbox_id="sb-abc123",
            task_id=sample_container.task_id,
        )
        store.save(container2)

        containers = store.get_by_task(sample_container.task_id)
        assert len(containers) == 2

    def test_list_active(self, store):
        """ContainerStore MUST list active containers."""
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.state = ContainerState.RUNNING
        store.save(c1)

        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.state = ContainerState.TERMINATED
        store.save(c2)

        c3 = Container.create(modal_sandbox_id="sb-3")
        c3.state = ContainerState.CREATED
        store.save(c3)

        active = store.list_active()
        assert len(active) == 2
        assert all(c.state in (ContainerState.RUNNING, ContainerState.CREATED) for c in active)

    def test_list_by_state(self, store):
        """ContainerStore MUST filter by state."""
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.state = ContainerState.RUNNING
        store.save(c1)

        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.state = ContainerState.ORPHANED
        store.save(c2)

        orphaned = store.list_by_state(ContainerState.ORPHANED)
        assert len(orphaned) == 1
        assert orphaned[0].state == ContainerState.ORPHANED

    def test_list_by_health(self, store):
        """ContainerStore MUST filter by health status."""
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.health_status = HealthStatus.HEALTHY
        store.save(c1)

        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.health_status = HealthStatus.UNHEALTHY
        store.save(c2)

        unhealthy = store.list_by_health(HealthStatus.UNHEALTHY)
        assert len(unhealthy) == 1
        assert unhealthy[0].health_status == HealthStatus.UNHEALTHY

    def test_list_without_heartbeat_since(self, store):
        """ContainerStore MUST find stale containers."""
        now = datetime.utcnow()
        threshold = now - timedelta(minutes=5)

        # Recent heartbeat
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.state = ContainerState.RUNNING
        c1.last_heartbeat_at = now - timedelta(minutes=1)
        store.save(c1)

        # Old heartbeat
        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.state = ContainerState.RUNNING
        c2.last_heartbeat_at = now - timedelta(minutes=10)
        store.save(c2)

        # No heartbeat
        c3 = Container.create(modal_sandbox_id="sb-3")
        c3.state = ContainerState.RUNNING
        c3.last_heartbeat_at = None
        store.save(c3)

        stale = store.list_without_heartbeat_since(threshold)
        assert len(stale) == 2
        stale_ids = {c.id for c in stale}
        assert c2.id in stale_ids
        assert c3.id in stale_ids

    def test_delete(self, store, sample_container):
        """ContainerStore MUST delete containers."""
        store.save(sample_container)
        assert store.get(sample_container.id) is not None

        deleted = store.delete(sample_container.id)
        assert deleted is True
        assert store.get(sample_container.id) is None

    def test_get_stats(self, store):
        """ContainerStore MUST return aggregate statistics."""
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.state = ContainerState.RUNNING
        c1.health_status = HealthStatus.HEALTHY
        c1.cost_accrued_usd = 1.50
        store.save(c1)

        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.state = ContainerState.TERMINATED
        c2.health_status = HealthStatus.DEAD
        c2.cost_accrued_usd = 0.75
        store.save(c2)

        stats = store.get_stats()
        assert stats.total == 2
        assert stats.by_state.get("running", 0) == 1
        assert stats.by_state.get("terminated", 0) == 1
        assert stats.by_health.get("healthy", 0) == 1
        assert stats.total_cost_usd == pytest.approx(2.25)


# =============================================================================
# EventStore Tests
# =============================================================================


class TestEventStore:
    """Tests for EventStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create an EventStore with temp database."""
        return EventStore(tmp_path / "test.db")

    def test_save_and_get(self, store):
        """EventStore MUST save and retrieve events."""
        event = Event.create(
            EventType.CONTAINER_CREATED,
            container_id="c-abc123",
            message="Test event",
        )
        event_id = store.save(event)

        retrieved = store.get(event_id)
        assert retrieved is not None
        assert retrieved.event_type == EventType.CONTAINER_CREATED
        assert retrieved.container_id == "c-abc123"
        assert retrieved.message == "Test event"

    def test_list_by_container(self, store):
        """EventStore MUST filter by container ID."""
        e1 = Event.create(EventType.CONTAINER_CREATED, container_id="c-1")
        e2 = Event.create(EventType.CONTAINER_STARTED, container_id="c-1")
        e3 = Event.create(EventType.CONTAINER_CREATED, container_id="c-2")

        store.save(e1)
        store.save(e2)
        store.save(e3)

        events = store.list(container_id="c-1")
        assert len(events) == 2
        assert all(e.container_id == "c-1" for e in events)

    def test_list_by_type(self, store):
        """EventStore MUST filter by event type."""
        e1 = Event.create(EventType.CONTAINER_CREATED)
        e2 = Event.create(EventType.CONTAINER_STARTED)
        e3 = Event.create(EventType.CONTAINER_CREATED)

        store.save(e1)
        store.save(e2)
        store.save(e3)

        events = store.list(event_type=EventType.CONTAINER_CREATED)
        assert len(events) == 2
        assert all(e.event_type == EventType.CONTAINER_CREATED for e in events)

    def test_list_by_time_range(self, store):
        """EventStore MUST filter by time range."""
        now = datetime.utcnow()

        e1 = Event(
            id=None,
            timestamp=now - timedelta(hours=2),
            event_type=EventType.CONTAINER_CREATED,
        )
        e2 = Event(
            id=None,
            timestamp=now - timedelta(hours=1),
            event_type=EventType.CONTAINER_STARTED,
        )
        e3 = Event(
            id=None,
            timestamp=now,
            event_type=EventType.CONTAINER_STOPPED,
        )

        store.save(e1)
        store.save(e2)
        store.save(e3)

        # Events in last 90 minutes
        since = now - timedelta(minutes=90)
        events = store.list(since=since)
        assert len(events) == 2

    def test_count(self, store):
        """EventStore MUST count events."""
        for _ in range(5):
            store.save(Event.create(EventType.CONTAINER_CREATED))
        for _ in range(3):
            store.save(Event.create(EventType.ERROR))

        total = store.count()
        assert total == 8

        errors = store.count(event_type=EventType.ERROR)
        assert errors == 3

    def test_delete_before(self, store):
        """EventStore MUST delete old events."""
        now = datetime.utcnow()

        old = Event(
            id=None,
            timestamp=now - timedelta(days=10),
            event_type=EventType.CONTAINER_CREATED,
        )
        recent = Event(
            id=None,
            timestamp=now - timedelta(hours=1),
            event_type=EventType.CONTAINER_STARTED,
        )

        store.save(old)
        store.save(recent)

        deleted = store.delete_before(now - timedelta(days=7))
        assert deleted == 1

        remaining = store.list()
        assert len(remaining) == 1


# =============================================================================
# HeartbeatStore Tests
# =============================================================================


class TestHeartbeatStore:
    """Tests for HeartbeatStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a HeartbeatStore with temp database."""
        return HeartbeatStore(tmp_path / "test.db")

    def test_save_and_get_latest(self, store):
        """HeartbeatStore MUST save and retrieve latest heartbeat."""
        hb1 = Heartbeat.create(
            container_id="c-abc123",
            cpu_percent=45.0,
            memory_percent=60.0,
        )
        store.save(hb1)

        # Wait and save another
        hb2 = Heartbeat.create(
            container_id="c-abc123",
            cpu_percent=50.0,
            memory_percent=65.0,
        )
        store.save(hb2)

        latest = store.get_latest("c-abc123")
        assert latest is not None
        assert latest.cpu_percent == 50.0  # Second heartbeat

    def test_list_history(self, store):
        """HeartbeatStore MUST return heartbeat history."""
        for i in range(5):
            hb = Heartbeat.create(
                container_id="c-abc123",
                cpu_percent=float(i * 10),
            )
            store.save(hb)

        history = store.list("c-abc123")
        assert len(history) == 5

    def test_list_since(self, store):
        """HeartbeatStore MUST filter by time."""
        now = datetime.utcnow()

        old = Heartbeat(
            id=None,
            container_id="c-abc123",
            timestamp=now - timedelta(hours=2),
        )
        recent = Heartbeat(
            id=None,
            container_id="c-abc123",
            timestamp=now - timedelta(minutes=30),
        )

        store.save(old)
        store.save(recent)

        since = now - timedelta(hours=1)
        history = store.list("c-abc123", since=since)
        assert len(history) == 1

    def test_delete_before(self, store):
        """HeartbeatStore MUST delete old heartbeats."""
        now = datetime.utcnow()

        old = Heartbeat(
            id=None,
            container_id="c-abc123",
            timestamp=now - timedelta(days=10),
        )
        recent = Heartbeat(
            id=None,
            container_id="c-abc123",
            timestamp=now - timedelta(hours=1),
        )

        store.save(old)
        store.save(recent)

        deleted = store.delete_before(now - timedelta(days=7))
        assert deleted == 1

        remaining = store.list("c-abc123")
        assert len(remaining) == 1


# =============================================================================
# StateStore Integration Tests
# =============================================================================


class TestStateStore:
    """Tests for unified StateStore interface."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a StateStore with temp database."""
        return StateStore(tmp_path / "test.db")

    def test_create_container_emits_event(self, store):
        """StateStore MUST emit event when creating container."""
        container = Container.create(
            modal_sandbox_id="sb-xyz789",
            task_id="task-abc123",
        )
        store.create_container(container)

        # Check container exists
        retrieved = store.get_container(container.id)
        assert retrieved is not None

        # Check event was emitted
        events = store.get_events(container_id=container.id)
        assert len(events) == 1
        assert events[0].event_type == EventType.CONTAINER_CREATED

    def test_update_container_state_emits_event(self, store):
        """StateStore MUST emit event when updating state."""
        container = Container.create(modal_sandbox_id="sb-xyz789")
        store.create_container(container)

        store.update_container_state(
            container.id,
            ContainerState.RUNNING,
        )

        # Check state updated
        retrieved = store.get_container(container.id)
        assert retrieved.state == ContainerState.RUNNING
        assert retrieved.started_at is not None

        # Check events (created + started)
        events = store.get_events(container_id=container.id)
        assert len(events) == 2
        assert events[0].event_type == EventType.CONTAINER_STARTED

    def test_update_container_health_emits_event(self, store):
        """StateStore MUST emit event when health changes."""
        container = Container.create(modal_sandbox_id="sb-xyz789")
        container.health_status = HealthStatus.HEALTHY
        store.create_container(container)

        store.update_container_health(container.id, HealthStatus.UNHEALTHY)

        # Check health updated
        retrieved = store.get_container(container.id)
        assert retrieved.health_status == HealthStatus.UNHEALTHY
        assert retrieved.consecutive_failures == 1

        # Check event emitted
        events = store.get_events(
            container_id=container.id,
            event_type=EventType.CONTAINER_UNHEALTHY,
        )
        assert len(events) == 1

    def test_health_recovery_emits_event(self, store):
        """StateStore MUST emit recovery event when health improves."""
        container = Container.create(modal_sandbox_id="sb-xyz789")
        container.health_status = HealthStatus.UNHEALTHY
        container.consecutive_failures = 3
        store.containers.save(container)

        store.update_container_health(container.id, HealthStatus.HEALTHY)

        # Check recovery
        retrieved = store.get_container(container.id)
        assert retrieved.health_status == HealthStatus.HEALTHY
        assert retrieved.consecutive_failures == 0

        # Check recovery event
        events = store.get_events(
            container_id=container.id,
            event_type=EventType.CONTAINER_RECOVERED,
        )
        assert len(events) == 1

    def test_record_heartbeat_updates_container(self, store):
        """StateStore MUST update container when heartbeat received."""
        container = Container.create(modal_sandbox_id="sb-xyz789")
        container.health_status = HealthStatus.DEGRADED
        store.containers.save(container)

        heartbeat = Heartbeat.create(
            container_id=container.id,
            cpu_percent=45.0,
            memory_percent=60.0,
        )
        store.record_heartbeat(heartbeat)

        # Check container updated
        retrieved = store.get_container(container.id)
        assert retrieved.last_heartbeat_at is not None
        assert retrieved.health_status == HealthStatus.HEALTHY

        # Check heartbeat stored
        history = store.get_heartbeat_history(container.id)
        assert len(history) == 1

    def test_log_event(self, store):
        """StateStore MUST log events."""
        event_id = store.log_event(
            EventType.ERROR,
            message="Something went wrong",
            source="test",
            error_code="E500",
        )

        events = store.get_events(event_type=EventType.ERROR)
        assert len(events) == 1
        assert events[0].message == "Something went wrong"
        assert events[0].details.get("error_code") == "E500"

    def test_get_container_stats(self, store):
        """StateStore MUST return container statistics."""
        c1 = Container.create(modal_sandbox_id="sb-1")
        c1.state = ContainerState.RUNNING
        c1.cost_accrued_usd = 1.00
        store.containers.save(c1)

        c2 = Container.create(modal_sandbox_id="sb-2")
        c2.state = ContainerState.RUNNING
        c2.cost_accrued_usd = 2.00
        store.containers.save(c2)

        stats = store.get_container_stats()
        assert stats.total == 2
        assert stats.by_state.get("running", 0) == 2
        assert stats.total_cost_usd == pytest.approx(3.00)

    def test_cleanup_old_data(self, store):
        """StateStore MUST clean up old data."""
        now = datetime.utcnow()

        # Create old data
        old_event = Event(
            id=None,
            timestamp=now - timedelta(days=10),
            event_type=EventType.CONTAINER_CREATED,
        )
        store.events.save(old_event)

        old_heartbeat = Heartbeat(
            id=None,
            container_id="c-old",
            timestamp=now - timedelta(days=10),
        )
        store.heartbeats.save(old_heartbeat)

        # Create recent data
        recent_event = Event.create(EventType.CONTAINER_STARTED)
        store.events.save(recent_event)

        recent_heartbeat = Heartbeat.create(container_id="c-new")
        store.heartbeats.save(recent_heartbeat)

        # Cleanup
        result = store.cleanup_old_data(retention_days=7)
        assert result["events_deleted"] == 1
        assert result["heartbeats_deleted"] == 1

        # Verify recent data preserved
        assert store.events.count() == 1
        assert len(store.heartbeats.list("c-new")) == 1


# =============================================================================
# Container Model Tests
# =============================================================================


class TestContainerModel:
    """Tests for Container dataclass."""

    def test_create_generates_id(self):
        """Container.create MUST generate unique ID."""
        c1 = Container.create(modal_sandbox_id="sb-1")
        c2 = Container.create(modal_sandbox_id="sb-2")

        assert c1.id.startswith("c-")
        assert c2.id.startswith("c-")
        assert c1.id != c2.id

    def test_create_sets_defaults(self):
        """Container.create MUST set sensible defaults."""
        container = Container.create(modal_sandbox_id="sb-xyz789")

        assert container.state == ContainerState.CREATED
        assert container.health_status == HealthStatus.UNKNOWN
        assert container.cost_accrued_usd == 0.0
        assert container.consecutive_failures == 0


# =============================================================================
# Event Model Tests
# =============================================================================


class TestEventModel:
    """Tests for Event dataclass."""

    def test_create_sets_timestamp(self):
        """Event.create MUST set current timestamp."""
        before = datetime.utcnow()
        event = Event.create(EventType.CONTAINER_CREATED)
        after = datetime.utcnow()

        assert before <= event.timestamp <= after

    def test_create_with_details(self):
        """Event.create MUST accept details."""
        event = Event.create(
            EventType.ERROR,
            message="Test error",
            details={"error_code": "E500"},
        )

        assert event.message == "Test error"
        assert event.details["error_code"] == "E500"


# =============================================================================
# Heartbeat Model Tests
# =============================================================================


class TestHeartbeatModel:
    """Tests for Heartbeat dataclass."""

    def test_create_sets_timestamp(self):
        """Heartbeat.create MUST set current timestamp."""
        before = datetime.utcnow()
        heartbeat = Heartbeat.create(container_id="c-abc123")
        after = datetime.utcnow()

        assert before <= heartbeat.timestamp <= after

    def test_create_with_metrics(self):
        """Heartbeat.create MUST accept metrics."""
        heartbeat = Heartbeat.create(
            container_id="c-abc123",
            cpu_percent=45.0,
            memory_percent=60.0,
            tmux_active=True,
        )

        assert heartbeat.cpu_percent == 45.0
        assert heartbeat.memory_percent == 60.0
        assert heartbeat.tmux_active is True
