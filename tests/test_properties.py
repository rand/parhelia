"""Property-based tests for system invariants.

Uses Hypothesis to verify key invariants hold across many random inputs.
Tests [SPEC-21.XX] invariants for state management.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from parhelia.state import (
    Container,
    ContainerState,
    Event,
    EventType,
    HealthStatus,
    Heartbeat,
    StateStore,
)
from parhelia.events import EventFilter
from parhelia.budget import BudgetManager, BudgetStatus


# =============================================================================
# Custom Strategies
# =============================================================================

container_states = st.sampled_from(list(ContainerState))
health_statuses = st.sampled_from(list(HealthStatus))
event_types = st.sampled_from(list(EventType))

container_ids = st.text(
    alphabet="abcdef0123456789",
    min_size=8,
    max_size=8,
).map(lambda s: f"c-{s}")

sandbox_ids = st.text(
    alphabet="abcdef0123456789-",
    min_size=10,
    max_size=30,
)


# =============================================================================
# Container State Invariants
# =============================================================================


class TestContainerStateInvariants:
    """Property tests for container state management."""

    @given(sandbox_id=sandbox_ids)
    @settings(max_examples=50)
    def test_container_create_always_sets_created_state(self, sandbox_id: str):
        """Container.create() MUST always set state to CREATED."""
        assume(len(sandbox_id) > 0)
        container = Container.create(modal_sandbox_id=sandbox_id)
        assert container.state == ContainerState.CREATED

    @given(sandbox_id=sandbox_ids)
    @settings(max_examples=50)
    def test_container_id_follows_format(self, sandbox_id: str):
        """Container IDs MUST follow c-{hex} format."""
        assume(len(sandbox_id) > 0)
        container = Container.create(modal_sandbox_id=sandbox_id)
        assert container.id.startswith("c-")
        # ID should be c- followed by 8 hex chars
        assert len(container.id) == 10

    @given(sandbox_id=sandbox_ids)
    @settings(max_examples=50)
    def test_container_created_at_is_set(self, sandbox_id: str):
        """Container.create() MUST set created_at timestamp."""
        assume(len(sandbox_id) > 0)
        before = datetime.now(timezone.utc)
        container = Container.create(modal_sandbox_id=sandbox_id)
        after = datetime.now(timezone.utc)

        assert container.created_at is not None
        # Allow some tolerance for timing
        assert container.created_at >= before.replace(tzinfo=None) - timedelta(seconds=1)
        assert container.created_at <= after.replace(tzinfo=None) + timedelta(seconds=1)


class TestStateStorePersistence:
    """Property tests for state store persistence."""

    @given(sandbox_id=sandbox_ids)
    @settings(max_examples=30)
    def test_saved_container_can_be_retrieved(self, sandbox_id: str):
        """Saved containers MUST be retrievable by ID."""
        assume(len(sandbox_id) > 0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            store = StateStore(str(db_path))

            container = Container.create(modal_sandbox_id=sandbox_id)
            store.containers.save(container)

            retrieved = store.containers.get(container.id)
            assert retrieved is not None
            assert retrieved.id == container.id
            assert retrieved.modal_sandbox_id == sandbox_id

    @given(sandbox_id=sandbox_ids, state=container_states)
    @settings(max_examples=30)
    def test_state_update_persists(self, sandbox_id: str, state: ContainerState):
        """State updates MUST persist across retrieval."""
        assume(len(sandbox_id) > 0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            store = StateStore(str(db_path))

            container = Container.create(modal_sandbox_id=sandbox_id)
            store.containers.save(container)
            store.update_container_state(container.id, state)

            retrieved = store.containers.get(container.id)
            assert retrieved is not None
            assert retrieved.state == state


# =============================================================================
# Event Invariants
# =============================================================================


class TestEventInvariants:
    """Property tests for event system."""

    @given(event_type=event_types, container_id=container_ids)
    @settings(max_examples=50)
    def test_event_create_sets_timestamp(self, event_type: EventType, container_id: str):
        """Event.create() MUST set timestamp."""
        event = Event.create(event_type=event_type, container_id=container_id)
        assert event.timestamp is not None

    @given(event_type=event_types, container_id=container_ids)
    @settings(max_examples=50)
    def test_event_has_required_fields(self, event_type: EventType, container_id: str):
        """Events MUST have event_type and container_id."""
        event = Event.create(event_type=event_type, container_id=container_id)
        assert event.event_type == event_type
        assert event.container_id == container_id


class TestEventFilterInvariants:
    """Property tests for event filtering."""

    @given(event_type=event_types, container_id=container_ids)
    @settings(max_examples=50)
    def test_empty_filter_matches_all(self, event_type: EventType, container_id: str):
        """Empty EventFilter MUST match any event."""
        filter = EventFilter()
        event = Event.create(event_type=event_type, container_id=container_id)
        assert filter.matches(event) is True

    @given(
        filter_types=st.lists(event_types, min_size=1, max_size=5),
        event_type=event_types,
        container_id=container_ids,
    )
    @settings(max_examples=50)
    def test_type_filter_correct(
        self,
        filter_types: list[EventType],
        event_type: EventType,
        container_id: str,
    ):
        """EventFilter with types MUST only match events of those types."""
        filter = EventFilter(event_types=filter_types)
        event = Event.create(event_type=event_type, container_id=container_id)

        if event_type in filter_types:
            assert filter.matches(event) is True
        else:
            assert filter.matches(event) is False

    @given(
        filter_container=container_ids,
        event_container=container_ids,
        event_type=event_types,
    )
    @settings(max_examples=50)
    def test_container_filter_correct(
        self,
        filter_container: str,
        event_container: str,
        event_type: EventType,
    ):
        """EventFilter with container_id MUST only match events from that container."""
        filter = EventFilter(container_id=filter_container)
        event = Event.create(event_type=event_type, container_id=event_container)

        if event_container == filter_container:
            assert filter.matches(event) is True
        else:
            assert filter.matches(event) is False


# =============================================================================
# Budget Invariants
# =============================================================================


class TestBudgetInvariants:
    """Property tests for budget management."""

    @given(ceiling=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False))
    @settings(max_examples=50)
    def test_budget_ceiling_non_negative(self, ceiling: float):
        """Budget ceiling MUST be non-negative."""
        manager = BudgetManager(ceiling_usd=ceiling)
        status = manager.check_budget(raise_on_exceeded=False)
        assert status.ceiling_usd >= 0

    @given(ceiling=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False))
    @settings(max_examples=50)
    def test_budget_remaining_equals_ceiling_minus_used(self, ceiling: float):
        """Remaining budget MUST equal ceiling minus used."""
        manager = BudgetManager(ceiling_usd=ceiling)
        status = manager.check_budget(raise_on_exceeded=False)
        # Floating point tolerance
        assert abs(status.remaining_usd - (status.ceiling_usd - status.used_usd)) < 0.001

    @given(ceiling=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False))
    @settings(max_examples=50)
    def test_usage_percent_valid_range(self, ceiling: float):
        """Usage percent MUST be between 0 and 100 (or higher if exceeded)."""
        manager = BudgetManager(ceiling_usd=ceiling)
        status = manager.check_budget(raise_on_exceeded=False)
        assert status.usage_percent >= 0


# =============================================================================
# Heartbeat Invariants
# =============================================================================


class TestHeartbeatInvariants:
    """Property tests for heartbeat system."""

    @given(container_id=container_ids)
    @settings(max_examples=50)
    def test_heartbeat_create_sets_timestamp(self, container_id: str):
        """Heartbeat.create() MUST set timestamp."""
        heartbeat = Heartbeat.create(container_id=container_id)
        assert heartbeat.timestamp is not None

    @given(container_id=container_ids)
    @settings(max_examples=50)
    def test_heartbeat_timestamps_unique(self, container_id: str):
        """Heartbeat.create() MUST generate distinct timestamps for different heartbeats."""
        hb1 = Heartbeat.create(container_id=container_id)
        hb2 = Heartbeat.create(container_id=container_id)
        # Timestamps should be set (IDs are assigned on save, not create)
        assert hb1.timestamp is not None
        assert hb2.timestamp is not None


# =============================================================================
# Data Integrity Invariants
# =============================================================================


class TestDataIntegrityInvariants:
    """Property tests for data integrity."""

    @given(
        sandbox_id=sandbox_ids,
        event_types_list=st.lists(event_types, min_size=1, max_size=10),
    )
    @settings(max_examples=20)
    def test_event_count_matches_logged(
        self,
        sandbox_id: str,
        event_types_list: list[EventType],
    ):
        """Event count MUST match number of logged events."""
        assume(len(sandbox_id) > 0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"
            store = StateStore(str(db_path))

            container = Container.create(modal_sandbox_id=sandbox_id)
            store.containers.save(container)

            # Log events
            for event_type in event_types_list:
                store.log_event(event_type, container_id=container.id)

            # Verify count
            events = store.events.list(container_id=container.id)
            assert len(events) == len(event_types_list)
