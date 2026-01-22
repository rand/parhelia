"""Tests for real-time events and streaming system.

Tests [SPEC-20.40], [SPEC-20.41], [SPEC-20.42], [SPEC-20.43].
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from parhelia.events import (
    EventExporter,
    EventFilter,
)
from parhelia.state import Event, EventType, StateStore


@pytest.fixture
def state_store(tmp_path: Path) -> StateStore:
    """Create a StateStore with temp database."""
    db_path = tmp_path / "test.db"
    return StateStore(str(db_path))


class TestEventFilter:
    """Tests for EventFilter matching."""

    def test_empty_filter_matches_all(self):
        """Empty filter matches any event."""
        filter = EventFilter()
        event = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="test-123",
        )
        assert filter.matches(event) is True

    def test_filter_by_event_type(self):
        """Filter by event type."""
        filter = EventFilter(event_types=[EventType.CONTAINER_CREATED])

        matching = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="test-123",
        )
        non_matching = Event.create(
            event_type=EventType.CONTAINER_TERMINATED,
            container_id="test-123",
        )

        assert filter.matches(matching) is True
        assert filter.matches(non_matching) is False

    def test_filter_by_multiple_event_types(self):
        """Filter by multiple event types."""
        filter = EventFilter(event_types=[
            EventType.CONTAINER_CREATED,
            EventType.CONTAINER_STARTED,
        ])

        created = Event.create(event_type=EventType.CONTAINER_CREATED, container_id="test")
        started = Event.create(event_type=EventType.CONTAINER_STARTED, container_id="test")
        terminated = Event.create(event_type=EventType.CONTAINER_TERMINATED, container_id="test")

        assert filter.matches(created) is True
        assert filter.matches(started) is True
        assert filter.matches(terminated) is False

    def test_filter_by_container_id(self):
        """Filter by container ID."""
        filter = EventFilter(container_id="container-abc")

        matching = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="container-abc",
        )
        non_matching = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="container-xyz",
        )

        assert filter.matches(matching) is True
        assert filter.matches(non_matching) is False

    def test_filter_by_time_range(self):
        """Filter by time range."""
        from datetime import datetime
        now = datetime.utcnow()
        filter = EventFilter(
            since=now - timedelta(hours=1),
            until=now + timedelta(hours=1),
        )

        # Event within range (created now)
        matching = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="test",
        )

        assert filter.matches(matching) is True

    def test_filter_combines_conditions(self):
        """Multiple filter conditions are ANDed."""
        filter = EventFilter(
            event_types=[EventType.CONTAINER_CREATED],
            container_id="specific-container",
        )

        # Matches both conditions
        matching = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="specific-container",
        )

        # Only matches one condition
        wrong_type = Event.create(
            event_type=EventType.CONTAINER_TERMINATED,
            container_id="specific-container",
        )
        wrong_container = Event.create(
            event_type=EventType.CONTAINER_CREATED,
            container_id="different-container",
        )

        assert filter.matches(matching) is True
        assert filter.matches(wrong_type) is False
        assert filter.matches(wrong_container) is False


class TestEventExporter:
    """Tests for EventExporter."""

    def test_to_jsonl_creates_file(self, state_store: StateStore):
        """Export to JSONL creates valid file."""
        # Create some events
        state_store.log_event(
            event_type=EventType.CONTAINER_CREATED,
            container_id="container-1",
        )
        state_store.log_event(
            event_type=EventType.CONTAINER_STARTED,
            container_id="container-1",
        )

        events = state_store.events.list()
        exporter = EventExporter()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            count = exporter.to_jsonl(events, f.name)

            assert count == len(events)

            # Verify file is valid JSONL
            with open(f.name) as rf:
                lines = rf.readlines()
                assert len(lines) == len(events)
                for line in lines:
                    parsed = json.loads(line)
                    assert "event_type" in parsed

    def test_to_json_returns_valid_json(self, state_store: StateStore):
        """Export to JSON returns valid JSON array."""
        state_store.log_event(
            event_type=EventType.CONTAINER_CREATED,
            container_id="container-1",
        )

        events = state_store.events.list()
        exporter = EventExporter()

        result = exporter.to_json(events)
        parsed = json.loads(result)

        assert isinstance(parsed, list)
        assert len(parsed) == len(events)

    def test_to_jsonl_empty_list(self):
        """Export empty list creates empty file."""
        exporter = EventExporter()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            count = exporter.to_jsonl([], f.name)
            assert count == 0

    def test_to_json_empty_list(self):
        """Export empty list returns empty array."""
        exporter = EventExporter()
        result = exporter.to_json([])
        assert result == "[]"


class TestStateStoreEventLogging:
    """Tests for event logging via StateStore."""

    def test_log_event_returns_id(self, state_store: StateStore):
        """StateStore.log_event returns event ID."""
        event_id = state_store.log_event(
            event_type=EventType.CONTAINER_CREATED,
            container_id="test-container",
        )

        assert isinstance(event_id, int)
        assert event_id > 0

        # Verify persisted via get
        event = state_store.events.get(event_id)
        assert event is not None
        assert event.event_type == EventType.CONTAINER_CREATED
        assert event.container_id == "test-container"

    def test_log_multiple_events(self, state_store: StateStore):
        """Multiple events can be logged."""
        id1 = state_store.log_event(EventType.CONTAINER_CREATED, container_id="c1")
        id2 = state_store.log_event(EventType.CONTAINER_STARTED, container_id="c1")
        id3 = state_store.log_event(EventType.CONTAINER_STOPPED, container_id="c1")

        # All IDs should be different
        assert len({id1, id2, id3}) == 3

        # Verify all persisted
        assert state_store.events.get(id1) is not None
        assert state_store.events.get(id2) is not None
        assert state_store.events.get(id3) is not None

    def test_events_list_by_container(self, state_store: StateStore):
        """Events can be listed by container ID."""
        state_store.log_event(EventType.CONTAINER_CREATED, container_id="c1")
        state_store.log_event(EventType.CONTAINER_STARTED, container_id="c1")
        state_store.log_event(EventType.CONTAINER_CREATED, container_id="c2")

        events_c1 = state_store.events.list(container_id="c1")
        events_c2 = state_store.events.list(container_id="c2")

        assert len(events_c1) == 2
        assert len(events_c2) == 1


class TestEventFilterByLevel:
    """Tests for filtering events by severity level."""

    def test_filter_error_level(self):
        """Filter by error level."""
        filter = EventFilter(levels=["error"])

        # Error-level events
        unhealthy = Event.create(EventType.CONTAINER_UNHEALTHY, "c1")

        # Non-error events
        created = Event.create(EventType.CONTAINER_CREATED, "c1")

        assert filter.matches(unhealthy) is True
        assert filter.matches(created) is False

    def test_filter_warning_level(self):
        """Filter by warning level."""
        filter = EventFilter(levels=["warning"])

        # Warning-level events
        degraded = Event.create(EventType.CONTAINER_DEGRADED, "c1")
        orphan = Event.create(EventType.ORPHAN_DETECTED, "c1")

        assert filter.matches(degraded) is True
        assert filter.matches(orphan) is True

    def test_filter_multiple_levels(self):
        """Filter by multiple levels."""
        filter = EventFilter(levels=["warning", "error"])

        degraded = Event.create(EventType.CONTAINER_DEGRADED, "c1")
        unhealthy = Event.create(EventType.CONTAINER_UNHEALTHY, "c1")
        created = Event.create(EventType.CONTAINER_CREATED, "c1")

        assert filter.matches(degraded) is True
        assert filter.matches(unhealthy) is True
        assert filter.matches(created) is False
