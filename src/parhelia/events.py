"""Real-time events and streaming for control plane introspection.

Provides centralized event logging with persistence, streaming, filtering,
and export capabilities.

Implements:
- [SPEC-20.40] Event Streaming Architecture
- [SPEC-20.41] Event Filtering
- [SPEC-20.42] Event Replay
- [SPEC-20.43] Event Export
"""

from __future__ import annotations

import asyncio
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

from parhelia.state import Event, EventStore, EventType, StateStore

if TYPE_CHECKING:
    pass


# =============================================================================
# Event Listener Protocol
# =============================================================================


@runtime_checkable
class EventListener(Protocol):
    """Protocol for event stream listeners.

    Listeners receive events as they are logged and can process them
    asynchronously (e.g., for MCP streaming, webhooks, or UI updates).
    """

    @abstractmethod
    async def on_event(self, event: Event) -> None:
        """Called when a new event is logged.

        Args:
            event: The event that was just logged.
        """
        ...


# =============================================================================
# Event Filter
# =============================================================================


@dataclass
class EventFilter:
    """Filter events by type, level, container, or time range.

    Used to select specific events for streaming, replay, or export.

    Attributes:
        event_types: List of event types to include (None = all).
        levels: List of severity levels to include (None = all).
        container_id: Filter to specific container.
        task_id: Filter to specific task.
        worker_id: Filter to specific worker.
        since: Only events after this timestamp.
        until: Only events before this timestamp.
    """

    event_types: list[EventType] | None = None
    levels: list[str] | None = None  # debug, info, warning, error
    container_id: str | None = None
    task_id: str | None = None
    worker_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None

    # Level to event type mapping for filtering
    _LEVEL_EVENT_TYPES: dict[str, list[EventType]] = field(
        default_factory=lambda: {
            "debug": [EventType.HEARTBEAT_RECEIVED],
            "info": [
                EventType.CONTAINER_CREATED,
                EventType.CONTAINER_STARTED,
                EventType.CONTAINER_STOPPED,
                EventType.CONTAINER_TERMINATED,
                EventType.CONTAINER_HEALTHY,
                EventType.CONTAINER_RECOVERED,
            ],
            "warning": [
                EventType.CONTAINER_DEGRADED,
                EventType.HEARTBEAT_MISSED,
                EventType.ORPHAN_DETECTED,
                EventType.STATE_DRIFT_CORRECTED,
            ],
            "error": [
                EventType.CONTAINER_UNHEALTHY,
                EventType.CONTAINER_DEAD,
                EventType.RECONCILE_FAILED,
                EventType.ERROR,
            ],
        },
        repr=False,
    )

    def __post_init__(self):
        """Initialize level mapping after dataclass init."""
        # Set up level mapping if not set
        object.__setattr__(
            self,
            "_LEVEL_EVENT_TYPES",
            {
                "debug": [EventType.HEARTBEAT_RECEIVED],
                "info": [
                    EventType.CONTAINER_CREATED,
                    EventType.CONTAINER_STARTED,
                    EventType.CONTAINER_STOPPED,
                    EventType.CONTAINER_TERMINATED,
                    EventType.CONTAINER_HEALTHY,
                    EventType.CONTAINER_RECOVERED,
                ],
                "warning": [
                    EventType.CONTAINER_DEGRADED,
                    EventType.HEARTBEAT_MISSED,
                    EventType.ORPHAN_DETECTED,
                    EventType.STATE_DRIFT_CORRECTED,
                ],
                "error": [
                    EventType.CONTAINER_UNHEALTHY,
                    EventType.CONTAINER_DEAD,
                    EventType.RECONCILE_FAILED,
                    EventType.ERROR,
                ],
            },
        )

    def matches(self, event: Event) -> bool:
        """Check if an event matches this filter.

        Args:
            event: Event to check.

        Returns:
            True if the event matches all filter criteria.
        """
        # Check event type filter
        if self.event_types is not None:
            if event.event_type not in self.event_types:
                return False

        # Check level filter
        if self.levels is not None:
            allowed_types = set()
            for level in self.levels:
                level_types = self._LEVEL_EVENT_TYPES.get(level, [])
                allowed_types.update(level_types)
            if event.event_type not in allowed_types:
                return False

        # Check container filter
        if self.container_id is not None:
            if event.container_id != self.container_id:
                return False

        # Check task filter
        if self.task_id is not None:
            if event.task_id != self.task_id:
                return False

        # Check worker filter
        if self.worker_id is not None:
            if event.worker_id != self.worker_id:
                return False

        # Check time range
        if self.since is not None:
            if event.timestamp < self.since:
                return False

        if self.until is not None:
            if event.timestamp > self.until:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert filter to dictionary representation."""
        return {
            "event_types": [t.value for t in self.event_types] if self.event_types else None,
            "levels": self.levels,
            "container_id": self.container_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "since": self.since.isoformat() if self.since else None,
            "until": self.until.isoformat() if self.until else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventFilter":
        """Create filter from dictionary."""
        event_types = None
        if data.get("event_types"):
            event_types = [EventType(t) for t in data["event_types"]]

        since = None
        if data.get("since"):
            since = datetime.fromisoformat(data["since"])

        until = None
        if data.get("until"):
            until = datetime.fromisoformat(data["until"])

        return cls(
            event_types=event_types,
            levels=data.get("levels"),
            container_id=data.get("container_id"),
            task_id=data.get("task_id"),
            worker_id=data.get("worker_id"),
            since=since,
            until=until,
        )


# =============================================================================
# Event Logger
# =============================================================================


class EventLogger:
    """Central event logging with persistence and streaming.

    Provides a unified interface for logging events, notifying listeners,
    and querying historical events.

    Implements [SPEC-20.40].

    Usage:
        logger = EventLogger(state_store)
        logger.add_listener(my_listener)

        # Log an event
        event = logger.log(
            EventType.CONTAINER_STARTED,
            container_id="c-abc123",
            data={"modal_sandbox_id": "sb-xyz"},
        )

        # Stream events
        async for event in logger.stream(filter=EventFilter(levels=["error"])):
            print(event)
    """

    def __init__(self, state_store: StateStore):
        """Initialize the event logger.

        Args:
            state_store: StateStore for event persistence.
        """
        self.state_store = state_store
        self._listeners: list[EventListener] = []
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._streaming: bool = False

    def log(
        self,
        event_type: EventType,
        container_id: str | None = None,
        data: dict[str, Any] | None = None,
        level: str = "info",
        message: str | None = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        session_id: str | None = None,
        source: str = "system",
    ) -> Event:
        """Log an event and notify listeners.

        Args:
            event_type: Type of event.
            container_id: Associated container ID.
            data: Additional event data.
            level: Severity level (debug, info, warning, error).
            message: Human-readable message.
            task_id: Associated task ID.
            worker_id: Associated worker ID.
            session_id: Associated session ID.
            source: Event source (system, user, reconciler, etc.).

        Returns:
            The created Event.
        """
        # Create the event
        event = Event.create(
            event_type=event_type,
            container_id=container_id,
            task_id=task_id,
            worker_id=worker_id,
            session_id=session_id,
            message=message,
            source=source,
            details=data or {},
        )

        # Persist to database
        event_id = self.state_store.events.save(event)
        event.id = event_id

        # Queue for streaming
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event if queue is full
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass

        # Notify listeners asynchronously
        asyncio.create_task(self._notify_listeners(event))

        return event

    async def _notify_listeners(self, event: Event) -> None:
        """Notify all listeners of a new event.

        Args:
            event: The event to broadcast.
        """
        for listener in self._listeners:
            try:
                await listener.on_event(event)
            except Exception:
                # Don't let listener errors break the logging
                pass

    def get_listeners(self) -> list[EventListener]:
        """Get all registered listeners.

        Returns:
            List of registered EventListener instances.
        """
        return list(self._listeners)

    def add_listener(self, listener: EventListener) -> None:
        """Register a listener for event notifications.

        Args:
            listener: EventListener to register.
        """
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        """Unregister a listener.

        Args:
            listener: EventListener to remove.
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def stream(
        self,
        filter: EventFilter | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[Event]:
        """Stream events as they are logged.

        Args:
            filter: Optional filter for events.
            timeout: Optional timeout in seconds.

        Yields:
            Events matching the filter.
        """
        self._streaming = True
        start_time = datetime.utcnow()

        try:
            while self._streaming:
                try:
                    # Wait for next event with timeout
                    wait_timeout = 1.0
                    if timeout:
                        elapsed = (datetime.utcnow() - start_time).total_seconds()
                        if elapsed >= timeout:
                            return
                        wait_timeout = min(1.0, timeout - elapsed)

                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=wait_timeout,
                    )

                    # Apply filter
                    if filter is None or filter.matches(event):
                        yield event

                except asyncio.TimeoutError:
                    # Continue waiting unless timeout specified
                    if timeout:
                        elapsed = (datetime.utcnow() - start_time).total_seconds()
                        if elapsed >= timeout:
                            return
                    continue
        finally:
            self._streaming = False

    def stop_streaming(self) -> None:
        """Stop the event stream."""
        self._streaming = False

    def get_events(
        self,
        filter: EventFilter | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        """Query historical events.

        Args:
            filter: Optional filter criteria.
            limit: Maximum events to return.
            offset: Number of events to skip.

        Returns:
            List of matching events.
        """
        # Build query parameters from filter
        kwargs: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if filter:
            if filter.container_id:
                kwargs["container_id"] = filter.container_id
            if filter.task_id:
                kwargs["task_id"] = filter.task_id
            if filter.event_types and len(filter.event_types) == 1:
                kwargs["event_type"] = filter.event_types[0]
            if filter.since:
                kwargs["since"] = filter.since
            if filter.until:
                kwargs["until"] = filter.until

        events = self.state_store.events.list(**kwargs)

        # Apply additional filtering that can't be done in SQL
        if filter:
            events = [e for e in events if filter.matches(e)]

        return events

    def replay(
        self,
        container_id: str,
        from_start: bool = False,
        since: datetime | None = None,
    ) -> list[Event]:
        """Replay historical events for a container.

        Args:
            container_id: Container to replay events for.
            from_start: If True, start from container creation.
            since: If set, start from this timestamp.

        Returns:
            List of events in chronological order.
        """
        filter = EventFilter(
            container_id=container_id,
            since=since if not from_start else None,
        )

        events = self.get_events(filter=filter, limit=1000)

        # Sort chronologically (events are returned in reverse order)
        events.sort(key=lambda e: e.timestamp)

        return events


# =============================================================================
# Event Exporter
# =============================================================================


class EventExporter:
    """Export events to various formats.

    Supports JSONL (JSON Lines) and JSON array formats for
    integration with external tools and analysis.
    """

    def to_jsonl(self, events: list[Event], file_path: str | Path) -> int:
        """Export events to JSONL format.

        JSONL (JSON Lines) is ideal for streaming and log analysis tools.
        Each line is a valid JSON object.

        Args:
            events: List of events to export.
            file_path: Output file path.

        Returns:
            Number of events written.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(file_path, "w") as f:
            for event in events:
                line = self._event_to_json(event)
                f.write(line + "\n")
                count += 1

        return count

    def to_json(self, events: list[Event]) -> str:
        """Export events to JSON array format.

        Returns a JSON array containing all events.

        Args:
            events: List of events to export.

        Returns:
            JSON string with event array.
        """
        event_dicts = [self._event_to_dict(e) for e in events]
        return json.dumps(event_dicts, indent=2)

    def _event_to_dict(self, event: Event) -> dict[str, Any]:
        """Convert event to dictionary.

        Args:
            event: Event to convert.

        Returns:
            Dictionary representation.
        """
        return {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "container_id": event.container_id,
            "worker_id": event.worker_id,
            "task_id": event.task_id,
            "session_id": event.session_id,
            "old_value": event.old_value,
            "new_value": event.new_value,
            "message": event.message,
            "details": event.details,
            "source": event.source,
        }

    def _event_to_json(self, event: Event) -> str:
        """Convert event to JSON string (single line).

        Args:
            event: Event to convert.

        Returns:
            Single-line JSON string.
        """
        return json.dumps(self._event_to_dict(event), separators=(",", ":"))


# =============================================================================
# Queue-based Listener for MCP Streaming
# =============================================================================


class QueueEventListener:
    """Event listener that queues events for async consumption.

    Used for MCP streaming and other async event consumers.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the queue listener.

        Args:
            max_size: Maximum queue size (oldest events dropped if full).
        """
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_size)
        self._filter: EventFilter | None = None

    def set_filter(self, filter: EventFilter | None) -> None:
        """Set the filter for this listener.

        Args:
            filter: Filter to apply to events.
        """
        self._filter = filter

    async def on_event(self, event: Event) -> None:
        """Handle incoming event.

        Args:
            event: Event to process.
        """
        # Apply filter
        if self._filter and not self._filter.matches(event):
            return

        # Queue the event
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass

    async def get(self, timeout: float | None = None) -> Event | None:
        """Get next event from queue.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            Next event or None if timeout.
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout,
                )
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if no events queued.
        """
        return self._queue.empty()


# =============================================================================
# Subscription Manager for MCP
# =============================================================================


@dataclass
class EventSubscription:
    """Represents an active event subscription.

    Used for MCP clients subscribing to event streams.
    """

    id: str
    filter: EventFilter
    listener: QueueEventListener
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subscription_id": self.id,
            "filter": self.filter.to_dict(),
            "created_at": self.created_at.isoformat(),
        }


class SubscriptionManager:
    """Manages event subscriptions for MCP streaming.

    Handles subscription lifecycle and event delivery to
    multiple concurrent clients.
    """

    def __init__(self, logger: EventLogger):
        """Initialize the subscription manager.

        Args:
            logger: EventLogger to subscribe to.
        """
        self.logger = logger
        self._subscriptions: dict[str, EventSubscription] = {}
        self._next_id = 1

    def subscribe(self, filter: EventFilter | None = None) -> EventSubscription:
        """Create a new subscription.

        Args:
            filter: Optional filter for events.

        Returns:
            EventSubscription for receiving events.
        """
        import uuid

        sub_id = f"sub-{uuid.uuid4().hex[:8]}"

        listener = QueueEventListener()
        if filter:
            listener.set_filter(filter)

        subscription = EventSubscription(
            id=sub_id,
            filter=filter or EventFilter(),
            listener=listener,
        )

        self._subscriptions[sub_id] = subscription
        self.logger.add_listener(listener)

        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """Cancel a subscription.

        Args:
            subscription_id: ID of subscription to cancel.

        Returns:
            True if subscription was found and removed.
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions.pop(subscription_id)
        self.logger.remove_listener(subscription.listener)

        return True

    def get_subscription(self, subscription_id: str) -> EventSubscription | None:
        """Get a subscription by ID.

        Args:
            subscription_id: ID of subscription.

        Returns:
            EventSubscription or None if not found.
        """
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self) -> list[EventSubscription]:
        """List all active subscriptions.

        Returns:
            List of active subscriptions.
        """
        return list(self._subscriptions.values())

    async def get_next_event(
        self,
        subscription_id: str,
        timeout: float = 30.0,
    ) -> Event | None:
        """Get next event for a subscription.

        Args:
            subscription_id: ID of subscription.
            timeout: Timeout in seconds.

        Returns:
            Next event or None if timeout or subscription not found.
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return None

        return await subscription.listener.get(timeout=timeout)
