"""Real-time event streaming for session status.

Implements:
- [SPEC-12.40] Event Stream Endpoint
- [SPEC-12.41] Status Watch
- [SPEC-12.42] Heartbeat Events
- [SPEC-11.22] Streaming Output

Provides real-time updates for task and session status monitoring.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

if TYPE_CHECKING:
    from parhelia.persistence import PersistentOrchestrator


class EventType(Enum):
    """Types of streaming events."""

    # Status events
    STATUS_CHANGE = "status_change"  # Task/worker status changed

    # Progress events
    PROGRESS = "progress"  # Progress percentage update
    PHASE = "phase"  # Entered a new phase

    # Activity events
    ACTIVITY = "activity"  # User/Claude activity detected
    OUTPUT = "output"  # New output from Claude

    # Lifecycle events
    HEARTBEAT = "heartbeat"  # Periodic keep-alive
    STARTED = "started"  # Task/session started
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"  # Task failed

    # System events
    WARNING = "warning"  # Budget warning, resource constraint
    ERROR = "error"  # Non-fatal error


@dataclass
class StreamEvent:
    """Base class for streaming events."""

    type: EventType = field(default=EventType.STATUS_CHANGE)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    session_id: str | None = None
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat() + "Z",
            "session_id": self.session_id,
            "task_id": self.task_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string (one line)."""
        return json.dumps(self.to_dict())


@dataclass
class StatusEvent(StreamEvent):
    """Status change event."""

    old_status: str | None = None
    new_status: str | None = None
    worker_state: str | None = None

    def __post_init__(self):
        self.type = EventType.STATUS_CHANGE

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "old_status": self.old_status,
            "new_status": self.new_status,
            "worker_state": self.worker_state,
        })
        return d


@dataclass
class ProgressEvent(StreamEvent):
    """Progress update event."""

    phase: str = ""
    percent: int = 0
    message: str = ""

    def __post_init__(self):
        self.type = EventType.PROGRESS

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "phase": self.phase,
            "percent": self.percent,
            "message": self.message,
        })
        return d


@dataclass
class HeartbeatEvent(StreamEvent):
    """Periodic heartbeat event."""

    uptime_seconds: int = 0
    cpu_percent: float | None = None
    memory_mb: int | None = None

    def __post_init__(self):
        self.type = EventType.HEARTBEAT

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "uptime_seconds": self.uptime_seconds,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
        })
        return d


@dataclass
class ActivityEvent(StreamEvent):
    """Activity detection event."""

    activity_type: str = ""  # "user_input", "claude_response", "file_change", "command"
    summary: str = ""

    def __post_init__(self):
        self.type = EventType.ACTIVITY

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "activity_type": self.activity_type,
            "summary": self.summary,
        })
        return d


@dataclass
class OutputEvent(StreamEvent):
    """Output from Claude event."""

    content: str = ""
    stream: str = "stdout"  # stdout or stderr

    def __post_init__(self):
        self.type = EventType.OUTPUT

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "content": self.content,
            "stream": self.stream,
        })
        return d


@dataclass
class CompletionEvent(StreamEvent):
    """Task completion event."""

    success: bool = True
    output_summary: str = ""
    duration_seconds: float = 0.0
    cost_usd: float = 0.0

    def __post_init__(self):
        self.type = EventType.COMPLETED if self.success else EventType.FAILED

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "success": self.success,
            "output_summary": self.output_summary,
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
        })
        return d


@dataclass
class WarningEvent(StreamEvent):
    """Warning event (budget, resources)."""

    warning_type: str = ""  # "budget_low", "timeout_near", "resource_constraint"
    message: str = ""
    threshold: float | None = None
    current: float | None = None

    def __post_init__(self):
        self.type = EventType.WARNING

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "warning_type": self.warning_type,
            "message": self.message,
            "threshold": self.threshold,
            "current": self.current,
        })
        return d


class EventStream:
    """Watches task/session status and emits events.

    Usage:
        stream = EventStream(orchestrator)
        async for event in stream.watch(task_id="task-123"):
            print(event.to_json())

    Events are emitted as JSON lines for easy parsing.
    """

    DEFAULT_POLL_INTERVAL_SECONDS = 2.0
    HEARTBEAT_INTERVAL_SECONDS = 10.0

    def __init__(
        self,
        orchestrator: "PersistentOrchestrator",
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        """Initialize event stream.

        Args:
            orchestrator: The orchestrator for status queries.
            poll_interval: How often to poll for status changes (seconds).
        """
        self.orchestrator = orchestrator
        self.poll_interval = poll_interval
        self._last_status: dict[str, str] = {}
        self._last_worker_state: dict[str, str] = {}
        self._start_time: datetime | None = None
        self._event_callback: Callable[[StreamEvent], None] | None = None

    def set_event_callback(self, callback: Callable[[StreamEvent], None]) -> None:
        """Set callback for event notifications."""
        self._event_callback = callback

    def _emit(self, event: StreamEvent) -> None:
        """Emit an event through callback if set."""
        if self._event_callback:
            self._event_callback(event)

    async def watch(
        self,
        task_id: str | None = None,
        session_id: str | None = None,
        include_heartbeat: bool = True,
        stop_on_complete: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Watch for events on a task or session.

        Args:
            task_id: Task ID to watch (optional).
            session_id: Session ID to watch (optional, defaults to task_id).
            include_heartbeat: Emit periodic heartbeat events.
            stop_on_complete: Stop iteration when task completes.

        Yields:
            StreamEvent objects.
        """
        self._start_time = datetime.now(UTC)
        last_heartbeat = datetime.now(UTC)

        # Initial status fetch
        if task_id:
            status = await self._get_task_status(task_id)
            if status:
                self._last_status[task_id] = status
                yield StatusEvent(
                    task_id=task_id,
                    session_id=session_id or task_id,
                    old_status=None,
                    new_status=status,
                )

        while True:
            # Check for status changes
            if task_id:
                status = await self._get_task_status(task_id)
                if status and status != self._last_status.get(task_id):
                    event = StatusEvent(
                        task_id=task_id,
                        session_id=session_id or task_id,
                        old_status=self._last_status.get(task_id),
                        new_status=status,
                    )
                    self._last_status[task_id] = status
                    yield event
                    self._emit(event)

                    # Check for completion
                    if stop_on_complete and status in ("completed", "failed"):
                        result = await self.orchestrator.collect_results(task_id)
                        yield CompletionEvent(
                            task_id=task_id,
                            session_id=session_id or task_id,
                            success=(status == "completed"),
                            output_summary=result.output[:200] if result else "",
                            duration_seconds=result.duration_seconds if result else 0,
                            cost_usd=result.cost_usd if result else 0,
                        )
                        return

                # Check worker state
                worker = self.orchestrator.worker_store.get_by_task(task_id)
                if worker:
                    state = worker.state.value
                    if state != self._last_worker_state.get(task_id):
                        event = StatusEvent(
                            task_id=task_id,
                            session_id=session_id or task_id,
                            old_status=self._last_worker_state.get(task_id),
                            new_status=state,
                            worker_state=state,
                        )
                        self._last_worker_state[task_id] = state
                        yield event
                        self._emit(event)

            # Emit heartbeat if needed
            if include_heartbeat:
                now = datetime.now(UTC)
                if (now - last_heartbeat).total_seconds() >= self.HEARTBEAT_INTERVAL_SECONDS:
                    uptime = int((now - self._start_time).total_seconds())
                    event = HeartbeatEvent(
                        task_id=task_id,
                        session_id=session_id or task_id,
                        uptime_seconds=uptime,
                    )
                    last_heartbeat = now
                    yield event

            await asyncio.sleep(self.poll_interval)

    async def watch_all(
        self,
        status_filter: str | None = None,
        include_heartbeat: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Watch for events across all tasks.

        Args:
            status_filter: Only watch tasks with this status.
            include_heartbeat: Emit periodic heartbeat events.

        Yields:
            StreamEvent objects.
        """
        self._start_time = datetime.now(UTC)
        last_heartbeat = datetime.now(UTC)

        while True:
            # Get current task list
            tasks = await self.orchestrator.list_tasks(status=status_filter)

            for task in tasks:
                task_id = task.id
                status = self.orchestrator.task_store.get_status(task_id)

                if status and status != self._last_status.get(task_id):
                    event = StatusEvent(
                        task_id=task_id,
                        session_id=task_id,
                        old_status=self._last_status.get(task_id),
                        new_status=status,
                    )
                    self._last_status[task_id] = status
                    yield event
                    self._emit(event)

            # Emit heartbeat
            if include_heartbeat:
                now = datetime.now(UTC)
                if (now - last_heartbeat).total_seconds() >= self.HEARTBEAT_INTERVAL_SECONDS:
                    uptime = int((now - self._start_time).total_seconds())
                    yield HeartbeatEvent(uptime_seconds=uptime)
                    last_heartbeat = now

            await asyncio.sleep(self.poll_interval)

    async def _get_task_status(self, task_id: str) -> str | None:
        """Get current task status."""
        return self.orchestrator.task_store.get_status(task_id)


class EventFormatter:
    """Formats events for output (JSON or human-readable)."""

    def __init__(self, json_mode: bool = True):
        """Initialize formatter.

        Args:
            json_mode: If True, output JSON lines. If False, human-readable.
        """
        self.json_mode = json_mode

    def format(self, event: StreamEvent) -> str:
        """Format an event for output."""
        if self.json_mode:
            return event.to_json()
        else:
            return self._format_human(event)

    def _format_human(self, event: StreamEvent) -> str:
        """Format event for human reading."""
        timestamp = event.timestamp.strftime("%H:%M:%S")

        if isinstance(event, StatusEvent):
            status = event.new_status or event.worker_state or "unknown"
            return f"[{timestamp}] Status: {status}"

        elif isinstance(event, ProgressEvent):
            bar = self._progress_bar(event.percent)
            return f"[{timestamp}] {event.phase}: {bar} {event.percent}% - {event.message}"

        elif isinstance(event, HeartbeatEvent):
            uptime = self._format_duration(event.uptime_seconds)
            return f"[{timestamp}] ♥ Running for {uptime}"

        elif isinstance(event, ActivityEvent):
            return f"[{timestamp}] Activity: {event.activity_type} - {event.summary}"

        elif isinstance(event, OutputEvent):
            preview = event.content[:80].replace("\n", " ")
            return f"[{timestamp}] Output: {preview}..."

        elif isinstance(event, CompletionEvent):
            icon = "✓" if event.success else "✗"
            duration = self._format_duration(int(event.duration_seconds))
            return f"[{timestamp}] {icon} {'Completed' if event.success else 'Failed'} ({duration}, ${event.cost_usd:.2f})"

        elif isinstance(event, WarningEvent):
            return f"[{timestamp}] ⚠ {event.warning_type}: {event.message}"

        else:
            return f"[{timestamp}] {event.type.value}"

    def _progress_bar(self, percent: int, width: int = 20) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percent / 100)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _format_duration(self, seconds: int) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"


async def watch_task(
    task_id: str,
    orchestrator: "PersistentOrchestrator",
    json_mode: bool = True,
    callback: Callable[[str], None] | None = None,
) -> None:
    """Convenience function to watch a task and print events.

    Args:
        task_id: Task ID to watch.
        orchestrator: The orchestrator instance.
        json_mode: Output JSON if True, human-readable if False.
        callback: Optional callback for each formatted event line.
    """
    stream = EventStream(orchestrator)
    formatter = EventFormatter(json_mode=json_mode)

    async for event in stream.watch(task_id=task_id):
        line = formatter.format(event)
        if callback:
            callback(line)
        else:
            print(line, flush=True)
