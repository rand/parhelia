"""Heartbeat monitoring for container health detection.

Implements:
- [SPEC-03.12] Heartbeat Monitoring
- [SPEC-21.12] Heartbeat History Schema
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from parhelia.state import StateStore


class HeartbeatState(Enum):
    """State of a monitored session's heartbeat."""

    HEALTHY = "healthy"
    WARNING = "warning"  # Missed 1-2 heartbeats
    CRITICAL = "critical"  # Missed threshold heartbeats
    DEAD = "dead"  # Confirmed failure


@dataclass
class HeartbeatInfo:
    """Information about a session's heartbeat status.

    Implements [SPEC-03.12].
    """

    session_id: str
    last_heartbeat: datetime
    missed_count: int = 0
    state: HeartbeatState = HeartbeatState.HEALTHY
    consecutive_healthy: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time_since_heartbeat(self) -> timedelta:
        """Get time since last heartbeat."""
        return datetime.now() - self.last_heartbeat

    @property
    def is_healthy(self) -> bool:
        """Check if session is healthy."""
        return self.state == HeartbeatState.HEALTHY


@dataclass
class HeartbeatEvent:
    """Event emitted when heartbeat state changes.

    Implements [SPEC-03.12].
    """

    session_id: str
    previous_state: HeartbeatState
    new_state: HeartbeatState
    missed_count: int
    timestamp: datetime = field(default_factory=datetime.now)


# Type alias for heartbeat callbacks
HeartbeatCallback = Callable[[HeartbeatEvent], Awaitable[None]]


class HeartbeatMonitor:
    """Monitor heartbeats from sessions to detect failures.

    Implements [SPEC-03.12].

    The heartbeat monitor:
    - Tracks heartbeats from registered sessions
    - Detects missed heartbeats
    - Triggers callbacks when state changes
    - Supports configurable intervals and thresholds
    """

    DEFAULT_INTERVAL = 30.0  # seconds
    DEFAULT_MISSED_THRESHOLD = 3

    def __init__(
        self,
        interval: float = DEFAULT_INTERVAL,
        missed_threshold: int = DEFAULT_MISSED_THRESHOLD,
        on_warning: HeartbeatCallback | None = None,
        on_critical: HeartbeatCallback | None = None,
        on_dead: HeartbeatCallback | None = None,
        on_recovery: HeartbeatCallback | None = None,
        state_store: "StateStore | None" = None,
    ):
        """Initialize the heartbeat monitor.

        Args:
            interval: Expected heartbeat interval in seconds.
            missed_threshold: Number of missed heartbeats before CRITICAL state.
            on_warning: Callback when session enters WARNING state.
            on_critical: Callback when session enters CRITICAL state.
            on_dead: Callback when session enters DEAD state.
            on_recovery: Callback when session recovers to HEALTHY.
            state_store: Optional state store for persisting heartbeats.
        """
        self.interval = interval
        self.missed_threshold = missed_threshold

        # Callbacks
        self._on_warning = on_warning
        self._on_critical = on_critical
        self._on_dead = on_dead
        self._on_recovery = on_recovery

        # State store for persistence
        self._state_store = state_store

        # Session tracking
        self._sessions: dict[str, HeartbeatInfo] = {}
        # Map session_id to container_id for StateStore lookups
        self._session_to_container: dict[str, str] = {}
        self._monitor_task: asyncio.Task | None = None
        self._running = False

    def register_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        container_id: str | None = None,
    ) -> HeartbeatInfo:
        """Register a session for heartbeat monitoring.

        Args:
            session_id: The session ID to monitor.
            metadata: Optional metadata to attach.
            container_id: Optional container ID for StateStore persistence.

        Returns:
            HeartbeatInfo for the session.
        """
        info = HeartbeatInfo(
            session_id=session_id,
            last_heartbeat=datetime.now(),
            metadata=metadata or {},
        )
        self._sessions[session_id] = info

        # Track container mapping for StateStore persistence
        if container_id:
            self._session_to_container[session_id] = container_id

        return info

    def unregister_session(self, session_id: str) -> HeartbeatInfo | None:
        """Stop monitoring a session.

        Args:
            session_id: The session to stop monitoring.

        Returns:
            The removed HeartbeatInfo, or None if not found.
        """
        # Clean up container mapping
        self._session_to_container.pop(session_id, None)
        return self._sessions.pop(session_id, None)

    def record_heartbeat(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> HeartbeatInfo | None:
        """Record a heartbeat from a session.

        Implements [SPEC-03.12] and [SPEC-21.12].

        Args:
            session_id: The session sending the heartbeat.
            metadata: Optional updated metadata.

        Returns:
            Updated HeartbeatInfo, or None if session not registered.
        """
        info = self._sessions.get(session_id)
        if not info:
            return None

        # Update heartbeat
        info.last_heartbeat = datetime.now()
        info.missed_count = 0
        info.consecutive_healthy += 1

        if metadata:
            info.metadata.update(metadata)

        # Persist heartbeat to StateStore if configured
        self._persist_heartbeat(session_id, metadata or {})

        # Handle recovery from non-healthy states
        if info.state != HeartbeatState.HEALTHY:
            previous = info.state
            info.state = HeartbeatState.HEALTHY
            # Schedule recovery callback
            if self._on_recovery:
                asyncio.create_task(
                    self._emit_event(
                        info,
                        previous,
                        HeartbeatState.HEALTHY,
                        self._on_recovery,
                    )
                )

        return info

    def _persist_heartbeat(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist heartbeat to StateStore.

        Implements [SPEC-21.12].

        Args:
            session_id: The session ID.
            metadata: Heartbeat metadata with optional metrics.
        """
        if not self._state_store:
            return

        container_id = self._session_to_container.get(session_id)
        if not container_id:
            return

        from parhelia.state import Heartbeat as StateHeartbeat

        heartbeat = StateHeartbeat.create(
            container_id=container_id,
            cpu_percent=metadata.get("cpu_percent"),
            memory_percent=metadata.get("memory_percent"),
            memory_mb=metadata.get("memory_mb"),
            disk_percent=metadata.get("disk_percent"),
            uptime_seconds=metadata.get("uptime_seconds"),
            tmux_active=metadata.get("tmux_active", False),
            claude_responsive=metadata.get("claude_responsive", False),
            metadata=metadata,
        )
        self._state_store.record_heartbeat(heartbeat)

    def get_session_info(self, session_id: str) -> HeartbeatInfo | None:
        """Get heartbeat info for a session.

        Args:
            session_id: The session ID.

        Returns:
            HeartbeatInfo or None if not found.
        """
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> list[HeartbeatInfo]:
        """Get all monitored sessions.

        Returns:
            List of HeartbeatInfo for all sessions.
        """
        return list(self._sessions.values())

    def get_unhealthy_sessions(self) -> list[HeartbeatInfo]:
        """Get sessions that are not healthy.

        Returns:
            List of unhealthy HeartbeatInfo.
        """
        return [s for s in self._sessions.values() if not s.is_healthy]

    async def check_heartbeats(self) -> list[HeartbeatEvent]:
        """Check all sessions for missed heartbeats.

        Implements [SPEC-03.12].

        Returns:
            List of HeartbeatEvents for state changes.
        """
        events = []
        now = datetime.now()
        expected_interval = timedelta(seconds=self.interval)

        for info in self._sessions.values():
            time_since = now - info.last_heartbeat

            # Calculate missed heartbeats
            if time_since > expected_interval:
                missed = int(time_since.total_seconds() / self.interval)
                if missed > info.missed_count:
                    info.missed_count = missed
                    info.consecutive_healthy = 0

                    # Determine new state
                    previous = info.state
                    if missed >= self.missed_threshold + 2:
                        new_state = HeartbeatState.DEAD
                    elif missed >= self.missed_threshold:
                        new_state = HeartbeatState.CRITICAL
                    elif missed >= 1:
                        new_state = HeartbeatState.WARNING
                    else:
                        new_state = HeartbeatState.HEALTHY

                    # Emit event if state changed
                    if new_state != previous:
                        info.state = new_state
                        event = HeartbeatEvent(
                            session_id=info.session_id,
                            previous_state=previous,
                            new_state=new_state,
                            missed_count=missed,
                        )
                        events.append(event)

                        # Trigger appropriate callback
                        callback = self._get_callback_for_state(new_state)
                        if callback:
                            await callback(event)

        return events

    async def start(self) -> None:
        """Start the heartbeat monitor background task.

        Implements [SPEC-03.12].
        """
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the heartbeat monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    async def _monitor_loop(self) -> None:
        """Background loop that checks heartbeats periodically."""
        while self._running:
            try:
                await self.check_heartbeats()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but keep running
                await asyncio.sleep(self.interval)

    async def _emit_event(
        self,
        info: HeartbeatInfo,
        previous: HeartbeatState,
        new_state: HeartbeatState,
        callback: HeartbeatCallback,
    ) -> None:
        """Emit a heartbeat event to a callback."""
        event = HeartbeatEvent(
            session_id=info.session_id,
            previous_state=previous,
            new_state=new_state,
            missed_count=info.missed_count,
        )
        try:
            await callback(event)
        except Exception:
            # Callback errors should not break the monitor
            pass

    def _get_callback_for_state(
        self, state: HeartbeatState
    ) -> HeartbeatCallback | None:
        """Get the appropriate callback for a state."""
        callbacks = {
            HeartbeatState.WARNING: self._on_warning,
            HeartbeatState.CRITICAL: self._on_critical,
            HeartbeatState.DEAD: self._on_dead,
        }
        return callbacks.get(state)


class HeartbeatSender:
    """Send heartbeats to a monitor (for use in containers).

    Implements [SPEC-03.12].
    """

    def __init__(
        self,
        session_id: str,
        interval: float = HeartbeatMonitor.DEFAULT_INTERVAL,
        send_func: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ):
        """Initialize the heartbeat sender.

        Args:
            session_id: The session ID to send heartbeats for.
            interval: Interval between heartbeats in seconds.
            send_func: Async function to send heartbeat. If None, uses local file.
        """
        self.session_id = session_id
        self.interval = interval
        self._send_func = send_func or self._default_send

        self._sender_task: asyncio.Task | None = None
        self._running = False
        self._metadata: dict[str, Any] = {}

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata to include in heartbeats.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value

    async def send_heartbeat(self) -> None:
        """Send a single heartbeat.

        Implements [SPEC-03.12].
        """
        metadata = {
            **self._metadata,
            "timestamp": datetime.now().isoformat(),
        }
        await self._send_func(self.session_id, metadata)

    async def start(self) -> None:
        """Start sending heartbeats periodically."""
        if self._running:
            return

        self._running = True
        self._sender_task = asyncio.create_task(self._sender_loop())

    async def stop(self) -> None:
        """Stop sending heartbeats."""
        self._running = False
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None

    @property
    def is_running(self) -> bool:
        """Check if sender is running."""
        return self._running

    async def _sender_loop(self) -> None:
        """Background loop that sends heartbeats."""
        while self._running:
            try:
                await self.send_heartbeat()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception:
                # Keep trying on errors
                await asyncio.sleep(self.interval)

    async def _default_send(
        self, session_id: str, metadata: dict[str, Any]
    ) -> None:
        """Default heartbeat send via local file (for testing)."""
        import json
        from pathlib import Path

        heartbeat_dir = Path("/tmp/claude/heartbeats")
        heartbeat_dir.mkdir(parents=True, exist_ok=True)

        heartbeat_file = heartbeat_dir / f"{session_id}.json"
        heartbeat_file.write_text(json.dumps(metadata))
