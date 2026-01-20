"""Human intervention signaling for Parhelia.

Implements:
- [SPEC-02.17] Human Intervention Signaling
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from parhelia.session import Session, SessionState


class InterventionReason(Enum):
    """Reasons for requesting human intervention."""

    CLAUDE_REQUESTED = "claude_requested"  # Claude explicitly asked for help
    TIMEOUT = "timeout"  # No progress for configurable duration
    ERROR = "error"  # Unrecoverable error occurred
    PERMISSION = "permission"  # Permission/authorization needed
    BUDGET = "budget"  # Budget threshold reached
    MANUAL = "manual"  # Manually triggered by operator


class InterventionState(Enum):
    """State of an intervention request."""

    PENDING = "pending"  # Waiting for human response
    ACKNOWLEDGED = "acknowledged"  # Human has seen the request
    RESOLVED = "resolved"  # Human has taken action
    DISMISSED = "dismissed"  # Human dismissed without action
    EXPIRED = "expired"  # Request expired without response


@dataclass
class InterventionAction:
    """An action that can be taken on an intervention."""

    label: str
    url: str
    description: str = ""


@dataclass
class InterventionRequest:
    """A request for human intervention.

    Implements [SPEC-02.17].
    """

    id: str
    session_id: str
    reason: InterventionReason
    context: str
    state: InterventionState = InterventionState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    suggested_action: str = ""
    actions: list[InterventionAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Response fields
    response: str | None = None
    responded_by: str | None = None


@dataclass
class InterventionConfig:
    """Configuration for intervention management."""

    timeout_minutes: float = 10.0  # Idle timeout before intervention
    check_interval_seconds: float = 60.0  # How often to check for timeout
    max_pending_interventions: int = 10  # Max pending per session
    auto_checkpoint_on_intervention: bool = True
    notification_channels: list[str] = field(default_factory=lambda: ["cli"])


# Type aliases for callbacks
InterventionCallback = Callable[[InterventionRequest], Awaitable[None]]
NotificationCallback = Callable[[str, str, list[InterventionAction]], Awaitable[None]]


class InterventionManager:
    """Manage human intervention requests.

    Implements [SPEC-02.17].

    Handles:
    - Detection of intervention signals from Claude output
    - Timeout-based intervention triggering
    - Notification dispatch
    - Intervention lifecycle management
    """

    # Signal patterns to watch for in Claude output
    NEEDS_HUMAN_PATTERN = '"type": "needs_human"'
    NEEDS_HUMAN_ALT_PATTERN = '"type":"needs_human"'

    def __init__(
        self,
        config: InterventionConfig | None = None,
    ):
        """Initialize the intervention manager.

        Args:
            config: Intervention configuration.
        """
        self.config = config or InterventionConfig()
        self._requests: dict[str, InterventionRequest] = {}
        self._session_requests: dict[str, list[str]] = {}  # session_id -> request_ids

        # Callbacks
        self._on_intervention: list[InterventionCallback] = []
        self._on_resolved: list[InterventionCallback] = []
        self._notification_handler: NotificationCallback | None = None

        # Request counter for ID generation
        self._request_counter = 0

    async def create_intervention(
        self,
        session_id: str,
        reason: InterventionReason,
        context: str,
        suggested_action: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> InterventionRequest:
        """Create a new intervention request.

        Implements [SPEC-02.17].

        Args:
            session_id: The session needing intervention.
            reason: Why intervention is needed.
            context: Detailed context for the human.
            suggested_action: Suggested action to take.
            metadata: Additional metadata.

        Returns:
            The created InterventionRequest.
        """
        self._request_counter += 1
        request_id = f"int-{session_id[:8]}-{self._request_counter:04d}"

        # Build default actions
        actions = [
            InterventionAction(
                label="Attach",
                url=f"parhelia://attach/{session_id}",
                description="Attach to the session interactively",
            ),
            InterventionAction(
                label="Dismiss",
                url=f"parhelia://dismiss/{request_id}",
                description="Dismiss this intervention request",
            ),
        ]

        request = InterventionRequest(
            id=request_id,
            session_id=session_id,
            reason=reason,
            context=context,
            suggested_action=suggested_action,
            actions=actions,
            metadata=metadata or {},
        )

        self._requests[request_id] = request

        # Track by session
        if session_id not in self._session_requests:
            self._session_requests[session_id] = []
        self._session_requests[session_id].append(request_id)

        # Trigger callbacks
        for callback in self._on_intervention:
            await callback(request)

        # Send notification
        if self._notification_handler:
            title = f"Session {session_id} needs attention"
            body = f"Reason: {reason.value}\n\nContext: {context}"
            if suggested_action:
                body += f"\n\nSuggested action: {suggested_action}"
            await self._notification_handler(title, body, actions)

        return request

    async def acknowledge(self, request_id: str) -> InterventionRequest:
        """Acknowledge an intervention request.

        Args:
            request_id: The request ID.

        Returns:
            Updated InterventionRequest.

        Raises:
            ValueError: If request not found.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Intervention request not found: {request_id}")

        request.state = InterventionState.ACKNOWLEDGED
        request.acknowledged_at = datetime.now()

        return request

    async def resolve(
        self,
        request_id: str,
        response: str = "",
        responded_by: str = "",
    ) -> InterventionRequest:
        """Resolve an intervention request.

        Args:
            request_id: The request ID.
            response: Human's response/action taken.
            responded_by: Who resolved it.

        Returns:
            Updated InterventionRequest.

        Raises:
            ValueError: If request not found.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Intervention request not found: {request_id}")

        request.state = InterventionState.RESOLVED
        request.resolved_at = datetime.now()
        request.response = response
        request.responded_by = responded_by

        # Trigger callbacks
        for callback in self._on_resolved:
            await callback(request)

        return request

    async def dismiss(self, request_id: str) -> InterventionRequest:
        """Dismiss an intervention request without action.

        Args:
            request_id: The request ID.

        Returns:
            Updated InterventionRequest.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Intervention request not found: {request_id}")

        request.state = InterventionState.DISMISSED
        request.resolved_at = datetime.now()

        return request

    async def get_request(self, request_id: str) -> InterventionRequest | None:
        """Get an intervention request by ID.

        Args:
            request_id: The request ID.

        Returns:
            InterventionRequest if found, None otherwise.
        """
        return self._requests.get(request_id)

    async def get_session_requests(
        self,
        session_id: str,
        state: InterventionState | None = None,
    ) -> list[InterventionRequest]:
        """Get all intervention requests for a session.

        Args:
            session_id: The session ID.
            state: Filter by state (optional).

        Returns:
            List of InterventionRequests.
        """
        request_ids = self._session_requests.get(session_id, [])
        requests = [self._requests[rid] for rid in request_ids if rid in self._requests]

        if state:
            requests = [r for r in requests if r.state == state]

        return requests

    async def get_pending_requests(self) -> list[InterventionRequest]:
        """Get all pending intervention requests.

        Returns:
            List of pending InterventionRequests.
        """
        return [r for r in self._requests.values() if r.state == InterventionState.PENDING]

    async def has_pending_intervention(self, session_id: str) -> bool:
        """Check if a session has pending intervention.

        Args:
            session_id: The session ID.

        Returns:
            True if session has pending intervention.
        """
        requests = await self.get_session_requests(session_id, InterventionState.PENDING)
        return len(requests) > 0

    def on_intervention(self, callback: InterventionCallback) -> None:
        """Register callback for new interventions."""
        self._on_intervention.append(callback)

    def on_resolved(self, callback: InterventionCallback) -> None:
        """Register callback for resolved interventions."""
        self._on_resolved.append(callback)

    def set_notification_handler(self, handler: NotificationCallback) -> None:
        """Set the notification handler."""
        self._notification_handler = handler


class OutputMonitor:
    """Monitor Claude Code output for intervention signals.

    Implements [SPEC-02.17] Claude-Requested Intervention.
    """

    def __init__(
        self,
        intervention_manager: InterventionManager,
    ):
        """Initialize the output monitor.

        Args:
            intervention_manager: Intervention manager instance.
        """
        self.intervention_manager = intervention_manager

    async def process_line(self, session_id: str, line: str) -> InterventionRequest | None:
        """Process a line of Claude output for intervention signals.

        Implements [SPEC-02.17].

        Args:
            session_id: The session ID.
            line: Line of output to process.

        Returns:
            InterventionRequest if signal detected, None otherwise.
        """
        # Check for needs_human signal
        if (InterventionManager.NEEDS_HUMAN_PATTERN in line or
                InterventionManager.NEEDS_HUMAN_ALT_PATTERN in line):
            try:
                event = json.loads(line)
                if event.get("type") == "needs_human":
                    return await self.intervention_manager.create_intervention(
                        session_id=session_id,
                        reason=InterventionReason.CLAUDE_REQUESTED,
                        context=event.get("context", ""),
                        suggested_action=event.get("suggested_action", ""),
                        metadata={
                            "reason_detail": event.get("reason", ""),
                            "raw_event": event,
                        },
                    )
            except json.JSONDecodeError:
                pass

        return None

    async def monitor_stream(
        self,
        session_id: str,
        stream: asyncio.StreamReader,
    ) -> None:
        """Monitor an output stream for intervention signals.

        Args:
            session_id: The session ID.
            stream: Async stream reader for output.
        """
        while True:
            line = await stream.readline()
            if not line:
                break

            line_str = line.decode("utf-8", errors="ignore").strip()
            await self.process_line(session_id, line_str)


class TimeoutMonitor:
    """Monitor sessions for timeout-based intervention.

    Implements [SPEC-02.17] Timeout-Based Intervention.
    """

    def __init__(
        self,
        intervention_manager: InterventionManager,
        config: InterventionConfig | None = None,
    ):
        """Initialize the timeout monitor.

        Args:
            intervention_manager: Intervention manager instance.
            config: Intervention configuration.
        """
        self.intervention_manager = intervention_manager
        self.config = config or InterventionConfig()

        self._monitoring: dict[str, datetime] = {}  # session_id -> last_activity
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start_monitoring(self, session_id: str) -> None:
        """Start monitoring a session for timeout.

        Args:
            session_id: The session to monitor.
        """
        self._monitoring[session_id] = datetime.now()

    async def stop_monitoring(self, session_id: str) -> None:
        """Stop monitoring a session.

        Args:
            session_id: The session to stop monitoring.
        """
        self._monitoring.pop(session_id, None)

    async def record_activity(self, session_id: str) -> None:
        """Record activity for a session.

        Args:
            session_id: The session with activity.
        """
        if session_id in self._monitoring:
            self._monitoring[session_id] = datetime.now()

    async def check_timeouts(self) -> list[InterventionRequest]:
        """Check all monitored sessions for timeouts.

        Returns:
            List of interventions created for timed-out sessions.
        """
        interventions = []
        now = datetime.now()
        timeout_seconds = self.config.timeout_minutes * 60

        for session_id, last_activity in list(self._monitoring.items()):
            idle_seconds = (now - last_activity).total_seconds()

            if idle_seconds > timeout_seconds:
                # Check if already has pending intervention
                has_pending = await self.intervention_manager.has_pending_intervention(session_id)
                if not has_pending:
                    minutes = idle_seconds / 60
                    intervention = await self.intervention_manager.create_intervention(
                        session_id=session_id,
                        reason=InterventionReason.TIMEOUT,
                        context=f"No progress for {minutes:.0f} minutes",
                        suggested_action="Check session status and provide guidance if needed",
                    )
                    interventions.append(intervention)

        return interventions

    async def start(self) -> None:
        """Start the timeout monitoring loop."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the timeout monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        """Background loop for timeout monitoring."""
        while self._running:
            await asyncio.sleep(self.config.check_interval_seconds)
            await self.check_timeouts()


class NotificationDispatcher:
    """Dispatch notifications for interventions.

    Supports multiple notification channels.
    """

    def __init__(self):
        """Initialize the notification dispatcher."""
        self._handlers: dict[str, NotificationCallback] = {}

    def register_handler(self, channel: str, handler: NotificationCallback) -> None:
        """Register a notification handler for a channel.

        Args:
            channel: Channel name (e.g., "cli", "webhook", "email").
            handler: Async callback to handle notifications.
        """
        self._handlers[channel] = handler

    async def send(
        self,
        title: str,
        body: str,
        actions: list[InterventionAction],
        channels: list[str] | None = None,
    ) -> None:
        """Send notification to specified channels.

        Args:
            title: Notification title.
            body: Notification body.
            actions: Available actions.
            channels: Channels to send to (None = all).
        """
        target_channels = channels or list(self._handlers.keys())

        for channel in target_channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    await handler(title, body, actions)
                except Exception:
                    # Log error but continue with other channels
                    pass


async def cli_notification_handler(
    title: str,
    body: str,
    actions: list[InterventionAction],
) -> None:
    """Default CLI notification handler.

    Prints notification to stdout.

    Args:
        title: Notification title.
        body: Notification body.
        actions: Available actions.
    """
    print(f"\n{'=' * 60}")
    print(f"⚠️  INTERVENTION NEEDED: {title}")
    print(f"{'=' * 60}")
    print(body)
    if actions:
        print("\nAvailable actions:")
        for action in actions:
            print(f"  - {action.label}: {action.url}")
    print(f"{'=' * 60}\n")
