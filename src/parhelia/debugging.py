"""Interactive debugging workflow for Parhelia.

Implements:
- [SPEC-02.15] Session Lifecycle Hooks (debugging integration)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from parhelia.session import Session, SessionState
from parhelia.tmux import TmuxManager, TmuxSession
from parhelia.ssh import SSHTunnelManager, TunnelInfo
from parhelia.intervention import InterventionManager, InterventionRequest


class DebugState(Enum):
    """State of a debugging session."""

    IDLE = "idle"  # Not debugging
    INSPECTING = "inspecting"  # Viewing state only
    ATTACHED = "attached"  # Interactive debugging
    STEPPING = "stepping"  # Step-by-step execution
    PAUSED = "paused"  # Execution paused


@dataclass
class SessionSnapshot:
    """Snapshot of session state for inspection.

    Captures current state of a running session for debugging.
    """

    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Session metadata
    state: SessionState = SessionState.RUNNING
    working_directory: str = ""
    uptime_seconds: float = 0.0

    # Claude Code state
    current_turn: int = 0
    tokens_used: int = 0
    last_tool: str | None = None
    pending_tool_calls: list[str] = field(default_factory=list)

    # Resource usage
    cpu_percent: float = 0.0
    memory_mb: float = 0.0

    # Output capture
    recent_output: str = ""
    last_error: str | None = None

    # Intervention status
    has_pending_intervention: bool = False
    intervention_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "working_directory": self.working_directory,
            "uptime_seconds": self.uptime_seconds,
            "current_turn": self.current_turn,
            "tokens_used": self.tokens_used,
            "last_tool": self.last_tool,
            "pending_tool_calls": self.pending_tool_calls,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "recent_output": self.recent_output,
            "last_error": self.last_error,
            "has_pending_intervention": self.has_pending_intervention,
            "intervention_reason": self.intervention_reason,
        }


@dataclass
class DebugSession:
    """An active debugging session.

    Tracks state of debugging activities on a session.
    """

    session_id: str
    state: DebugState = DebugState.IDLE
    started_at: datetime = field(default_factory=datetime.now)
    attached_at: datetime | None = None
    snapshots: list[SessionSnapshot] = field(default_factory=list)
    breakpoints: list[str] = field(default_factory=list)  # Tool names to break on
    metadata: dict[str, Any] = field(default_factory=dict)


# Type alias for debug callbacks
DebugCallback = Callable[[DebugSession, SessionSnapshot], Awaitable[None]]


class DebugManager:
    """Manage interactive debugging sessions.

    Implements [SPEC-02.15] debugging workflow.

    Coordinates:
    - Session state inspection
    - SSH tunnel attachment
    - tmux session interaction
    - Intervention handling
    """

    def __init__(
        self,
        tmux_manager: TmuxManager | None = None,
        tunnel_manager: SSHTunnelManager | None = None,
        intervention_manager: InterventionManager | None = None,
    ):
        """Initialize the debug manager.

        Args:
            tmux_manager: tmux session manager.
            tunnel_manager: SSH tunnel manager.
            intervention_manager: Intervention manager.
        """
        self.tmux_manager = tmux_manager or TmuxManager()
        self.tunnel_manager = tunnel_manager or SSHTunnelManager()
        self.intervention_manager = intervention_manager or InterventionManager()

        self._debug_sessions: dict[str, DebugSession] = {}
        self._on_snapshot: list[DebugCallback] = []
        self._on_breakpoint: list[DebugCallback] = []

    async def start_debug(self, session_id: str) -> DebugSession:
        """Start a debugging session.

        Args:
            session_id: The session to debug.

        Returns:
            DebugSession instance.
        """
        debug_session = DebugSession(
            session_id=session_id,
            state=DebugState.INSPECTING,
        )
        self._debug_sessions[session_id] = debug_session

        # Take initial snapshot
        await self.take_snapshot(session_id)

        return debug_session

    async def stop_debug(self, session_id: str) -> None:
        """Stop a debugging session.

        Args:
            session_id: The session to stop debugging.
        """
        debug_session = self._debug_sessions.pop(session_id, None)
        if debug_session and debug_session.state == DebugState.ATTACHED:
            await self.detach(session_id)

    async def take_snapshot(self, session_id: str) -> SessionSnapshot:
        """Take a snapshot of session state.

        Args:
            session_id: The session to snapshot.

        Returns:
            SessionSnapshot of current state.
        """
        snapshot = SessionSnapshot(session_id=session_id)

        # Get tmux session info
        tmux_session = await self.tmux_manager.get_session(session_id)
        if tmux_session:
            snapshot.working_directory = tmux_session.working_directory

        # Capture recent output
        try:
            output = await self.tmux_manager.capture_pane(
                session_id,
                start_line=-50,  # Last 50 lines
            )
            snapshot.recent_output = output
        except Exception:
            pass

        # Check for pending interventions
        has_intervention = await self.intervention_manager.has_pending_intervention(
            session_id
        )
        snapshot.has_pending_intervention = has_intervention

        if has_intervention:
            requests = await self.intervention_manager.get_session_requests(session_id)
            if requests:
                snapshot.intervention_reason = requests[0].context

        # Store snapshot
        debug_session = self._debug_sessions.get(session_id)
        if debug_session:
            debug_session.snapshots.append(snapshot)

        # Trigger callbacks
        for callback in self._on_snapshot:
            if debug_session:
                await callback(debug_session, snapshot)

        return snapshot

    async def attach(
        self,
        session_id: str,
        tunnel_host: str,
        tunnel_port: int,
    ) -> TunnelInfo:
        """Attach to a session for interactive debugging.

        Implements [SPEC-02.15].

        Args:
            session_id: The session to attach to.
            tunnel_host: SSH tunnel host.
            tunnel_port: SSH tunnel port.

        Returns:
            TunnelInfo for the connection.
        """
        debug_session = self._debug_sessions.get(session_id)
        if not debug_session:
            debug_session = await self.start_debug(session_id)

        # Create SSH tunnel
        tunnel = await self.tunnel_manager.create_tunnel(
            session_id=session_id,
            host=tunnel_host,
            port=tunnel_port,
        )

        # Update tmux session state
        await self.tmux_manager.attach(session_id)

        # Update debug session state
        debug_session.state = DebugState.ATTACHED
        debug_session.attached_at = datetime.now()

        return tunnel

    async def detach(self, session_id: str) -> None:
        """Detach from a debugging session.

        Args:
            session_id: The session to detach from.
        """
        debug_session = self._debug_sessions.get(session_id)

        # Disconnect SSH tunnel
        await self.tunnel_manager.disconnect(session_id)

        # Update tmux session
        await self.tmux_manager.detach(session_id)

        # Update debug session state
        if debug_session:
            debug_session.state = DebugState.INSPECTING

    async def pause(self, session_id: str) -> SessionSnapshot:
        """Pause execution for inspection.

        Sends Ctrl+C to pause Claude Code.

        Args:
            session_id: The session to pause.

        Returns:
            SessionSnapshot after pausing.
        """
        # Send Ctrl+C to pause
        await self.tmux_manager.send_keys(session_id, "C-c", enter=False)

        debug_session = self._debug_sessions.get(session_id)
        if debug_session:
            debug_session.state = DebugState.PAUSED

        # Take snapshot
        await asyncio.sleep(0.5)  # Allow time for pause to take effect
        return await self.take_snapshot(session_id)

    async def resume(self, session_id: str) -> None:
        """Resume paused execution.

        Args:
            session_id: The session to resume.
        """
        debug_session = self._debug_sessions.get(session_id)
        if debug_session:
            debug_session.state = DebugState.ATTACHED

        # Resume is handled by the session continuing naturally
        # or by sending a command if needed

    async def send_input(self, session_id: str, text: str) -> None:
        """Send input to a debugging session.

        Args:
            session_id: The session.
            text: Text to send.
        """
        await self.tmux_manager.send_keys(session_id, text)

    async def set_breakpoint(self, session_id: str, tool_name: str) -> None:
        """Set a breakpoint on a tool.

        Args:
            session_id: The session.
            tool_name: Tool name to break on.
        """
        debug_session = self._debug_sessions.get(session_id)
        if debug_session and tool_name not in debug_session.breakpoints:
            debug_session.breakpoints.append(tool_name)

    async def clear_breakpoint(self, session_id: str, tool_name: str) -> None:
        """Clear a breakpoint.

        Args:
            session_id: The session.
            tool_name: Tool to clear.
        """
        debug_session = self._debug_sessions.get(session_id)
        if debug_session and tool_name in debug_session.breakpoints:
            debug_session.breakpoints.remove(tool_name)

    async def get_debug_session(self, session_id: str) -> DebugSession | None:
        """Get a debugging session.

        Args:
            session_id: The session ID.

        Returns:
            DebugSession if found, None otherwise.
        """
        return self._debug_sessions.get(session_id)

    async def list_debug_sessions(self) -> list[DebugSession]:
        """List all active debugging sessions.

        Returns:
            List of DebugSession instances.
        """
        return list(self._debug_sessions.values())

    async def get_attach_command(self, session_id: str) -> list[str] | None:
        """Get the command to attach to a session.

        Args:
            session_id: The session ID.

        Returns:
            Command list or None if not found.
        """
        tunnel = await self.tunnel_manager.get_tunnel(session_id)
        if not tunnel:
            return None

        return self.tunnel_manager.build_attach_command(
            tunnel=tunnel,
            tmux_session=session_id,
        )

    def on_snapshot(self, callback: DebugCallback) -> None:
        """Register callback for snapshots."""
        self._on_snapshot.append(callback)

    def on_breakpoint(self, callback: DebugCallback) -> None:
        """Register callback for breakpoint hits."""
        self._on_breakpoint.append(callback)


class InspectFormatter:
    """Format session inspection output.

    Provides various output formats for session state.
    """

    @staticmethod
    def format_snapshot(snapshot: SessionSnapshot, format: str = "text") -> str:
        """Format a snapshot for display.

        Args:
            snapshot: The snapshot to format.
            format: Output format (text, json, compact).

        Returns:
            Formatted string.
        """
        if format == "json":
            return json.dumps(snapshot.to_dict(), indent=2)

        if format == "compact":
            return (
                f"{snapshot.session_id} | "
                f"turn={snapshot.current_turn} | "
                f"tokens={snapshot.tokens_used} | "
                f"cpu={snapshot.cpu_percent:.1f}% | "
                f"mem={snapshot.memory_mb:.0f}MB"
            )

        # Default: text format
        lines = [
            f"Session: {snapshot.session_id}",
            f"State: {snapshot.state.value}",
            f"Working Directory: {snapshot.working_directory}",
            f"Uptime: {snapshot.uptime_seconds:.0f}s",
            "",
            "Claude Code:",
            f"  Turn: {snapshot.current_turn}",
            f"  Tokens: {snapshot.tokens_used:,}",
            f"  Last Tool: {snapshot.last_tool or 'None'}",
            "",
            "Resources:",
            f"  CPU: {snapshot.cpu_percent:.1f}%",
            f"  Memory: {snapshot.memory_mb:.0f} MB",
        ]

        if snapshot.has_pending_intervention:
            lines.extend([
                "",
                "⚠️  INTERVENTION PENDING",
                f"  Reason: {snapshot.intervention_reason}",
            ])

        if snapshot.last_error:
            lines.extend([
                "",
                "❌ Last Error:",
                f"  {snapshot.last_error}",
            ])

        if snapshot.recent_output:
            lines.extend([
                "",
                "Recent Output (last 10 lines):",
                "-" * 40,
            ])
            output_lines = snapshot.recent_output.strip().split("\n")[-10:]
            lines.extend(output_lines)

        return "\n".join(lines)

    @staticmethod
    def format_debug_session(debug_session: DebugSession) -> str:
        """Format debug session info.

        Args:
            debug_session: The debug session.

        Returns:
            Formatted string.
        """
        lines = [
            f"Debug Session: {debug_session.session_id}",
            f"State: {debug_session.state.value}",
            f"Started: {debug_session.started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Snapshots: {len(debug_session.snapshots)}",
        ]

        if debug_session.breakpoints:
            lines.append(f"Breakpoints: {', '.join(debug_session.breakpoints)}")

        if debug_session.attached_at:
            lines.append(
                f"Attached: {debug_session.attached_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        return "\n".join(lines)


class DebugWorkflow:
    """High-level debugging workflow orchestration.

    Provides simplified API for common debugging tasks.
    """

    def __init__(self, debug_manager: DebugManager):
        """Initialize the debug workflow.

        Args:
            debug_manager: Debug manager instance.
        """
        self.debug_manager = debug_manager
        self.formatter = InspectFormatter()

    async def inspect_session(
        self,
        session_id: str,
        format: str = "text",
    ) -> str:
        """Inspect a session and return formatted state.

        Args:
            session_id: The session to inspect.
            format: Output format.

        Returns:
            Formatted inspection output.
        """
        # Ensure debug session exists
        debug_session = await self.debug_manager.get_debug_session(session_id)
        if not debug_session:
            await self.debug_manager.start_debug(session_id)

        # Take fresh snapshot
        snapshot = await self.debug_manager.take_snapshot(session_id)

        return self.formatter.format_snapshot(snapshot, format)

    async def quick_attach(
        self,
        session_id: str,
        tunnel_host: str,
        tunnel_port: int,
    ) -> dict[str, Any]:
        """Quick attach workflow.

        Args:
            session_id: The session.
            tunnel_host: SSH tunnel host.
            tunnel_port: SSH tunnel port.

        Returns:
            Dict with connection info.
        """
        # Attach to session
        tunnel = await self.debug_manager.attach(
            session_id=session_id,
            tunnel_host=tunnel_host,
            tunnel_port=tunnel_port,
        )

        # Get attach command
        cmd = await self.debug_manager.get_attach_command(session_id)

        return {
            "session_id": session_id,
            "tunnel_url": tunnel.ssh_url,
            "attach_command": cmd,
        }

    async def diagnose_session(self, session_id: str) -> dict[str, Any]:
        """Run diagnostics on a session.

        Args:
            session_id: The session to diagnose.

        Returns:
            Diagnostic results.
        """
        snapshot = await self.debug_manager.take_snapshot(session_id)

        issues = []
        recommendations = []

        # Check for interventions
        if snapshot.has_pending_intervention:
            issues.append("Pending intervention request")
            recommendations.append("Attach to session to resolve intervention")

        # Check for errors
        if snapshot.last_error:
            issues.append(f"Last error: {snapshot.last_error}")
            recommendations.append("Review error and provide guidance")

        # Check resource usage
        if snapshot.cpu_percent > 90:
            issues.append("High CPU usage")
            recommendations.append("Consider pausing or investigating workload")

        if snapshot.memory_mb > 14000:  # Assuming 16GB container
            issues.append("High memory usage")
            recommendations.append("Risk of OOM - consider checkpoint")

        return {
            "session_id": session_id,
            "snapshot": snapshot.to_dict(),
            "issues": issues,
            "recommendations": recommendations,
            "healthy": len(issues) == 0,
        }
