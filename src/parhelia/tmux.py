"""tmux session management for Parhelia.

Implements:
- [SPEC-02.11] tmux Server Configuration
- [SPEC-02.12] Session Creation
- [SPEC-02.13] Headless Execution
- [SPEC-02.15] Session Lifecycle Hooks
"""

from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

from parhelia.session import Session, SessionState


class TmuxSessionState(Enum):
    """tmux session states."""

    CREATED = "created"
    RUNNING = "running"
    ATTACHED = "attached"
    DETACHED = "detached"
    KILLED = "killed"


@dataclass
class TmuxConfig:
    """tmux server configuration.

    Implements [SPEC-02.11].
    """

    default_shell: str = "/bin/bash"
    history_limit: int = 50000
    escape_time: int = 0
    base_index: int = 1
    renumber_windows: bool = True
    mouse_enabled: bool = True
    status_right: str = "#{session_name} | CPU: #{cpu_percentage} | MEM: #{mem_percentage}"
    checkpoint_on_detach: bool = True
    config_path: str = "/vol/parhelia/config/tmux.conf"

    def generate_config(self) -> str:
        """Generate tmux configuration file content."""
        lines = [
            "# Parhelia tmux configuration",
            "# Auto-generated - do not edit manually",
            "",
            "# Server options",
            f"set-option -g default-shell {self.default_shell}",
            f"set-option -g history-limit {self.history_limit}",
            f"set-option -g escape-time {self.escape_time}",
            "",
            "# Session options",
            f"set-option -g base-index {self.base_index}",
            f"set-option -g renumber-windows {'on' if self.renumber_windows else 'off'}",
            "",
            "# Enable mouse for interactive sessions",
            f"set-option -g mouse {'on' if self.mouse_enabled else 'off'}",
            "",
            "# Status bar shows session ID and resource usage",
            f"set-option -g status-right '{self.status_right}'",
        ]

        if self.checkpoint_on_detach:
            lines.extend([
                "",
                "# Parhelia-specific: hook for checkpoint on detach",
                'set-hook -g client-detached \'run-shell "parhelia-checkpoint #{session_name}"\'',
            ])

        return "\n".join(lines) + "\n"


@dataclass
class TmuxSession:
    """A tmux session instance.

    Tracks state of a tmux session within Parhelia.
    """

    name: str
    state: TmuxSessionState = TmuxSessionState.CREATED
    working_directory: str = "/vol/parhelia/workspaces"
    environment: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    attached_at: datetime | None = None
    detached_at: datetime | None = None
    pid: int | None = None
    window_count: int = 1


# Type alias for lifecycle hooks
LifecycleHook = Callable[[TmuxSession], Awaitable[None]]


class TmuxManager:
    """Manage tmux sessions for Parhelia.

    Implements [SPEC-02.11-13].

    Handles:
    - Session creation and destruction
    - Attachment and detachment
    - Environment variable injection
    - Command execution within sessions
    - Lifecycle hook management
    """

    DEFAULT_SOCKET_NAME = "parhelia"

    def __init__(
        self,
        config: TmuxConfig | None = None,
        socket_name: str | None = None,
    ):
        """Initialize the tmux manager.

        Args:
            config: tmux configuration.
            socket_name: tmux socket name for session isolation.
        """
        self.config = config or TmuxConfig()
        self.socket_name = socket_name or self.DEFAULT_SOCKET_NAME
        self._sessions: dict[str, TmuxSession] = {}

        # Lifecycle hooks
        self._on_created: list[LifecycleHook] = []
        self._on_attached: list[LifecycleHook] = []
        self._on_detached: list[LifecycleHook] = []
        self._on_killed: list[LifecycleHook] = []

    async def setup(self) -> None:
        """Setup tmux environment.

        Writes configuration file and ensures tmux server is ready.
        """
        config_path = Path(self.config.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(self.config.generate_config())

    async def create_session(
        self,
        task_id: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> TmuxSession:
        """Create a new tmux session.

        Implements [SPEC-02.12].

        Args:
            task_id: Task identifier for session naming.
            working_dir: Working directory for the session.
            env: Environment variables to set in the session.

        Returns:
            TmuxSession instance.
        """
        # Generate session name per spec
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        session_name = f"ph-{task_id}-{timestamp}"

        work_dir = working_dir or "/vol/parhelia/workspaces"

        # Create tmux session
        cmd = [
            "tmux",
            "-L", self.socket_name,  # Use dedicated socket
            "-f", self.config.config_path,  # Use our config
            "new-session",
            "-d",  # Detached
            "-s", session_name,  # Session name
            "-c", work_dir,  # Working directory
        ]

        await self._run_command(cmd)

        # Set environment variables
        if env:
            for key, value in env.items():
                await self.set_environment(session_name, key, value)

        session = TmuxSession(
            name=session_name,
            state=TmuxSessionState.CREATED,
            working_directory=work_dir,
            environment=env or {},
        )

        self._sessions[session_name] = session

        # Trigger hooks
        for hook in self._on_created:
            await hook(session)

        return session

    async def kill_session(self, session_name: str) -> None:
        """Kill a tmux session.

        Args:
            session_name: Name of the session to kill.

        Raises:
            ValueError: If session not found.
        """
        session = self._sessions.get(session_name)
        if not session:
            raise ValueError(f"Session not found: {session_name}")

        cmd = [
            "tmux",
            "-L", self.socket_name,
            "kill-session",
            "-t", session_name,
        ]

        await self._run_command(cmd)

        session.state = TmuxSessionState.KILLED

        # Trigger hooks
        for hook in self._on_killed:
            await hook(session)

        del self._sessions[session_name]

    async def attach(self, session_name: str) -> TmuxSession:
        """Mark a session as attached.

        Note: Actual terminal attachment happens via SSH.
        This tracks the attachment state.

        Args:
            session_name: Name of the session.

        Returns:
            Updated TmuxSession.
        """
        session = self._sessions.get(session_name)
        if not session:
            raise ValueError(f"Session not found: {session_name}")

        session.state = TmuxSessionState.ATTACHED
        session.attached_at = datetime.now()

        # Trigger hooks
        for hook in self._on_attached:
            await hook(session)

        return session

    async def detach(self, session_name: str) -> TmuxSession:
        """Detach all clients from a session.

        Implements [SPEC-02.15] - triggers checkpoint on detach.

        Args:
            session_name: Name of the session.

        Returns:
            Updated TmuxSession.
        """
        session = self._sessions.get(session_name)
        if not session:
            raise ValueError(f"Session not found: {session_name}")

        # Force detach all clients
        cmd = [
            "tmux",
            "-L", self.socket_name,
            "detach-client",
            "-s", session_name,
        ]

        try:
            await self._run_command(cmd)
        except Exception:
            # May fail if no clients attached
            pass

        session.state = TmuxSessionState.DETACHED
        session.detached_at = datetime.now()

        # Trigger hooks (checkpoint happens here per spec)
        for hook in self._on_detached:
            await hook(session)

        return session

    async def send_keys(
        self,
        session_name: str,
        keys: str,
        enter: bool = True,
    ) -> None:
        """Send keystrokes to a tmux session.

        Implements [SPEC-02.13] for sending commands.

        Args:
            session_name: Name of the session.
            keys: Keys/text to send.
            enter: Whether to send Enter after keys.
        """
        if session_name not in self._sessions:
            raise ValueError(f"Session not found: {session_name}")

        cmd = [
            "tmux",
            "-L", self.socket_name,
            "send-keys",
            "-t", session_name,
            keys,
        ]

        if enter:
            cmd.append("Enter")

        await self._run_command(cmd)

    async def run_command_in_session(
        self,
        session_name: str,
        command: list[str],
    ) -> None:
        """Run a command in a tmux session.

        Args:
            session_name: Name of the session.
            command: Command to run (as list).
        """
        # Build quoted command string
        cmd_str = " ".join(shlex.quote(c) for c in command)
        await self.send_keys(session_name, cmd_str)

    async def set_environment(
        self,
        session_name: str,
        key: str,
        value: str,
    ) -> None:
        """Set an environment variable in a session.

        Args:
            session_name: Name of the session.
            key: Environment variable name.
            value: Environment variable value.
        """
        cmd = [
            "tmux",
            "-L", self.socket_name,
            "set-environment",
            "-t", session_name,
            key,
            value,
        ]

        await self._run_command(cmd)

        # Update local tracking
        session = self._sessions.get(session_name)
        if session:
            session.environment[key] = value

    async def get_session(self, session_name: str) -> TmuxSession | None:
        """Get a session by name.

        Args:
            session_name: Name of the session.

        Returns:
            TmuxSession if found, None otherwise.
        """
        return self._sessions.get(session_name)

    async def list_sessions(self) -> list[TmuxSession]:
        """List all managed sessions.

        Returns:
            List of TmuxSession instances.
        """
        return list(self._sessions.values())

    async def session_exists(self, session_name: str) -> bool:
        """Check if a tmux session exists.

        Args:
            session_name: Name of the session.

        Returns:
            True if session exists in tmux server.
        """
        cmd = [
            "tmux",
            "-L", self.socket_name,
            "has-session",
            "-t", session_name,
        ]

        try:
            await self._run_command(cmd)
            return True
        except Exception:
            return False

    async def capture_pane(
        self,
        session_name: str,
        start_line: int = 0,
        end_line: int = -1,
    ) -> str:
        """Capture content from a tmux pane.

        Useful for reading output without attachment.

        Args:
            session_name: Name of the session.
            start_line: Starting line (0 = top of history).
            end_line: Ending line (-1 = current position).

        Returns:
            Captured text content.
        """
        cmd = [
            "tmux",
            "-L", self.socket_name,
            "capture-pane",
            "-t", session_name,
            "-p",  # Print to stdout
            "-S", str(start_line),
            "-E", str(end_line),
        ]

        result = await self._run_command(cmd, capture_output=True)
        return result

    def on_created(self, hook: LifecycleHook) -> None:
        """Register a hook for session creation."""
        self._on_created.append(hook)

    def on_attached(self, hook: LifecycleHook) -> None:
        """Register a hook for session attachment."""
        self._on_attached.append(hook)

    def on_detached(self, hook: LifecycleHook) -> None:
        """Register a hook for session detachment."""
        self._on_detached.append(hook)

    def on_killed(self, hook: LifecycleHook) -> None:
        """Register a hook for session termination."""
        self._on_killed.append(hook)

    async def _run_command(
        self,
        cmd: list[str],
        capture_output: bool = False,
    ) -> str:
        """Run a tmux command.

        Args:
            cmd: Command to run.
            capture_output: Whether to capture and return stdout.

        Returns:
            Stdout if capture_output=True, empty string otherwise.

        Raises:
            RuntimeError: If command fails.
        """
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE if capture_output else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"tmux command failed: {error}")

        if capture_output and stdout:
            return stdout.decode()
        return ""


class HeadlessRunner:
    """Run Claude Code headlessly in tmux sessions.

    Implements [SPEC-02.13].
    """

    DEFAULT_MAX_TURNS = 50
    DEFAULT_OUTPUT_FORMAT = "stream-json"

    def __init__(
        self,
        tmux_manager: TmuxManager,
    ):
        """Initialize the headless runner.

        Args:
            tmux_manager: tmux manager instance.
        """
        self.tmux_manager = tmux_manager

    async def run(
        self,
        session_name: str,
        prompt: str,
        allowed_tools: list[str] | None = None,
        max_turns: int | None = None,
        output_format: str | None = None,
    ) -> None:
        """Run Claude Code headlessly in a session.

        Implements [SPEC-02.13].

        Args:
            session_name: Name of the tmux session.
            prompt: Prompt to send to Claude.
            allowed_tools: List of allowed tools (optional).
            max_turns: Maximum agentic turns.
            output_format: Output format (default: stream-json).
        """
        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", output_format or self.DEFAULT_OUTPUT_FORMAT,
            "--max-turns", str(max_turns or self.DEFAULT_MAX_TURNS),
        ]

        if allowed_tools:
            cmd.extend(["--allowedTools", ",".join(allowed_tools)])

        # Update session state
        session = await self.tmux_manager.get_session(session_name)
        if session:
            session.state = TmuxSessionState.RUNNING

        # Send command to tmux session
        await self.tmux_manager.run_command_in_session(session_name, cmd)

    async def run_in_new_session(
        self,
        task_id: str,
        prompt: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        allowed_tools: list[str] | None = None,
        max_turns: int | None = None,
    ) -> TmuxSession:
        """Create a session and run Claude Code headlessly.

        Args:
            task_id: Task identifier.
            prompt: Prompt for Claude.
            working_dir: Working directory.
            env: Environment variables.
            allowed_tools: Allowed tools.
            max_turns: Maximum turns.

        Returns:
            The created TmuxSession.
        """
        session = await self.tmux_manager.create_session(
            task_id=task_id,
            working_dir=working_dir,
            env=env,
        )

        await self.run(
            session_name=session.name,
            prompt=prompt,
            allowed_tools=allowed_tools,
            max_turns=max_turns,
        )

        return session


class SessionMonitor:
    """Monitor tmux sessions for activity and completion.

    Implements [SPEC-02.17] progress monitoring.
    """

    DEFAULT_POLL_INTERVAL = 5.0  # seconds
    DEFAULT_IDLE_TIMEOUT = 600.0  # 10 minutes

    def __init__(
        self,
        tmux_manager: TmuxManager,
        poll_interval: float | None = None,
        idle_timeout: float | None = None,
    ):
        """Initialize the session monitor.

        Args:
            tmux_manager: tmux manager instance.
            poll_interval: Seconds between activity checks.
            idle_timeout: Seconds of inactivity before timeout.
        """
        self.tmux_manager = tmux_manager
        self.poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        self.idle_timeout = idle_timeout or self.DEFAULT_IDLE_TIMEOUT

        self._monitoring: set[str] = set()
        self._last_activity: dict[str, datetime] = {}
        self._last_content: dict[str, str] = {}

    async def start_monitoring(self, session_name: str) -> None:
        """Start monitoring a session.

        Args:
            session_name: Name of the session to monitor.
        """
        self._monitoring.add(session_name)
        self._last_activity[session_name] = datetime.now()
        self._last_content[session_name] = ""

    async def stop_monitoring(self, session_name: str) -> None:
        """Stop monitoring a session.

        Args:
            session_name: Name of the session.
        """
        self._monitoring.discard(session_name)
        self._last_activity.pop(session_name, None)
        self._last_content.pop(session_name, None)

    async def check_activity(self, session_name: str) -> bool:
        """Check if a session has had recent activity.

        Args:
            session_name: Name of the session.

        Returns:
            True if activity detected, False otherwise.
        """
        try:
            content = await self.tmux_manager.capture_pane(
                session_name,
                start_line=-100,  # Last 100 lines
            )

            last_content = self._last_content.get(session_name, "")

            if content != last_content:
                self._last_content[session_name] = content
                self._last_activity[session_name] = datetime.now()
                return True

            return False

        except Exception:
            return False

    async def is_idle(self, session_name: str) -> bool:
        """Check if a session has exceeded idle timeout.

        Args:
            session_name: Name of the session.

        Returns:
            True if idle timeout exceeded.
        """
        last_activity = self._last_activity.get(session_name)
        if not last_activity:
            return False

        idle_seconds = (datetime.now() - last_activity).total_seconds()
        return idle_seconds > self.idle_timeout

    async def get_idle_time(self, session_name: str) -> float:
        """Get seconds since last activity.

        Args:
            session_name: Name of the session.

        Returns:
            Seconds since last activity, or 0 if unknown.
        """
        last_activity = self._last_activity.get(session_name)
        if not last_activity:
            return 0.0

        return (datetime.now() - last_activity).total_seconds()
