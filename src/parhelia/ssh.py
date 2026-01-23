"""SSH tunnel management for container attachment.

Implements:
- [SPEC-02.12] Session Creation (SSH tunnel setup)
- [SPEC-02.14] Interactive Attachment
"""

from __future__ import annotations

import asyncio
import os
import pwd
import shlex
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    pass


class TunnelState(Enum):
    """SSH tunnel connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class TunnelInfo:
    """Information about an SSH tunnel connection.

    Implements [SPEC-02.14].
    """

    session_id: str
    host: str
    port: int
    user: str = "root"
    state: TunnelState = TunnelState.DISCONNECTED
    created_at: datetime = field(default_factory=datetime.now)
    connected_at: datetime | None = None
    error: str | None = None

    @property
    def ssh_url(self) -> str:
        """Get the SSH connection URL."""
        return f"ssh://{self.user}@{self.host}:{self.port}"

    @property
    def ssh_command(self) -> list[str]:
        """Get the SSH command to connect."""
        return [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "TCPKeepAlive=yes",
            "-o", "ConnectionAttempts=3",
            "-p", str(self.port),
            f"{self.user}@{self.host}",
        ]


@dataclass
class SSHServerConfig:
    """Configuration for SSH server in container.

    Implements [SPEC-02.14].
    """

    port: int = 2222
    user: str = "root"
    allow_password_auth: bool = False
    authorized_keys: list[str] = field(default_factory=list)
    host_key_path: str = "/etc/ssh/ssh_host_ed25519_key"


class SSHTunnelManager:
    """Manage SSH tunnels for container attachment.

    Implements [SPEC-02.14].

    This manager handles:
    - SSH tunnel creation via Modal's forward() API
    - Connection lifecycle management
    - SSH keepalive configuration
    - tmux session attachment
    """

    # Default SSH options for resilience (no mosh available)
    DEFAULT_SSH_OPTIONS = {
        "StrictHostKeyChecking": "no",
        "UserKnownHostsFile": "/dev/null",
        "ServerAliveInterval": "30",
        "ServerAliveCountMax": "3",
        "TCPKeepAlive": "yes",
        "ConnectionAttempts": "3",
    }

    def __init__(
        self,
        ssh_config_path: str | None = None,
        default_user: str = "root",
    ):
        """Initialize the SSH tunnel manager.

        Args:
            ssh_config_path: Path to SSH config file for custom options.
            default_user: Default SSH user for connections.
        """
        self.ssh_config_path = ssh_config_path
        self.default_user = default_user
        self._tunnels: dict[str, TunnelInfo] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def create_tunnel(
        self,
        session_id: str,
        host: str,
        port: int,
        user: str | None = None,
    ) -> TunnelInfo:
        """Create a tunnel info record for a session.

        Args:
            session_id: The session to tunnel to.
            host: The tunnel hostname (from Modal).
            port: The tunnel port (from Modal).
            user: SSH user (defaults to manager default).

        Returns:
            TunnelInfo with connection details.
        """
        tunnel = TunnelInfo(
            session_id=session_id,
            host=host,
            port=port,
            user=user or self.default_user,
        )
        self._tunnels[session_id] = tunnel
        return tunnel

    async def connect(
        self,
        session_id: str,
        tmux_session: str | None = None,
    ) -> TunnelInfo:
        """Connect to a session via SSH tunnel.

        Args:
            session_id: The session to connect to.
            tmux_session: Optional tmux session name to attach.

        Returns:
            Updated TunnelInfo with connection state.

        Raises:
            ValueError: If tunnel not found for session.
            ConnectionError: If connection fails.
        """
        tunnel = self._tunnels.get(session_id)
        if not tunnel:
            raise ValueError(f"No tunnel found for session: {session_id}")

        tunnel.state = TunnelState.CONNECTING

        try:
            # Build SSH command
            cmd = list(tunnel.ssh_command)

            # Add tmux attach if specified
            if tmux_session:
                cmd.append(f"tmux attach-session -t {shlex.quote(tmux_session)}")

            # Start SSH process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self._processes[session_id] = process
            tunnel.state = TunnelState.CONNECTED
            tunnel.connected_at = datetime.now()

        except Exception as e:
            tunnel.state = TunnelState.ERROR
            tunnel.error = str(e)
            raise ConnectionError(f"Failed to connect: {e}") from e

        return tunnel

    async def disconnect(self, session_id: str) -> None:
        """Disconnect from a session.

        Args:
            session_id: The session to disconnect from.
        """
        process = self._processes.pop(session_id, None)
        if process:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()

        tunnel = self._tunnels.get(session_id)
        if tunnel:
            tunnel.state = TunnelState.DISCONNECTED

    async def get_tunnel(self, session_id: str) -> TunnelInfo | None:
        """Get tunnel info for a session.

        Args:
            session_id: The session ID.

        Returns:
            TunnelInfo if found, None otherwise.
        """
        return self._tunnels.get(session_id)

    async def list_tunnels(self) -> list[TunnelInfo]:
        """List all active tunnels.

        Returns:
            List of TunnelInfo for all sessions.
        """
        return list(self._tunnels.values())

    def build_attach_command(
        self,
        tunnel: TunnelInfo,
        tmux_session: str,
    ) -> list[str]:
        """Build the full command to attach to a tmux session via SSH.

        Implements [SPEC-02.14] Local Attachment Command.

        Args:
            tunnel: The tunnel connection info.
            tmux_session: The tmux session name.

        Returns:
            Command list for subprocess.
        """
        cmd = [
            "ssh",
            "-t",  # Force pseudo-terminal allocation
        ]

        # Add SSH options for resilience
        for key, value in self.DEFAULT_SSH_OPTIONS.items():
            cmd.extend(["-o", f"{key}={value}"])

        # Add custom config if specified
        if self.ssh_config_path and Path(self.ssh_config_path).exists():
            cmd.extend(["-F", self.ssh_config_path])

        # Add connection details
        cmd.extend([
            "-p", str(tunnel.port),
            f"{tunnel.user}@{tunnel.host}",
        ])

        # Add tmux attach command
        cmd.append(f"tmux attach-session -t {shlex.quote(tmux_session)}")

        return cmd


class SSHServerSetup:
    """Setup SSH server in Modal container.

    Implements [SPEC-02.14].

    This handles:
    - Generating ephemeral host keys
    - Starting SSH daemon
    - Configuring authorized keys
    """

    # SSH configuration for non-root parhelia user
    # Claude Code requires non-root for --dangerously-skip-permissions
    CONTAINER_USER = "parhelia"
    CONTAINER_HOME = "/home/parhelia"

    DEFAULT_SSHD_CONFIG = """
# Parhelia SSH server configuration
Port {port}
HostKey /etc/ssh/ssh_host_ed25519_key
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile %h/.ssh/authorized_keys
Subsystem sftp /usr/lib/openssh/sftp-server
UsePAM no
"""

    def __init__(self, config: SSHServerConfig | None = None):
        """Initialize SSH server setup.

        Args:
            config: SSH server configuration.
        """
        self.config = config or SSHServerConfig()
        self._sshd_process: subprocess.Popen[bytes] | None = None

    async def setup(self) -> None:
        """Setup the SSH server.

        This generates host keys and writes configuration.
        """
        # Ensure .ssh directory exists for parhelia user
        ssh_dir = Path(f"{self.CONTAINER_HOME}/.ssh")
        ssh_dir.mkdir(parents=True, exist_ok=True)
        ssh_dir.chmod(0o700)
        # Set ownership to parhelia user (uid/gid lookup)
        try:
            pw = pwd.getpwnam(self.CONTAINER_USER)
            os.chown(ssh_dir, pw.pw_uid, pw.pw_gid)
        except KeyError:
            pass  # User doesn't exist yet (during image build)

        # Generate host key if not exists
        host_key = Path(self.config.host_key_path)
        if not host_key.exists():
            await self._generate_host_key()

        # Write authorized keys
        if self.config.authorized_keys:
            authorized_keys_file = ssh_dir / "authorized_keys"
            authorized_keys_file.write_text(
                "\n".join(self.config.authorized_keys) + "\n"
            )
            authorized_keys_file.chmod(0o600)
            # Set ownership to parhelia user
            try:
                pw = pwd.getpwnam(self.CONTAINER_USER)
                os.chown(authorized_keys_file, pw.pw_uid, pw.pw_gid)
            except KeyError:
                pass

        # Write sshd config
        await self._write_sshd_config()

    async def _generate_host_key(self) -> None:
        """Generate ephemeral host key."""
        process = await asyncio.create_subprocess_exec(
            "ssh-keygen",
            "-t", "ed25519",
            "-f", self.config.host_key_path,
            "-N", "",  # No passphrase
            "-q",  # Quiet
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()

    async def _write_sshd_config(self) -> None:
        """Write SSHD configuration file."""
        config_content = self.DEFAULT_SSHD_CONFIG.format(
            port=self.config.port,
        )

        config_path = Path("/etc/ssh/sshd_config.d/parhelia.conf")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_content)

    async def start(self) -> int:
        """Start the SSH daemon.

        Returns:
            The PID of the SSH daemon.
        """
        # Start sshd in foreground mode (for container)
        self._sshd_process = subprocess.Popen(
            ["/usr/sbin/sshd", "-D", "-p", str(self.config.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return self._sshd_process.pid

    async def stop(self) -> None:
        """Stop the SSH daemon."""
        if self._sshd_process:
            self._sshd_process.terminate()
            try:
                self._sshd_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._sshd_process.kill()
            self._sshd_process = None

    @property
    def is_running(self) -> bool:
        """Check if SSH daemon is running."""
        if self._sshd_process is None:
            return False
        return self._sshd_process.poll() is None


@asynccontextmanager
async def modal_ssh_tunnel(
    internal_port: int = 2222,
) -> AsyncIterator[tuple[str, int]]:
    """Context manager for Modal SSH tunnel.

    Implements [SPEC-02.14] SSH Tunnel Characteristics.

    Note: This is a mock implementation. In production, this would use
    modal.forward() which is an experimental API.

    Args:
        internal_port: The SSH port inside the container.

    Yields:
        Tuple of (hostname, port) for the tunnel endpoint.
    """
    # In production, this would be:
    # async with modal.forward(internal_port, unencrypted=True) as tunnel:
    #     host, port = tunnel.tcp_socket
    #     yield host, port

    # Mock implementation for testing
    mock_host = "r3.modal.host"
    mock_port = 23447  # Random port as per spec

    yield mock_host, mock_port


class AttachmentManager:
    """Manage interactive attachment to sessions.

    Implements [SPEC-02.14] Interactive Attachment.

    Coordinates:
    - Tunnel creation
    - SSH connection
    - tmux session attachment
    - Detachment handling
    """

    def __init__(
        self,
        tunnel_manager: SSHTunnelManager | None = None,
    ):
        """Initialize the attachment manager.

        Args:
            tunnel_manager: SSH tunnel manager instance.
        """
        self.tunnel_manager = tunnel_manager or SSHTunnelManager()

    async def attach_to_session(
        self,
        session_id: str,
        tunnel_host: str,
        tunnel_port: int,
        tmux_session: str | None = None,
    ) -> TunnelInfo:
        """Attach to a running session.

        Implements [SPEC-02.14].

        Args:
            session_id: The session to attach to.
            tunnel_host: The Modal tunnel hostname.
            tunnel_port: The Modal tunnel port.
            tmux_session: tmux session name (defaults to session_id).

        Returns:
            TunnelInfo with connection details.
        """
        # Create tunnel record
        tunnel = await self.tunnel_manager.create_tunnel(
            session_id=session_id,
            host=tunnel_host,
            port=tunnel_port,
        )

        # Connect via SSH
        await self.tunnel_manager.connect(
            session_id=session_id,
            tmux_session=tmux_session or session_id,
        )

        return tunnel

    async def detach_from_session(self, session_id: str) -> None:
        """Detach from a session.

        This triggers checkpoint per [SPEC-02.15].

        Args:
            session_id: The session to detach from.
        """
        await self.tunnel_manager.disconnect(session_id)

    async def get_attach_command(
        self,
        session_id: str,
        tmux_session: str | None = None,
    ) -> list[str] | None:
        """Get the command to attach to a session.

        Args:
            session_id: The session ID.
            tmux_session: tmux session name (defaults to session_id).

        Returns:
            Command list for shell execution, or None if not found.
        """
        tunnel = await self.tunnel_manager.get_tunnel(session_id)
        if not tunnel:
            return None

        return self.tunnel_manager.build_attach_command(
            tunnel=tunnel,
            tmux_session=tmux_session or session_id,
        )
