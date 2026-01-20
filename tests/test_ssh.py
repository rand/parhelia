"""Tests for Parhelia SSH tunnel management.

Tests SSH tunnel creation, connection, and attachment per SPEC-02.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.ssh import (
    AttachmentManager,
    SSHServerConfig,
    SSHServerSetup,
    SSHTunnelManager,
    TunnelInfo,
    TunnelState,
    modal_ssh_tunnel,
)


# =============================================================================
# TunnelInfo Tests
# =============================================================================


class TestTunnelInfo:
    """Tests for TunnelInfo data class."""

    def test_tunnel_info_creation(self):
        """TunnelInfo MUST initialize with required fields."""
        tunnel = TunnelInfo(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        assert tunnel.session_id == "session-1"
        assert tunnel.host == "r3.modal.host"
        assert tunnel.port == 23447
        assert tunnel.user == "root"
        assert tunnel.state == TunnelState.DISCONNECTED

    def test_tunnel_info_ssh_url(self):
        """TunnelInfo MUST provide correct SSH URL."""
        tunnel = TunnelInfo(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
            user="developer",
        )

        assert tunnel.ssh_url == "ssh://developer@r3.modal.host:23447"

    def test_tunnel_info_ssh_command(self):
        """TunnelInfo MUST provide correct SSH command."""
        tunnel = TunnelInfo(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        cmd = tunnel.ssh_command
        assert "ssh" in cmd
        assert "-p" in cmd
        assert "23447" in cmd
        assert "root@r3.modal.host" in cmd
        # Should include keepalive options
        assert "-o" in cmd
        assert "ServerAliveInterval=30" in " ".join(cmd)

    def test_tunnel_info_timestamps(self):
        """TunnelInfo MUST track creation time."""
        before = datetime.now()
        tunnel = TunnelInfo(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )
        after = datetime.now()

        assert before <= tunnel.created_at <= after
        assert tunnel.connected_at is None


# =============================================================================
# SSHTunnelManager Tests
# =============================================================================


class TestSSHTunnelManager:
    """Tests for SSHTunnelManager class."""

    @pytest.fixture
    def manager(self):
        """Create a tunnel manager for testing."""
        return SSHTunnelManager()

    @pytest.mark.asyncio
    async def test_create_tunnel(self, manager):
        """create_tunnel MUST create TunnelInfo record."""
        tunnel = await manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        assert tunnel.session_id == "session-1"
        assert tunnel.host == "r3.modal.host"
        assert tunnel.port == 23447
        assert tunnel.state == TunnelState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_create_tunnel_custom_user(self, manager):
        """create_tunnel MUST accept custom user."""
        tunnel = await manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
            user="developer",
        )

        assert tunnel.user == "developer"

    @pytest.mark.asyncio
    async def test_get_tunnel(self, manager):
        """get_tunnel MUST return existing tunnel."""
        await manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        tunnel = await manager.get_tunnel("session-1")
        assert tunnel is not None
        assert tunnel.session_id == "session-1"

    @pytest.mark.asyncio
    async def test_get_tunnel_not_found(self, manager):
        """get_tunnel MUST return None for unknown session."""
        tunnel = await manager.get_tunnel("nonexistent")
        assert tunnel is None

    @pytest.mark.asyncio
    async def test_list_tunnels(self, manager):
        """list_tunnels MUST return all tunnels."""
        await manager.create_tunnel("session-1", "host1.modal.host", 10001)
        await manager.create_tunnel("session-2", "host2.modal.host", 10002)

        tunnels = await manager.list_tunnels()
        assert len(tunnels) == 2

    @pytest.mark.asyncio
    async def test_connect_updates_state(self, manager):
        """connect MUST update tunnel state."""
        await manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_exec.return_value = mock_process

            tunnel = await manager.connect("session-1")

            assert tunnel.state == TunnelState.CONNECTED
            assert tunnel.connected_at is not None

    @pytest.mark.asyncio
    async def test_connect_not_found(self, manager):
        """connect MUST raise ValueError for unknown session."""
        with pytest.raises(ValueError, match="No tunnel found"):
            await manager.connect("nonexistent")

    @pytest.mark.asyncio
    async def test_connect_with_tmux(self, manager):
        """connect MUST include tmux command when specified."""
        await manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_exec.return_value = mock_process

            await manager.connect("session-1", tmux_session="my-session")

            # Check that tmux attach was included in command
            call_args = mock_exec.call_args
            cmd_parts = call_args[0]
            assert any("tmux" in str(part) for part in cmd_parts)

    @pytest.mark.asyncio
    async def test_disconnect(self, manager):
        """disconnect MUST terminate connection."""
        await manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process

            await manager.connect("session-1")
            await manager.disconnect("session-1")

            mock_process.terminate.assert_called_once()

        tunnel = await manager.get_tunnel("session-1")
        assert tunnel.state == TunnelState.DISCONNECTED

    def test_build_attach_command(self, manager):
        """build_attach_command MUST create correct SSH command."""
        tunnel = TunnelInfo(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        cmd = manager.build_attach_command(tunnel, "my-session")

        assert cmd[0] == "ssh"
        assert "-t" in cmd  # Force pseudo-terminal
        assert "-p" in cmd
        assert "23447" in cmd
        assert "root@r3.modal.host" in cmd
        assert "tmux attach-session" in cmd[-1]
        assert "my-session" in cmd[-1]

    def test_build_attach_command_with_options(self, manager):
        """build_attach_command MUST include keepalive options."""
        tunnel = TunnelInfo(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        cmd = manager.build_attach_command(tunnel, "my-session")
        cmd_str = " ".join(cmd)

        # Check SSH options per SPEC-02.14
        assert "ServerAliveInterval=30" in cmd_str
        assert "ServerAliveCountMax=3" in cmd_str
        assert "TCPKeepAlive=yes" in cmd_str


# =============================================================================
# SSHServerConfig Tests
# =============================================================================


class TestSSHServerConfig:
    """Tests for SSHServerConfig class."""

    def test_default_config(self):
        """SSHServerConfig MUST have sensible defaults."""
        config = SSHServerConfig()

        assert config.port == 2222
        assert config.user == "root"
        assert config.allow_password_auth is False

    def test_custom_config(self):
        """SSHServerConfig MUST accept custom values."""
        config = SSHServerConfig(
            port=2200,
            user="developer",
            authorized_keys=["ssh-ed25519 AAAA... user@host"],
        )

        assert config.port == 2200
        assert config.user == "developer"
        assert len(config.authorized_keys) == 1


# =============================================================================
# SSHServerSetup Tests
# =============================================================================


class TestSSHServerSetup:
    """Tests for SSHServerSetup class."""

    @pytest.fixture
    def setup(self):
        """Create an SSH server setup for testing."""
        return SSHServerSetup()

    @pytest.mark.asyncio
    async def test_setup_creates_directories(self, setup, tmp_path):
        """setup MUST create .ssh directory."""
        with patch("parhelia.ssh.Path") as mock_path:
            mock_ssh_dir = MagicMock()
            mock_path.return_value = mock_ssh_dir

            with patch.object(setup, "_generate_host_key", new_callable=AsyncMock):
                with patch.object(setup, "_write_sshd_config", new_callable=AsyncMock):
                    await setup.setup()

    @pytest.mark.asyncio
    async def test_generate_host_key(self, setup):
        """_generate_host_key MUST run ssh-keygen."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process

            await setup._generate_host_key()

            mock_exec.assert_called_once()
            call_args = mock_exec.call_args[0]
            assert "ssh-keygen" in call_args
            assert "-t" in call_args
            assert "ed25519" in call_args

    @pytest.mark.asyncio
    async def test_start_sshd(self, setup):
        """start MUST launch SSH daemon."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_popen.return_value = mock_process

            pid = await setup.start()

            assert pid == 12345
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args[0][0]
            assert "/usr/sbin/sshd" in call_args

    @pytest.mark.asyncio
    async def test_stop_sshd(self, setup):
        """stop MUST terminate SSH daemon."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_popen.return_value = mock_process

            await setup.start()
            await setup.stop()

            mock_process.terminate.assert_called_once()

    def test_is_running(self, setup):
        """is_running MUST check process status."""
        assert setup.is_running is False

        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            mock_popen.return_value = mock_process

            import asyncio
            asyncio.run(setup.start())

            assert setup.is_running is True


# =============================================================================
# Modal SSH Tunnel Context Manager Tests
# =============================================================================


class TestModalSSHTunnel:
    """Tests for modal_ssh_tunnel context manager."""

    @pytest.mark.asyncio
    async def test_tunnel_yields_endpoint(self):
        """modal_ssh_tunnel MUST yield host and port."""
        async with modal_ssh_tunnel(2222) as (host, port):
            assert host == "r3.modal.host"
            assert isinstance(port, int)

    @pytest.mark.asyncio
    async def test_tunnel_port_is_random(self):
        """modal_ssh_tunnel port MUST be randomly assigned per spec."""
        async with modal_ssh_tunnel(2222) as (host, port):
            # Port should not be the internal port (per spec, Modal assigns random)
            assert port != 2222


# =============================================================================
# AttachmentManager Tests
# =============================================================================


class TestAttachmentManager:
    """Tests for AttachmentManager class."""

    @pytest.fixture
    def manager(self):
        """Create an attachment manager for testing."""
        return AttachmentManager()

    @pytest.mark.asyncio
    async def test_attach_to_session(self, manager):
        """attach_to_session MUST create tunnel and connect."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_exec.return_value = mock_process

            tunnel = await manager.attach_to_session(
                session_id="session-1",
                tunnel_host="r3.modal.host",
                tunnel_port=23447,
            )

            assert tunnel.session_id == "session-1"
            assert tunnel.state == TunnelState.CONNECTED

    @pytest.mark.asyncio
    async def test_attach_with_tmux_session(self, manager):
        """attach_to_session MUST use provided tmux session."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_exec.return_value = mock_process

            await manager.attach_to_session(
                session_id="session-1",
                tunnel_host="r3.modal.host",
                tunnel_port=23447,
                tmux_session="custom-session",
            )

            # Verify tmux session was used
            call_args = mock_exec.call_args[0]
            assert any("custom-session" in str(part) for part in call_args)

    @pytest.mark.asyncio
    async def test_detach_from_session(self, manager):
        """detach_from_session MUST disconnect tunnel."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process

            await manager.attach_to_session(
                session_id="session-1",
                tunnel_host="r3.modal.host",
                tunnel_port=23447,
            )

            await manager.detach_from_session("session-1")

            mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_attach_command(self, manager):
        """get_attach_command MUST return command for existing tunnel."""
        await manager.tunnel_manager.create_tunnel(
            session_id="session-1",
            host="r3.modal.host",
            port=23447,
        )

        cmd = await manager.get_attach_command("session-1")

        assert cmd is not None
        assert "ssh" in cmd

    @pytest.mark.asyncio
    async def test_get_attach_command_not_found(self, manager):
        """get_attach_command MUST return None for unknown session."""
        cmd = await manager.get_attach_command("nonexistent")
        assert cmd is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSSHIntegration:
    """Integration tests for SSH tunnel flow."""

    @pytest.mark.asyncio
    async def test_full_attach_detach_flow(self):
        """Full attach/detach flow MUST work correctly."""
        manager = AttachmentManager()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.wait = AsyncMock()
            mock_exec.return_value = mock_process

            # Attach
            tunnel = await manager.attach_to_session(
                session_id="session-1",
                tunnel_host="r3.modal.host",
                tunnel_port=23447,
                tmux_session="my-session",
            )

            assert tunnel.state == TunnelState.CONNECTED

            # Detach
            await manager.detach_from_session("session-1")

            tunnel = await manager.tunnel_manager.get_tunnel("session-1")
            assert tunnel.state == TunnelState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        """Manager MUST handle multiple concurrent sessions."""
        manager = AttachmentManager()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_exec.return_value = mock_process

            # Attach to multiple sessions
            tunnel1 = await manager.attach_to_session(
                session_id="session-1",
                tunnel_host="host1.modal.host",
                tunnel_port=10001,
            )
            tunnel2 = await manager.attach_to_session(
                session_id="session-2",
                tunnel_host="host2.modal.host",
                tunnel_port=10002,
            )

            assert tunnel1.session_id == "session-1"
            assert tunnel2.session_id == "session-2"

            tunnels = await manager.tunnel_manager.list_tunnels()
            assert len(tunnels) == 2
