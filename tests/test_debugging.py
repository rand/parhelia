"""Tests for Parhelia interactive debugging workflow.

Tests debugging session management, inspection, and workflows per SPEC-02.15.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.debugging import (
    DebugManager,
    DebugSession,
    DebugState,
    DebugWorkflow,
    InspectFormatter,
    SessionSnapshot,
)
from parhelia.session import SessionState
from parhelia.tmux import TmuxManager
from parhelia.ssh import SSHTunnelManager
from parhelia.intervention import InterventionManager


# =============================================================================
# SessionSnapshot Tests
# =============================================================================


class TestSessionSnapshot:
    """Tests for SessionSnapshot data class."""

    def test_snapshot_creation(self):
        """SessionSnapshot MUST initialize with session ID."""
        snapshot = SessionSnapshot(session_id="session-1")

        assert snapshot.session_id == "session-1"
        assert snapshot.state == SessionState.RUNNING
        assert snapshot.current_turn == 0

    def test_snapshot_timestamps(self):
        """SessionSnapshot MUST track timestamp."""
        before = datetime.now()
        snapshot = SessionSnapshot(session_id="session-1")
        after = datetime.now()

        assert before <= snapshot.timestamp <= after

    def test_snapshot_to_dict(self):
        """to_dict MUST convert all fields."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            current_turn=5,
            tokens_used=10000,
            cpu_percent=45.5,
        )

        result = snapshot.to_dict()

        assert result["session_id"] == "session-1"
        assert result["current_turn"] == 5
        assert result["tokens_used"] == 10000
        assert result["cpu_percent"] == 45.5

    def test_snapshot_with_intervention(self):
        """SessionSnapshot MUST track intervention status."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            has_pending_intervention=True,
            intervention_reason="Session idle",
        )

        assert snapshot.has_pending_intervention is True
        assert snapshot.intervention_reason == "Session idle"


# =============================================================================
# DebugSession Tests
# =============================================================================


class TestDebugSession:
    """Tests for DebugSession data class."""

    def test_debug_session_creation(self):
        """DebugSession MUST initialize with session ID."""
        debug = DebugSession(session_id="session-1")

        assert debug.session_id == "session-1"
        assert debug.state == DebugState.IDLE
        assert len(debug.snapshots) == 0
        assert len(debug.breakpoints) == 0

    def test_debug_session_timestamps(self):
        """DebugSession MUST track started_at."""
        before = datetime.now()
        debug = DebugSession(session_id="session-1")
        after = datetime.now()

        assert before <= debug.started_at <= after
        assert debug.attached_at is None


# =============================================================================
# DebugManager Tests
# =============================================================================


class TestDebugManager:
    """Tests for DebugManager class."""

    @pytest.fixture
    def mock_tmux(self):
        """Create a mock tmux manager."""
        manager = MagicMock(spec=TmuxManager)
        manager.get_session = AsyncMock(return_value=None)
        manager.capture_pane = AsyncMock(return_value="sample output")
        manager.attach = AsyncMock()
        manager.detach = AsyncMock()
        manager.send_keys = AsyncMock()
        return manager

    @pytest.fixture
    def mock_tunnel(self):
        """Create a mock tunnel manager."""
        manager = MagicMock(spec=SSHTunnelManager)
        manager.create_tunnel = AsyncMock()
        manager.disconnect = AsyncMock()
        manager.get_tunnel = AsyncMock(return_value=None)
        manager.build_attach_command = MagicMock(return_value=["ssh", "-t", "host"])
        return manager

    @pytest.fixture
    def mock_intervention(self):
        """Create a mock intervention manager."""
        manager = MagicMock(spec=InterventionManager)
        manager.has_pending_intervention = AsyncMock(return_value=False)
        manager.get_session_requests = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def manager(self, mock_tmux, mock_tunnel, mock_intervention):
        """Create a debug manager for testing."""
        return DebugManager(
            tmux_manager=mock_tmux,
            tunnel_manager=mock_tunnel,
            intervention_manager=mock_intervention,
        )

    @pytest.mark.asyncio
    async def test_start_debug(self, manager):
        """start_debug MUST create debug session."""
        debug = await manager.start_debug("session-1")

        assert debug.session_id == "session-1"
        assert debug.state == DebugState.INSPECTING
        assert len(debug.snapshots) == 1  # Initial snapshot

    @pytest.mark.asyncio
    async def test_stop_debug(self, manager):
        """stop_debug MUST remove debug session."""
        await manager.start_debug("session-1")
        await manager.stop_debug("session-1")

        assert await manager.get_debug_session("session-1") is None

    @pytest.mark.asyncio
    async def test_take_snapshot(self, manager, mock_tmux):
        """take_snapshot MUST capture session state."""
        await manager.start_debug("session-1")

        snapshot = await manager.take_snapshot("session-1")

        assert snapshot.session_id == "session-1"
        assert snapshot.recent_output == "sample output"
        mock_tmux.capture_pane.assert_called()

    @pytest.mark.asyncio
    async def test_take_snapshot_with_intervention(self, manager, mock_intervention):
        """take_snapshot MUST check for interventions."""
        mock_intervention.has_pending_intervention = AsyncMock(return_value=True)
        mock_intervention.get_session_requests = AsyncMock(return_value=[
            MagicMock(context="Session is idle")
        ])

        await manager.start_debug("session-1")
        snapshot = await manager.take_snapshot("session-1")

        assert snapshot.has_pending_intervention is True
        assert snapshot.intervention_reason == "Session is idle"

    @pytest.mark.asyncio
    async def test_attach(self, manager, mock_tunnel, mock_tmux):
        """attach MUST create tunnel and update state."""
        await manager.start_debug("session-1")

        tunnel = await manager.attach(
            "session-1",
            tunnel_host="r3.modal.host",
            tunnel_port=23447,
        )

        mock_tunnel.create_tunnel.assert_called_once()
        mock_tmux.attach.assert_called_once()

        debug = await manager.get_debug_session("session-1")
        assert debug.state == DebugState.ATTACHED

    @pytest.mark.asyncio
    async def test_attach_auto_starts_debug(self, manager, mock_tunnel):
        """attach MUST auto-start debug session if needed."""
        tunnel = await manager.attach(
            "session-1",
            tunnel_host="r3.modal.host",
            tunnel_port=23447,
        )

        debug = await manager.get_debug_session("session-1")
        assert debug is not None
        assert debug.state == DebugState.ATTACHED

    @pytest.mark.asyncio
    async def test_detach(self, manager, mock_tunnel, mock_tmux):
        """detach MUST disconnect and update state."""
        await manager.start_debug("session-1")
        await manager.attach("session-1", "host", 22)

        await manager.detach("session-1")

        mock_tunnel.disconnect.assert_called_once()
        mock_tmux.detach.assert_called_once()

        debug = await manager.get_debug_session("session-1")
        assert debug.state == DebugState.INSPECTING

    @pytest.mark.asyncio
    async def test_pause(self, manager, mock_tmux):
        """pause MUST send Ctrl+C and update state."""
        await manager.start_debug("session-1")

        snapshot = await manager.pause("session-1")

        mock_tmux.send_keys.assert_called()
        debug = await manager.get_debug_session("session-1")
        assert debug.state == DebugState.PAUSED

    @pytest.mark.asyncio
    async def test_resume(self, manager):
        """resume MUST update state."""
        await manager.start_debug("session-1")
        debug = await manager.get_debug_session("session-1")
        debug.state = DebugState.PAUSED

        await manager.resume("session-1")

        assert debug.state == DebugState.ATTACHED

    @pytest.mark.asyncio
    async def test_send_input(self, manager, mock_tmux):
        """send_input MUST send to tmux."""
        await manager.start_debug("session-1")

        await manager.send_input("session-1", "test input")

        mock_tmux.send_keys.assert_called_with("session-1", "test input")

    @pytest.mark.asyncio
    async def test_set_breakpoint(self, manager):
        """set_breakpoint MUST add to breakpoints."""
        await manager.start_debug("session-1")

        await manager.set_breakpoint("session-1", "Bash")

        debug = await manager.get_debug_session("session-1")
        assert "Bash" in debug.breakpoints

    @pytest.mark.asyncio
    async def test_set_breakpoint_no_duplicates(self, manager):
        """set_breakpoint MUST not add duplicates."""
        await manager.start_debug("session-1")

        await manager.set_breakpoint("session-1", "Bash")
        await manager.set_breakpoint("session-1", "Bash")

        debug = await manager.get_debug_session("session-1")
        assert debug.breakpoints.count("Bash") == 1

    @pytest.mark.asyncio
    async def test_clear_breakpoint(self, manager):
        """clear_breakpoint MUST remove breakpoint."""
        await manager.start_debug("session-1")
        await manager.set_breakpoint("session-1", "Bash")

        await manager.clear_breakpoint("session-1", "Bash")

        debug = await manager.get_debug_session("session-1")
        assert "Bash" not in debug.breakpoints

    @pytest.mark.asyncio
    async def test_list_debug_sessions(self, manager):
        """list_debug_sessions MUST return all sessions."""
        await manager.start_debug("session-1")
        await manager.start_debug("session-2")

        sessions = await manager.list_debug_sessions()

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_on_snapshot_callback(self, manager):
        """on_snapshot callback MUST be called."""
        callback_called = False

        async def on_snapshot(debug, snapshot):
            nonlocal callback_called
            callback_called = True

        manager.on_snapshot(on_snapshot)
        await manager.start_debug("session-1")
        await manager.take_snapshot("session-1")

        assert callback_called is True


# =============================================================================
# InspectFormatter Tests
# =============================================================================


class TestInspectFormatter:
    """Tests for InspectFormatter class."""

    def test_format_snapshot_text(self):
        """format_snapshot MUST produce text output."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            current_turn=5,
            tokens_used=10000,
        )

        result = InspectFormatter.format_snapshot(snapshot, "text")

        assert "session-1" in result
        assert "Turn: 5" in result
        assert "10,000" in result

    def test_format_snapshot_json(self):
        """format_snapshot MUST produce JSON output."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            current_turn=5,
        )

        result = InspectFormatter.format_snapshot(snapshot, "json")

        import json
        data = json.loads(result)
        assert data["session_id"] == "session-1"
        assert data["current_turn"] == 5

    def test_format_snapshot_compact(self):
        """format_snapshot MUST produce compact output."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            current_turn=5,
            tokens_used=10000,
            cpu_percent=45.5,
            memory_mb=2048,
        )

        result = InspectFormatter.format_snapshot(snapshot, "compact")

        assert "session-1" in result
        assert "turn=5" in result
        assert "tokens=10000" in result
        assert "|" in result  # Separator

    def test_format_snapshot_with_intervention(self):
        """format_snapshot MUST show intervention warning."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            has_pending_intervention=True,
            intervention_reason="Session idle",
        )

        result = InspectFormatter.format_snapshot(snapshot, "text")

        assert "INTERVENTION" in result
        assert "Session idle" in result

    def test_format_snapshot_with_error(self):
        """format_snapshot MUST show error."""
        snapshot = SessionSnapshot(
            session_id="session-1",
            last_error="Connection refused",
        )

        result = InspectFormatter.format_snapshot(snapshot, "text")

        assert "Error" in result
        assert "Connection refused" in result

    def test_format_debug_session(self):
        """format_debug_session MUST show session info."""
        debug = DebugSession(
            session_id="session-1",
            state=DebugState.ATTACHED,
            breakpoints=["Bash", "Read"],
        )

        result = InspectFormatter.format_debug_session(debug)

        assert "session-1" in result
        assert "attached" in result
        assert "Bash" in result


# =============================================================================
# DebugWorkflow Tests
# =============================================================================


class TestDebugWorkflow:
    """Tests for DebugWorkflow class."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock debug manager."""
        manager = MagicMock(spec=DebugManager)
        manager.get_debug_session = AsyncMock(return_value=None)
        manager.start_debug = AsyncMock(return_value=DebugSession("session-1"))
        manager.take_snapshot = AsyncMock(return_value=SessionSnapshot("session-1"))
        manager.attach = AsyncMock()
        manager.get_attach_command = AsyncMock(return_value=["ssh", "host"])
        return manager

    @pytest.fixture
    def workflow(self, mock_manager):
        """Create a debug workflow for testing."""
        return DebugWorkflow(mock_manager)

    @pytest.mark.asyncio
    async def test_inspect_session(self, workflow, mock_manager):
        """inspect_session MUST return formatted output."""
        result = await workflow.inspect_session("session-1")

        assert "session-1" in result
        mock_manager.take_snapshot.assert_called()

    @pytest.mark.asyncio
    async def test_inspect_session_json(self, workflow, mock_manager):
        """inspect_session MUST support JSON format."""
        result = await workflow.inspect_session("session-1", format="json")

        import json
        data = json.loads(result)
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_quick_attach(self, workflow, mock_manager):
        """quick_attach MUST return connection info."""
        mock_tunnel = MagicMock()
        mock_tunnel.ssh_url = "ssh://root@host:22"
        mock_manager.attach = AsyncMock(return_value=mock_tunnel)

        result = await workflow.quick_attach(
            "session-1",
            tunnel_host="r3.modal.host",
            tunnel_port=23447,
        )

        assert result["session_id"] == "session-1"
        assert "tunnel_url" in result
        assert "attach_command" in result

    @pytest.mark.asyncio
    async def test_diagnose_session_healthy(self, workflow, mock_manager):
        """diagnose_session MUST report healthy status."""
        mock_manager.take_snapshot = AsyncMock(return_value=SessionSnapshot(
            session_id="session-1",
            cpu_percent=30.0,
            memory_mb=4000,
        ))

        result = await workflow.diagnose_session("session-1")

        assert result["healthy"] is True
        assert len(result["issues"]) == 0

    @pytest.mark.asyncio
    async def test_diagnose_session_with_intervention(self, workflow, mock_manager):
        """diagnose_session MUST detect interventions."""
        mock_manager.take_snapshot = AsyncMock(return_value=SessionSnapshot(
            session_id="session-1",
            has_pending_intervention=True,
        ))

        result = await workflow.diagnose_session("session-1")

        assert result["healthy"] is False
        assert any("intervention" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_diagnose_session_high_cpu(self, workflow, mock_manager):
        """diagnose_session MUST detect high CPU."""
        mock_manager.take_snapshot = AsyncMock(return_value=SessionSnapshot(
            session_id="session-1",
            cpu_percent=95.0,
        ))

        result = await workflow.diagnose_session("session-1")

        assert result["healthy"] is False
        assert any("cpu" in i.lower() for i in result["issues"])

    @pytest.mark.asyncio
    async def test_diagnose_session_high_memory(self, workflow, mock_manager):
        """diagnose_session MUST detect high memory."""
        mock_manager.take_snapshot = AsyncMock(return_value=SessionSnapshot(
            session_id="session-1",
            memory_mb=15000,
        ))

        result = await workflow.diagnose_session("session-1")

        assert result["healthy"] is False
        assert any("memory" in i.lower() for i in result["issues"])


# =============================================================================
# Integration Tests
# =============================================================================


class TestDebugIntegration:
    """Integration tests for debugging workflow."""

    @pytest.mark.asyncio
    async def test_full_debug_workflow(self):
        """Full debug workflow MUST work correctly."""
        # Create real managers with mocked internals
        tmux = TmuxManager()
        tunnel = SSHTunnelManager()
        intervention = InterventionManager()

        manager = DebugManager(
            tmux_manager=tmux,
            tunnel_manager=tunnel,
            intervention_manager=intervention,
        )

        with patch.object(tmux, "_run_command", new_callable=AsyncMock):
            with patch.object(tmux, "capture_pane", new_callable=AsyncMock) as mock_capture:
                mock_capture.return_value = "Claude Code running..."

                # Start debug
                debug = await manager.start_debug("session-1")
                assert debug.state == DebugState.INSPECTING

                # Take snapshot
                snapshot = await manager.take_snapshot("session-1")
                assert snapshot.recent_output == "Claude Code running..."

                # Set breakpoint
                await manager.set_breakpoint("session-1", "Bash")
                assert "Bash" in debug.breakpoints

                # Stop debug
                await manager.stop_debug("session-1")
                assert await manager.get_debug_session("session-1") is None
