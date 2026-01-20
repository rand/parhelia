"""Tests for Parhelia tmux session management.

Tests tmux session creation, lifecycle, and headless execution per SPEC-02.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.tmux import (
    HeadlessRunner,
    SessionMonitor,
    TmuxConfig,
    TmuxManager,
    TmuxSession,
    TmuxSessionState,
)


# =============================================================================
# TmuxConfig Tests
# =============================================================================


class TestTmuxConfig:
    """Tests for TmuxConfig class."""

    def test_default_config(self):
        """TmuxConfig MUST have sensible defaults."""
        config = TmuxConfig()

        assert config.default_shell == "/bin/bash"
        assert config.history_limit == 50000
        assert config.escape_time == 0
        assert config.mouse_enabled is True
        assert config.checkpoint_on_detach is True

    def test_custom_config(self):
        """TmuxConfig MUST accept custom values."""
        config = TmuxConfig(
            history_limit=10000,
            mouse_enabled=False,
        )

        assert config.history_limit == 10000
        assert config.mouse_enabled is False

    def test_generate_config(self):
        """generate_config MUST produce valid tmux config."""
        config = TmuxConfig()
        content = config.generate_config()

        assert "set-option -g default-shell /bin/bash" in content
        assert "set-option -g history-limit 50000" in content
        assert "set-option -g mouse on" in content

    def test_generate_config_checkpoint_hook(self):
        """generate_config MUST include checkpoint hook when enabled."""
        config = TmuxConfig(checkpoint_on_detach=True)
        content = config.generate_config()

        assert "parhelia-checkpoint" in content
        assert "client-detached" in content

    def test_generate_config_no_checkpoint_hook(self):
        """generate_config MUST exclude checkpoint hook when disabled."""
        config = TmuxConfig(checkpoint_on_detach=False)
        content = config.generate_config()

        assert "parhelia-checkpoint" not in content


# =============================================================================
# TmuxSession Tests
# =============================================================================


class TestTmuxSession:
    """Tests for TmuxSession data class."""

    def test_session_creation(self):
        """TmuxSession MUST initialize with required fields."""
        session = TmuxSession(name="ph-test-20260120T120000")

        assert session.name == "ph-test-20260120T120000"
        assert session.state == TmuxSessionState.CREATED
        assert session.working_directory == "/vol/parhelia/workspaces"

    def test_session_timestamps(self):
        """TmuxSession MUST track timestamps."""
        before = datetime.now()
        session = TmuxSession(name="ph-test-20260120T120000")
        after = datetime.now()

        assert before <= session.created_at <= after
        assert session.attached_at is None
        assert session.detached_at is None


# =============================================================================
# TmuxManager Tests
# =============================================================================


class TestTmuxManager:
    """Tests for TmuxManager class."""

    @pytest.fixture
    def manager(self):
        """Create a tmux manager for testing."""
        return TmuxManager()

    @pytest.mark.asyncio
    async def test_setup_writes_config(self, manager, tmp_path):
        """setup MUST write configuration file."""
        config_path = tmp_path / "tmux.conf"
        manager.config.config_path = str(config_path)

        await manager.setup()

        assert config_path.exists()
        content = config_path.read_text()
        assert "set-option" in content

    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        """create_session MUST create tmux session."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_run:
            session = await manager.create_session(
                task_id="test",
                working_dir="/tmp/test",
            )

            assert session.name.startswith("ph-test-")
            assert session.state == TmuxSessionState.CREATED
            assert session.working_directory == "/tmp/test"
            mock_run.assert_called()

    @pytest.mark.asyncio
    async def test_create_session_with_env(self, manager):
        """create_session MUST set environment variables."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(
                task_id="test",
                env={"API_KEY": "secret123"},
            )

            assert session.environment["API_KEY"] == "secret123"

    @pytest.mark.asyncio
    async def test_create_session_name_format(self, manager):
        """create_session MUST use correct name format."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="fix-auth")

            # Format: ph-{task_id}-{timestamp}
            assert session.name.startswith("ph-fix-auth-")
            # Timestamp format: YYYYMMDDTHHMMSS
            parts = session.name.split("-")
            assert len(parts) >= 3

    @pytest.mark.asyncio
    async def test_kill_session(self, manager):
        """kill_session MUST terminate session."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")
            session_name = session.name

            await manager.kill_session(session_name)

            assert session.state == TmuxSessionState.KILLED
            assert session_name not in manager._sessions

    @pytest.mark.asyncio
    async def test_kill_session_not_found(self, manager):
        """kill_session MUST raise error for unknown session."""
        with pytest.raises(ValueError, match="Session not found"):
            await manager.kill_session("nonexistent")

    @pytest.mark.asyncio
    async def test_attach(self, manager):
        """attach MUST update session state."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            updated = await manager.attach(session.name)

            assert updated.state == TmuxSessionState.ATTACHED
            assert updated.attached_at is not None

    @pytest.mark.asyncio
    async def test_detach(self, manager):
        """detach MUST update session state."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")
            await manager.attach(session.name)

            updated = await manager.detach(session.name)

            assert updated.state == TmuxSessionState.DETACHED
            assert updated.detached_at is not None

    @pytest.mark.asyncio
    async def test_send_keys(self, manager):
        """send_keys MUST send to correct session."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_run:
            session = await manager.create_session(task_id="test")

            await manager.send_keys(session.name, "echo hello")

            # Verify send-keys command was called
            calls = mock_run.call_args_list
            send_keys_call = [c for c in calls if "send-keys" in c[0][0]]
            assert len(send_keys_call) > 0

    @pytest.mark.asyncio
    async def test_send_keys_with_enter(self, manager):
        """send_keys MUST include Enter by default."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_run:
            session = await manager.create_session(task_id="test")

            await manager.send_keys(session.name, "ls", enter=True)

            # Check that Enter is in the command
            calls = mock_run.call_args_list
            last_call = calls[-1][0][0]
            assert "Enter" in last_call

    @pytest.mark.asyncio
    async def test_send_keys_no_enter(self, manager):
        """send_keys MUST exclude Enter when disabled."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_run:
            session = await manager.create_session(task_id="test")

            await manager.send_keys(session.name, "partial", enter=False)

            # Check that Enter is not in the command
            calls = mock_run.call_args_list
            last_call = calls[-1][0][0]
            assert "Enter" not in last_call

    @pytest.mark.asyncio
    async def test_set_environment(self, manager):
        """set_environment MUST update session environment."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            await manager.set_environment(session.name, "MY_VAR", "my_value")

            assert session.environment["MY_VAR"] == "my_value"

    @pytest.mark.asyncio
    async def test_get_session(self, manager):
        """get_session MUST return existing session."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            result = await manager.get_session(session.name)

            assert result is session

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager):
        """get_session MUST return None for unknown session."""
        result = await manager.get_session("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager):
        """list_sessions MUST return all sessions."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            await manager.create_session(task_id="test1")
            await manager.create_session(task_id="test2")

            sessions = await manager.list_sessions()

            assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_session_exists_true(self, manager):
        """session_exists MUST return True for existing session."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            result = await manager.session_exists("some-session")
            assert result is True

    @pytest.mark.asyncio
    async def test_session_exists_false(self, manager):
        """session_exists MUST return False when tmux returns error."""
        async def raise_error(*args, **kwargs):
            raise RuntimeError("no session")

        with patch.object(manager, "_run_command", side_effect=raise_error):
            result = await manager.session_exists("nonexistent")
            assert result is False

    @pytest.mark.asyncio
    async def test_capture_pane(self, manager):
        """capture_pane MUST return pane content."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "captured content\n"

            content = await manager.capture_pane("session-1", start_line=-10)

            assert content == "captured content\n"

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_created(self, manager):
        """on_created hook MUST be called when session created."""
        hook_called = False
        hook_session = None

        async def on_created(session):
            nonlocal hook_called, hook_session
            hook_called = True
            hook_session = session

        manager.on_created(on_created)

        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            assert hook_called is True
            assert hook_session is session

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_detached(self, manager):
        """on_detached hook MUST be called when session detached."""
        hook_called = False

        async def on_detached(session):
            nonlocal hook_called
            hook_called = True

        manager.on_detached(on_detached)

        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")
            await manager.detach(session.name)

            assert hook_called is True


# =============================================================================
# HeadlessRunner Tests
# =============================================================================


class TestHeadlessRunner:
    """Tests for HeadlessRunner class."""

    @pytest.fixture
    def manager(self):
        """Create a tmux manager for testing."""
        return TmuxManager()

    @pytest.fixture
    def runner(self, manager):
        """Create a headless runner for testing."""
        return HeadlessRunner(manager)

    @pytest.mark.asyncio
    async def test_run_basic(self, runner, manager):
        """run MUST send claude command to session."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            await runner.run(session.name, "Fix the bug")

            # Verify command was sent
            calls = manager._run_command.call_args_list
            send_keys_calls = [c for c in calls if "send-keys" in c[0][0]]
            assert len(send_keys_calls) > 0

            # Verify claude command format
            last_call = send_keys_calls[-1][0][0]
            assert "claude" in " ".join(last_call)

    @pytest.mark.asyncio
    async def test_run_with_allowed_tools(self, runner, manager):
        """run MUST include allowed tools when specified."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            await runner.run(
                session.name,
                "Run tests",
                allowed_tools=["Read", "Bash"],
            )

            calls = manager._run_command.call_args_list
            send_keys_calls = [c for c in calls if "send-keys" in c[0][0]]
            last_call_str = " ".join(send_keys_calls[-1][0][0])
            assert "allowedTools" in last_call_str

    @pytest.mark.asyncio
    async def test_run_with_max_turns(self, runner, manager):
        """run MUST include max turns."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            await runner.run(session.name, "Build project", max_turns=100)

            calls = manager._run_command.call_args_list
            send_keys_calls = [c for c in calls if "send-keys" in c[0][0]]
            last_call_str = " ".join(send_keys_calls[-1][0][0])
            assert "max-turns" in last_call_str
            assert "100" in last_call_str

    @pytest.mark.asyncio
    async def test_run_updates_session_state(self, runner, manager):
        """run MUST update session state to RUNNING."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await manager.create_session(task_id="test")

            await runner.run(session.name, "Do something")

            assert session.state == TmuxSessionState.RUNNING

    @pytest.mark.asyncio
    async def test_run_in_new_session(self, runner, manager):
        """run_in_new_session MUST create session and run."""
        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session = await runner.run_in_new_session(
                task_id="test",
                prompt="Fix the bug",
                working_dir="/tmp/test",
            )

            assert session.name.startswith("ph-test-")
            assert session.state == TmuxSessionState.RUNNING


# =============================================================================
# SessionMonitor Tests
# =============================================================================


class TestSessionMonitor:
    """Tests for SessionMonitor class."""

    @pytest.fixture
    def manager(self):
        """Create a tmux manager for testing."""
        return TmuxManager()

    @pytest.fixture
    def monitor(self, manager):
        """Create a session monitor for testing."""
        return SessionMonitor(manager, idle_timeout=60.0)

    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor):
        """start_monitoring MUST add session to monitoring."""
        await monitor.start_monitoring("session-1")

        assert "session-1" in monitor._monitoring
        assert "session-1" in monitor._last_activity

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """stop_monitoring MUST remove session from monitoring."""
        await monitor.start_monitoring("session-1")
        await monitor.stop_monitoring("session-1")

        assert "session-1" not in monitor._monitoring
        assert "session-1" not in monitor._last_activity

    @pytest.mark.asyncio
    async def test_check_activity_new_content(self, monitor, manager):
        """check_activity MUST return True when content changes."""
        await monitor.start_monitoring("session-1")

        with patch.object(manager, "capture_pane", new_callable=AsyncMock) as mock_capture:
            mock_capture.return_value = "new output"

            result = await monitor.check_activity("session-1")

            assert result is True

    @pytest.mark.asyncio
    async def test_check_activity_same_content(self, monitor, manager):
        """check_activity MUST return False when content unchanged."""
        await monitor.start_monitoring("session-1")
        monitor._last_content["session-1"] = "same content"

        with patch.object(manager, "capture_pane", new_callable=AsyncMock) as mock_capture:
            mock_capture.return_value = "same content"

            result = await monitor.check_activity("session-1")

            assert result is False

    @pytest.mark.asyncio
    async def test_is_idle_not_exceeded(self, monitor):
        """is_idle MUST return False when within timeout."""
        await monitor.start_monitoring("session-1")

        result = await monitor.is_idle("session-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_idle_exceeded(self, monitor):
        """is_idle MUST return True when timeout exceeded."""
        await monitor.start_monitoring("session-1")
        # Set last activity to past the timeout
        from datetime import timedelta
        monitor._last_activity["session-1"] = datetime.now() - timedelta(seconds=120)

        result = await monitor.is_idle("session-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_get_idle_time(self, monitor):
        """get_idle_time MUST return seconds since activity."""
        await monitor.start_monitoring("session-1")
        from datetime import timedelta
        monitor._last_activity["session-1"] = datetime.now() - timedelta(seconds=30)

        idle_time = await monitor.get_idle_time("session-1")

        assert 29 <= idle_time <= 32  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_get_idle_time_unknown_session(self, monitor):
        """get_idle_time MUST return 0 for unknown session."""
        idle_time = await monitor.get_idle_time("unknown")
        assert idle_time == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestTmuxIntegration:
    """Integration tests for tmux management."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self):
        """Full session lifecycle MUST work correctly."""
        manager = TmuxManager()
        runner = HeadlessRunner(manager)

        # Track lifecycle events
        events = []

        async def on_created(s):
            events.append(("created", s.name))

        async def on_attached(s):
            events.append(("attached", s.name))

        async def on_detached(s):
            events.append(("detached", s.name))

        async def on_killed(s):
            events.append(("killed", s.name))

        manager.on_created(on_created)
        manager.on_attached(on_attached)
        manager.on_detached(on_detached)
        manager.on_killed(on_killed)

        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            # Create session
            session = await manager.create_session(task_id="test")
            assert ("created", session.name) in events

            # Run headless
            await runner.run(session.name, "Do work")
            assert session.state == TmuxSessionState.RUNNING

            # Attach
            await manager.attach(session.name)
            assert ("attached", session.name) in events

            # Detach
            await manager.detach(session.name)
            assert ("detached", session.name) in events

            # Kill
            await manager.kill_session(session.name)
            assert ("killed", session.name) in events

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        """Manager MUST handle multiple sessions."""
        manager = TmuxManager()

        with patch.object(manager, "_run_command", new_callable=AsyncMock):
            session1 = await manager.create_session(task_id="task1")
            session2 = await manager.create_session(task_id="task2")
            session3 = await manager.create_session(task_id="task3")

            sessions = await manager.list_sessions()
            assert len(sessions) == 3

            # Kill middle session
            await manager.kill_session(session2.name)

            sessions = await manager.list_sessions()
            assert len(sessions) == 2
            assert session2 not in sessions
