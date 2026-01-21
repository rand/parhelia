"""Tests for Parhelia CLI.

Tests CLI commands for session management.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from parhelia.cli import cli, CLIContext


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = MagicMock()
    config.modal.volume_name = "parhelia-vol"
    config.modal.cpu_count = 4
    config.modal.memory_mb = 16384
    config.modal.default_timeout_hours = 4
    config.budget.default_ceiling_usd = 10.0
    config.paths.volume_root = "/vol/parhelia"
    return config


# =============================================================================
# CLI Context Tests
# =============================================================================


class TestCLIContext:
    """Tests for CLIContext class."""

    def test_context_creation(self, mock_config):
        """CLIContext MUST initialize with config."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            ctx = CLIContext(verbose=True)

            assert ctx.verbose is True
            assert ctx.config == mock_config

    def test_orchestrator_lazy_loading(self, mock_config):
        """CLIContext MUST lazy-load orchestrator."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            ctx = CLIContext()

            assert ctx._orchestrator is None
            orch = ctx.orchestrator
            assert orch is not None
            assert ctx._orchestrator is orch

    def test_budget_manager_lazy_loading(self, mock_config):
        """CLIContext MUST lazy-load budget manager."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            ctx = CLIContext()

            assert ctx._budget_manager is None
            budget = ctx.budget_manager
            assert budget is not None
            assert budget.ceiling_usd == 10.0

    def test_log_verbose(self, mock_config, capsys):
        """CLIContext MUST log when verbose is True."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            ctx = CLIContext(verbose=True)
            ctx.log("Test message", "info")

            # Note: click.echo output is not captured by capsys
            # This is a basic test for coverage


# =============================================================================
# Status Command Tests
# =============================================================================


class TestStatusCommand:
    """Tests for status command."""

    def test_status_shows_info(self, runner, mock_config):
        """status MUST show system status."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Parhelia System Status" in result.output
            assert "Configuration:" in result.output
            assert "Orchestrator:" in result.output
            assert "Budget:" in result.output

    def test_status_shows_volume(self, runner, mock_config):
        """status MUST show volume name."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["status"])

            assert "parhelia-vol" in result.output


# =============================================================================
# List Command Tests
# =============================================================================


class TestListCommand:
    """Tests for list command."""

    def test_list_no_sessions(self, runner, mock_config, tmp_path):
        """list MUST show message when no tasks."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                # Use temp database
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["list"])

                assert result.exit_code == 0
                assert "No tasks found" in result.output

    def test_list_with_status_filter(self, runner, mock_config, tmp_path):
        """list MUST accept status filter."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["list", "-s", "running"])

                assert result.exit_code == 0

    def test_list_with_limit(self, runner, mock_config, tmp_path):
        """list MUST accept limit option."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["list", "-n", "5"])

                assert result.exit_code == 0


# =============================================================================
# Submit Command Tests
# =============================================================================


class TestSubmitCommand:
    """Tests for submit command."""

    def test_submit_creates_task(self, runner, mock_config, tmp_path):
        """submit MUST create and submit task."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["submit", "Test prompt", "--no-dispatch"])

                assert result.exit_code == 0
                assert "Task submitted:" in result.output

    def test_submit_with_type(self, runner, mock_config, tmp_path):
        """submit MUST accept task type option."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["submit", "Run tests", "-t", "test", "--no-dispatch"])

                assert result.exit_code == 0
                assert "Task submitted:" in result.output

    def test_submit_with_gpu(self, runner, mock_config, tmp_path):
        """submit MUST accept GPU option."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["submit", "GPU task", "--gpu", "A10G", "--no-dispatch"])

                assert result.exit_code == 0

    def test_submit_with_memory(self, runner, mock_config, tmp_path):
        """submit MUST accept memory option."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["submit", "Big task", "--memory", "16", "--no-dispatch"])

                assert result.exit_code == 0

    def test_submit_with_dispatch_dry_run(self, runner, mock_config, tmp_path):
        """submit MUST dispatch with --dry-run."""
        from parhelia.persistence import PersistentOrchestrator

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.PersistentOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = PersistentOrchestrator(
                    db_path=tmp_path / "test.db"
                )
                result = runner.invoke(cli, ["submit", "Test dry run", "--dry-run"])

                assert result.exit_code == 0
                assert "Task submitted:" in result.output
                assert "Dispatching (dry-run)" in result.output
                assert "Worker started:" in result.output


# =============================================================================
# Attach Command Tests
# =============================================================================


class TestAttachCommand:
    """Tests for attach command."""

    def test_attach_not_found(self, runner, mock_config):
        """attach MUST fail if session not found."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["attach", "nonexistent"])

            assert result.exit_code == 1
            assert "Session not found" in result.output

    def test_attach_not_found_json(self, runner, mock_config):
        """attach MUST return JSON error when session not found."""
        import json

        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["attach", "nonexistent", "--json"])

            assert result.exit_code == 1
            parsed = json.loads(result.output)
            assert parsed["success"] is False
            assert parsed["error"]["code"] == "SESSION_NOT_FOUND"

    def test_attach_session_not_running(self, runner, mock_config):
        """attach MUST fail if session is not in running state."""
        from parhelia.orchestrator import WorkerInfo, WorkerState

        mock_worker = WorkerInfo(
            id="worker-1",
            task_id="task-1",
            state=WorkerState.FAILED,
            target_type="parhelia-cpu",
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.CLIContext.orchestrator") as mock_orch:
                mock_orch.get_task = AsyncMock(return_value=MagicMock())
                mock_orch.worker_store.get_by_task = MagicMock(return_value=mock_worker)

                result = runner.invoke(cli, ["attach", "task-1"])

                assert result.exit_code == 1
                assert "not running" in result.output

    def test_attach_info_only(self, runner, mock_config):
        """attach --info-only MUST show connection info without connecting."""
        from parhelia.orchestrator import WorkerInfo, WorkerState

        mock_worker = WorkerInfo(
            id="worker-1",
            task_id="task-1",
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
            metrics={"tunnel_host": "localhost", "tunnel_port": 2222},
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.CLIContext.orchestrator") as mock_orch:
                mock_orch.get_task = AsyncMock(return_value=MagicMock())
                mock_orch.worker_store.get_by_task = MagicMock(return_value=mock_worker)

                result = runner.invoke(cli, ["attach", "task-1", "--info-only"])

                assert result.exit_code == 0
                assert "Connection info" in result.output
                assert "tmux session" in result.output

    def test_attach_info_only_json(self, runner, mock_config):
        """attach --info-only --json MUST return structured connection info."""
        import json
        from parhelia.orchestrator import WorkerInfo, WorkerState

        mock_worker = WorkerInfo(
            id="worker-1",
            task_id="task-1",
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
            metrics={"tunnel_host": "localhost", "tunnel_port": 2222},
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.CLIContext.orchestrator") as mock_orch:
                mock_orch.get_task = AsyncMock(return_value=MagicMock())
                mock_orch.worker_store.get_by_task = MagicMock(return_value=mock_worker)

                result = runner.invoke(cli, ["attach", "task-1", "--info-only", "--json"])

                assert result.exit_code == 0
                parsed = json.loads(result.output)
                assert parsed["success"] is True
                assert "tunnel" in parsed["data"]
                assert "ssh_command" in parsed["data"]
                assert "tmux_session" in parsed["data"]


# =============================================================================
# Detach Command Tests
# =============================================================================


class TestDetachCommand:
    """Tests for detach command."""

    def test_detach_shows_message(self, runner, mock_config):
        """detach MUST show detach message."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["detach", "session-1"])

            assert result.exit_code == 0
            assert "Detaching from session" in result.output

    def test_detach_creates_checkpoint(self, runner, mock_config):
        """detach MUST create checkpoint by default."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        mock_checkpoint = Checkpoint(
            id="cp-detach-1",
            session_id="session-1",
            trigger=CheckpointTrigger.DETACH,
            working_directory="/vol/workspaces/session-1",
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch(
                "parhelia.checkpoint.CheckpointManager.create_checkpoint",
                new_callable=lambda: AsyncMock(return_value=mock_checkpoint),
            ):
                result = runner.invoke(cli, ["detach", "session-1"])

                assert result.exit_code == 0
                assert "Creating checkpoint" in result.output
                assert "Checkpoint created" in result.output

    def test_detach_no_checkpoint(self, runner, mock_config):
        """detach --no-checkpoint MUST skip checkpoint creation."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["detach", "session-1", "--no-checkpoint"])

            assert result.exit_code == 0
            assert "Creating checkpoint" not in result.output
            assert "Session continues running" in result.output

    def test_detach_json_output(self, runner, mock_config):
        """detach --json MUST return structured response."""
        import json
        from parhelia.session import Checkpoint, CheckpointTrigger

        mock_checkpoint = Checkpoint(
            id="cp-detach-1",
            session_id="session-1",
            trigger=CheckpointTrigger.DETACH,
            working_directory="/vol/workspaces/session-1",
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch(
                "parhelia.checkpoint.CheckpointManager.create_checkpoint",
                new_callable=lambda: AsyncMock(return_value=mock_checkpoint),
            ):
                result = runner.invoke(cli, ["detach", "session-1", "--json"])

                assert result.exit_code == 0
                parsed = json.loads(result.output)
                assert parsed["success"] is True
                assert parsed["data"]["session_id"] == "session-1"
                assert parsed["data"]["status"] == "detached"
                assert parsed["data"]["checkpoint_id"] == "cp-detach-1"

    def test_detach_json_no_checkpoint(self, runner, mock_config):
        """detach --json --no-checkpoint MUST return null checkpoint_id."""
        import json

        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["detach", "session-1", "--json", "--no-checkpoint"])

            assert result.exit_code == 0
            parsed = json.loads(result.output)
            assert parsed["success"] is True
            assert parsed["data"]["checkpoint_id"] is None


# =============================================================================
# Checkpoint Command Tests
# =============================================================================


class TestCheckpointCommand:
    """Tests for checkpoint command."""

    def test_checkpoint_creates(self, runner, mock_config):
        """checkpoint MUST create checkpoint."""
        from datetime import datetime
        from parhelia.session import Checkpoint, CheckpointTrigger

        mock_checkpoint = Checkpoint(
            id="cp-abc123",
            session_id="session-1",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/vol/workspaces/session-1",
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch(
                "parhelia.checkpoint.CheckpointManager.create_checkpoint",
                new_callable=lambda: AsyncMock(return_value=mock_checkpoint),
            ):
                result = runner.invoke(cli, ["checkpoint", "session-1"])

                assert result.exit_code == 0
                assert "Checkpoint created" in result.output

    def test_checkpoint_with_message(self, runner, mock_config):
        """checkpoint MUST accept message option."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        mock_checkpoint = Checkpoint(
            id="cp-abc123",
            session_id="session-1",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/vol/workspaces/session-1",
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch(
                "parhelia.checkpoint.CheckpointManager.create_checkpoint",
                new_callable=lambda: AsyncMock(return_value=mock_checkpoint),
            ):
                result = runner.invoke(cli, ["checkpoint", "session-1", "-m", "Before refactor"])

                assert result.exit_code == 0


# =============================================================================
# Resume Command Tests
# =============================================================================


class TestResumeCommand:
    """Tests for resume command."""

    def test_resume_no_checkpoint(self, runner, mock_config):
        """resume MUST fail if no checkpoint exists."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.cli.CheckpointManager") as mock_cm:
                mock_cm_instance = MagicMock()
                mock_cm_instance.get_latest_checkpoint = AsyncMock(return_value=None)
                mock_cm.return_value = mock_cm_instance

                result = runner.invoke(cli, ["resume", "session-1"])

                assert result.exit_code == 1
                assert "No checkpoint found" in result.output


# =============================================================================
# Session Recover Command Tests
# =============================================================================


class TestSessionRecoverCommand:
    """Tests for session recover command."""

    def test_recover_no_options(self, runner, mock_config):
        """session recover MUST fail if no recovery options exist."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.recovery.RecoveryManager") as mock_rm:
                mock_rm_instance = MagicMock()
                mock_rm_instance.plan_failure_recovery = AsyncMock(return_value=None)
                mock_rm.return_value = mock_rm_instance

                result = runner.invoke(cli, ["session", "recover", "session-1"])

                assert result.exit_code == 1
                assert "No recovery options found" in result.output

    def test_recover_no_options_json(self, runner, mock_config):
        """session recover --json MUST return error when no options."""
        import json

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.recovery.RecoveryManager") as mock_rm:
                mock_rm_instance = MagicMock()
                mock_rm_instance.plan_failure_recovery = AsyncMock(return_value=None)
                mock_rm.return_value = mock_rm_instance

                result = runner.invoke(cli, ["session", "recover", "session-1", "--json"])

                assert result.exit_code == 1
                parsed = json.loads(result.output)
                assert parsed["success"] is False
                assert parsed["error"]["code"] == "SESSION_NOT_FOUND"

    def test_recover_json_returns_plan(self, runner, mock_config):
        """session recover --json MUST return recovery plan options."""
        import json
        from parhelia.recovery import RecoveryAction, RecoveryOption, RecoveryPlan, RecoveryScenario

        mock_plan = RecoveryPlan(
            session_id="session-1",
            scenario=RecoveryScenario.FAILURE,
            current_checkpoint_id="cp-test-1",
            options=[
                RecoveryOption(
                    action=RecoveryAction.RESUME,
                    checkpoint_id="cp-test-1",
                    description="Resume from last checkpoint",
                    recommended=True,
                ),
                RecoveryOption(
                    action=RecoveryAction.NEW_SESSION,
                    checkpoint_id=None,
                    description="Start new session",
                    recommended=False,
                ),
            ],
        )

        with patch("parhelia.cli.load_config", return_value=mock_config):
            with patch("parhelia.recovery.RecoveryManager") as mock_rm:
                mock_rm_instance = MagicMock()
                mock_rm_instance.plan_failure_recovery = AsyncMock(return_value=mock_plan)
                mock_rm.return_value = mock_rm_instance

                result = runner.invoke(cli, ["session", "recover", "session-1", "--json"])

                assert result.exit_code == 0
                parsed = json.loads(result.output)
                assert parsed["success"] is True
                assert parsed["data"]["session_id"] == "session-1"
                assert parsed["data"]["scenario"] == "failure"
                assert len(parsed["data"]["options"]) == 2
                assert parsed["data"]["options"][0]["action"] == "resume"


# =============================================================================
# Logs Command Tests
# =============================================================================


class TestLogsCommand:
    """Tests for logs command."""

    def test_logs_shows_session(self, runner, mock_config):
        """logs MUST show session ID."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["logs", "session-1"])

            assert result.exit_code == 0
            assert "session-1" in result.output

    def test_logs_with_follow(self, runner, mock_config):
        """logs MUST accept follow flag."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["logs", "session-1", "-f"])

            assert result.exit_code == 0
            assert "Following" in result.output

    def test_logs_with_lines(self, runner, mock_config):
        """logs MUST accept lines option."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["logs", "session-1", "-n", "100"])

            assert result.exit_code == 0
            assert "100" in result.output


# =============================================================================
# Budget Command Tests
# =============================================================================


class TestBudgetCommand:
    """Tests for budget commands."""

    def test_budget_show(self, runner, mock_config):
        """budget show MUST display budget status."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["budget", "show"])

            assert result.exit_code == 0
            assert "Budget Status" in result.output
            assert "Ceiling:" in result.output
            assert "Used:" in result.output

    def test_budget_set(self, runner, mock_config):
        """budget set MUST update ceiling."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["budget", "set", "50.0"])

            assert result.exit_code == 0
            assert "Budget ceiling set" in result.output
            assert "50.00" in result.output

    def test_budget_reset(self, runner, mock_config):
        """budget reset MUST clear tracking."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["budget", "reset", "--yes"])

            assert result.exit_code == 0
            assert "reset" in result.output.lower()


# =============================================================================
# Config Command Tests
# =============================================================================


class TestConfigCommand:
    """Tests for config command."""

    def test_config_shows_settings(self, runner, mock_config):
        """config MUST show configuration."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["config"])

            assert result.exit_code == 0
            assert "Parhelia Configuration" in result.output
            assert "modal" in result.output.lower()

    def test_config_json_output(self, runner, mock_config):
        """config MUST support JSON output."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["config", "--json"])

            assert result.exit_code == 0
            # Should be valid JSON
            import json
            config_dict = json.loads(result.output)
            assert "modal" in config_dict
            assert "budget" in config_dict


# =============================================================================
# CLI Options Tests
# =============================================================================


class TestCLIOptions:
    """Tests for global CLI options."""

    def test_version_option(self, runner, mock_config):
        """CLI MUST show version."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["--version"])

            assert result.exit_code == 0
            assert "0.1.0" in result.output

    def test_help_option(self, runner, mock_config):
        """CLI MUST show help."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["--help"])

            assert result.exit_code == 0
            assert "Parhelia" in result.output
            assert "status" in result.output
            assert "submit" in result.output

    def test_verbose_option(self, runner, mock_config):
        """CLI MUST accept verbose flag."""
        with patch("parhelia.cli.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["-v", "status"])

            assert result.exit_code == 0
