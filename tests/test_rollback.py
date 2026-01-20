"""Tests for coordinated workspace rollback.

@trace SPEC-07.40.01 - Rollback Command
@trace SPEC-07.40.02 - Safety Guarantees
@trace SPEC-07.40.03 - Rollback Scope
@trace SPEC-07.40.04 - Rollback Recovery
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.checkpoint import Checkpoint, CheckpointTrigger
from parhelia.rollback import (
    GitStashResult,
    RollbackPhase,
    RollbackPlan,
    RollbackResult,
    WorkspaceRollback,
)


class TestGitStashResult:
    """Tests for GitStashResult dataclass."""

    def test_stashed(self):
        """GitStashResult MUST capture stash state."""
        result = GitStashResult(
            stashed=True,
            stash_ref="stash@{0}",
            message="Changes stashed",
        )

        assert result.stashed is True
        assert result.stash_ref == "stash@{0}"
        assert "stashed" in result.message.lower()

    def test_not_stashed(self):
        """GitStashResult MUST handle no changes case."""
        result = GitStashResult(
            stashed=False,
            message="No changes to stash",
        )

        assert result.stashed is False
        assert result.stash_ref is None


class TestRollbackResult:
    """Tests for RollbackResult dataclass."""

    def test_success_result(self):
        """RollbackResult MUST capture success details."""
        result = RollbackResult(
            success=True,
            target_checkpoint_id="cp-abc123",
            safety_checkpoint_id="cp-safe001",
            stash_ref="stash@{0}",
            files_restored=["src/auth.py", "tests/test_auth.py"],
            phase_reached=RollbackPhase.COMPLETE,
        )

        assert result.success is True
        assert result.target_checkpoint_id == "cp-abc123"
        assert result.safety_checkpoint_id == "cp-safe001"
        assert len(result.files_restored) == 2

    def test_failure_result(self):
        """RollbackResult MUST capture failure details."""
        result = RollbackResult(
            success=False,
            target_checkpoint_id="cp-abc123",
            phase_reached=RollbackPhase.FAILED,
            error_message="Checkpoint not found",
        )

        assert result.success is False
        assert result.error_message == "Checkpoint not found"
        assert result.phase_reached == RollbackPhase.FAILED

    def test_to_dict(self):
        """RollbackResult MUST serialize to dict."""
        result = RollbackResult(
            success=True,
            target_checkpoint_id="cp-abc123",
            phase_reached=RollbackPhase.COMPLETE,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["target_checkpoint_id"] == "cp-abc123"
        assert data["phase_reached"] == "complete"


class TestRollbackPlan:
    """Tests for RollbackPlan dataclass."""

    def test_plan_creation(self):
        """RollbackPlan MUST capture plan details."""
        checkpoint = MagicMock(spec=Checkpoint)
        checkpoint.id = "cp-abc123"

        plan = RollbackPlan(
            target_checkpoint=checkpoint,
            files_to_restore=["file1.py", "file2.py"],
            has_uncommitted_changes=True,
            requires_confirmation=True,
        )

        assert plan.target_checkpoint.id == "cp-abc123"
        assert len(plan.files_to_restore) == 2
        assert plan.has_uncommitted_changes is True


class TestWorkspaceRollback:
    """Tests for WorkspaceRollback."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize as git repo
            import subprocess

            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmpdir,
                capture_output=True,
            )
            # Create initial commit
            (Path(tmpdir) / "README.md").write_text("# Test")
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial"],
                cwd=tmpdir,
                capture_output=True,
            )
            yield tmpdir

    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Create mock checkpoint manager."""
        manager = MagicMock()
        manager.create_checkpoint = AsyncMock()
        manager.list_checkpoints = AsyncMock(return_value=[])
        manager.restore_checkpoint = AsyncMock()
        return manager

    @pytest.fixture
    def rollback(self, mock_checkpoint_manager, temp_workspace):
        """Create WorkspaceRollback instance."""
        return WorkspaceRollback(
            checkpoint_manager=mock_checkpoint_manager,
            workspace_dir=temp_workspace,
            session_id="test-session-123",
        )

    @pytest.mark.asyncio
    async def test_check_no_uncommitted_changes(self, rollback, temp_workspace):
        """@trace SPEC-07.40.02 - Check MUST detect clean workspace."""
        has_changes, files = await rollback.check_uncommitted_changes()

        assert has_changes is False
        assert files == []

    @pytest.mark.asyncio
    async def test_check_uncommitted_changes(self, rollback, temp_workspace):
        """@trace SPEC-07.40.02 - Check MUST detect modified files."""
        # Create uncommitted change
        (Path(temp_workspace) / "new_file.py").write_text("print('hello')")

        has_changes, files = await rollback.check_uncommitted_changes()

        assert has_changes is True
        assert "new_file.py" in files

    @pytest.mark.asyncio
    async def test_stash_changes(self, rollback, temp_workspace):
        """@trace SPEC-07.40.02 - Stash MUST save uncommitted changes."""
        # Create uncommitted change
        (Path(temp_workspace) / "new_file.py").write_text("print('hello')")

        result = await rollback.stash_changes("test-stash")

        assert result.stashed is True
        assert result.stash_ref is not None

        # Verify file is now clean
        has_changes, _ = await rollback.check_uncommitted_changes()
        assert has_changes is False

    @pytest.mark.asyncio
    async def test_stash_no_changes(self, rollback):
        """@trace SPEC-07.40.02 - Stash MUST handle clean workspace."""
        result = await rollback.stash_changes()

        assert result.stashed is False
        assert "no changes" in result.message.lower()

    @pytest.mark.asyncio
    async def test_pop_stash(self, rollback, temp_workspace):
        """@trace SPEC-07.40.04 - Pop MUST restore stashed changes."""
        # Create and stash a change
        new_file = Path(temp_workspace) / "stashed_file.py"
        new_file.write_text("stashed content")

        stash_result = await rollback.stash_changes()
        assert stash_result.stashed is True

        # Pop the stash
        success = await rollback.pop_stash()
        assert success is True

        # Verify file is back
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_create_safety_checkpoint(self, rollback, mock_checkpoint_manager):
        """@trace SPEC-07.40.02 - Safety checkpoint MUST be created."""
        safety_cp = MagicMock(spec=Checkpoint)
        safety_cp.id = "cp-safety-001"
        mock_checkpoint_manager.create_checkpoint.return_value = safety_cp

        result = await rollback.create_safety_checkpoint()

        assert result.id == "cp-safety-001"
        mock_checkpoint_manager.create_checkpoint.assert_called_once()
        call_kwargs = mock_checkpoint_manager.create_checkpoint.call_args.kwargs
        assert call_kwargs["trigger"] == CheckpointTrigger.MANUAL
        assert "safety" in call_kwargs["reason"].lower()

    @pytest.mark.asyncio
    async def test_verify_checkpoint_readable_snapshot(self, rollback, temp_workspace):
        """@trace SPEC-07.40.02 - Verify MUST check snapshot exists."""
        checkpoint = MagicMock(spec=Checkpoint)
        checkpoint.workspace_snapshot = str(Path(temp_workspace) / "snapshot.tar.gz")
        checkpoint.workspace_root = None

        # Snapshot doesn't exist
        readable, error = await rollback.verify_checkpoint_readable(checkpoint)

        assert readable is False
        assert "not found" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_checkpoint_readable_success(self, rollback, temp_workspace):
        """@trace SPEC-07.40.02 - Verify MUST succeed for valid checkpoint."""
        # Create a fake snapshot file
        snapshot_path = Path(temp_workspace) / "snapshot.tar.gz"
        snapshot_path.write_text("fake snapshot")

        checkpoint = MagicMock(spec=Checkpoint)
        checkpoint.workspace_snapshot = str(snapshot_path)
        checkpoint.workspace_root = None

        readable, error = await rollback.verify_checkpoint_readable(checkpoint)

        assert readable is True
        assert error is None

    @pytest.mark.asyncio
    async def test_verify_checkpoint_no_data(self, rollback):
        """@trace SPEC-07.40.02 - Verify MUST fail for checkpoint without data."""
        checkpoint = MagicMock(spec=Checkpoint)
        checkpoint.workspace_snapshot = None
        checkpoint.workspace_root = None

        readable, error = await rollback.verify_checkpoint_readable(checkpoint)

        assert readable is False
        assert "no workspace data" in error.lower()

    @pytest.mark.asyncio
    async def test_plan_rollback(self, rollback, mock_checkpoint_manager):
        """@trace SPEC-07.40.01 - Plan MUST analyze rollback requirements."""
        target_cp = MagicMock(spec=Checkpoint)
        target_cp.id = "cp-target-001"
        target_cp.uncommitted_changes = ["file1.py", "file2.py"]

        mock_checkpoint_manager.list_checkpoints.return_value = [target_cp]

        plan = await rollback.plan_rollback("cp-target-001")

        assert plan is not None
        assert plan.target_checkpoint.id == "cp-target-001"
        assert len(plan.files_to_restore) == 2

    @pytest.mark.asyncio
    async def test_plan_rollback_not_found(self, rollback, mock_checkpoint_manager):
        """@trace SPEC-07.40.01 - Plan MUST return None for missing checkpoint."""
        mock_checkpoint_manager.list_checkpoints.return_value = []

        plan = await rollback.plan_rollback("cp-nonexistent")

        assert plan is None

    @pytest.mark.asyncio
    async def test_rollback_success(self, rollback, mock_checkpoint_manager, temp_workspace):
        """@trace SPEC-07.40.01 - Rollback MUST restore workspace."""
        # Setup checkpoints
        target_cp = MagicMock(spec=Checkpoint)
        target_cp.id = "cp-target-001"
        target_cp.uncommitted_changes = ["restored.py"]
        target_cp.workspace_snapshot = str(Path(temp_workspace) / "snap.tar.gz")
        target_cp.workspace_root = None

        safety_cp = MagicMock(spec=Checkpoint)
        safety_cp.id = "cp-safety-001"

        # Create fake snapshot
        (Path(temp_workspace) / "snap.tar.gz").write_text("fake")

        mock_checkpoint_manager.list_checkpoints.return_value = [target_cp]
        mock_checkpoint_manager.create_checkpoint.return_value = safety_cp

        result = await rollback.rollback(
            "cp-target-001",
            skip_confirmation=True,
        )

        assert result.success is True
        assert result.target_checkpoint_id == "cp-target-001"
        assert result.safety_checkpoint_id == "cp-safety-001"
        assert result.phase_reached == RollbackPhase.COMPLETE
        mock_checkpoint_manager.restore_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_checkpoint_not_found(self, rollback, mock_checkpoint_manager):
        """@trace SPEC-07.40.01 - Rollback MUST fail for missing checkpoint."""
        mock_checkpoint_manager.list_checkpoints.return_value = []

        result = await rollback.rollback(
            "cp-nonexistent",
            skip_confirmation=True,
        )

        assert result.success is False
        assert "not found" in result.error_message.lower()
        assert result.phase_reached == RollbackPhase.FAILED

    @pytest.mark.asyncio
    async def test_rollback_user_cancelled(self, rollback, mock_checkpoint_manager, temp_workspace):
        """@trace SPEC-07.40.02 - Rollback MUST respect user cancellation."""
        target_cp = MagicMock(spec=Checkpoint)
        target_cp.id = "cp-target-001"
        target_cp.uncommitted_changes = []
        target_cp.workspace_snapshot = str(Path(temp_workspace) / "snap.tar.gz")
        target_cp.workspace_root = None

        safety_cp = MagicMock(spec=Checkpoint)
        safety_cp.id = "cp-safety-001"

        (Path(temp_workspace) / "snap.tar.gz").write_text("fake")

        mock_checkpoint_manager.list_checkpoints.return_value = [target_cp]
        mock_checkpoint_manager.create_checkpoint.return_value = safety_cp

        # User cancels
        result = await rollback.rollback(
            "cp-target-001",
            skip_confirmation=False,
            confirm_callback=lambda: False,
        )

        assert result.success is False
        assert "cancelled" in result.error_message.lower()
        assert result.recovery_performed is True

    @pytest.mark.asyncio
    async def test_rollback_recovery_on_failure(
        self, rollback, mock_checkpoint_manager, temp_workspace
    ):
        """@trace SPEC-07.40.04 - Rollback MUST attempt recovery on failure."""
        target_cp = MagicMock(spec=Checkpoint)
        target_cp.id = "cp-target-001"
        target_cp.uncommitted_changes = []
        target_cp.workspace_snapshot = str(Path(temp_workspace) / "snap.tar.gz")
        target_cp.workspace_root = None

        safety_cp = MagicMock(spec=Checkpoint)
        safety_cp.id = "cp-safety-001"

        (Path(temp_workspace) / "snap.tar.gz").write_text("fake")

        mock_checkpoint_manager.list_checkpoints.return_value = [target_cp]
        mock_checkpoint_manager.create_checkpoint.return_value = safety_cp
        # Restore fails on first call (target), succeeds on second (recovery)
        mock_checkpoint_manager.restore_checkpoint.side_effect = [
            Exception("Restore failed"),
            None,  # Recovery succeeds
        ]

        result = await rollback.rollback(
            "cp-target-001",
            skip_confirmation=True,
        )

        assert result.success is False
        assert result.phase_reached == RollbackPhase.FAILED
        assert result.recovery_performed is True
        # Verify restore was called twice (target + recovery)
        assert mock_checkpoint_manager.restore_checkpoint.call_count == 2

    def test_format_rollback_result_success(self, rollback):
        """@trace SPEC-07.40.01 - Format MUST show success details."""
        result = RollbackResult(
            success=True,
            target_checkpoint_id="cp-abc123",
            safety_checkpoint_id="cp-safe001",
            stash_ref="stash@{0}",
            files_restored=["src/auth.py"],
            phase_reached=RollbackPhase.COMPLETE,
        )

        output = rollback.format_rollback_result(result)

        assert "cp-abc123" in output
        assert "cp-safe001" in output
        assert "stash@{0}" in output
        assert "src/auth.py" in output
        assert "restored" in output.lower()

    def test_format_rollback_result_failure(self, rollback):
        """@trace SPEC-07.40.01 - Format MUST show failure details."""
        result = RollbackResult(
            success=False,
            target_checkpoint_id="cp-abc123",
            phase_reached=RollbackPhase.FAILED,
            error_message="Checkpoint corrupted",
            recovery_performed=True,
        )

        output = rollback.format_rollback_result(result)

        assert "failed" in output.lower()
        assert "Checkpoint corrupted" in output
        assert "recovery" in output.lower()

    def test_progress_callback(self, rollback):
        """@trace SPEC-07.40.01 - Rollback MUST report progress."""
        progress_calls: list[tuple[str, RollbackPhase]] = []

        def callback(msg: str, phase: RollbackPhase) -> None:
            progress_calls.append((msg, phase))

        rollback.set_progress_callback(callback)
        rollback._report_progress("Test message", RollbackPhase.INIT)

        assert len(progress_calls) == 1
        assert progress_calls[0][0] == "Test message"
        assert progress_calls[0][1] == RollbackPhase.INIT
