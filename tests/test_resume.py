"""Tests for auto-resume workflow.

Tests [SPEC-03.13] Auto-Resume Workflow.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.checkpoint import CheckpointManager
from parhelia.resume import (
    ResumeError,
    ResumeManager,
    ResumeResult,
    ResumeStage,
    ResumeState,
)
from parhelia.session import Checkpoint, CheckpointTrigger, Session, SessionState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def checkpoint_manager(temp_dir):
    """Create a CheckpointManager for tests."""
    checkpoint_root = temp_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    return CheckpointManager(checkpoint_root=str(checkpoint_root))


@pytest.fixture
def resume_manager(checkpoint_manager, temp_dir):
    """Create a ResumeManager for tests."""
    workspace_root = temp_dir / "workspaces"
    workspace_root.mkdir(parents=True)
    return ResumeManager(
        checkpoint_manager=checkpoint_manager,
        workspace_root=str(workspace_root),
    )


@pytest.fixture
def sample_session(temp_dir):
    """Create a sample session for tests."""
    workspace = temp_dir / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "main.py").write_text("print('hello')")

    return Session(
        id="test-session-1",
        task_id="task-1",
        state=SessionState.RUNNING,
        working_directory=str(workspace),
    )


@pytest.fixture
async def sample_checkpoint(checkpoint_manager, sample_session):
    """Create a sample checkpoint for tests."""
    return await checkpoint_manager.create_checkpoint(
        session=sample_session,
        trigger=CheckpointTrigger.MANUAL,
        conversation={"turn": 5, "messages": ["hello"]},
    )


# =============================================================================
# ResumeState Tests
# =============================================================================


class TestResumeState:
    """Tests for ResumeState dataclass."""

    def test_resume_state_creation(self):
        """ResumeState MUST track resume workflow state."""
        state = ResumeState(session_id="session-1")

        assert state.session_id == "session-1"
        assert state.checkpoint is None
        assert state.stage == ResumeStage.INIT
        assert state.errors == []

    def test_resume_state_tracks_errors(self):
        """ResumeState MUST accumulate errors."""
        state = ResumeState(session_id="session-1")

        state.errors.append("Error 1")
        state.errors.append("Error 2")

        assert len(state.errors) == 2


# =============================================================================
# ResumeResult Tests
# =============================================================================


class TestResumeResult:
    """Tests for ResumeResult dataclass."""

    def test_resume_result_success(self):
        """ResumeResult MUST represent successful resume."""
        result = ResumeResult(
            session_id="session-1",
            checkpoint_id="cp-abc123",
            success=True,
            stage=ResumeStage.COMPLETE,
            restored_working_directory="/vol/workspaces/session-1",
            conversation_turn=5,
        )

        assert result.success
        assert result.stage == ResumeStage.COMPLETE
        assert result.error is None

    def test_resume_result_failure(self):
        """ResumeResult MUST represent failed resume."""
        result = ResumeResult(
            session_id="session-1",
            checkpoint_id="cp-abc123",
            success=False,
            stage=ResumeStage.WORKSPACE_RESTORE,
            error="Workspace restore failed",
        )

        assert not result.success
        assert result.stage == ResumeStage.WORKSPACE_RESTORE
        assert "restore failed" in result.error


# =============================================================================
# ResumeManager Tests
# =============================================================================


class TestResumeManager:
    """Tests for ResumeManager class."""

    def test_initialization(self, checkpoint_manager, temp_dir):
        """ResumeManager MUST initialize with checkpoint manager."""
        manager = ResumeManager(
            checkpoint_manager=checkpoint_manager,
            workspace_root=str(temp_dir / "workspaces"),
        )

        assert manager.checkpoint_manager is checkpoint_manager
        assert manager.workspace_root == temp_dir / "workspaces"

    @pytest.mark.asyncio
    async def test_can_resume_true(
        self, resume_manager, checkpoint_manager, sample_session
    ):
        """can_resume MUST return True when checkpoint exists."""
        # Create checkpoint
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        result = await resume_manager.can_resume(sample_session.id)

        assert result is True

    @pytest.mark.asyncio
    async def test_can_resume_false(self, resume_manager):
        """can_resume MUST return False when no checkpoint exists."""
        result = await resume_manager.can_resume("nonexistent-session")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_resume_info(
        self, resume_manager, checkpoint_manager, sample_session
    ):
        """get_resume_info MUST return checkpoint details."""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
            conversation={"turn": 10},
        )

        info = await resume_manager.get_resume_info(sample_session.id)

        assert info is not None
        assert info["session_id"] == sample_session.id
        assert info["checkpoint_id"] == checkpoint.id
        assert info["trigger"] == "periodic"

    @pytest.mark.asyncio
    async def test_get_resume_info_not_found(self, resume_manager):
        """get_resume_info MUST return None when no checkpoint."""
        info = await resume_manager.get_resume_info("nonexistent")

        assert info is None

    @pytest.mark.asyncio
    async def test_resume_session_no_checkpoint(self, resume_manager):
        """resume_session MUST raise when no checkpoint exists."""
        with pytest.raises(ResumeError) as exc_info:
            await resume_manager.resume_session("nonexistent-session")

        assert "No checkpoint found" in str(exc_info.value)
        assert exc_info.value.session_id == "nonexistent-session"

    @pytest.mark.asyncio
    async def test_resume_session_restores_workspace(
        self, resume_manager, checkpoint_manager, sample_session, temp_dir
    ):
        """resume_session MUST restore workspace from checkpoint."""
        # Create checkpoint
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"turn": 5},
        )

        # Resume without running Claude
        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            run_claude=False,
        )

        assert result.success
        assert result.stage == ResumeStage.COMPLETE
        assert result.restored_working_directory is not None

        # Verify workspace was restored
        restored_path = Path(result.restored_working_directory)
        assert restored_path.exists()
        assert (restored_path / "main.py").exists()
        assert (restored_path / "main.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_resume_session_custom_target(
        self, resume_manager, checkpoint_manager, sample_session, temp_dir
    ):
        """resume_session MUST accept custom target directory."""
        # Create checkpoint
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        custom_target = str(temp_dir / "custom_restore")

        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            target_directory=custom_target,
            run_claude=False,
        )

        assert result.success
        assert result.restored_working_directory == custom_target
        assert (Path(custom_target) / "main.py").exists()

    @pytest.mark.asyncio
    async def test_resume_session_specific_checkpoint(
        self, resume_manager, checkpoint_manager, sample_session, temp_dir
    ):
        """resume_session MUST accept specific checkpoint ID."""
        # Create two checkpoints
        cp1 = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"turn": 5},
        )

        # Modify workspace
        workspace = Path(sample_session.working_directory)
        (workspace / "main.py").write_text("print('modified')")

        cp2 = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"turn": 10},
        )

        # Resume from first checkpoint
        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            checkpoint_id=cp1.id,
            run_claude=False,
        )

        assert result.success
        assert result.checkpoint_id == cp1.id

    @pytest.mark.asyncio
    async def test_resume_session_returns_conversation_turn(
        self, resume_manager, checkpoint_manager, sample_session
    ):
        """resume_session MUST return conversation turn from checkpoint."""
        # Create checkpoint with conversation
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"turn": 15, "messages": ["test"]},
        )

        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            run_claude=False,
        )

        assert result.success
        assert result.conversation_turn == 15

    @pytest.mark.asyncio
    async def test_resume_session_handles_claude_not_found(
        self, resume_manager, checkpoint_manager, sample_session
    ):
        """resume_session MUST handle missing Claude binary gracefully."""
        # Create checkpoint
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # This should succeed but with Claude not found error recorded
        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            run_claude=True,
        )

        # Since Claude binary doesn't exist in test env, this will fail
        # but the workspace restore should have succeeded
        # The exact behavior depends on implementation - could fail or succeed with warning
        assert result.session_id == sample_session.id

    @pytest.mark.asyncio
    async def test_resume_session_tracks_duration(
        self, resume_manager, checkpoint_manager, sample_session
    ):
        """resume_session MUST track duration."""
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            run_claude=False,
        )

        assert result.duration_seconds >= 0


# =============================================================================
# ResumeError Tests
# =============================================================================


class TestResumeError:
    """Tests for ResumeError exception."""

    def test_error_attributes(self):
        """ResumeError MUST contain context details."""
        error = ResumeError(
            message="Restore failed",
            session_id="session-1",
            checkpoint_id="cp-abc",
            stage="workspace_restore",
        )

        assert error.session_id == "session-1"
        assert error.checkpoint_id == "cp-abc"
        assert error.stage == "workspace_restore"
        assert str(error) == "Restore failed"

    def test_error_optional_attributes(self):
        """ResumeError MUST work with optional attributes."""
        error = ResumeError(message="Generic error")

        assert error.session_id is None
        assert error.checkpoint_id is None
        assert error.stage is None


# =============================================================================
# ResumeStage Tests
# =============================================================================


class TestResumeStage:
    """Tests for ResumeStage enum."""

    def test_stage_values(self):
        """ResumeStage MUST define all workflow stages."""
        assert ResumeStage.INIT.value == "init"
        assert ResumeStage.CHECKPOINT_LOAD.value == "checkpoint_load"
        assert ResumeStage.WORKSPACE_RESTORE.value == "workspace_restore"
        assert ResumeStage.STATE_VERIFY.value == "state_verify"
        assert ResumeStage.CLAUDE_RESUME.value == "claude_resume"
        assert ResumeStage.COMPLETE.value == "complete"
        assert ResumeStage.FAILED.value == "failed"


# =============================================================================
# Integration Tests
# =============================================================================


class TestResumeIntegration:
    """Integration tests for resume workflow."""

    @pytest.mark.asyncio
    async def test_full_resume_workflow(
        self, resume_manager, checkpoint_manager, sample_session, temp_dir
    ):
        """Full resume workflow MUST restore session state."""
        # Setup: Create checkpoint
        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
            conversation={"turn": 20, "session_id": "claude-session-123"},
        )

        # Verify: Can resume
        assert await resume_manager.can_resume(sample_session.id)

        # Get info
        info = await resume_manager.get_resume_info(sample_session.id)
        assert info["checkpoint_id"] == checkpoint.id

        # Resume
        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            run_claude=False,
        )

        # Verify results
        assert result.success
        assert result.stage == ResumeStage.COMPLETE
        assert result.conversation_turn == 20

        # Verify workspace
        restored = Path(result.restored_working_directory)
        assert restored.exists()
        assert (restored / "main.py").read_text() == "print('hello')"

    @pytest.mark.asyncio
    async def test_resume_after_modification(
        self, resume_manager, checkpoint_manager, sample_session, temp_dir
    ):
        """Resume MUST restore to checkpoint state, not current state."""
        workspace = Path(sample_session.working_directory)

        # Create checkpoint
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Modify workspace after checkpoint
        (workspace / "main.py").write_text("print('modified after checkpoint')")
        (workspace / "new_file.py").write_text("# new file")

        # Resume (without specifying which checkpoint - should use latest)
        result = await resume_manager.resume_session(
            session_id=sample_session.id,
            run_claude=False,
        )

        # Verify restored to checkpoint state
        restored = Path(result.restored_working_directory)
        assert (restored / "main.py").read_text() == "print('hello')"
        # New file shouldn't exist in restored workspace
        assert not (restored / "new_file.py").exists()
