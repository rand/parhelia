"""Tests for session state management.

@trace SPEC-03.10 - Checkpoint State Schema
@trace SPEC-03.11 - Conversation State Capture
"""

from datetime import datetime
from pathlib import Path

import pytest


class TestSessionState:
    """Tests for SessionState enum - SPEC-03.10."""

    def test_session_states_defined(self):
        """@trace SPEC-03.10 - SessionState MUST define required states."""
        from parhelia.session import SessionState

        # All required states must be defined
        assert SessionState.STARTING
        assert SessionState.RUNNING
        assert SessionState.SUSPENDED
        assert SessionState.COMPLETED
        assert SessionState.FAILED

    def test_session_state_values(self):
        """@trace SPEC-03.10 - SessionState values MUST be string literals."""
        from parhelia.session import SessionState

        assert SessionState.STARTING.value == "starting"
        assert SessionState.RUNNING.value == "running"
        assert SessionState.SUSPENDED.value == "suspended"
        assert SessionState.COMPLETED.value == "completed"
        assert SessionState.FAILED.value == "failed"


class TestCheckpointTrigger:
    """Tests for CheckpointTrigger enum - SPEC-03.10."""

    def test_checkpoint_triggers_defined(self):
        """@trace SPEC-03.10 - CheckpointTrigger MUST define required triggers."""
        from parhelia.session import CheckpointTrigger

        assert CheckpointTrigger.PERIODIC
        assert CheckpointTrigger.DETACH
        assert CheckpointTrigger.ERROR
        assert CheckpointTrigger.COMPLETE
        assert CheckpointTrigger.SHUTDOWN
        assert CheckpointTrigger.MANUAL


class TestSession:
    """Tests for Session dataclass - SPEC-03.10."""

    def test_session_creation(self):
        """@trace SPEC-03.10 - Session MUST capture required metadata."""
        from parhelia.session import Session, SessionState

        session = Session(
            id="test-session-123",
            task_id="ph-test-task",
            state=SessionState.STARTING,
            working_directory="/vol/parhelia/workspaces/test",
        )

        assert session.id == "test-session-123"
        assert session.task_id == "ph-test-task"
        assert session.state == SessionState.STARTING
        assert session.working_directory == "/vol/parhelia/workspaces/test"

    def test_session_has_timestamps(self):
        """@trace SPEC-03.10 - Session MUST have creation timestamp."""
        from parhelia.session import Session, SessionState

        session = Session(
            id="test-session",
            task_id="test-task",
            state=SessionState.STARTING,
            working_directory="/tmp",
        )

        assert session.created_at is not None
        assert isinstance(session.created_at, datetime)

    def test_session_tracks_activity(self):
        """@trace SPEC-03.10 - Session MUST track last activity."""
        from parhelia.session import Session, SessionState

        session = Session(
            id="test-session",
            task_id="test-task",
            state=SessionState.RUNNING,
            working_directory="/tmp",
        )

        old_activity = session.last_activity
        session.update_activity()

        assert session.last_activity >= old_activity


class TestCheckpoint:
    """Tests for Checkpoint dataclass - SPEC-03.10."""

    def test_checkpoint_creation(self):
        """@trace SPEC-03.10 - Checkpoint MUST capture session state."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        checkpoint = Checkpoint(
            id="cp-123",
            session_id="session-456",
            trigger=CheckpointTrigger.PERIODIC,
            working_directory="/vol/parhelia/workspaces/test",
        )

        assert checkpoint.id == "cp-123"
        assert checkpoint.session_id == "session-456"
        assert checkpoint.trigger == CheckpointTrigger.PERIODIC

    def test_checkpoint_has_timestamp(self):
        """@trace SPEC-03.10 - Checkpoint MUST have creation timestamp."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        checkpoint = Checkpoint(
            id="cp-123",
            session_id="session-456",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
        )

        assert checkpoint.created_at is not None
        assert isinstance(checkpoint.created_at, datetime)

    def test_checkpoint_tracks_metrics(self):
        """@trace SPEC-03.10 - Checkpoint MUST track token usage and cost."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        checkpoint = Checkpoint(
            id="cp-123",
            session_id="session-456",
            trigger=CheckpointTrigger.PERIODIC,
            working_directory="/tmp",
            tokens_used=45000,
            cost_estimate=0.45,
        )

        assert checkpoint.tokens_used == 45000
        assert checkpoint.cost_estimate == 0.45


class TestConversationState:
    """Tests for ConversationState - SPEC-03.11."""

    def test_conversation_state_creation(self):
        """@trace SPEC-03.11 - ConversationState MUST capture conversation."""
        from parhelia.session import ConversationState, Message

        messages = [
            Message(role="user", content="Fix the auth bug"),
            Message(role="assistant", content="I'll look at the auth code."),
        ]

        state = ConversationState(
            messages=messages,
            system_prompt="You are a helpful assistant.",
            session_id="claude-session-123",
        )

        assert len(state.messages) == 2
        assert state.system_prompt == "You are a helpful assistant."
        assert state.session_id == "claude-session-123"

    def test_conversation_state_tracks_resume_token(self):
        """@trace SPEC-03.11 - ConversationState MUST track resume token."""
        from parhelia.session import ConversationState

        state = ConversationState(
            messages=[],
            system_prompt="",
            session_id="session-123",
            resume_token="resume-abc123",
        )

        assert state.resume_token == "resume-abc123"


class TestGitState:
    """Tests for GitState - SPEC-03.10."""

    def test_git_state_captures_branch(self):
        """@trace SPEC-03.10 - GitState MUST capture current branch."""
        from parhelia.session import GitState

        git_state = GitState(
            branch="feature/fix-auth",
            head_commit="a1b2c3d4e5f6",
            is_dirty=True,
            has_stash=False,
        )

        assert git_state.branch == "feature/fix-auth"
        assert git_state.head_commit == "a1b2c3d4e5f6"
        assert git_state.is_dirty is True
        assert git_state.has_stash is False
