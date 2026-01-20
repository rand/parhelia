"""Tests for checkpoint comparison and diff.

@trace SPEC-07.41.01 - Diff Command
@trace SPEC-07.41.02 - File Diff
@trace SPEC-07.41.03 - Conversation Diff
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from parhelia.checkpoint import Checkpoint, CheckpointTrigger
from parhelia.checkpoint_diff import (
    ApprovalDiff,
    CheckpointComparison,
    CheckpointDiffer,
    ConversationDiff,
    ConversationStats,
    EnvironmentDiff,
    FileChange,
)
from parhelia.session import ApprovalStatus, CheckpointApproval


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_creation(self):
        """FileChange MUST capture path and change details."""
        fc = FileChange(
            path="src/auth.py",
            change_type="modified",
            additions=45,
            deletions=12,
        )

        assert fc.path == "src/auth.py"
        assert fc.change_type == "modified"
        assert fc.additions == 45
        assert fc.deletions == 12

    def test_to_dict(self):
        """FileChange MUST serialize to dict."""
        fc = FileChange(path="test.py", change_type="added", additions=100)
        data = fc.to_dict()

        assert data["path"] == "test.py"
        assert data["change_type"] == "added"


class TestConversationStats:
    """Tests for ConversationStats dataclass."""

    def test_creation(self):
        """ConversationStats MUST capture metrics."""
        stats = ConversationStats(
            turns=15,
            tokens=45000,
            cost_usd=0.45,
        )

        assert stats.turns == 15
        assert stats.tokens == 45000
        assert stats.cost_usd == 0.45

    def test_to_dict(self):
        """ConversationStats MUST serialize to dict."""
        stats = ConversationStats(turns=10, tokens=30000, cost_usd=0.30)
        data = stats.to_dict()

        assert data["turns"] == 10
        assert data["tokens"] == 30000


class TestConversationDiff:
    """Tests for ConversationDiff dataclass."""

    def test_creation(self):
        """ConversationDiff MUST capture differences."""
        diff = ConversationDiff(
            turns_added=5,
            tokens_added=15000,
            cost_added=0.15,
        )

        assert diff.turns_added == 5
        assert diff.tokens_added == 15000
        assert diff.cost_added == 0.15

    def test_to_dict(self):
        """ConversationDiff MUST serialize to dict."""
        diff = ConversationDiff(turns_added=3, tokens_added=5000, cost_added=0.05)
        data = diff.to_dict()

        assert data["turns_added"] == 3


class TestEnvironmentDiff:
    """Tests for EnvironmentDiff dataclass."""

    def test_no_changes(self):
        """EnvironmentDiff MUST handle no changes."""
        diff = EnvironmentDiff()

        assert diff.has_changes is False
        assert diff.claude_code_changed is False
        assert diff.plugins_changed == []

    def test_with_changes(self):
        """EnvironmentDiff MUST capture all change types."""
        diff = EnvironmentDiff(
            has_changes=True,
            claude_code_changed=True,
            plugins_changed=["plugin1", "plugin2"],
            mcp_servers_changed=["server1"],
            packages_changed=["pkg1", "pkg2", "pkg3"],
        )

        assert diff.has_changes is True
        assert diff.claude_code_changed is True
        assert len(diff.plugins_changed) == 2

    def test_to_dict(self):
        """EnvironmentDiff MUST serialize to dict."""
        diff = EnvironmentDiff(has_changes=True, plugins_changed=["p1"])
        data = diff.to_dict()

        assert data["has_changes"] is True
        assert data["plugins_changed"] == ["p1"]


class TestApprovalDiff:
    """Tests for ApprovalDiff dataclass."""

    def test_creation(self):
        """ApprovalDiff MUST capture status changes."""
        diff = ApprovalDiff(
            from_status="approved",
            from_user="rand",
            to_status="pending",
            to_user=None,
        )

        assert diff.from_status == "approved"
        assert diff.from_user == "rand"
        assert diff.to_status == "pending"

    def test_to_dict(self):
        """ApprovalDiff MUST serialize to dict."""
        diff = ApprovalDiff(from_status="pending", to_status="approved")
        data = diff.to_dict()

        assert data["from_status"] == "pending"
        assert data["to_status"] == "approved"


class TestCheckpointComparison:
    """Tests for CheckpointComparison dataclass."""

    def test_creation(self):
        """@trace SPEC-07.41.01 - Comparison MUST include all sections."""
        comparison = CheckpointComparison(
            from_checkpoint_id="cp-abc121",
            to_checkpoint_id="cp-abc123",
            from_time=datetime(2026, 1, 20, 14, 0, 0),
            to_time=datetime(2026, 1, 20, 14, 30, 0),
            time_delta=timedelta(minutes=30),
            conversation=ConversationDiff(),
            files_changed=[FileChange("file.py", "modified")],
            environment=EnvironmentDiff(),
            approval=ApprovalDiff(),
        )

        assert comparison.from_checkpoint_id == "cp-abc121"
        assert comparison.to_checkpoint_id == "cp-abc123"
        assert comparison.time_delta.total_seconds() == 1800

    def test_to_dict(self):
        """@trace SPEC-07.41.01 - Comparison MUST serialize to dict."""
        comparison = CheckpointComparison(
            from_checkpoint_id="cp-a",
            to_checkpoint_id="cp-b",
            from_time=datetime(2026, 1, 20, 14, 0, 0),
            to_time=datetime(2026, 1, 20, 15, 0, 0),
            time_delta=timedelta(hours=1),
            conversation=ConversationDiff(),
            files_changed=[],
            environment=EnvironmentDiff(),
            approval=ApprovalDiff(),
        )

        data = comparison.to_dict()

        assert data["from_checkpoint_id"] == "cp-a"
        assert data["time_delta_seconds"] == 3600


class TestCheckpointDiffer:
    """Tests for CheckpointDiffer."""

    @pytest.fixture
    def differ(self) -> CheckpointDiffer:
        """Create CheckpointDiffer instance."""
        return CheckpointDiffer()

    @pytest.fixture
    def from_checkpoint(self) -> Checkpoint:
        """Create source checkpoint."""
        cp = Checkpoint(
            id="cp-abc121",
            session_id="test-session",
            created_at=datetime(2026, 1, 20, 14, 0, 0),
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/workspace",
            tokens_used=30000,
            cost_estimate=0.30,
            uncommitted_changes=["src/auth.py", "tests/test_auth.py"],
        )
        cp.approval = CheckpointApproval(
            status=ApprovalStatus.APPROVED,
            user="rand",
            timestamp=datetime(2026, 1, 20, 14, 5, 0),
        )
        return cp

    @pytest.fixture
    def to_checkpoint(self) -> Checkpoint:
        """Create target checkpoint."""
        cp = Checkpoint(
            id="cp-abc123",
            session_id="test-session",
            created_at=datetime(2026, 1, 20, 14, 30, 0),
            trigger=CheckpointTrigger.PERIODIC,
            working_directory="/workspace",
            tokens_used=45000,
            cost_estimate=0.45,
            uncommitted_changes=[
                "src/auth.py",
                "tests/test_auth.py",
                ".github/workflows/ci.yml",
            ],
        )
        return cp

    def test_compare(self, differ, from_checkpoint, to_checkpoint):
        """@trace SPEC-07.41.01 - Compare MUST generate full comparison."""
        comparison = differ.compare(from_checkpoint, to_checkpoint)

        assert comparison.from_checkpoint_id == "cp-abc121"
        assert comparison.to_checkpoint_id == "cp-abc123"
        assert comparison.time_delta == timedelta(minutes=30)

    def test_compare_conversation(self, differ, from_checkpoint, to_checkpoint):
        """@trace SPEC-07.41.01 - Compare MUST include conversation diff."""
        comparison = differ.compare(from_checkpoint, to_checkpoint)

        assert comparison.conversation.tokens_added == 15000
        assert comparison.conversation.cost_added == pytest.approx(0.15)

    def test_compare_files(self, differ, from_checkpoint, to_checkpoint):
        """@trace SPEC-07.41.01 - Compare MUST include file changes."""
        comparison = differ.compare(from_checkpoint, to_checkpoint)

        # Should have 3 files: 2 modified, 1 added
        paths = [f.path for f in comparison.files_changed]
        assert ".github/workflows/ci.yml" in paths
        assert "src/auth.py" in paths

    def test_compare_approval(self, differ, from_checkpoint, to_checkpoint):
        """@trace SPEC-07.41.01 - Compare MUST include approval diff."""
        comparison = differ.compare(from_checkpoint, to_checkpoint)

        assert comparison.approval.from_status == "approved"
        assert comparison.approval.from_user == "rand"
        assert comparison.approval.to_status is None

    def test_diff_file(self, differ):
        """@trace SPEC-07.41.02 - File diff MUST generate unified diff."""
        from_content = "line1\nline2\nline3\n"
        to_content = "line1\nmodified\nline3\n"

        diff = differ.diff_file(from_content, to_content, "test.py")

        assert "---" in diff
        assert "+++" in diff
        assert "-line2" in diff
        assert "+modified" in diff

    def test_diff_file_no_changes(self, differ):
        """@trace SPEC-07.41.02 - File diff MUST handle no changes."""
        content = "same content\n"
        diff = differ.diff_file(content, content, "test.py")

        assert diff == ""

    def test_diff_file_stats(self, differ):
        """@trace SPEC-07.41.02 - File diff MUST calculate stats."""
        from_content = "line1\nline2\nline3"
        to_content = "line1\nmodified\nline3\nnew_line"

        additions, deletions = differ.diff_file_stats(from_content, to_content)

        assert additions == 2  # "modified" and "new_line"
        assert deletions == 1  # "line2"

    def test_diff_conversation(self, differ):
        """@trace SPEC-07.41.03 - Conversation diff MUST show added turns."""
        from_turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        to_turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure"},
        ]

        added = differ.diff_conversation(from_turns, to_turns)

        assert len(added) == 2
        assert added[0]["content"] == "Help me"
        assert added[1]["content"] == "Sure"

    def test_diff_conversation_no_changes(self, differ):
        """@trace SPEC-07.41.03 - Conversation diff MUST handle no changes."""
        turns = [{"role": "user", "content": "Hello"}]
        added = differ.diff_conversation(turns, turns)

        assert added == []

    def test_format_comparison(self, differ, from_checkpoint, to_checkpoint):
        """@trace SPEC-07.41.01 - Format MUST produce readable output."""
        comparison = differ.compare(from_checkpoint, to_checkpoint)
        output = differ.format_comparison(comparison)

        assert "Checkpoint Comparison:" in output
        assert "cp-abc121" in output
        assert "cp-abc123" in output
        assert "Time:" in output
        assert "30 minutes" in output
        assert "Conversation:" in output
        assert "Tokens:" in output
        assert "Files Changed:" in output
        assert "Approval:" in output
        assert "approved" in output

    def test_format_file_diff(self, differ):
        """@trace SPEC-07.41.02 - Format MUST include header."""
        diff_text = "--- a/test.py\n+++ b/test.py\n-old\n+new"
        output = differ.format_file_diff(diff_text, "cp-a", "cp-b", "test.py")

        assert "File Diff: test.py" in output
        assert "cp-a → cp-b" in output
        assert "--- a/test.py" in output

    def test_format_file_diff_no_changes(self, differ):
        """@trace SPEC-07.41.02 - Format MUST handle no changes."""
        output = differ.format_file_diff("", "cp-a", "cp-b", "test.py")

        assert "(no changes)" in output

    def test_format_conversation_diff(self, differ):
        """@trace SPEC-07.41.03 - Format MUST show added turns."""
        added_turns = [
            {"role": "user", "content": "Help me with auth"},
            {"role": "assistant", "content": "I'll implement JWT authentication."},
        ]

        output = differ.format_conversation_diff(added_turns, "cp-a", "cp-b")

        assert "Conversation Diff:" in output
        assert "Added 2 turns" in output
        assert "user:" in output
        assert "assistant:" in output
        assert "Help me with auth" in output

    def test_format_conversation_diff_truncates_long(self, differ):
        """@trace SPEC-07.41.03 - Format MUST truncate long content."""
        added_turns = [
            {"role": "assistant", "content": "x" * 500},
        ]

        output = differ.format_conversation_diff(added_turns, "cp-a", "cp-b")

        assert "..." in output
        assert len(output) < 600  # Should be truncated


class TestCheckpointDifferEnvironment:
    """Tests for environment comparison."""

    @pytest.fixture
    def differ(self) -> CheckpointDiffer:
        """Create CheckpointDiffer instance."""
        return CheckpointDiffer()

    def test_compare_no_environment(self, differ):
        """Environment diff MUST handle missing environment data."""
        from_cp = Checkpoint(
            id="cp-a",
            session_id="test",
            created_at=datetime.now(),
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/workspace",
        )
        to_cp = Checkpoint(
            id="cp-b",
            session_id="test",
            created_at=datetime.now(),
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/workspace",
        )

        comparison = differ.compare(from_cp, to_cp)

        assert comparison.environment.has_changes is False

    def test_format_environment_changes(self, differ):
        """Format MUST show environment changes."""
        comparison = CheckpointComparison(
            from_checkpoint_id="cp-a",
            to_checkpoint_id="cp-b",
            from_time=datetime.now(),
            to_time=datetime.now(),
            time_delta=timedelta(minutes=10),
            conversation=ConversationDiff(),
            files_changed=[],
            environment=EnvironmentDiff(
                has_changes=True,
                claude_code_changed=True,
                plugins_changed=["mcp-git", "mcp-fs"],
            ),
            approval=ApprovalDiff(),
        )

        output = differ.format_comparison(comparison)

        assert "Claude Code version changed" in output
        assert "mcp-git" in output
        assert "mcp-fs" in output
