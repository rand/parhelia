"""Tests for checkpoint management.

@trace SPEC-03.11 - Conversation State Capture
@trace SPEC-03.12 - Checkpoint Triggers
@trace SPEC-03.13 - Workspace Snapshot
@trace SPEC-07.11 - Checkpoint Metadata Schema v1.2
"""

import json
from datetime import datetime
from pathlib import Path

import pytest


class TestCheckpointManager:
    """Tests for CheckpointManager - SPEC-03.11, SPEC-03.12."""

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        return cp_dir

    @pytest.fixture
    def checkpoint_manager(self, checkpoint_dir: Path):
        """Create CheckpointManager instance."""
        from parhelia.checkpoint import CheckpointManager

        return CheckpointManager(checkpoint_root=str(checkpoint_dir))

    @pytest.fixture
    def sample_session(self):
        """Create a sample session for testing."""
        from parhelia.session import Session, SessionState

        return Session(
            id="test-session-123",
            task_id="ph-test",
            state=SessionState.RUNNING,
            working_directory="/tmp/workspace",
            environment={"FOO": "bar"},
        )

    def test_checkpoint_manager_initialization(self, checkpoint_manager):
        """@trace SPEC-03.11 - CheckpointManager MUST initialize with root path."""
        assert checkpoint_manager is not None
        assert checkpoint_manager.checkpoint_root is not None

    @pytest.mark.asyncio
    async def test_create_checkpoint_returns_checkpoint(
        self, checkpoint_manager, sample_session, tmp_path: Path
    ):
        """@trace SPEC-03.11 - create_checkpoint MUST return Checkpoint."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        # Create a workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "test.py").write_text("print('hello')")
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        assert isinstance(checkpoint, Checkpoint)
        assert checkpoint.session_id == sample_session.id
        assert checkpoint.trigger == CheckpointTrigger.MANUAL

    @pytest.mark.asyncio
    async def test_checkpoint_saves_manifest(
        self, checkpoint_manager, sample_session, checkpoint_dir: Path, tmp_path: Path
    ):
        """@trace SPEC-03.11 - Checkpoint MUST save manifest.json."""
        from parhelia.session import CheckpointTrigger

        # Create a workspace
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("content")
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
        )

        manifest_path = (
            checkpoint_dir / sample_session.id / checkpoint.id / "manifest.json"
        )
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["session_id"] == sample_session.id
        assert manifest["trigger"] == "periodic"

    @pytest.mark.asyncio
    async def test_checkpoint_saves_session_state(
        self, checkpoint_manager, sample_session, checkpoint_dir: Path, tmp_path: Path
    ):
        """@trace SPEC-03.11 - Checkpoint MUST save session_state.json."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        state_path = (
            checkpoint_dir / sample_session.id / checkpoint.id / "session_state.json"
        )
        assert state_path.exists()

        state = json.loads(state_path.read_text())
        assert state["id"] == sample_session.id
        assert state["task_id"] == sample_session.task_id

    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self, checkpoint_manager, sample_session, tmp_path: Path
    ):
        """@trace SPEC-03.12 - MUST be able to list checkpoints for session."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sample_session.working_directory = str(workspace)

        # Create multiple checkpoints
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
        )
        await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        checkpoints = await checkpoint_manager.list_checkpoints(sample_session.id)
        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(
        self, checkpoint_manager, sample_session, tmp_path: Path
    ):
        """@trace SPEC-03.12 - MUST be able to get latest checkpoint."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sample_session.working_directory = str(workspace)

        cp1 = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
        )

        cp2 = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        latest = await checkpoint_manager.get_latest_checkpoint(sample_session.id)
        assert latest is not None
        assert latest.id == cp2.id


class TestWorkspaceSnapshot:
    """Tests for workspace snapshotting - SPEC-03.13."""

    @pytest.fixture
    def checkpoint_manager(self, tmp_path: Path):
        """Create CheckpointManager instance."""
        from parhelia.checkpoint import CheckpointManager

        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        return CheckpointManager(checkpoint_root=str(cp_dir))

    @pytest.mark.asyncio
    async def test_snapshot_workspace_creates_archive(
        self, checkpoint_manager, tmp_path: Path
    ):
        """@trace SPEC-03.13 - snapshot_workspace MUST create compressed archive."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file1.txt").write_text("content1")
        (workspace / "file2.py").write_text("print('hello')")

        snapshot_path = await checkpoint_manager.snapshot_workspace(
            workspace_dir=str(workspace),
            session_id="test-session",
            checkpoint_id="cp-123",
        )

        assert Path(snapshot_path).exists()
        assert snapshot_path.endswith(".tar.zst") or snapshot_path.endswith(".tar.gz")

    @pytest.mark.asyncio
    async def test_snapshot_excludes_large_directories(
        self, checkpoint_manager, tmp_path: Path
    ):
        """@trace SPEC-03.13 - snapshot MUST exclude node_modules, .git, etc."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "main.py").write_text("code")

        # Create directories that should be excluded
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "big-package.js").write_text("x" * 10000)
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "cache.pyc").write_bytes(b"bytecode")

        snapshot_path = await checkpoint_manager.snapshot_workspace(
            workspace_dir=str(workspace),
            session_id="test-session",
            checkpoint_id="cp-456",
        )

        # Archive should exist and be small (excluding large dirs)
        size = Path(snapshot_path).stat().st_size
        assert size < 5000  # Should be much smaller without node_modules


class TestCASIntegratedCheckpoint:
    """Tests for CAS-integrated checkpoint system - SPEC-08.14."""

    @pytest.fixture
    def cas_and_checkpoint_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create temporary CAS and checkpoint directories."""
        cas_dir = tmp_path / "cas"
        cas_dir.mkdir()
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        return cas_dir, cp_dir

    @pytest.fixture
    def checkpoint_manager_with_cas(self, cas_and_checkpoint_dirs: tuple[Path, Path]):
        """Create CheckpointManager with CAS enabled."""
        from parhelia.checkpoint import CheckpointManager

        cas_dir, cp_dir = cas_and_checkpoint_dirs
        return CheckpointManager(
            checkpoint_root=str(cp_dir),
            cas_root=str(cas_dir),
        )

    @pytest.fixture
    def sample_session(self):
        """Create a sample session for testing."""
        from parhelia.session import Session, SessionState

        return Session(
            id="test-session-cas",
            task_id="ph-cas-test",
            state=SessionState.RUNNING,
            working_directory="/tmp/workspace",
            environment={"FOO": "bar"},
        )

    @pytest.mark.asyncio
    async def test_checkpoint_with_cas_stores_merkle_root(
        self, checkpoint_manager_with_cas, sample_session, tmp_path: Path
    ):
        """@trace SPEC-08.14 - Checkpoint MUST store workspace as Merkle tree root."""
        from parhelia.cas import Digest
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("content")
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Should have workspace_root digest instead of tar.gz path
        assert checkpoint.workspace_root is not None
        assert isinstance(checkpoint.workspace_root, Digest)

    @pytest.mark.asyncio
    async def test_checkpoint_with_cas_stores_blobs_in_cas(
        self, checkpoint_manager_with_cas, sample_session, cas_and_checkpoint_dirs, tmp_path: Path
    ):
        """@trace SPEC-08.14 - Checkpoint content MUST be stored in CAS."""
        from parhelia.cas import ContentAddressableStorage, Digest
        from parhelia.session import CheckpointTrigger

        cas_dir, _ = cas_and_checkpoint_dirs

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        content = "test content for CAS"
        (workspace / "file.txt").write_text(content)
        sample_session.working_directory = str(workspace)

        await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Verify content is in CAS
        cas = ContentAddressableStorage(root_path=str(cas_dir))
        file_digest = Digest.from_content(content.encode())
        assert await cas.contains(file_digest)

    @pytest.mark.asyncio
    async def test_checkpoint_with_cas_tracks_changed_files(
        self, checkpoint_manager_with_cas, sample_session, tmp_path: Path
    ):
        """@trace SPEC-08.14 - Checkpoint MUST track changed files from previous."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "original.txt").write_text("original")
        sample_session.working_directory = str(workspace)

        # First checkpoint
        cp1 = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
        )

        # Modify workspace
        (workspace / "original.txt").write_text("modified")
        (workspace / "new.txt").write_text("new file")

        # Second checkpoint with previous
        cp2 = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
            previous_checkpoint=cp1,
        )

        # Should track changed files
        assert "original.txt" in cp2.uncommitted_changes or len(cp2.uncommitted_changes) > 0

    @pytest.mark.asyncio
    async def test_restore_checkpoint_from_cas(
        self, checkpoint_manager_with_cas, sample_session, tmp_path: Path
    ):
        """@trace SPEC-08.14 - restore_checkpoint MUST restore from CAS."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("restore me")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.txt").write_text("nested content")
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Restore to new location
        restore_dir = tmp_path / "restored"
        restore_dir.mkdir()
        await checkpoint_manager_with_cas.restore_checkpoint(checkpoint, str(restore_dir))

        assert (restore_dir / "file.txt").read_text() == "restore me"
        assert (restore_dir / "subdir" / "nested.txt").read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_incremental_restore_from_cas(
        self, checkpoint_manager_with_cas, sample_session, tmp_path: Path
    ):
        """@trace SPEC-08.14 - incremental restore MUST only transfer changes."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "unchanged.txt").write_text("stays same")
        (workspace / "changed.txt").write_text("original")
        sample_session.working_directory = str(workspace)

        cp1 = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
        )

        # Modify workspace
        (workspace / "changed.txt").write_text("modified")
        (workspace / "added.txt").write_text("new file")

        cp2 = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.PERIODIC,
            previous_checkpoint=cp1,
        )

        # Restore incrementally
        restore_dir = tmp_path / "restored"
        restore_dir.mkdir()

        # First full restore
        await checkpoint_manager_with_cas.restore_checkpoint(cp1, str(restore_dir))
        assert (restore_dir / "changed.txt").read_text() == "original"

        # Incremental restore
        await checkpoint_manager_with_cas.restore_checkpoint(
            cp2, str(restore_dir), from_checkpoint=cp1
        )
        assert (restore_dir / "unchanged.txt").read_text() == "stays same"
        assert (restore_dir / "changed.txt").read_text() == "modified"
        assert (restore_dir / "added.txt").read_text() == "new file"

    @pytest.mark.asyncio
    async def test_checkpoint_manifest_includes_workspace_root(
        self, checkpoint_manager_with_cas, sample_session, cas_and_checkpoint_dirs, tmp_path: Path
    ):
        """@trace SPEC-08.14 - Manifest MUST include workspace_root digest."""
        from parhelia.session import CheckpointTrigger

        _, cp_dir = cas_and_checkpoint_dirs

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("content")
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        manifest_path = cp_dir / sample_session.id / checkpoint.id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert "workspace_root" in manifest
        assert manifest["workspace_root"]["hash"] is not None
        assert manifest["workspace_root"]["size_bytes"] is not None

    @pytest.mark.asyncio
    async def test_backward_compatible_no_tar_archive(
        self, checkpoint_manager_with_cas, sample_session, cas_and_checkpoint_dirs, tmp_path: Path
    ):
        """@trace SPEC-08.14 - CAS mode SHOULD NOT create tar.gz archives."""
        _, cp_dir = cas_and_checkpoint_dirs
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("content")
        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Should not create tar.gz when using CAS
        checkpoint_path = cp_dir / sample_session.id / checkpoint.id
        tar_files = list(checkpoint_path.glob("*.tar.gz")) + list(checkpoint_path.glob("*.tar.zst"))
        assert len(tar_files) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_excludes_large_directories_cas(
        self, checkpoint_manager_with_cas, sample_session, tmp_path: Path
    ):
        """@trace SPEC-08.14 - CAS MUST exclude node_modules, .git, etc."""
        from parhelia.session import CheckpointTrigger

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "main.py").write_text("code")

        # Create excluded directories
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "big-package.js").write_text("x" * 10000)
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "cache.pyc").write_bytes(b"bytecode")

        sample_session.working_directory = str(workspace)

        checkpoint = await checkpoint_manager_with_cas.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Restore and verify excluded dirs not restored
        restore_dir = tmp_path / "restored"
        restore_dir.mkdir()
        await checkpoint_manager_with_cas.restore_checkpoint(checkpoint, str(restore_dir))

        assert (restore_dir / "main.py").exists()
        assert not (restore_dir / "node_modules").exists()
        assert not (restore_dir / "__pycache__").exists()


class TestCheckpointMetadataSchemaV12:
    """Tests for checkpoint metadata schema v1.2 - SPEC-07.11."""

    def test_approval_status_enum(self):
        """@trace SPEC-07.11.03 - ApprovalStatus MUST have four states."""
        from parhelia.session import ApprovalStatus

        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.AUTO_APPROVED.value == "auto_approved"

    def test_checkpoint_approval_creation(self):
        """@trace SPEC-07.11.03 - CheckpointApproval MUST store approval data."""
        from parhelia.session import ApprovalStatus, CheckpointApproval

        approval = CheckpointApproval(
            status=ApprovalStatus.APPROVED,
            user="test-user",
            timestamp=datetime(2026, 1, 20, 12, 0, 0),
            reason="Looks good",
            policy="auto",
        )

        assert approval.status == ApprovalStatus.APPROVED
        assert approval.user == "test-user"
        assert approval.reason == "Looks good"
        assert approval.policy == "auto"

    def test_checkpoint_approval_serialization(self):
        """@trace SPEC-07.11.03 - CheckpointApproval MUST serialize/deserialize."""
        from parhelia.session import ApprovalStatus, CheckpointApproval

        original = CheckpointApproval(
            status=ApprovalStatus.REJECTED,
            user="reviewer",
            timestamp=datetime(2026, 1, 20, 14, 30, 0),
            reason="Needs more tests",
            policy="strict",
        )

        data = original.to_dict()
        restored = CheckpointApproval.from_dict(data)

        assert restored.status == original.status
        assert restored.user == original.user
        assert restored.timestamp == original.timestamp
        assert restored.reason == original.reason
        assert restored.policy == original.policy

    def test_checkpoint_annotation_creation(self):
        """@trace SPEC-07.11.05 - CheckpointAnnotation MUST store annotation data."""
        from parhelia.session import CheckpointAnnotation

        annotation = CheckpointAnnotation(
            timestamp=datetime(2026, 1, 20, 15, 0, 0),
            user="developer",
            text="This checkpoint includes the auth fix",
        )

        assert annotation.user == "developer"
        assert annotation.text == "This checkpoint includes the auth fix"

    def test_checkpoint_annotation_serialization(self):
        """@trace SPEC-07.11.05 - CheckpointAnnotation MUST serialize/deserialize."""
        from parhelia.session import CheckpointAnnotation

        original = CheckpointAnnotation(
            timestamp=datetime(2026, 1, 20, 15, 0, 0),
            user="developer",
            text="Important milestone reached",
        )

        data = original.to_dict()
        restored = CheckpointAnnotation.from_dict(data)

        assert restored.timestamp == original.timestamp
        assert restored.user == original.user
        assert restored.text == original.text

    def test_linked_issue_creation(self):
        """@trace SPEC-07.11.05 - LinkedIssue MUST store issue reference."""
        from parhelia.session import LinkedIssue

        issue = LinkedIssue(
            tracker="github",
            id="123",
            url="https://github.com/org/repo/issues/123",
        )

        assert issue.tracker == "github"
        assert issue.id == "123"
        assert issue.url == "https://github.com/org/repo/issues/123"

    def test_linked_issue_serialization(self):
        """@trace SPEC-07.11.05 - LinkedIssue MUST serialize/deserialize."""
        from parhelia.session import LinkedIssue

        original = LinkedIssue(
            tracker="beads",
            id="ph-abc",
            url=None,
        )

        data = original.to_dict()
        restored = LinkedIssue.from_dict(data)

        assert restored.tracker == original.tracker
        assert restored.id == original.id
        assert restored.url == original.url

    def test_checkpoint_provenance_fields(self):
        """@trace SPEC-07.11.02 - Checkpoint MUST have provenance fields."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        checkpoint = Checkpoint(
            id="cp-test123",
            session_id="session-456",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp/workspace",
            parent_checkpoint_id="cp-parent789",
            checkpoint_chain_depth=5,
        )

        assert checkpoint.parent_checkpoint_id == "cp-parent789"
        assert checkpoint.checkpoint_chain_depth == 5

    def test_checkpoint_provenance_defaults(self):
        """@trace SPEC-07.11.02 - Provenance fields MUST have sensible defaults."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        checkpoint = Checkpoint(
            id="cp-first",
            session_id="session-new",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp/workspace",
        )

        assert checkpoint.parent_checkpoint_id is None
        assert checkpoint.checkpoint_chain_depth == 1

    def test_checkpoint_approval_field(self):
        """@trace SPEC-07.11.03 - Checkpoint MUST support approval field."""
        from parhelia.session import (
            ApprovalStatus,
            Checkpoint,
            CheckpointApproval,
            CheckpointTrigger,
        )

        approval = CheckpointApproval(
            status=ApprovalStatus.AUTO_APPROVED,
            policy="auto",
        )

        checkpoint = Checkpoint(
            id="cp-approved",
            session_id="session-123",
            trigger=CheckpointTrigger.COMPLETE,
            working_directory="/tmp/workspace",
            approval=approval,
        )

        assert checkpoint.approval is not None
        assert checkpoint.approval.status == ApprovalStatus.AUTO_APPROVED

    def test_checkpoint_annotation_fields(self):
        """@trace SPEC-07.11.05 - Checkpoint MUST support annotation fields."""
        from parhelia.session import (
            Checkpoint,
            CheckpointAnnotation,
            CheckpointTrigger,
            LinkedIssue,
        )

        checkpoint = Checkpoint(
            id="cp-annotated",
            session_id="session-123",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp/workspace",
            tags=["milestone/v1.0", "stable"],
            annotations=[
                CheckpointAnnotation(
                    timestamp=datetime(2026, 1, 20, 12, 0, 0),
                    user="dev",
                    text="Release candidate",
                )
            ],
            linked_issues=[
                LinkedIssue(tracker="github", id="42", url="https://github.com/org/repo/issues/42")
            ],
        )

        assert checkpoint.tags == ["milestone/v1.0", "stable"]
        assert len(checkpoint.annotations) == 1
        assert checkpoint.annotations[0].text == "Release candidate"
        assert len(checkpoint.linked_issues) == 1
        assert checkpoint.linked_issues[0].id == "42"

    def test_checkpoint_annotation_defaults(self):
        """@trace SPEC-07.11.05 - Annotation fields MUST default to empty lists."""
        from parhelia.session import Checkpoint, CheckpointTrigger

        checkpoint = Checkpoint(
            id="cp-plain",
            session_id="session-123",
            trigger=CheckpointTrigger.PERIODIC,
            working_directory="/tmp/workspace",
        )

        assert checkpoint.tags == []
        assert checkpoint.annotations == []
        assert checkpoint.linked_issues == []
