"""Checkpoint management for session persistence.

Implements:
- [SPEC-03.11] Conversation State Capture
- [SPEC-03.12] Checkpoint Triggers
- [SPEC-03.13] Workspace Snapshot
- [SPEC-07.10] Environment Versioning
- [SPEC-07.11] Checkpoint Metadata Schema v1.2
- [SPEC-08.14] Incremental Workspace Checkpoint (CAS integration)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parhelia.session import Session

from parhelia.cas import (
    ContentAddressableStorage,
    Digest,
    IncrementalCheckpointManager as CASCheckpointManager,
    MerkleTreeBuilder,
    MerkleTreeDiff,
)
from parhelia.environment import EnvironmentCapture, EnvironmentSnapshot
from parhelia.session import (
    ApprovalStatus,
    Checkpoint,
    CheckpointAnnotation,
    CheckpointApproval,
    CheckpointTrigger,
    LinkedIssue,
)


class CheckpointManager:
    """Manage checkpoint creation and retrieval.

    Implements [SPEC-03.11], [SPEC-03.12], [SPEC-03.13], [SPEC-08.14].

    Supports two modes:
    - Legacy mode (cas_root=None): Uses tar.gz archives
    - CAS mode (cas_root set): Uses Content-Addressable Storage with Merkle trees
    """

    # Directories to exclude from workspace snapshots
    EXCLUDE_PATTERNS = [
        "node_modules",
        ".git/objects",
        "__pycache__",
        "target",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
    ]

    def __init__(
        self,
        checkpoint_root: str = "/vol/parhelia/checkpoints",
        cas_root: str | None = None,
        capture_environment: bool = True,
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_root: Root directory for checkpoint storage.
            cas_root: Root directory for CAS storage. If set, enables CAS mode.
            capture_environment: Whether to capture environment at checkpoint time.
        """
        self.checkpoint_root = Path(checkpoint_root)
        self.cas_root = cas_root
        self.capture_environment = capture_environment

        # Initialize CAS components if CAS mode enabled
        if cas_root:
            self._cas = ContentAddressableStorage(root_path=cas_root)
            self._tree_builder = MerkleTreeBuilder(self._cas)
            self._tree_diff = MerkleTreeDiff(self._cas)
            self._cas_checkpoint_mgr = CASCheckpointManager(self._cas)
        else:
            self._cas = None
            self._tree_builder = None
            self._tree_diff = None
            self._cas_checkpoint_mgr = None

        # Environment capture [SPEC-07.10]
        self._env_capture = EnvironmentCapture() if capture_environment else None

    async def create_checkpoint(
        self,
        session: Session,
        trigger: CheckpointTrigger,
        conversation: dict | None = None,
        previous_checkpoint: Checkpoint | None = None,
    ) -> Checkpoint:
        """Create a checkpoint of the session state.

        Implements [SPEC-03.11], [SPEC-03.12], [SPEC-08.14].

        Args:
            session: The session to checkpoint.
            trigger: What triggered this checkpoint.
            conversation: Optional conversation state to include.
            previous_checkpoint: Previous checkpoint for incremental CAS mode.

        Returns:
            The created Checkpoint.
        """
        # Generate checkpoint ID
        checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"

        # Create checkpoint directory
        checkpoint_dir = self.checkpoint_root / session.id / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create workspace snapshot - CAS or legacy mode
        workspace_snapshot = None
        workspace_root = None
        uncommitted_changes: list[str] = []

        if session.working_directory and Path(session.working_directory).exists():
            if self._cas and self._tree_builder:
                # CAS mode: Build Merkle tree [SPEC-08.14]
                workspace_root = await self._tree_builder.build_tree(
                    session.working_directory
                )

                # Track changed files if we have a previous checkpoint
                if previous_checkpoint and previous_checkpoint.workspace_root:
                    diff = await self._tree_diff.diff(
                        previous_checkpoint.workspace_root,
                        workspace_root,
                    )
                    uncommitted_changes = diff.added + diff.modified + diff.deleted
            else:
                # Legacy mode: Create tar.gz archive
                workspace_snapshot = await self.snapshot_workspace(
                    workspace_dir=session.working_directory,
                    session_id=session.id,
                    checkpoint_id=checkpoint_id,
                )

        # Capture environment [SPEC-07.10]
        environment_snapshot: EnvironmentSnapshot | None = None
        if self._env_capture:
            try:
                environment_snapshot = await self._env_capture.capture()
            except Exception:
                # Don't fail checkpoint if environment capture fails
                pass

        # Compute provenance [SPEC-07.11.02]
        parent_checkpoint_id: str | None = None
        checkpoint_chain_depth: int = 1
        if previous_checkpoint:
            parent_checkpoint_id = previous_checkpoint.id
            checkpoint_chain_depth = previous_checkpoint.checkpoint_chain_depth + 1

        # Create checkpoint object
        checkpoint = Checkpoint(
            id=checkpoint_id,
            session_id=session.id,
            trigger=trigger,
            working_directory=session.working_directory,
            environment=session.environment,
            workspace_snapshot=workspace_snapshot,
            workspace_root=workspace_root,
            uncommitted_changes=uncommitted_changes,
            environment_snapshot=environment_snapshot,
            parent_checkpoint_id=parent_checkpoint_id,
            checkpoint_chain_depth=checkpoint_chain_depth,
        )

        # Save manifest
        await self._save_manifest(checkpoint, checkpoint_dir)

        # Save session state
        await self._save_session_state(session, checkpoint_dir)

        # Save conversation if provided
        if conversation:
            await self._save_conversation(conversation, checkpoint_dir)

        return checkpoint

    async def snapshot_workspace(
        self,
        workspace_dir: str,
        session_id: str,
        checkpoint_id: str,
    ) -> str:
        """Create compressed snapshot of workspace.

        Implements [SPEC-03.13].

        Args:
            workspace_dir: Directory to snapshot.
            session_id: Session ID for path.
            checkpoint_id: Checkpoint ID for path.

        Returns:
            Path to the created snapshot archive.
        """
        checkpoint_dir = self.checkpoint_root / session_id / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Use tar.gz for compatibility (tar.zst requires zstd)
        snapshot_path = checkpoint_dir / "workspace.tar.gz"

        # Build exclude arguments
        exclude_args = []
        for pattern in self.EXCLUDE_PATTERNS:
            exclude_args.extend(["--exclude", pattern])

        # Create compressed archive
        cmd = [
            "tar",
            "-czf",
            str(snapshot_path),
            "-C",
            workspace_dir,
            ".",
            *exclude_args,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

        return str(snapshot_path)

    async def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """List all checkpoints for a session.

        Args:
            session_id: The session ID.

        Returns:
            List of checkpoints, sorted by creation time.
        """
        session_dir = self.checkpoint_root / session_id
        if not session_dir.exists():
            return []

        checkpoints = []
        for cp_dir in session_dir.iterdir():
            if cp_dir.is_dir():
                manifest_path = cp_dir / "manifest.json"
                if manifest_path.exists():
                    checkpoint = await self._load_checkpoint(manifest_path)
                    if checkpoint:
                        checkpoints.append(checkpoint)

        # Sort by creation time
        checkpoints.sort(key=lambda c: c.created_at)
        return checkpoints

    async def get_latest_checkpoint(self, session_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a session.

        Args:
            session_id: The session ID.

        Returns:
            The latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = await self.list_checkpoints(session_id)
        if not checkpoints:
            return None
        return checkpoints[-1]

    async def restore_checkpoint(
        self,
        checkpoint: Checkpoint,
        target_dir: str,
        from_checkpoint: Checkpoint | None = None,
    ) -> None:
        """Restore workspace from checkpoint.

        Implements [SPEC-08.14].

        Args:
            checkpoint: The checkpoint to restore.
            target_dir: Directory to restore to.
            from_checkpoint: Previous checkpoint for incremental restore (CAS mode).
        """
        from parhelia.cas import Checkpoint as CASCheckpoint

        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        if checkpoint.workspace_root and self._cas_checkpoint_mgr:
            # CAS mode: Use IncrementalCheckpointManager [SPEC-08.14]
            cas_cp = CASCheckpoint(
                workspace_root=checkpoint.workspace_root,
                changed_files=checkpoint.uncommitted_changes,
            )

            from_cas_cp = None
            if from_checkpoint and from_checkpoint.workspace_root:
                from_cas_cp = CASCheckpoint(
                    workspace_root=from_checkpoint.workspace_root,
                    changed_files=from_checkpoint.uncommitted_changes,
                )

            await self._cas_checkpoint_mgr.restore_checkpoint(
                cas_cp, target_dir, from_checkpoint=from_cas_cp
            )

        elif checkpoint.workspace_snapshot:
            # Legacy mode: Extract tar.gz archive
            await self._extract_workspace_archive(
                checkpoint.workspace_snapshot, target_dir
            )

    async def _extract_workspace_archive(
        self, archive_path: str, target_dir: str
    ) -> None:
        """Extract legacy tar.gz workspace archive."""
        cmd = ["tar", "-xzf", archive_path, "-C", target_dir]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()

    async def _save_manifest(self, checkpoint: Checkpoint, checkpoint_dir: Path) -> None:
        """Save checkpoint manifest."""
        manifest = {
            "version": "1.2",  # Bumped for environment versioning [SPEC-07.11]
            "id": checkpoint.id,
            "session_id": checkpoint.session_id,
            "created_at": checkpoint.created_at.isoformat(),
            "trigger": checkpoint.trigger.value,
            "working_directory": checkpoint.working_directory,
            "workspace_snapshot": checkpoint.workspace_snapshot,
            "tokens_used": checkpoint.tokens_used,
            "cost_estimate": checkpoint.cost_estimate,
            "uncommitted_changes": checkpoint.uncommitted_changes,
        }

        # Add workspace_root if using CAS [SPEC-08.14]
        if checkpoint.workspace_root:
            manifest["workspace_root"] = {
                "hash": checkpoint.workspace_root.hash,
                "size_bytes": checkpoint.workspace_root.size_bytes,
            }

        # Add environment snapshot [SPEC-07.10]
        if checkpoint.environment_snapshot:
            manifest["environment"] = checkpoint.environment_snapshot.to_dict()

        # Add provenance [SPEC-07.11.02]
        manifest["parent_checkpoint_id"] = checkpoint.parent_checkpoint_id
        manifest["checkpoint_chain_depth"] = checkpoint.checkpoint_chain_depth

        # Add approval [SPEC-07.11.03]
        if checkpoint.approval:
            manifest["approval"] = checkpoint.approval.to_dict()

        # Add annotations [SPEC-07.11.05]
        if checkpoint.tags:
            manifest["tags"] = checkpoint.tags
        if checkpoint.annotations:
            manifest["annotations"] = [a.to_dict() for a in checkpoint.annotations]
        if checkpoint.linked_issues:
            manifest["linked_issues"] = [i.to_dict() for i in checkpoint.linked_issues]

        manifest_path = checkpoint_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    async def _save_session_state(self, session: Session, checkpoint_dir: Path) -> None:
        """Save session state."""
        state = {
            "id": session.id,
            "task_id": session.task_id,
            "state": session.state.value,
            "working_directory": session.working_directory,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "environment": session.environment,
            "container_id": session.container_id,
            "gpu": session.gpu,
        }

        state_path = checkpoint_dir / "session_state.json"
        state_path.write_text(json.dumps(state, indent=2))

    async def _save_conversation(
        self, conversation: dict, checkpoint_dir: Path
    ) -> None:
        """Save conversation state."""
        conv_path = checkpoint_dir / "conversation.json"
        conv_path.write_text(json.dumps(conversation, indent=2))

    async def _load_checkpoint(self, manifest_path: Path) -> Checkpoint | None:
        """Load checkpoint from manifest file."""
        try:
            data = json.loads(manifest_path.read_text())

            # Load workspace_root if present [SPEC-08.14]
            workspace_root = None
            if data.get("workspace_root"):
                workspace_root = Digest(
                    hash=data["workspace_root"]["hash"],
                    size_bytes=data["workspace_root"]["size_bytes"],
                )

            # Load environment snapshot if present [SPEC-07.10]
            environment_snapshot = None
            if data.get("environment"):
                environment_snapshot = EnvironmentSnapshot.from_dict(data["environment"])

            # Load approval if present [SPEC-07.11.03]
            approval = None
            if data.get("approval"):
                approval = CheckpointApproval.from_dict(data["approval"])

            # Load annotations [SPEC-07.11.05]
            annotations = []
            for ann in data.get("annotations", []):
                annotations.append(CheckpointAnnotation.from_dict(ann))

            linked_issues = []
            for issue in data.get("linked_issues", []):
                linked_issues.append(LinkedIssue.from_dict(issue))

            return Checkpoint(
                id=data["id"],
                session_id=data["session_id"],
                trigger=CheckpointTrigger(data["trigger"]),
                working_directory=data.get("working_directory", ""),
                created_at=datetime.fromisoformat(data["created_at"]),
                workspace_snapshot=data.get("workspace_snapshot"),
                workspace_root=workspace_root,
                uncommitted_changes=data.get("uncommitted_changes", []),
                tokens_used=data.get("tokens_used", 0),
                cost_estimate=data.get("cost_estimate", 0.0),
                environment_snapshot=environment_snapshot,
                # Provenance [SPEC-07.11.02] - defaults for v1.0/v1.1 compatibility
                parent_checkpoint_id=data.get("parent_checkpoint_id"),
                checkpoint_chain_depth=data.get("checkpoint_chain_depth", 1),
                # Approval [SPEC-07.11.03]
                approval=approval,
                # Annotations [SPEC-07.11.05]
                tags=data.get("tags", []),
                annotations=annotations,
                linked_issues=linked_issues,
            )
        except Exception:
            return None
