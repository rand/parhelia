"""Coordinated workspace rollback.

Implements:
- [SPEC-07.40.01] Rollback Command
- [SPEC-07.40.02] Safety Guarantees
- [SPEC-07.40.03] Rollback Scope
- [SPEC-07.40.04] Rollback Recovery
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from parhelia.checkpoint import Checkpoint, CheckpointManager, CheckpointTrigger


class RollbackPhase(Enum):
    """Phases of the rollback process."""

    INIT = "init"
    SAFETY_CHECKPOINT = "safety_checkpoint"
    STASH_CHANGES = "stash_changes"
    VERIFY_TARGET = "verify_target"
    RESTORE_FILES = "restore_files"
    COMPLETE = "complete"
    FAILED = "failed"
    RECOVERED = "recovered"


@dataclass
class GitStashResult:
    """Result of git stash operation."""

    stashed: bool
    stash_ref: str | None = None
    message: str = ""


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    success: bool
    target_checkpoint_id: str
    safety_checkpoint_id: str | None = None
    stash_ref: str | None = None
    files_restored: list[str] = field(default_factory=list)
    phase_reached: RollbackPhase = RollbackPhase.INIT
    error_message: str | None = None
    recovery_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "target_checkpoint_id": self.target_checkpoint_id,
            "safety_checkpoint_id": self.safety_checkpoint_id,
            "stash_ref": self.stash_ref,
            "files_restored": self.files_restored,
            "phase_reached": self.phase_reached.value,
            "error_message": self.error_message,
            "recovery_performed": self.recovery_performed,
        }


@dataclass
class RollbackPlan:
    """Plan for a rollback operation before execution."""

    target_checkpoint: Checkpoint
    files_to_restore: list[str]
    has_uncommitted_changes: bool
    requires_confirmation: bool = True


class WorkspaceRollback:
    """Coordinate safe workspace rollback to checkpoint state.

    Implements [SPEC-07.40].
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        workspace_dir: str | Path,
        session_id: str,
    ):
        """Initialize the rollback coordinator.

        Args:
            checkpoint_manager: Manager for checkpoint operations.
            workspace_dir: Current workspace directory.
            session_id: Current session ID.
        """
        self.checkpoint_manager = checkpoint_manager
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self._progress_callback: Callable[[str, RollbackPhase], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[str, RollbackPhase], None],
    ) -> None:
        """Set callback for progress updates.

        Args:
            callback: Function called with (message, phase) during rollback.
        """
        self._progress_callback = callback

    def _report_progress(self, message: str, phase: RollbackPhase) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(message, phase)

    # =========================================================================
    # Safety Operations [SPEC-07.40.02]
    # =========================================================================

    async def check_uncommitted_changes(self) -> tuple[bool, list[str]]:
        """Check for uncommitted changes in workspace.

        Returns:
            Tuple of (has_changes, list of changed files).
        """
        proc = await asyncio.create_subprocess_exec(
            "git",
            "status",
            "--porcelain",
            cwd=self.workspace_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            # Not a git repo or git error
            return False, []

        output = stdout.decode().strip()
        if not output:
            return False, []

        changed_files = [line[3:] for line in output.split("\n") if line]
        return True, changed_files

    async def stash_changes(self, message: str | None = None) -> GitStashResult:
        """Stash uncommitted changes.

        Implements [SPEC-07.40.02] - stash any uncommitted changes.

        Args:
            message: Optional stash message.

        Returns:
            GitStashResult with stash details.
        """
        has_changes, _ = await self.check_uncommitted_changes()
        if not has_changes:
            return GitStashResult(stashed=False, message="No changes to stash")

        stash_msg = message or f"parhelia-rollback-{datetime.now().isoformat()}"
        # Use -u to include untracked files
        cmd = ["git", "stash", "push", "-u", "-m", stash_msg]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return GitStashResult(
                stashed=False,
                message=f"Stash failed: {stderr.decode()}",
            )

        # Get the stash ref
        proc = await asyncio.create_subprocess_exec(
            "git",
            "stash",
            "list",
            "-1",
            cwd=self.workspace_dir,
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        stash_ref = stdout.decode().split(":")[0] if stdout else "stash@{0}"

        return GitStashResult(
            stashed=True,
            stash_ref=stash_ref,
            message=f"Changes stashed as {stash_ref}",
        )

    async def pop_stash(self, stash_ref: str | None = None) -> bool:
        """Pop stashed changes back to workspace.

        Implements [SPEC-07.40.04] - pop stashed changes on recovery.

        Args:
            stash_ref: Specific stash to pop, or latest if None.

        Returns:
            True if successful.
        """
        cmd = ["git", "stash", "pop"]
        if stash_ref:
            cmd.append(stash_ref)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def create_safety_checkpoint(self) -> Checkpoint:
        """Create a safety checkpoint before rollback.

        Implements [SPEC-07.40.02] - create safety checkpoint of current state.

        Returns:
            The created safety checkpoint.
        """
        checkpoint = await self.checkpoint_manager.create_checkpoint(
            session_id=self.session_id,
            trigger=CheckpointTrigger.MANUAL,
            working_directory=str(self.workspace_dir),
            reason="Pre-rollback safety checkpoint",
        )
        return checkpoint

    async def verify_checkpoint_readable(
        self,
        checkpoint: Checkpoint,
    ) -> tuple[bool, str | None]:
        """Verify that a checkpoint can be read and restored.

        Implements [SPEC-07.40.02] - verify target checkpoint is readable.

        Args:
            checkpoint: Checkpoint to verify.

        Returns:
            Tuple of (is_readable, error_message).
        """
        # Check if workspace snapshot exists
        if checkpoint.workspace_snapshot:
            snapshot_path = Path(checkpoint.workspace_snapshot)
            if not snapshot_path.exists():
                return False, f"Workspace snapshot not found: {snapshot_path}"

        # Check if CAS workspace root exists
        if checkpoint.workspace_root:
            # Verify the CAS store has the required data
            if not self.checkpoint_manager._cas_checkpoint_mgr:
                return False, "CAS checkpoint manager not available"

        if not checkpoint.workspace_snapshot and not checkpoint.workspace_root:
            return False, "Checkpoint has no workspace data"

        return True, None

    # =========================================================================
    # Rollback Operations [SPEC-07.40.01, SPEC-07.40.03]
    # =========================================================================

    async def plan_rollback(
        self,
        target_checkpoint_id: str,
    ) -> RollbackPlan | None:
        """Plan a rollback operation without executing.

        Args:
            target_checkpoint_id: ID of checkpoint to rollback to.

        Returns:
            RollbackPlan, or None if checkpoint not found.
        """
        # Find the target checkpoint
        checkpoints = await self.checkpoint_manager.list_checkpoints(self.session_id)
        target = None
        for cp in checkpoints:
            if cp.id == target_checkpoint_id:
                target = cp
                break

        if not target:
            return None

        # Check for uncommitted changes
        has_changes, changed_files = await self.check_uncommitted_changes()

        # Determine files to restore
        files_to_restore = target.uncommitted_changes or []

        return RollbackPlan(
            target_checkpoint=target,
            files_to_restore=files_to_restore,
            has_uncommitted_changes=has_changes,
            requires_confirmation=True,
        )

    async def rollback(
        self,
        target_checkpoint_id: str,
        skip_confirmation: bool = False,
        confirm_callback: Callable[[], bool] | None = None,
    ) -> RollbackResult:
        """Execute rollback to a checkpoint.

        Implements [SPEC-07.40.01] and [SPEC-07.40.03].

        Args:
            target_checkpoint_id: ID of checkpoint to rollback to.
            skip_confirmation: Skip user confirmation (--yes flag).
            confirm_callback: Callback for user confirmation.

        Returns:
            RollbackResult with details of the operation.
        """
        result = RollbackResult(
            success=False,
            target_checkpoint_id=target_checkpoint_id,
            phase_reached=RollbackPhase.INIT,
        )

        safety_checkpoint: Checkpoint | None = None
        stash_result: GitStashResult | None = None

        try:
            # Phase 1: Plan and verify
            self._report_progress(
                f"Planning rollback to {target_checkpoint_id}...",
                RollbackPhase.INIT,
            )

            plan = await self.plan_rollback(target_checkpoint_id)
            if not plan:
                result.error_message = f"Checkpoint not found: {target_checkpoint_id}"
                result.phase_reached = RollbackPhase.FAILED
                return result

            # Verify target is readable
            readable, error = await self.verify_checkpoint_readable(
                plan.target_checkpoint
            )
            if not readable:
                result.error_message = error
                result.phase_reached = RollbackPhase.FAILED
                return result

            # Phase 2: Create safety checkpoint
            self._report_progress(
                "Creating safety checkpoint...",
                RollbackPhase.SAFETY_CHECKPOINT,
            )
            result.phase_reached = RollbackPhase.SAFETY_CHECKPOINT

            safety_checkpoint = await self.create_safety_checkpoint()
            result.safety_checkpoint_id = safety_checkpoint.id

            # Phase 3: Stash changes
            self._report_progress(
                "Stashing uncommitted changes...",
                RollbackPhase.STASH_CHANGES,
            )
            result.phase_reached = RollbackPhase.STASH_CHANGES

            if plan.has_uncommitted_changes:
                stash_result = await self.stash_changes(
                    f"Pre-rollback to {target_checkpoint_id}"
                )
                result.stash_ref = stash_result.stash_ref

            # Phase 4: Confirm with user
            if not skip_confirmation:
                self._report_progress(
                    "Waiting for confirmation...",
                    RollbackPhase.VERIFY_TARGET,
                )
                result.phase_reached = RollbackPhase.VERIFY_TARGET

                if confirm_callback and not confirm_callback():
                    # User cancelled - recover
                    await self._recover_from_rollback(
                        safety_checkpoint, stash_result, result
                    )
                    result.error_message = "Rollback cancelled by user"
                    result.phase_reached = RollbackPhase.RECOVERED
                    result.recovery_performed = True
                    return result

            # Phase 5: Restore files
            self._report_progress(
                "Restoring workspace files...",
                RollbackPhase.RESTORE_FILES,
            )
            result.phase_reached = RollbackPhase.RESTORE_FILES

            # Perform the actual restore
            await self.checkpoint_manager.restore_checkpoint(
                plan.target_checkpoint,
                str(self.workspace_dir),
            )

            result.files_restored = plan.files_to_restore

            # Phase 6: Complete
            self._report_progress(
                "Rollback complete",
                RollbackPhase.COMPLETE,
            )
            result.phase_reached = RollbackPhase.COMPLETE
            result.success = True

        except Exception as e:
            result.error_message = str(e)
            result.phase_reached = RollbackPhase.FAILED

            # Attempt recovery
            if safety_checkpoint or stash_result:
                self._report_progress(
                    "Rollback failed, attempting recovery...",
                    RollbackPhase.FAILED,
                )
                await self._recover_from_rollback(
                    safety_checkpoint, stash_result, result
                )

        return result

    async def _recover_from_rollback(
        self,
        safety_checkpoint: Checkpoint | None,
        stash_result: GitStashResult | None,
        result: RollbackResult,
    ) -> None:
        """Recover from a failed or cancelled rollback.

        Implements [SPEC-07.40.04].

        Args:
            safety_checkpoint: The safety checkpoint to restore from.
            stash_result: The stash to pop.
            result: Result object to update.
        """
        try:
            # Restore from safety checkpoint if we have one
            if safety_checkpoint:
                self._report_progress(
                    f"Restoring from safety checkpoint {safety_checkpoint.id}...",
                    RollbackPhase.RECOVERED,
                )
                await self.checkpoint_manager.restore_checkpoint(
                    safety_checkpoint,
                    str(self.workspace_dir),
                )

            # Pop stashed changes
            if stash_result and stash_result.stashed:
                self._report_progress(
                    f"Popping stash {stash_result.stash_ref}...",
                    RollbackPhase.RECOVERED,
                )
                await self.pop_stash(stash_result.stash_ref)

            result.recovery_performed = True

        except Exception as recovery_error:
            if result.error_message:
                result.error_message += f"; Recovery also failed: {recovery_error}"
            else:
                result.error_message = f"Recovery failed: {recovery_error}"

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def list_available_checkpoints(self) -> list[Checkpoint]:
        """List checkpoints available for rollback.

        Returns:
            List of checkpoints ordered by creation time.
        """
        return await self.checkpoint_manager.list_checkpoints(self.session_id)

    def format_rollback_result(self, result: RollbackResult) -> str:
        """Format rollback result for display.

        Args:
            result: The rollback result.

        Returns:
            Formatted string for CLI output.
        """
        lines: list[str] = []

        if result.success:
            lines.append(f"Rolling back to checkpoint {result.target_checkpoint_id}...")
            lines.append("")
            lines.append("Pre-rollback safety:")
            if result.safety_checkpoint_id:
                lines.append(f"  Creating safety checkpoint... {result.safety_checkpoint_id}")
            if result.stash_ref:
                lines.append(f"  Stashing current changes... {result.stash_ref}")
            lines.append("")
            lines.append("Restoring workspace:")
            if result.files_restored:
                lines.append(f"  Reverting {len(result.files_restored)} files to checkpoint state...")
                for f in result.files_restored[:10]:  # Show first 10
                    lines.append(f"  - {f} (restored)")
                if len(result.files_restored) > 10:
                    lines.append(f"  ... and {len(result.files_restored) - 10} more")
            else:
                lines.append("  Workspace restored")
            lines.append("")
            lines.append("Post-rollback:")
            lines.append(f"  Workspace restored to {result.target_checkpoint_id} state")
            if result.safety_checkpoint_id:
                lines.append(f"  Safety checkpoint: {result.safety_checkpoint_id}")
            if result.stash_ref:
                lines.append(f"  Stashed changes: {result.stash_ref}")
        else:
            lines.append(f"Rollback to {result.target_checkpoint_id} failed")
            lines.append("")
            if result.error_message:
                lines.append(f"Error: {result.error_message}")
            lines.append(f"Phase reached: {result.phase_reached.value}")
            if result.recovery_performed:
                lines.append("")
                lines.append("Recovery was performed - workspace should be in original state")

        return "\n".join(lines)
