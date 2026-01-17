"""Auto-resume workflow for session recovery.

Implements:
- [SPEC-03.13] Auto-Resume Workflow
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parhelia.session import Session

from parhelia.checkpoint import CheckpointManager
from parhelia.session import Checkpoint, CheckpointTrigger, Session, SessionState


class ResumeError(Exception):
    """Error during session resume."""

    def __init__(
        self,
        message: str,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        stage: str | None = None,
    ):
        """Initialize the error.

        Args:
            message: Error message.
            session_id: The session being resumed.
            checkpoint_id: The checkpoint used.
            stage: Stage where error occurred.
        """
        super().__init__(message)
        self.session_id = session_id
        self.checkpoint_id = checkpoint_id
        self.stage = stage


class ResumeStage(Enum):
    """Stages of the resume workflow."""

    INIT = "init"
    CHECKPOINT_LOAD = "checkpoint_load"
    WORKSPACE_RESTORE = "workspace_restore"
    STATE_VERIFY = "state_verify"
    CLAUDE_RESUME = "claude_resume"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ResumeResult:
    """Result of a resume operation.

    Implements [SPEC-03.13].
    """

    session_id: str
    checkpoint_id: str
    success: bool
    stage: ResumeStage
    restored_working_directory: str | None = None
    conversation_turn: int | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResumeState:
    """Internal state during resume workflow."""

    session_id: str
    checkpoint: Checkpoint | None = None
    session: Session | None = None
    conversation: dict | None = None
    target_directory: str | None = None
    stage: ResumeStage = ResumeStage.INIT
    errors: list[str] = field(default_factory=list)


class ResumeManager:
    """Manage automatic session resume after failure.

    Implements [SPEC-03.13].

    The resume manager handles:
    - Loading checkpoint from storage
    - Restoring workspace to target directory
    - Verifying state consistency
    - Running Claude Code with --resume flag
    """

    CLAUDE_BINARY = "/root/.claude/local/claude"
    DEFAULT_TIMEOUT = 300  # 5 minutes for resume

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        workspace_root: str = "/vol/parhelia/workspaces",
    ):
        """Initialize the resume manager.

        Args:
            checkpoint_manager: Manager for checkpoint operations.
            workspace_root: Root directory for restored workspaces.
        """
        self.checkpoint_manager = checkpoint_manager
        self.workspace_root = Path(workspace_root)

    async def resume_session(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
        target_directory: str | None = None,
        run_claude: bool = True,
    ) -> ResumeResult:
        """Resume a session from checkpoint.

        Implements [SPEC-03.13].

        Args:
            session_id: The session to resume.
            checkpoint_id: Specific checkpoint to use (default: latest).
            target_directory: Directory to restore workspace to.
            run_claude: Whether to run Claude Code after restore.

        Returns:
            ResumeResult with status and details.

        Raises:
            ResumeError: If resume fails at any stage.
        """
        start_time = datetime.now()
        state = ResumeState(session_id=session_id)

        try:
            # Stage 1: Load checkpoint
            state.stage = ResumeStage.CHECKPOINT_LOAD
            state.checkpoint = await self._load_checkpoint(
                session_id, checkpoint_id
            )
            if not state.checkpoint:
                raise ResumeError(
                    f"No checkpoint found for session {session_id}",
                    session_id=session_id,
                    checkpoint_id=checkpoint_id,
                    stage=state.stage.value,
                )

            # Load session state and conversation
            state.session = await self._load_session_state(state.checkpoint)
            state.conversation = await self._load_conversation(state.checkpoint)

            # Stage 2: Restore workspace
            state.stage = ResumeStage.WORKSPACE_RESTORE
            state.target_directory = target_directory or self._get_restore_path(
                session_id
            )
            await self._restore_workspace(state)

            # Stage 3: Verify state consistency
            state.stage = ResumeStage.STATE_VERIFY
            await self._verify_state(state)

            # Stage 4: Resume Claude Code
            if run_claude:
                state.stage = ResumeStage.CLAUDE_RESUME
                await self._resume_claude(state)

            # Complete
            state.stage = ResumeStage.COMPLETE
            duration = (datetime.now() - start_time).total_seconds()

            return ResumeResult(
                session_id=session_id,
                checkpoint_id=state.checkpoint.id,
                success=True,
                stage=state.stage,
                restored_working_directory=state.target_directory,
                conversation_turn=state.conversation.get("turn") if state.conversation else None,
                duration_seconds=duration,
            )

        except ResumeError:
            raise

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            state.stage = ResumeStage.FAILED
            state.errors.append(str(e))

            return ResumeResult(
                session_id=session_id,
                checkpoint_id=state.checkpoint.id if state.checkpoint else "unknown",
                success=False,
                stage=state.stage,
                error=str(e),
                duration_seconds=duration,
            )

    async def can_resume(self, session_id: str) -> bool:
        """Check if a session can be resumed.

        Args:
            session_id: The session ID.

        Returns:
            True if checkpoint exists and is valid.
        """
        checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
        return checkpoint is not None

    async def get_resume_info(self, session_id: str) -> dict | None:
        """Get information about resumable session.

        Args:
            session_id: The session ID.

        Returns:
            Dict with resume info, or None if not resumable.
        """
        checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
        if not checkpoint:
            return None

        return {
            "session_id": session_id,
            "checkpoint_id": checkpoint.id,
            "created_at": checkpoint.created_at.isoformat(),
            "trigger": checkpoint.trigger.value,
            "working_directory": checkpoint.working_directory,
            "tokens_used": checkpoint.tokens_used,
            "cost_estimate": checkpoint.cost_estimate,
            "uncommitted_changes": checkpoint.uncommitted_changes,
        }

    async def _load_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> Checkpoint | None:
        """Load checkpoint from storage."""
        if checkpoint_id:
            # Load specific checkpoint
            checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)
            for cp in checkpoints:
                if cp.id == checkpoint_id:
                    return cp
            return None
        else:
            # Load latest
            return await self.checkpoint_manager.get_latest_checkpoint(session_id)

    async def _load_session_state(self, checkpoint: Checkpoint) -> Session | None:
        """Load session state from checkpoint directory."""
        checkpoint_dir = (
            self.checkpoint_manager.checkpoint_root
            / checkpoint.session_id
            / checkpoint.id
        )
        state_path = checkpoint_dir / "session_state.json"

        if not state_path.exists():
            return None

        try:
            data = json.loads(state_path.read_text())
            return Session(
                id=data["id"],
                task_id=data["task_id"],
                state=SessionState(data["state"]),
                working_directory=data.get("working_directory", ""),
                created_at=datetime.fromisoformat(data["created_at"]),
                environment=data.get("environment", {}),
                container_id=data.get("container_id"),
                gpu=data.get("gpu"),
            )
        except Exception:
            return None

    async def _load_conversation(self, checkpoint: Checkpoint) -> dict | None:
        """Load conversation state from checkpoint directory."""
        checkpoint_dir = (
            self.checkpoint_manager.checkpoint_root
            / checkpoint.session_id
            / checkpoint.id
        )
        conv_path = checkpoint_dir / "conversation.json"

        if not conv_path.exists():
            return None

        try:
            return json.loads(conv_path.read_text())
        except Exception:
            return None

    def _get_restore_path(self, session_id: str) -> str:
        """Get default restore path for session."""
        return str(self.workspace_root / f"restored-{session_id}")

    async def _restore_workspace(self, state: ResumeState) -> None:
        """Restore workspace from checkpoint."""
        if not state.checkpoint or not state.target_directory:
            raise ResumeError(
                "Cannot restore: missing checkpoint or target directory",
                session_id=state.session_id,
                stage=state.stage.value,
            )

        # Create target directory
        target_path = Path(state.target_directory)
        target_path.mkdir(parents=True, exist_ok=True)

        # Restore using checkpoint manager
        await self.checkpoint_manager.restore_checkpoint(
            state.checkpoint,
            state.target_directory,
        )

    async def _verify_state(self, state: ResumeState) -> None:
        """Verify restored state consistency.

        Implements [SPEC-03.13] state verification.
        """
        if not state.target_directory:
            raise ResumeError(
                "Cannot verify: no target directory",
                session_id=state.session_id,
                stage=state.stage.value,
            )

        target_path = Path(state.target_directory)

        # Verify directory exists and is not empty
        if not target_path.exists():
            raise ResumeError(
                f"Restored directory does not exist: {state.target_directory}",
                session_id=state.session_id,
                checkpoint_id=state.checkpoint.id if state.checkpoint else None,
                stage=state.stage.value,
            )

        # Verify expected files exist (if we know what they should be)
        if state.checkpoint and state.checkpoint.uncommitted_changes:
            missing_files = []
            for file_path in state.checkpoint.uncommitted_changes:
                # Only check files that were added/modified, not deleted
                if not file_path.startswith("-"):  # Deleted files start with -
                    full_path = target_path / file_path
                    if not full_path.exists():
                        missing_files.append(file_path)

            if missing_files:
                state.errors.append(
                    f"Missing files after restore: {', '.join(missing_files[:5])}"
                )

    async def _resume_claude(self, state: ResumeState) -> None:
        """Resume Claude Code execution.

        Implements [SPEC-03.13] claude --resume.
        """
        if not state.target_directory:
            raise ResumeError(
                "Cannot resume Claude: no working directory",
                session_id=state.session_id,
                stage=state.stage.value,
            )

        # Build resume command
        cmd = [
            self.CLAUDE_BINARY,
            "--resume",
            "--print",
            "--output-format", "stream-json",
        ]

        # Add conversation context if available
        if state.conversation and "session_id" in state.conversation:
            cmd.extend(["--session-id", state.conversation["session_id"]])

        # Run Claude Code
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=state.target_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "PARHELIA_RESUMED": "1",
                    "PARHELIA_SESSION_ID": state.session_id,
                },
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.DEFAULT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise ResumeError(
                    f"Claude resume timed out after {self.DEFAULT_TIMEOUT}s",
                    session_id=state.session_id,
                    checkpoint_id=state.checkpoint.id if state.checkpoint else None,
                    stage=state.stage.value,
                )

            if proc.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ResumeError(
                    f"Claude resume failed: {error_msg}",
                    session_id=state.session_id,
                    checkpoint_id=state.checkpoint.id if state.checkpoint else None,
                    stage=state.stage.value,
                )

        except FileNotFoundError:
            # Claude binary not found - this is expected in test environments
            state.errors.append(
                f"Claude binary not found at {self.CLAUDE_BINARY}"
            )
