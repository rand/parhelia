"""Task dispatch to Modal sandboxes.

Implements:
- [SPEC-05.11] Dispatch Logic
- [SPEC-05.12] Worker Lifecycle Management

Connects the persistence layer to Modal execution.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from parhelia.orchestrator import (
    Task,
    TaskResult,
    WorkerInfo,
    WorkerState,
)
from parhelia.persistence import PersistentOrchestrator

if TYPE_CHECKING:
    import modal


class DispatchError(Exception):
    """Error during task dispatch."""


class DispatchMode(Enum):
    """How to dispatch tasks."""

    SYNC = "sync"  # Wait for completion
    ASYNC = "async"  # Fire and forget
    BACKGROUND = "background"  # Dispatch and track in background


@dataclass
class DispatchResult:
    """Result of a dispatch operation."""

    task_id: str
    worker_id: str
    sandbox_id: str | None = None
    success: bool = True
    error: str | None = None
    output: str | None = None


class TaskDispatcher:
    """Dispatches tasks to Modal sandboxes.

    Connects PersistentOrchestrator to Modal execution.

    Usage:
        dispatcher = TaskDispatcher(orchestrator)
        result = await dispatcher.dispatch(task)
    """

    READY_TIMEOUT_SECONDS = 60
    CLAUDE_BIN = "/root/.claude/local/claude"

    def __init__(
        self,
        orchestrator: PersistentOrchestrator,
        skip_modal: bool = False,
    ):
        """Initialize the dispatcher.

        Args:
            orchestrator: The persistent orchestrator for state management.
            skip_modal: If True, skip actual Modal calls (for testing/dry-run).
        """
        self.orchestrator = orchestrator
        self.skip_modal = skip_modal
        self._progress_callback: Callable[[str], None] | None = None

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _log(self, message: str) -> None:
        """Log progress message."""
        if self._progress_callback:
            self._progress_callback(message)

    async def dispatch(
        self,
        task: Task,
        mode: DispatchMode = DispatchMode.ASYNC,
        timeout_hours: int | None = None,
    ) -> DispatchResult:
        """Dispatch a task to Modal for execution.

        Args:
            task: The task to dispatch.
            mode: Dispatch mode (sync waits, async returns immediately).
            timeout_hours: Sandbox timeout in hours.

        Returns:
            DispatchResult with worker/sandbox info.

        Raises:
            DispatchError: If dispatch fails.
        """
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"

        # Update task status to running
        self.orchestrator.task_store.update_status(task.id, "running")

        if self.skip_modal:
            return await self._dispatch_dry_run(task, worker_id)

        try:
            return await self._dispatch_to_modal(task, worker_id, mode, timeout_hours)
        except Exception as e:
            # Mark task as failed
            self.orchestrator.mark_task_failed(task.id, str(e))
            raise DispatchError(f"Failed to dispatch task {task.id}: {e}") from e

    async def _dispatch_dry_run(
        self,
        task: Task,
        worker_id: str,
    ) -> DispatchResult:
        """Dry-run dispatch without Modal (for testing)."""
        self._log(f"[dry-run] Would dispatch task {task.id}")
        self._log(f"[dry-run] Prompt: {task.prompt[:100]}...")

        # Register a mock worker
        worker = WorkerInfo(
            id=worker_id,
            task_id=task.id,
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
            gpu_type=task.requirements.gpu_type,
        )
        self.orchestrator.register_worker(worker)

        return DispatchResult(
            task_id=task.id,
            worker_id=worker_id,
            sandbox_id="dry-run-sandbox",
            success=True,
        )

    async def _dispatch_to_modal(
        self,
        task: Task,
        worker_id: str,
        mode: DispatchMode,
        timeout_hours: int | None,
    ) -> DispatchResult:
        """Dispatch task to Modal sandbox."""
        from parhelia.modal_app import create_claude_sandbox, run_in_sandbox

        self._log(f"Creating sandbox for task {task.id}...")

        # Determine GPU requirement
        gpu = task.requirements.gpu_type if task.requirements.needs_gpu else None

        # Create sandbox
        sandbox = await create_claude_sandbox(
            task_id=task.id,
            gpu=gpu,
            timeout_hours=timeout_hours,
        )

        sandbox_id = str(sandbox.object_id) if hasattr(sandbox, "object_id") else worker_id

        self._log(f"Sandbox created: {sandbox_id}")

        # Register worker
        target_type = "parhelia-gpu" if gpu else "parhelia-cpu"
        worker = WorkerInfo(
            id=worker_id,
            task_id=task.id,
            state=WorkerState.RUNNING,
            target_type=target_type,
            gpu_type=gpu,
            metrics={"sandbox_id": sandbox_id},
        )
        self.orchestrator.register_worker(worker)

        # Wait for container to be ready
        self._log("Waiting for container ready...")
        await self._wait_for_ready(sandbox)

        # Run Claude Code with the task prompt
        self._log("Starting Claude Code...")
        if mode == DispatchMode.SYNC:
            # Wait for completion
            output = await self._run_claude_and_wait(sandbox, task)
            return DispatchResult(
                task_id=task.id,
                worker_id=worker_id,
                sandbox_id=sandbox_id,
                success=True,
                output=output,
            )
        else:
            # Fire and forget - start Claude in background
            asyncio.create_task(self._run_claude_background(sandbox, task, worker_id))
            return DispatchResult(
                task_id=task.id,
                worker_id=worker_id,
                sandbox_id=sandbox_id,
                success=True,
            )

    async def _wait_for_ready(self, sandbox: "modal.Sandbox") -> None:
        """Wait for sandbox to signal readiness."""
        from parhelia.modal_app import run_in_sandbox

        start = datetime.now()
        while (datetime.now() - start).seconds < self.READY_TIMEOUT_SECONDS:
            try:
                output = await run_in_sandbox(sandbox, ["cat", "/tmp/ready"])
                if "PARHELIA_READY" in output:
                    self._log("Container ready")
                    return
            except Exception:
                pass
            await asyncio.sleep(1)

        raise DispatchError("Sandbox did not become ready in time")

    async def _run_claude_and_wait(
        self,
        sandbox: "modal.Sandbox",
        task: Task,
    ) -> str:
        """Run Claude Code and wait for completion."""
        from parhelia.modal_app import run_in_sandbox

        # Build Claude command
        # Use --print for non-interactive mode with prompt
        cmd = [
            self.CLAUDE_BIN,
            "--print",  # Non-interactive mode
            task.prompt,
        ]

        # Add working directory if specified
        if task.requirements.working_directory:
            cmd = ["bash", "-c", f"cd {task.requirements.working_directory} && {' '.join(cmd)}"]

        self._log(f"Running: {' '.join(cmd[:3])}...")

        output = await run_in_sandbox(sandbox, cmd)

        # Mark task complete
        result = TaskResult(
            task_id=task.id,
            status="completed",
            output=output,
            cost_usd=0.0,  # Would need to track actual cost
            duration_seconds=0.0,
        )
        self.orchestrator.mark_task_complete(task.id, result)

        return output

    async def _run_claude_background(
        self,
        sandbox: "modal.Sandbox",
        task: Task,
        worker_id: str,
    ) -> None:
        """Run Claude Code in background (async dispatch)."""
        try:
            output = await self._run_claude_and_wait(sandbox, task)
            self._log(f"Task {task.id} completed")
        except Exception as e:
            self._log(f"Task {task.id} failed: {e}")
            self.orchestrator.mark_task_failed(task.id, str(e))
            self.orchestrator.worker_store.update_state(worker_id, WorkerState.FAILED)

    async def dispatch_pending(
        self,
        limit: int = 10,
        mode: DispatchMode = DispatchMode.ASYNC,
    ) -> list[DispatchResult]:
        """Dispatch all pending tasks.

        Args:
            limit: Maximum number of tasks to dispatch.
            mode: Dispatch mode for all tasks.

        Returns:
            List of DispatchResults.
        """
        pending = await self.orchestrator.get_pending_tasks()
        results = []

        for task in pending[:limit]:
            try:
                result = await self.dispatch(task, mode=mode)
                results.append(result)
            except DispatchError as e:
                results.append(DispatchResult(
                    task_id=task.id,
                    worker_id="",
                    success=False,
                    error=str(e),
                ))

        return results


async def dispatch_task(
    task: Task,
    orchestrator: PersistentOrchestrator | None = None,
    skip_modal: bool = False,
    mode: DispatchMode = DispatchMode.ASYNC,
) -> DispatchResult:
    """Convenience function to dispatch a single task.

    Args:
        task: The task to dispatch.
        orchestrator: Orchestrator to use (creates default if None).
        skip_modal: Skip actual Modal calls.
        mode: Dispatch mode.

    Returns:
        DispatchResult.
    """
    if orchestrator is None:
        orchestrator = PersistentOrchestrator()

    dispatcher = TaskDispatcher(orchestrator, skip_modal=skip_modal)
    return await dispatcher.dispatch(task, mode=mode)
