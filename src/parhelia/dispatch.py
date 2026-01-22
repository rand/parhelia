"""Task dispatch to Modal sandboxes.

Implements:
- [SPEC-05.11] Dispatch Logic
- [SPEC-05.12] Worker Lifecycle Management
- [SPEC-10.50] Hook Framework

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
from parhelia.state import Container, ContainerState, StateStore

if TYPE_CHECKING:
    import modal

    from parhelia.budget import BudgetManager
    from parhelia.hook_executor import HookExecutor


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
    CLAUDE_BIN = "/root/.local/bin/claude"

    def __init__(
        self,
        orchestrator: PersistentOrchestrator,
        skip_modal: bool = False,
        hook_executor: "HookExecutor | None" = None,
        budget_manager: "BudgetManager | None" = None,
        state_store: StateStore | None = None,
    ):
        """Initialize the dispatcher.

        Args:
            orchestrator: The persistent orchestrator for state management.
            skip_modal: If True, skip actual Modal calls (for testing/dry-run).
            hook_executor: Optional hook executor for pre/post dispatch hooks.
            budget_manager: Optional budget manager for hook context.
            state_store: Optional state store for container lifecycle tracking.
        """
        self.orchestrator = orchestrator
        self.skip_modal = skip_modal
        self.hook_executor = hook_executor
        self.budget_manager = budget_manager
        self.state_store = state_store
        self._progress_callback: Callable[[str], None] | None = None

    def set_progress_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback
        if self.hook_executor:
            self.hook_executor.set_log_callback(callback)

    def _log(self, message: str) -> None:
        """Log progress message."""
        if self._progress_callback:
            self._progress_callback(message)

    def _create_container_record(
        self,
        task: Task,
        worker_id: str,
        sandbox_id: str,
        session_id: str | None = None,
    ) -> Container | None:
        """Create a container record in the state store.

        Args:
            task: The task being dispatched.
            worker_id: The worker ID.
            sandbox_id: The Modal sandbox ID.
            session_id: Optional session ID.

        Returns:
            The created Container, or None if state_store not configured.
        """
        if not self.state_store:
            return None

        container = Container.create(
            modal_sandbox_id=sandbox_id,
            task_id=task.id,
            worker_id=worker_id,
            session_id=session_id,
        )
        self.state_store.create_container(container)
        self._log(f"Container record created: {container.id}")
        return container

    def _update_container_state(
        self,
        worker_id: str,
        new_state: ContainerState,
        exit_code: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Update container state based on worker state change.

        Args:
            worker_id: The worker ID whose container to update.
            new_state: The new container state.
            exit_code: Optional exit code for termination.
            reason: Optional reason for state change.
        """
        if not self.state_store:
            return

        # Get worker to find container_id
        worker = self.orchestrator.get_worker(worker_id)
        if not worker or not worker.container_id:
            return

        container = self.state_store.get_container(worker.container_id)
        if not container:
            return

        # Update container state
        if exit_code is not None:
            container.exit_code = exit_code
        self.state_store.update_container_state(
            container.id,
            new_state,
            reason=reason,
        )
        self._log(f"Container {container.id} state updated to {new_state.value}")

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
            DispatchError: If dispatch fails or is rejected by pre-dispatch hook.
        """
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"

        # Run pre-dispatch hook
        if self.hook_executor:
            hook_result = await self._run_pre_dispatch_hook(task)
            if hook_result and hook_result.get("rejected"):
                raise DispatchError(f"Pre-dispatch hook rejected: {hook_result.get('message', 'Unknown reason')}")

        # Update task status to running
        self.orchestrator.task_store.update_status(task.id, "running")

        if self.skip_modal:
            result = await self._dispatch_dry_run(task, worker_id)
        else:
            try:
                result = await self._dispatch_to_modal(task, worker_id, mode, timeout_hours)
            except Exception as e:
                # Mark task as failed
                self.orchestrator.mark_task_failed(task.id, str(e))
                raise DispatchError(f"Failed to dispatch task {task.id}: {e}") from e

        # Run post-dispatch hook
        if self.hook_executor and result.success:
            await self._run_post_dispatch_hook(task, result)

        return result

    async def _run_pre_dispatch_hook(self, task: Task) -> dict[str, Any] | None:
        """Run pre-dispatch hook and return result.

        Returns:
            Dict with 'rejected' bool and 'message' if hook ran, None otherwise.
        """
        from parhelia.hook_executor import (
            HookResult,
            HookType,
            create_hook_context_from_task,
        )

        # Get budget info for hook context
        budget_remaining = None
        estimated_cost = None
        if self.budget_manager:
            budget_remaining = self.budget_manager.remaining_usd

            # Estimate cost based on requirements
            hours = task.requirements.estimated_duration_minutes / 60
            if task.requirements.needs_gpu:
                # GPU pricing ~$2-4/hr depending on type
                estimated_cost = hours * 3.0
            else:
                # CPU pricing ~$0.20/hr
                estimated_cost = hours * 0.20

        context = create_hook_context_from_task(
            task=task,
            hook_type=HookType.PRE_DISPATCH,
            budget_remaining_usd=budget_remaining,
            estimated_cost_usd=estimated_cost,
        )

        output = await self.hook_executor.run_pre_dispatch(context)

        if output.result == HookResult.REJECT:
            self._log(f"Pre-dispatch hook rejected: {output.message}")
            return {"rejected": True, "message": output.message}
        elif output.result == HookResult.WARN:
            self._log(f"Pre-dispatch hook warning: {output.message}")
        elif output.result == HookResult.ERROR:
            self._log(f"Pre-dispatch hook error: {output.message}")
            # Continue based on fail_open setting (handled in HookExecutor)

        return {"rejected": False, "message": output.message}

    async def _run_post_dispatch_hook(self, task: Task, result: "DispatchResult") -> None:
        """Run post-dispatch hook for audit/logging."""
        from parhelia.hook_executor import HookType, create_hook_context_from_task

        context = create_hook_context_from_task(
            task=task,
            hook_type=HookType.POST_DISPATCH,
            worker_id=result.worker_id,
        )

        output = await self.hook_executor.run_post_dispatch(context)

        if output.message:
            self._log(f"Post-dispatch: {output.message}")

    async def _dispatch_dry_run(
        self,
        task: Task,
        worker_id: str,
    ) -> DispatchResult:
        """Dry-run dispatch without Modal (for testing)."""
        self._log(f"[dry-run] Would dispatch task {task.id}")
        self._log(f"[dry-run] Prompt: {task.prompt[:100]}...")

        # Use unique sandbox_id based on worker_id
        sandbox_id = f"dry-run-{worker_id}"

        # Create container record
        container = self._create_container_record(
            task=task,
            worker_id=worker_id,
            sandbox_id=sandbox_id,
        )

        # Register a mock worker with container_id link
        worker = WorkerInfo(
            id=worker_id,
            task_id=task.id,
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
            gpu_type=task.requirements.gpu_type,
            container_id=container.id if container else None,
        )
        self.orchestrator.register_worker(worker)

        return DispatchResult(
            task_id=task.id,
            worker_id=worker_id,
            sandbox_id=sandbox_id,
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

        # Create container record
        container = self._create_container_record(
            task=task,
            worker_id=worker_id,
            sandbox_id=sandbox_id,
        )

        # Register worker with container_id link
        target_type = "parhelia-gpu" if gpu else "parhelia-cpu"
        worker = WorkerInfo(
            id=worker_id,
            task_id=task.id,
            state=WorkerState.RUNNING,
            target_type=target_type,
            gpu_type=gpu,
            metrics={"sandbox_id": sandbox_id},
            container_id=container.id if container else None,
        )
        self.orchestrator.register_worker(worker)

        # Wait for container to be ready
        self._log("Waiting for container ready...")
        await self._wait_for_ready(sandbox)

        # Update container state to RUNNING after it's ready
        if container and self.state_store:
            self.state_store.update_container_state(
                container.id,
                ContainerState.RUNNING,
            )

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
        """Initialize sandbox and verify readiness.

        Runs the entrypoint script on first use to:
        - Link Claude config from volume
        - Verify Claude installation
        - Start MCP servers (if configured)
        - Initialize tmux session
        - Create /tmp/ready marker

        Uses /tmp/parhelia_init marker to avoid re-running on subsequent commands.
        """
        from parhelia.modal_app import run_in_sandbox

        # Check if already initialized
        try:
            output = await run_in_sandbox(
                sandbox, ["test", "-f", "/tmp/parhelia_init", "&&", "echo", "initialized"]
            )
            if "initialized" in output:
                self._log("Sandbox already initialized")
                return
        except Exception:
            pass

        # Run entrypoint script (set PARHELIA_INTERACTIVE=true to avoid tail -f wait)
        self._log("Running entrypoint initialization...")
        try:
            output = await run_in_sandbox(
                sandbox,
                ["bash", "-c", "PARHELIA_INTERACTIVE=true /entrypoint.sh && touch /tmp/parhelia_init"],
                timeout_seconds=30,
            )
            self._log("Entrypoint completed")
        except Exception as e:
            self._log(f"Entrypoint failed: {e}, falling back to direct verification")

        # Verify readiness - check /tmp/ready or fall back to Claude --version
        start = datetime.now()
        while (datetime.now() - start).seconds < self.READY_TIMEOUT_SECONDS:
            try:
                # First try /tmp/ready (created by entrypoint)
                output = await run_in_sandbox(sandbox, ["cat", "/tmp/ready"])
                if "PARHELIA_READY" in output:
                    self._log("Container ready (entrypoint signaled)")
                    return

                # Fall back to Claude --version check
                output = await run_in_sandbox(sandbox, [self.CLAUDE_BIN, "--version"])
                if "Claude Code" in output:
                    self._log("Container ready (Claude Code verified)")
                    return
            except Exception:
                pass
            await asyncio.sleep(2)

        raise DispatchError("Sandbox did not become ready in time")

    async def _run_claude_and_wait(
        self,
        sandbox: "modal.Sandbox",
        task: Task,
    ) -> str:
        """Run Claude Code and wait for completion."""
        from parhelia.modal_app import run_in_sandbox

        # Build Claude command
        # Use -p for non-interactive print mode with prompt
        cmd = [
            self.CLAUDE_BIN,
            "-p",  # Non-interactive print mode
            task.prompt,
            "--max-turns", "10",  # Reasonable limit for autonomous work
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
            # Update container state to terminated (successful completion)
            self._update_container_state(
                worker_id,
                ContainerState.TERMINATED,
                exit_code=0,
                reason="Task completed successfully",
            )
        except Exception as e:
            self._log(f"Task {task.id} failed: {e}")
            self.orchestrator.mark_task_failed(task.id, str(e))
            self.orchestrator.worker_store.update_state(worker_id, WorkerState.FAILED)
            # Update container state to terminated (failure)
            self._update_container_state(
                worker_id,
                ContainerState.TERMINATED,
                exit_code=1,
                reason=str(e),
            )

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
