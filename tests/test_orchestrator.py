"""Tests for local orchestrator.

@trace SPEC-05.10 - Task Decomposition
@trace SPEC-05.11 - Dispatch Logic
@trace SPEC-05.12 - Worker Lifecycle
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestTaskDataclass:
    """Tests for Task dataclass - SPEC-05.10."""

    def test_task_creation(self):
        """@trace SPEC-05.10 - Task MUST have id, prompt, and requirements."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        requirements = TaskRequirements()
        task = Task(
            id="task-123",
            prompt="Fix the bug in auth.py",
            task_type=TaskType.CODE_FIX,
            requirements=requirements,
        )

        assert task.id == "task-123"
        assert task.prompt == "Fix the bug in auth.py"
        assert task.task_type == TaskType.CODE_FIX
        assert task.requirements == requirements

    def test_task_has_optional_parent_id(self):
        """@trace SPEC-05.10 - Task SHOULD support parent_id for subtasks."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        subtask = Task(
            id="subtask-1",
            prompt="Fix auth",
            task_type=TaskType.CODE_FIX,
            requirements=TaskRequirements(),
            parent_id="parent-task",
        )

        assert subtask.parent_id == "parent-task"

    def test_task_has_metadata(self):
        """@trace SPEC-05.10 - Task SHOULD support arbitrary metadata."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-123",
            prompt="Do work",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
            metadata={"source": "cli", "priority": "high"},
        )

        assert task.metadata["source"] == "cli"
        assert task.metadata["priority"] == "high"


class TestTaskRequirements:
    """Tests for TaskRequirements dataclass - SPEC-05.10."""

    def test_default_requirements(self):
        """@trace SPEC-05.10 - TaskRequirements MUST have sensible defaults."""
        from parhelia.orchestrator import TaskRequirements

        req = TaskRequirements()

        assert req.needs_gpu is False
        assert req.gpu_type is None
        assert req.min_memory_gb >= 1
        assert req.min_cpu >= 1
        assert req.needs_network is True

    def test_gpu_requirements(self):
        """@trace SPEC-05.10 - TaskRequirements MUST support GPU specification."""
        from parhelia.orchestrator import TaskRequirements

        req = TaskRequirements(needs_gpu=True, gpu_type="A100")

        assert req.needs_gpu is True
        assert req.gpu_type == "A100"

    def test_working_directory_requirement(self):
        """@trace SPEC-05.10 - TaskRequirements SHOULD support working_directory."""
        from parhelia.orchestrator import TaskRequirements

        req = TaskRequirements(working_directory="/path/to/workspace")

        assert req.working_directory == "/path/to/workspace"


class TestLocalOrchestrator:
    """Tests for LocalOrchestrator - SPEC-05.10."""

    @pytest.fixture
    def orchestrator(self):
        """Create LocalOrchestrator instance."""
        from parhelia.orchestrator import LocalOrchestrator

        return LocalOrchestrator()

    def test_orchestrator_initialization(self, orchestrator):
        """@trace SPEC-05.10 - LocalOrchestrator MUST initialize with empty state."""
        assert orchestrator is not None
        assert len(orchestrator.pending_tasks) == 0
        assert len(orchestrator.active_workers) == 0

    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator):
        """@trace SPEC-05.10 - LocalOrchestrator MUST accept task submission."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-1",
            prompt="Run tests",
            task_type=TaskType.TEST_RUN,
            requirements=TaskRequirements(),
        )

        task_id = await orchestrator.submit_task(task)

        assert task_id == "task-1"
        assert task_id in orchestrator.pending_tasks

    @pytest.mark.asyncio
    async def test_get_workers_returns_list(self, orchestrator):
        """@trace SPEC-05.10 - get_workers MUST return list of workers."""
        workers = await orchestrator.get_workers()

        assert isinstance(workers, list)

    @pytest.mark.asyncio
    async def test_collect_results_for_task(self, orchestrator):
        """@trace SPEC-05.10 - collect_results MUST return results for task."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType, TaskResult

        task = Task(
            id="task-1",
            prompt="Do work",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

        # Submit and mark complete with mock result
        await orchestrator.submit_task(task)
        orchestrator.completed_results["task-1"] = TaskResult(
            task_id="task-1",
            status="success",
            output="Task completed",
        )

        result = await orchestrator.collect_results("task-1")

        assert result is not None
        assert result.task_id == "task-1"
        assert result.status == "success"


class TestTaskType:
    """Tests for TaskType enum - SPEC-05.10."""

    def test_task_types_defined(self):
        """@trace SPEC-05.10 - TaskType MUST define common task categories."""
        from parhelia.orchestrator import TaskType

        assert TaskType.GENERIC
        assert TaskType.CODE_FIX
        assert TaskType.TEST_RUN
        assert TaskType.BUILD
        assert TaskType.INTERACTIVE


class TestWorkerTracking:
    """Tests for worker tracking - SPEC-05.12."""

    @pytest.fixture
    def orchestrator(self):
        """Create LocalOrchestrator instance."""
        from parhelia.orchestrator import LocalOrchestrator

        return LocalOrchestrator()

    def test_register_worker(self, orchestrator):
        """@trace SPEC-05.12 - Orchestrator MUST track active workers."""
        from parhelia.orchestrator import WorkerInfo, WorkerState

        worker = WorkerInfo(
            id="worker-1",
            task_id="task-1",
            state=WorkerState.RUNNING,
            target_type="parhelia-cpu",
        )

        orchestrator.register_worker(worker)

        assert "worker-1" in orchestrator.active_workers
        assert orchestrator.active_workers["worker-1"] == worker

    def test_unregister_worker(self, orchestrator):
        """@trace SPEC-05.12 - Orchestrator MUST allow worker removal."""
        from parhelia.orchestrator import WorkerInfo, WorkerState

        worker = WorkerInfo(
            id="worker-1",
            task_id="task-1",
            state=WorkerState.RUNNING,
            target_type="local",
        )

        orchestrator.register_worker(worker)
        removed = orchestrator.unregister_worker("worker-1")

        assert removed == worker
        assert "worker-1" not in orchestrator.active_workers

    def test_get_worker_by_id(self, orchestrator):
        """@trace SPEC-05.12 - Orchestrator MUST retrieve worker by ID."""
        from parhelia.orchestrator import WorkerInfo, WorkerState

        worker = WorkerInfo(
            id="worker-1",
            task_id="task-1",
            state=WorkerState.IDLE,
            target_type="parhelia-gpu",
        )

        orchestrator.register_worker(worker)
        retrieved = orchestrator.get_worker("worker-1")

        assert retrieved == worker


class TestWorkerState:
    """Tests for WorkerState enum - SPEC-05.12."""

    def test_worker_states_defined(self):
        """@trace SPEC-05.12 - WorkerState MUST define lifecycle states."""
        from parhelia.orchestrator import WorkerState

        assert WorkerState.IDLE
        assert WorkerState.RUNNING
        assert WorkerState.COMPLETED
        assert WorkerState.FAILED
        assert WorkerState.TERMINATED
