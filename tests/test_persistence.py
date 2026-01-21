"""Tests for Parhelia persistence layer.

Tests SQLite-backed storage for tasks and workers.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from parhelia.orchestrator import (
    Task,
    TaskRequirements,
    TaskResult,
    TaskType,
    WorkerInfo,
    WorkerState,
)
from parhelia.persistence import (
    PersistenceError,
    PersistentOrchestrator,
    TaskStore,
    WorkerStore,
)


# =============================================================================
# TaskStore Tests
# =============================================================================


class TestTaskStore:
    """Tests for TaskStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a TaskStore with temp database."""
        return TaskStore(tmp_path / "test.db")

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return Task(
            id="task-abc123",
            prompt="Fix the bug",
            task_type=TaskType.CODE_FIX,
            requirements=TaskRequirements(
                needs_gpu=False,
                min_memory_gb=4,
            ),
            metadata={"priority": "high"},
        )

    def test_save_and_get(self, store, sample_task):
        """TaskStore MUST save and retrieve tasks."""
        store.save(sample_task)

        retrieved = store.get(sample_task.id)
        assert retrieved is not None
        assert retrieved.id == sample_task.id
        assert retrieved.prompt == sample_task.prompt
        assert retrieved.task_type == sample_task.task_type

    def test_get_nonexistent(self, store):
        """TaskStore MUST return None for missing tasks."""
        result = store.get("nonexistent-task")
        assert result is None

    def test_save_with_status(self, store, sample_task):
        """TaskStore MUST save with custom status."""
        store.save(sample_task, status="running")

        status = store.get_status(sample_task.id)
        assert status == "running"

    def test_update_status(self, store, sample_task):
        """TaskStore MUST update task status."""
        store.save(sample_task, status="pending")
        store.update_status(sample_task.id, "completed")

        status = store.get_status(sample_task.id)
        assert status == "completed"

    def test_list_by_status(self, store):
        """TaskStore MUST filter tasks by status."""
        task1 = Task(id="task-1", prompt="Task 1", task_type=TaskType.GENERIC, requirements=TaskRequirements())
        task2 = Task(id="task-2", prompt="Task 2", task_type=TaskType.GENERIC, requirements=TaskRequirements())
        task3 = Task(id="task-3", prompt="Task 3", task_type=TaskType.GENERIC, requirements=TaskRequirements())

        store.save(task1, status="pending")
        store.save(task2, status="running")
        store.save(task3, status="pending")

        pending = store.list_by_status("pending")
        assert len(pending) == 2
        assert all(store.get_status(t.id) == "pending" for t in pending)

        running = store.list_by_status("running")
        assert len(running) == 1
        assert running[0].id == "task-2"

    def test_list_pending(self, store):
        """TaskStore MUST list pending tasks."""
        task1 = Task(id="task-1", prompt="Task 1", task_type=TaskType.GENERIC, requirements=TaskRequirements())
        task2 = Task(id="task-2", prompt="Task 2", task_type=TaskType.GENERIC, requirements=TaskRequirements())

        store.save(task1, status="pending")
        store.save(task2, status="completed")

        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0].id == "task-1"

    def test_delete(self, store, sample_task):
        """TaskStore MUST delete tasks."""
        store.save(sample_task)
        assert store.get(sample_task.id) is not None

        deleted = store.delete(sample_task.id)
        assert deleted is True
        assert store.get(sample_task.id) is None

    def test_delete_nonexistent(self, store):
        """TaskStore MUST return False for missing deletes."""
        deleted = store.delete("nonexistent")
        assert deleted is False

    def test_save_result(self, store, sample_task):
        """TaskStore MUST save task results."""
        store.save(sample_task)

        result = TaskResult(
            task_id=sample_task.id,
            status="completed",
            output="Bug fixed successfully",
            cost_usd=0.05,
            duration_seconds=120.5,
            artifacts=["src/fix.py"],
        )
        store.save_result(result)

        retrieved = store.get_result(sample_task.id)
        assert retrieved is not None
        assert retrieved.status == "completed"
        assert retrieved.output == "Bug fixed successfully"
        assert retrieved.cost_usd == 0.05
        assert retrieved.artifacts == ["src/fix.py"]

        # Status should be updated too
        assert store.get_status(sample_task.id) == "completed"

    def test_count_by_status(self, store):
        """TaskStore MUST count tasks by status."""
        store.save(Task(id="t1", prompt="1", task_type=TaskType.GENERIC, requirements=TaskRequirements()), status="pending")
        store.save(Task(id="t2", prompt="2", task_type=TaskType.GENERIC, requirements=TaskRequirements()), status="pending")
        store.save(Task(id="t3", prompt="3", task_type=TaskType.GENERIC, requirements=TaskRequirements()), status="running")
        store.save(Task(id="t4", prompt="4", task_type=TaskType.GENERIC, requirements=TaskRequirements()), status="completed")

        counts = store.count_by_status()
        assert counts["pending"] == 2
        assert counts["running"] == 1
        assert counts["completed"] == 1

    def test_preserves_requirements(self, store, sample_task):
        """TaskStore MUST preserve task requirements."""
        task = Task(
            id="task-gpu",
            prompt="ML training",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(
                needs_gpu=True,
                gpu_type="A100",
                min_memory_gb=32,
            ),
        )
        store.save(task)

        retrieved = store.get(task.id)
        assert retrieved.requirements.needs_gpu is True
        assert retrieved.requirements.gpu_type == "A100"
        assert retrieved.requirements.min_memory_gb == 32


# =============================================================================
# WorkerStore Tests
# =============================================================================


class TestWorkerStore:
    """Tests for WorkerStore class."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a WorkerStore with temp database."""
        return WorkerStore(tmp_path / "test.db")

    @pytest.fixture
    def sample_worker(self):
        """Create a sample worker."""
        return WorkerInfo(
            id="worker-xyz789",
            task_id="task-abc123",
            state=WorkerState.RUNNING,
            target_type="modal",
            gpu_type="A10G",
            metrics={"tokens": 1000, "cost": 0.02},
        )

    def test_save_and_get(self, store, sample_worker):
        """WorkerStore MUST save and retrieve workers."""
        store.save(sample_worker)

        retrieved = store.get(sample_worker.id)
        assert retrieved is not None
        assert retrieved.id == sample_worker.id
        assert retrieved.task_id == sample_worker.task_id
        assert retrieved.state == sample_worker.state

    def test_get_nonexistent(self, store):
        """WorkerStore MUST return None for missing workers."""
        result = store.get("nonexistent")
        assert result is None

    def test_get_by_task(self, store, sample_worker):
        """WorkerStore MUST find worker by task ID."""
        store.save(sample_worker)

        retrieved = store.get_by_task(sample_worker.task_id)
        assert retrieved is not None
        assert retrieved.id == sample_worker.id

    def test_update_state(self, store, sample_worker):
        """WorkerStore MUST update worker state."""
        store.save(sample_worker)
        store.update_state(sample_worker.id, WorkerState.COMPLETED)

        retrieved = store.get(sample_worker.id)
        assert retrieved.state == WorkerState.COMPLETED

    def test_update_metrics(self, store, sample_worker):
        """WorkerStore MUST update worker metrics."""
        store.save(sample_worker)
        store.update_metrics(sample_worker.id, {"tokens": 5000, "cost": 0.10})

        retrieved = store.get(sample_worker.id)
        assert retrieved.metrics["tokens"] == 5000
        assert retrieved.metrics["cost"] == 0.10

    def test_list_active(self, store):
        """WorkerStore MUST list active workers."""
        w1 = WorkerInfo(id="w1", task_id="t1", state=WorkerState.IDLE, target_type="modal")
        w2 = WorkerInfo(id="w2", task_id="t2", state=WorkerState.RUNNING, target_type="modal")
        w3 = WorkerInfo(id="w3", task_id="t3", state=WorkerState.COMPLETED, target_type="modal")

        store.save(w1)
        store.save(w2)
        store.save(w3)

        active = store.list_active()
        assert len(active) == 2
        assert all(w.state in (WorkerState.IDLE, WorkerState.RUNNING) for w in active)

    def test_list_by_state(self, store):
        """WorkerStore MUST filter by state."""
        w1 = WorkerInfo(id="w1", task_id="t1", state=WorkerState.RUNNING, target_type="modal")
        w2 = WorkerInfo(id="w2", task_id="t2", state=WorkerState.RUNNING, target_type="modal")
        w3 = WorkerInfo(id="w3", task_id="t3", state=WorkerState.FAILED, target_type="modal")

        store.save(w1)
        store.save(w2)
        store.save(w3)

        running = store.list_by_state(WorkerState.RUNNING)
        assert len(running) == 2

    def test_delete(self, store, sample_worker):
        """WorkerStore MUST delete workers."""
        store.save(sample_worker)
        deleted = store.delete(sample_worker.id)

        assert deleted is True
        assert store.get(sample_worker.id) is None

    def test_count_by_state(self, store):
        """WorkerStore MUST count by state."""
        store.save(WorkerInfo(id="w1", task_id="t1", state=WorkerState.RUNNING, target_type="modal"))
        store.save(WorkerInfo(id="w2", task_id="t2", state=WorkerState.RUNNING, target_type="modal"))
        store.save(WorkerInfo(id="w3", task_id="t3", state=WorkerState.COMPLETED, target_type="modal"))

        counts = store.count_by_state()
        assert counts["running"] == 2
        assert counts["completed"] == 1


# =============================================================================
# PersistentOrchestrator Tests
# =============================================================================


class TestPersistentOrchestrator:
    """Tests for PersistentOrchestrator class."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator with temp database."""
        return PersistentOrchestrator(db_path=tmp_path / "test.db")

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return Task(
            id="task-orch-123",
            prompt="Test task",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator, sample_task):
        """PersistentOrchestrator MUST submit and persist tasks."""
        task_id = await orchestrator.submit_task(sample_task)

        assert task_id == sample_task.id

        # Verify persisted
        retrieved = await orchestrator.get_task(task_id)
        assert retrieved is not None
        assert retrieved.prompt == sample_task.prompt

    @pytest.mark.asyncio
    async def test_get_pending_tasks(self, orchestrator):
        """PersistentOrchestrator MUST list pending tasks."""
        t1 = Task(id="t1", prompt="Task 1", task_type=TaskType.GENERIC, requirements=TaskRequirements())
        t2 = Task(id="t2", prompt="Task 2", task_type=TaskType.GENERIC, requirements=TaskRequirements())

        await orchestrator.submit_task(t1)
        await orchestrator.submit_task(t2)

        pending = await orchestrator.get_pending_tasks()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_register_worker(self, orchestrator, sample_task):
        """PersistentOrchestrator MUST register workers."""
        await orchestrator.submit_task(sample_task)

        worker = WorkerInfo(
            id="worker-1",
            task_id=sample_task.id,
            state=WorkerState.RUNNING,
            target_type="modal",
        )
        orchestrator.register_worker(worker)

        # Task should be running
        task = await orchestrator.get_task(sample_task.id)
        status = orchestrator.task_store.get_status(task.id)
        assert status == "running"

        # Worker should be retrievable
        retrieved = orchestrator.get_worker(worker.id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_mark_task_complete(self, orchestrator, sample_task):
        """PersistentOrchestrator MUST mark tasks complete."""
        await orchestrator.submit_task(sample_task)

        worker = WorkerInfo(
            id="worker-1",
            task_id=sample_task.id,
            state=WorkerState.RUNNING,
            target_type="modal",
        )
        orchestrator.register_worker(worker)

        result = TaskResult(
            task_id=sample_task.id,
            status="completed",
            output="Done",
            cost_usd=0.05,
            duration_seconds=60,
        )
        orchestrator.mark_task_complete(sample_task.id, result)

        # Task status should be completed
        status = orchestrator.task_store.get_status(sample_task.id)
        assert status == "completed"

        # Worker should be completed
        w = orchestrator.get_worker(worker.id)
        assert w.state == WorkerState.COMPLETED

    @pytest.mark.asyncio
    async def test_get_stats(self, orchestrator):
        """PersistentOrchestrator MUST return statistics."""
        t1 = Task(id="t1", prompt="Task 1", task_type=TaskType.GENERIC, requirements=TaskRequirements())
        t2 = Task(id="t2", prompt="Task 2", task_type=TaskType.GENERIC, requirements=TaskRequirements())

        await orchestrator.submit_task(t1)
        await orchestrator.submit_task(t2)

        stats = orchestrator.get_stats()
        assert stats["tasks"]["pending"] == 2
        assert stats["tasks"]["total"] == 2

    def test_get_pending_count(self, orchestrator):
        """PersistentOrchestrator MUST count pending tasks."""
        orchestrator.task_store.save(
            Task(id="t1", prompt="1", task_type=TaskType.GENERIC, requirements=TaskRequirements()),
            status="pending"
        )
        orchestrator.task_store.save(
            Task(id="t2", prompt="2", task_type=TaskType.GENERIC, requirements=TaskRequirements()),
            status="running"
        )

        assert orchestrator.get_pending_count() == 1

    def test_get_active_worker_count(self, orchestrator):
        """PersistentOrchestrator MUST count active workers."""
        orchestrator.worker_store.save(
            WorkerInfo(id="w1", task_id="t1", state=WorkerState.RUNNING, target_type="modal")
        )
        orchestrator.worker_store.save(
            WorkerInfo(id="w2", task_id="t2", state=WorkerState.IDLE, target_type="modal")
        )
        orchestrator.worker_store.save(
            WorkerInfo(id="w3", task_id="t3", state=WorkerState.COMPLETED, target_type="modal")
        )

        assert orchestrator.get_active_worker_count() == 2


# =============================================================================
# Persistence Across Sessions Tests
# =============================================================================


class TestPersistenceAcrossSessions:
    """Tests verifying data persists across orchestrator instances."""

    def test_task_persists_across_instances(self, tmp_path):
        """Tasks MUST persist across orchestrator instances."""
        db_path = tmp_path / "persist.db"

        # Create first orchestrator and submit task
        orch1 = PersistentOrchestrator(db_path=db_path)
        import asyncio
        task = Task(id="persist-task", prompt="Persist me", task_type=TaskType.GENERIC, requirements=TaskRequirements())
        asyncio.run(orch1.submit_task(task))

        # Create second orchestrator and verify task exists
        orch2 = PersistentOrchestrator(db_path=db_path)
        retrieved = asyncio.run(orch2.get_task("persist-task"))

        assert retrieved is not None
        assert retrieved.prompt == "Persist me"

    def test_worker_persists_across_instances(self, tmp_path):
        """Workers MUST persist across orchestrator instances."""
        db_path = tmp_path / "persist.db"

        # Create first orchestrator and register worker
        orch1 = PersistentOrchestrator(db_path=db_path)
        worker = WorkerInfo(
            id="persist-worker",
            task_id="task-1",
            state=WorkerState.RUNNING,
            target_type="modal",
        )
        orch1.register_worker(worker)

        # Create second orchestrator and verify worker exists
        orch2 = PersistentOrchestrator(db_path=db_path)
        retrieved = orch2.get_worker("persist-worker")

        assert retrieved is not None
        assert retrieved.state == WorkerState.RUNNING
