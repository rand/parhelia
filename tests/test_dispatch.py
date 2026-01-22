"""Tests for task dispatch module.

Tests dispatch logic connecting persistence to Modal.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from parhelia.dispatch import (
    DispatchError,
    DispatchMode,
    DispatchResult,
    TaskDispatcher,
    dispatch_task,
)
from parhelia.orchestrator import (
    Task,
    TaskRequirements,
    TaskType,
    WorkerState,
)
from parhelia.persistence import PersistentOrchestrator


# =============================================================================
# TaskDispatcher Tests
# =============================================================================


class TestTaskDispatcher:
    """Tests for TaskDispatcher class."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator with temp database."""
        return PersistentOrchestrator(db_path=tmp_path / "test.db")

    @pytest.fixture
    def dispatcher(self, orchestrator):
        """Create dispatcher in dry-run mode."""
        return TaskDispatcher(orchestrator, skip_modal=True)

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return Task(
            id="task-dispatch-123",
            prompt="Test dispatch",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

    @pytest.mark.asyncio
    async def test_dispatch_dry_run(self, dispatcher, orchestrator, sample_task):
        """TaskDispatcher MUST handle dry-run mode."""
        # Submit task first
        await orchestrator.submit_task(sample_task)

        # Dispatch in dry-run
        result = await dispatcher.dispatch(sample_task)

        assert result.success is True
        assert result.task_id == sample_task.id
        assert result.worker_id.startswith("worker-")
        assert result.sandbox_id.startswith("dry-run-")

    @pytest.mark.asyncio
    async def test_dispatch_registers_worker(self, dispatcher, orchestrator, sample_task):
        """TaskDispatcher MUST register worker on dispatch."""
        await orchestrator.submit_task(sample_task)
        result = await dispatcher.dispatch(sample_task)

        # Worker should be registered
        worker = orchestrator.get_worker(result.worker_id)
        assert worker is not None
        assert worker.task_id == sample_task.id
        assert worker.state == WorkerState.RUNNING

    @pytest.mark.asyncio
    async def test_dispatch_updates_task_status(self, dispatcher, orchestrator, sample_task):
        """TaskDispatcher MUST update task status to running."""
        await orchestrator.submit_task(sample_task)
        await dispatcher.dispatch(sample_task)

        # Task should be running
        status = orchestrator.task_store.get_status(sample_task.id)
        assert status == "running"

    @pytest.mark.asyncio
    async def test_dispatch_with_gpu_requirement(self, dispatcher, orchestrator):
        """TaskDispatcher MUST handle GPU requirements."""
        task = Task(
            id="task-gpu-dispatch",
            prompt="GPU task",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(
                needs_gpu=True,
                gpu_type="A100",
            ),
        )
        await orchestrator.submit_task(task)
        result = await dispatcher.dispatch(task)

        assert result.success is True
        worker = orchestrator.get_worker(result.worker_id)
        assert worker.gpu_type == "A100"

    @pytest.mark.asyncio
    async def test_dispatch_pending_tasks(self, dispatcher, orchestrator):
        """TaskDispatcher MUST dispatch multiple pending tasks."""
        tasks = [
            Task(id=f"task-batch-{i}", prompt=f"Task {i}", task_type=TaskType.GENERIC, requirements=TaskRequirements())
            for i in range(3)
        ]
        for task in tasks:
            await orchestrator.submit_task(task)

        results = await dispatcher.dispatch_pending(limit=10)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dispatch_pending_respects_limit(self, dispatcher, orchestrator):
        """TaskDispatcher MUST respect limit on pending dispatch."""
        tasks = [
            Task(id=f"task-limit-{i}", prompt=f"Task {i}", task_type=TaskType.GENERIC, requirements=TaskRequirements())
            for i in range(5)
        ]
        for task in tasks:
            await orchestrator.submit_task(task)

        results = await dispatcher.dispatch_pending(limit=2)

        assert len(results) == 2

    def test_progress_callback(self, dispatcher, orchestrator, sample_task):
        """TaskDispatcher MUST call progress callback."""
        messages = []
        dispatcher.set_progress_callback(lambda msg: messages.append(msg))

        import asyncio
        asyncio.run(orchestrator.submit_task(sample_task))
        asyncio.run(dispatcher.dispatch(sample_task))

        assert len(messages) > 0
        assert any("dry-run" in msg for msg in messages)


# =============================================================================
# DispatchResult Tests
# =============================================================================


class TestDispatchResult:
    """Tests for DispatchResult dataclass."""

    def test_dispatch_result_success(self):
        """DispatchResult MUST represent successful dispatch."""
        result = DispatchResult(
            task_id="task-123",
            worker_id="worker-456",
            sandbox_id="sandbox-789",
            success=True,
        )
        assert result.success is True
        assert result.error is None

    def test_dispatch_result_failure(self):
        """DispatchResult MUST represent failed dispatch."""
        result = DispatchResult(
            task_id="task-123",
            worker_id="",
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDispatchTask:
    """Tests for dispatch_task convenience function."""

    @pytest.mark.asyncio
    async def test_dispatch_task_creates_orchestrator(self, tmp_path):
        """dispatch_task MUST create orchestrator if not provided."""
        task = Task(
            id="task-conv-123",
            prompt="Test",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

        with patch("parhelia.dispatch.PersistentOrchestrator") as mock_orch:
            mock_instance = MagicMock()
            mock_instance.submit_task = AsyncMock(return_value=task.id)
            mock_instance.task_store = MagicMock()
            mock_instance.task_store.update_status = MagicMock()
            mock_instance.register_worker = MagicMock()
            mock_orch.return_value = mock_instance

            result = await dispatch_task(task, skip_modal=True)

            assert result.success is True
            mock_orch.assert_called_once()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestDispatchErrors:
    """Tests for dispatch error handling."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator with temp database."""
        return PersistentOrchestrator(db_path=tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_dispatch_error_marks_task_failed(self, orchestrator):
        """TaskDispatcher MUST mark task as failed on error."""
        task = Task(
            id="task-error-test",
            prompt="Will fail",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )
        await orchestrator.submit_task(task)

        # Create dispatcher that will fail
        dispatcher = TaskDispatcher(orchestrator, skip_modal=False)

        with patch("parhelia.modal_app.create_claude_sandbox", side_effect=Exception("Modal error")):
            with pytest.raises(DispatchError, match="Modal error"):
                await dispatcher.dispatch(task)

        # Task should be marked failed
        status = orchestrator.task_store.get_status(task.id)
        assert status == "failed"
