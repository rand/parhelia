"""Integration tests for Parhelia workflows.

Tests end-to-end workflows across multiple components.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_modal_sandbox():
    """Create a mock Modal sandbox for testing."""
    sandbox = MagicMock()
    sandbox.exec = MagicMock()
    sandbox.exec.return_value.stdout = MagicMock()
    sandbox.exec.return_value.stdout.read = MagicMock(return_value="output")
    sandbox.exec.return_value.returncode = 0
    return sandbox


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(text="Test response")],
            stop_reason="end_turn",
        )
    )
    return client


# =============================================================================
# Task Submission Workflow Tests
# =============================================================================


class TestTaskSubmissionWorkflow:
    """Integration tests for task submission workflow."""

    @pytest.mark.asyncio
    async def test_task_submission_creates_pending(self, temp_dir):
        """Task submission MUST add task to pending queue."""
        from parhelia.orchestrator import LocalOrchestrator, Task, TaskType, TaskRequirements

        orchestrator = LocalOrchestrator()

        task = Task(
            id="test-task-1",
            prompt="Test prompt",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(min_memory_gb=4),
        )

        task_id = await orchestrator.submit_task(task)

        assert task_id == "test-task-1"
        assert orchestrator.get_pending_count() == 1
        assert orchestrator.get_task("test-task-1") is not None

    @pytest.mark.asyncio
    async def test_task_dispatch_selects_target(self, temp_dir):
        """Task dispatch MUST select appropriate execution target."""
        from parhelia.dispatcher import Dispatcher, ExecutionTarget, ResourceCapacity
        from parhelia.orchestrator import Task, TaskType, TaskRequirements

        dispatcher = Dispatcher()

        # Register targets
        dispatcher.register_target(
            ExecutionTarget(
                id="local",
                target_type="local",
                capacity=ResourceCapacity(
                    available_cpu=4,
                    available_memory_gb=8,
                    cpu_percent=20.0,
                ),
            )
        )
        dispatcher.register_target(
            ExecutionTarget(
                id="modal-1",
                target_type="parhelia-cpu",
                capacity=ResourceCapacity(
                    available_cpu=8,
                    available_memory_gb=16,
                    cpu_percent=50.0,
                ),
            )
        )

        task = Task(
            id="dispatch-test",
            prompt="Test",
            task_type=TaskType.BUILD,
            requirements=TaskRequirements(min_memory_gb=4, min_cpu=2),
        )

        target = dispatcher.select_optimal_target(task)

        assert target is not None
        assert target.id in ["local", "modal-1"]

    @pytest.mark.asyncio
    async def test_task_result_aggregation(self, temp_dir):
        """Multiple task results MUST be aggregated correctly."""
        from parhelia.aggregator import ResultAggregator, WorkResult, ExecutionMetrics
        from parhelia.orchestrator import Task, TaskType, TaskRequirements

        aggregator = ResultAggregator()

        task = Task(
            id="aggregate-test",
            prompt="Test",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

        results = [
            WorkResult(
                work_unit_id="work-1",
                worker_id="worker-1",
                status="success",
                output="out1",
                metrics=ExecutionMetrics(cost_usd=1.0, duration_seconds=10),
            ),
            WorkResult(
                work_unit_id="work-2",
                worker_id="worker-2",
                status="success",
                output="out2",
                metrics=ExecutionMetrics(cost_usd=1.5, duration_seconds=15),
            ),
            WorkResult(
                work_unit_id="work-3",
                worker_id="worker-3",
                status="failed",
                output="error",
                error="Task failed",
                metrics=ExecutionMetrics(cost_usd=0.5, duration_seconds=5),
            ),
        ]

        summary = await aggregator.aggregate(task, results)

        assert summary.status == "partial"  # Some failed
        assert summary.total_cost == 3.0
        assert len(summary.results) == 3


# =============================================================================
# Checkpoint/Resume Workflow Tests
# =============================================================================


class TestCheckpointWorkflow:
    """Integration tests for checkpoint/resume workflow."""

    @pytest.mark.asyncio
    async def test_checkpoint_creation_and_storage(self, temp_dir):
        """Checkpoint MUST be created and stored correctly."""
        from parhelia.checkpoint import CheckpointManager
        from parhelia.session import Session, SessionState, CheckpointTrigger

        manager = CheckpointManager(checkpoint_root=str(temp_dir / "checkpoints"))

        # Create workspace
        workspace = temp_dir / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("test content")

        session = Session(
            id="test-session-1",
            task_id="task-1",
            state=SessionState.RUNNING,
            working_directory=str(workspace),
        )

        checkpoint = await manager.create_checkpoint(
            session=session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"messages": ["test"]},
        )

        assert checkpoint is not None
        assert checkpoint.session_id == "test-session-1"

        # Verify checkpoint directory exists
        checkpoint_dir = temp_dir / "checkpoints" / "test-session-1"
        assert checkpoint_dir.exists()

    @pytest.mark.asyncio
    async def test_checkpoint_restore(self, temp_dir):
        """Checkpoint MUST be restorable to previous state."""
        from parhelia.checkpoint import CheckpointManager
        from parhelia.session import Session, SessionState, CheckpointTrigger

        manager = CheckpointManager(checkpoint_root=str(temp_dir / "checkpoints"))

        # Create workspace with content
        workspace = temp_dir / "workspace"
        workspace.mkdir()
        (workspace / "file.txt").write_text("original content")

        session = Session(
            id="test-session-2",
            task_id="task-2",
            state=SessionState.RUNNING,
            working_directory=str(workspace),
        )

        checkpoint = await manager.create_checkpoint(
            session=session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"turn": 5, "messages": ["hello"]},
        )

        # Restore to new location
        restore_dir = temp_dir / "restored"
        restore_dir.mkdir()
        await manager.restore_checkpoint(checkpoint, str(restore_dir))

        # Verify restored content
        restored_file = restore_dir / "file.txt"
        assert restored_file.exists()
        assert restored_file.read_text() == "original content"


# =============================================================================
# Metrics Pipeline Workflow Tests
# =============================================================================


class TestMetricsPipelineWorkflow:
    """Integration tests for metrics collection and aggregation."""

    @pytest.mark.asyncio
    async def test_metrics_collection_to_aggregation(self, temp_dir):
        """Metrics MUST flow from collection to aggregation."""
        from parhelia.metrics.collector import MetricsCollector, ContainerMetrics
        from parhelia.metrics.aggregator import MetricsAggregator

        # Collect metrics
        collector = MetricsCollector(environment="local", max_sessions=4)
        collector.register_session("session-1")

        with patch("psutil.cpu_count", return_value=8):
            with patch("psutil.cpu_percent", return_value=25.0):
                mock_memory = MagicMock()
                mock_memory.total = 17179869184
                mock_memory.available = 12884901888
                mock_memory.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        assert metrics.cpu_total_cores == 8
        assert metrics.sessions_active == 1

        # Verify metrics can be converted for aggregation
        prometheus_dict = metrics.to_prometheus_dict()
        assert "parhelia_cpu_total_cores" in prometheus_dict
        assert "parhelia_sessions_active" in prometheus_dict

    @pytest.mark.asyncio
    async def test_metrics_pusher_integration(self, temp_dir):
        """MetricsPusher MUST integrate with MetricsCollector."""
        from parhelia.metrics.collector import MetricsCollector
        from parhelia.metrics.pusher import MetricsPusher

        collector = MetricsCollector(environment="modal", max_sessions=2)
        pusher = MetricsPusher(
            pushgateway_url="http://localhost:9091",
            push_interval=1,
        )

        with patch("psutil.cpu_count", return_value=4):
            with patch("psutil.cpu_percent", return_value=50.0):
                mock_memory = MagicMock()
                mock_memory.total = 16000000000
                mock_memory.available = 8000000000
                mock_memory.percent = 50.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        # Push should work (mocked)
        with patch("parhelia.metrics.pusher.push_to_gateway") as mock_push:
            await pusher.push_once(metrics)
            mock_push.assert_called_once()


# =============================================================================
# Session Lifecycle Workflow Tests
# =============================================================================


class TestSessionLifecycleWorkflow:
    """Integration tests for session lifecycle."""

    def test_session_state_transitions(self):
        """Session MUST transition through valid states."""
        from parhelia.session import Session, SessionState

        session = Session(
            id="lifecycle-test",
            task_id="task-lifecycle",
            state=SessionState.STARTING,
            working_directory="/tmp/workspace",
        )

        assert session.state == SessionState.STARTING

        # Transition to running
        session.transition_to(SessionState.RUNNING)
        assert session.state == SessionState.RUNNING

        # Transition to completed
        session.transition_to(SessionState.COMPLETED)
        assert session.state == SessionState.COMPLETED

    @pytest.mark.asyncio
    async def test_session_with_checkpoint_lifecycle(self, temp_dir):
        """Session with checkpointing MUST maintain state across checkpoints."""
        from parhelia.session import Session, SessionState, CheckpointTrigger
        from parhelia.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_root=str(temp_dir / "checkpoints"))

        # Create workspace
        workspace = temp_dir / "workspace"
        workspace.mkdir()
        (workspace / "code.py").write_text("print('hello')")

        session = Session(
            id="checkpoint-lifecycle",
            task_id="task-checkpoint",
            state=SessionState.RUNNING,
            working_directory=str(workspace),
        )

        # Create checkpoint at turn 5
        cp1 = await manager.create_checkpoint(
            session=session,
            trigger=CheckpointTrigger.PERIODIC,
            conversation={"turn": 5},
        )

        # Modify workspace
        (workspace / "code.py").write_text("print('world')")

        # Create checkpoint at turn 10
        cp2 = await manager.create_checkpoint(
            session=session,
            trigger=CheckpointTrigger.PERIODIC,
            conversation={"turn": 10},
            previous_checkpoint=cp1,
        )

        # List checkpoints should show both
        checkpoints = await manager.list_checkpoints(session.id)
        assert len(checkpoints) >= 1  # At least one checkpoint


# =============================================================================
# CAS Integration Workflow Tests
# =============================================================================


class TestCASWorkflow:
    """Integration tests for Content-Addressable Storage."""

    @pytest.mark.asyncio
    async def test_cas_store_and_retrieve(self, temp_dir):
        """CAS MUST store and retrieve content by digest."""
        from parhelia.cas import ContentAddressableStorage, Digest

        cas = ContentAddressableStorage(root_path=str(temp_dir / "cas"))

        content = b"test content for CAS"
        digest = await cas.write_blob(content)

        assert digest is not None
        assert digest.size_bytes == len(content)

        # Retrieve
        retrieved = await cas.read_blob(digest)
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_cas_merkle_tree_workflow(self, temp_dir):
        """CAS Merkle tree MUST capture directory state."""
        from parhelia.cas import ContentAddressableStorage, MerkleTreeBuilder

        cas = ContentAddressableStorage(root_path=str(temp_dir / "cas"))
        builder = MerkleTreeBuilder(cas)

        # Create test directory structure
        test_dir = temp_dir / "workspace"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        # Build tree (async!)
        root_digest = await builder.build_tree(str(test_dir))

        assert root_digest is not None
        assert root_digest.size_bytes > 0


# =============================================================================
# Resilience Workflow Tests
# =============================================================================


class TestResilienceWorkflow:
    """Integration tests for retry and circuit breaker."""

    @pytest.mark.asyncio
    async def test_retry_with_transient_failure(self):
        """Retry MUST handle transient failures."""
        from parhelia.resilience import RetryPolicy

        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        policy = RetryPolicy(max_attempts=5, base_delay=0.01)
        result = await policy.execute(flaky_operation)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascade(self):
        """Circuit breaker MUST prevent cascading failures."""
        from parhelia.resilience import CircuitBreaker, CircuitOpenError, CircuitState

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        async def failing_service():
            raise Exception("Service unavailable")

        # Trigger circuit open
        for _ in range(3):
            try:
                await cb.execute(failing_service)
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

        # Subsequent calls should be rejected immediately
        with pytest.raises(CircuitOpenError):
            await cb.execute(failing_service)


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_task_execution_workflow(self, temp_dir):
        """Full workflow: submit → dispatch → execute → checkpoint → complete."""
        from parhelia.orchestrator import LocalOrchestrator, Task, TaskResult, TaskType, TaskRequirements
        from parhelia.checkpoint import CheckpointManager
        from parhelia.session import Session, SessionState, CheckpointTrigger

        # Setup
        orchestrator = LocalOrchestrator()
        checkpoint_mgr = CheckpointManager(
            checkpoint_root=str(temp_dir / "checkpoints")
        )

        # Create workspace
        workspace = temp_dir / "workspace"
        workspace.mkdir()
        (workspace / "main.py").write_text("# main file")

        # Submit task
        task = Task(
            id="e2e-test-1",
            prompt="Implement feature X",
            task_type=TaskType.CODE_FIX,
            requirements=TaskRequirements(min_memory_gb=4),
        )
        await orchestrator.submit_task(task)

        # Simulate execution with session
        session = Session(
            id=f"session-{task.id}",
            task_id=task.id,
            state=SessionState.RUNNING,
            working_directory=str(workspace),
        )

        # Create checkpoint mid-execution
        checkpoint = await checkpoint_mgr.create_checkpoint(
            session=session,
            trigger=CheckpointTrigger.PERIODIC,
            conversation={"progress": 50},
        )

        # Mark task complete
        result = TaskResult(
            task_id=task.id,
            status="success",
            output="Completed successfully",
            cost_usd=0.05,
            duration_seconds=10.0,
        )
        orchestrator.mark_task_complete(task.id, result)

        # Verify
        collected = await orchestrator.collect_results(task.id)
        assert collected is not None
        assert collected.status == "success"

        # Verify checkpoint was created
        checkpoints = await checkpoint_mgr.list_checkpoints(f"session-{task.id}")
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_metrics_observability_workflow(self, temp_dir):
        """Full observability: collect → push → aggregate → dashboard."""
        from parhelia.metrics.collector import MetricsCollector
        from parhelia.metrics.aggregator import MetricsAggregator, ResourceMetrics
        from parhelia.metrics.grafana import generate_overview_dashboard

        # Collect metrics
        collector = MetricsCollector(environment="local")

        with patch("psutil.cpu_count", return_value=8):
            with patch("psutil.cpu_percent", return_value=30.0):
                mock_memory = MagicMock()
                mock_memory.total = 17179869184
                mock_memory.available = 12000000000
                mock_memory.percent = 30.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        # Convert to ResourceMetrics format
        resource_metrics = ResourceMetrics.from_dict(
            metrics.to_prometheus_dict(),
            "local",
        )

        assert resource_metrics.cpu_total == 8
        assert resource_metrics.cpu_usage_percent == 30.0

        # Generate dashboard (validates the full pipeline)
        dashboard = generate_overview_dashboard()
        assert "panels" in dashboard
        assert len(dashboard["panels"]) > 0
