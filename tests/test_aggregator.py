"""Tests for result aggregation.

@trace SPEC-05.13 - Result Aggregation
"""

import pytest


class TestWorkResult:
    """Tests for WorkResult dataclass - SPEC-05.13."""

    def test_work_result_creation(self):
        """@trace SPEC-05.13 - WorkResult MUST capture execution results."""
        from parhelia.aggregator import WorkResult

        result = WorkResult(
            work_unit_id="unit-1",
            worker_id="worker-1",
            status="success",
            output="Task completed",
        )

        assert result.work_unit_id == "unit-1"
        assert result.worker_id == "worker-1"
        assert result.status == "success"
        assert result.output == "Task completed"

    def test_work_result_with_artifacts(self):
        """@trace SPEC-05.13 - WorkResult SHOULD support artifacts."""
        from parhelia.aggregator import Artifact, WorkResult

        artifact = Artifact(path="src/file.py", content="code", action="modified")
        result = WorkResult(
            work_unit_id="unit-1",
            worker_id="worker-1",
            status="success",
            output="Done",
            artifacts=[artifact],
        )

        assert len(result.artifacts) == 1
        assert result.artifacts[0].path == "src/file.py"

    def test_work_result_with_metrics(self):
        """@trace SPEC-05.13 - WorkResult SHOULD include execution metrics."""
        from parhelia.aggregator import ExecutionMetrics, WorkResult

        metrics = ExecutionMetrics(duration_seconds=45.0, cost_usd=0.12)
        result = WorkResult(
            work_unit_id="unit-1",
            worker_id="worker-1",
            status="success",
            output="Done",
            metrics=metrics,
        )

        assert result.metrics.duration_seconds == 45.0
        assert result.metrics.cost_usd == 0.12


class TestArtifact:
    """Tests for Artifact dataclass."""

    def test_artifact_creation(self):
        """Artifact MUST capture file path and content."""
        from parhelia.aggregator import Artifact

        artifact = Artifact(
            path="src/main.py",
            content="print('hello')",
            action="created",
        )

        assert artifact.path == "src/main.py"
        assert artifact.content == "print('hello')"
        assert artifact.action == "created"


class TestResultAggregator:
    """Tests for ResultAggregator class - SPEC-05.13."""

    @pytest.fixture
    def aggregator(self):
        """Create ResultAggregator instance."""
        from parhelia.aggregator import ResultAggregator

        return ResultAggregator()

    def test_aggregator_initialization(self, aggregator):
        """@trace SPEC-05.13 - ResultAggregator MUST initialize properly."""
        assert aggregator is not None

    @pytest.mark.asyncio
    async def test_aggregate_success_results(self, aggregator):
        """@trace SPEC-05.13 - MUST aggregate successful results."""
        from parhelia.aggregator import ExecutionMetrics, WorkResult
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-1",
            prompt="Fix bugs",
            task_type=TaskType.CODE_FIX,
            requirements=TaskRequirements(),
        )

        results = [
            WorkResult(
                work_unit_id="unit-1",
                worker_id="worker-1",
                status="success",
                output="Fixed file1.py",
                metrics=ExecutionMetrics(cost_usd=0.10),
            ),
            WorkResult(
                work_unit_id="unit-2",
                worker_id="worker-2",
                status="success",
                output="Fixed file2.py",
                metrics=ExecutionMetrics(cost_usd=0.15),
            ),
        ]

        aggregated = await aggregator.aggregate(task, results)

        assert aggregated.task_id == "task-1"
        assert aggregated.status == "success"
        assert aggregated.total_cost == 0.25

    @pytest.mark.asyncio
    async def test_aggregate_partial_failure(self, aggregator):
        """@trace SPEC-05.13 - MUST handle partial failures."""
        from parhelia.aggregator import WorkResult
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-1",
            prompt="Run tests",
            task_type=TaskType.TEST_RUN,
            requirements=TaskRequirements(),
        )

        results = [
            WorkResult(
                work_unit_id="unit-1",
                worker_id="worker-1",
                status="success",
                output="Tests passed",
            ),
            WorkResult(
                work_unit_id="unit-2",
                worker_id="worker-2",
                status="failed",
                output="Tests failed",
                error="AssertionError",
            ),
        ]

        aggregated = await aggregator.aggregate(task, results)

        assert aggregated.status == "partial"

    @pytest.mark.asyncio
    async def test_aggregate_all_failed(self, aggregator):
        """@trace SPEC-05.13 - MUST report failed if all failed."""
        from parhelia.aggregator import WorkResult
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-1",
            prompt="Build",
            task_type=TaskType.BUILD,
            requirements=TaskRequirements(),
        )

        results = [
            WorkResult(
                work_unit_id="unit-1",
                worker_id="worker-1",
                status="failed",
                error="Build error",
            ),
            WorkResult(
                work_unit_id="unit-2",
                worker_id="worker-2",
                status="failed",
                error="Compile error",
            ),
        ]

        aggregated = await aggregator.aggregate(task, results)

        assert aggregated.status == "failed"

    @pytest.mark.asyncio
    async def test_aggregate_merges_artifacts(self, aggregator):
        """@trace SPEC-05.13 - MUST merge artifacts from results."""
        from parhelia.aggregator import Artifact, WorkResult
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-1",
            prompt="Fix bugs",
            task_type=TaskType.CODE_FIX,
            requirements=TaskRequirements(),
        )

        results = [
            WorkResult(
                work_unit_id="unit-1",
                worker_id="worker-1",
                status="success",
                output="Done",
                artifacts=[Artifact(path="file1.py", content="code1", action="modified")],
            ),
            WorkResult(
                work_unit_id="unit-2",
                worker_id="worker-2",
                status="success",
                output="Done",
                artifacts=[Artifact(path="file2.py", content="code2", action="created")],
            ),
        ]

        aggregated = await aggregator.aggregate(task, results)

        assert len(aggregated.artifacts) == 2


class TestAggregatedResult:
    """Tests for AggregatedResult dataclass."""

    def test_aggregated_result_creation(self):
        """AggregatedResult MUST capture aggregation outcome."""
        from parhelia.aggregator import AggregatedResult

        result = AggregatedResult(
            task_id="task-1",
            status="success",
            summary="All subtasks completed",
            total_cost=0.50,
        )

        assert result.task_id == "task-1"
        assert result.status == "success"
        assert result.summary == "All subtasks completed"
        assert result.total_cost == 0.50
