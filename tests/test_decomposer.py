"""Tests for task decomposition.

Tests [SPEC-05.11] Task Decomposition.
"""

from __future__ import annotations

import pytest

from parhelia.decomposer import (
    DecompositionResult,
    DecompositionScheduler,
    Subtask,
    SubtaskRelation,
    TaskDecomposer,
)
from parhelia.orchestrator import Task, TaskRequirements, TaskType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="task-1",
        prompt="Execute the test suite",
        task_type=TaskType.TEST_RUN,
        requirements=TaskRequirements(min_memory_gb=4),
    )


@pytest.fixture
def decomposer():
    """Create a TaskDecomposer for testing."""
    return TaskDecomposer()


# =============================================================================
# Subtask Tests
# =============================================================================


class TestSubtask:
    """Tests for Subtask dataclass."""

    def test_creation(self):
        """Subtask MUST track decomposition details."""
        subtask = Subtask(
            id="task-1-sub-0",
            parent_id="task-1",
            prompt="Run unit tests",
            task_type=TaskType.TEST_RUN,
            requirements=TaskRequirements(),
            order=0,
        )

        assert subtask.id == "task-1-sub-0"
        assert subtask.parent_id == "task-1"
        assert subtask.order == 0
        assert subtask.depends_on == []

    def test_to_task(self):
        """Subtask MUST convert to Task for execution."""
        subtask = Subtask(
            id="task-1-sub-0",
            parent_id="task-1",
            prompt="Run unit tests",
            task_type=TaskType.TEST_RUN,
            requirements=TaskRequirements(min_memory_gb=8),
            order=1,
            depends_on=["task-1-sub-prev"],
        )

        task = subtask.to_task()

        assert task.id == subtask.id
        assert task.prompt == subtask.prompt
        assert task.parent_id == "task-1"
        assert task.metadata["order"] == 1
        assert "task-1-sub-prev" in task.metadata["depends_on"]


# =============================================================================
# DecompositionResult Tests
# =============================================================================


class TestDecompositionResult:
    """Tests for DecompositionResult dataclass."""

    def test_creation(self):
        """DecompositionResult MUST capture decomposition output."""
        subtasks = [
            Subtask(
                id="task-1-sub-0",
                parent_id="task-1",
                prompt="Part 1",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
            Subtask(
                id="task-1-sub-1",
                parent_id="task-1",
                prompt="Part 2",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
        ]

        result = DecompositionResult(
            task_id="task-1",
            subtasks=subtasks,
            is_decomposed=True,
            strategy="parallel",
            reasoning="Split into parallel parts",
        )

        assert result.task_id == "task-1"
        assert result.subtask_count == 2
        assert result.is_decomposed
        assert result.strategy == "parallel"

    def test_get_independent_subtasks(self):
        """get_independent_subtasks MUST return tasks with no dependencies."""
        subtasks = [
            Subtask(
                id="sub-0",
                parent_id="task-1",
                prompt="Independent",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
            Subtask(
                id="sub-1",
                parent_id="task-1",
                prompt="Dependent",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
                depends_on=["sub-0"],
            ),
        ]

        result = DecompositionResult(
            task_id="task-1",
            subtasks=subtasks,
            is_decomposed=True,
            strategy="sequential",
        )

        independent = result.get_independent_subtasks()

        assert len(independent) == 1
        assert independent[0].id == "sub-0"

    def test_get_dependent_subtasks(self):
        """get_dependent_subtasks MUST return tasks with satisfied deps."""
        subtasks = [
            Subtask(
                id="sub-0",
                parent_id="task-1",
                prompt="First",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
            Subtask(
                id="sub-1",
                parent_id="task-1",
                prompt="Second",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
                depends_on=["sub-0"],
            ),
            Subtask(
                id="sub-2",
                parent_id="task-1",
                prompt="Third",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
                depends_on=["sub-1"],
            ),
        ]

        result = DecompositionResult(
            task_id="task-1",
            subtasks=subtasks,
            is_decomposed=True,
            strategy="sequential",
        )

        # With sub-0 complete, sub-1 should be ready
        ready = result.get_dependent_subtasks({"sub-0"})
        assert len(ready) == 1
        assert ready[0].id == "sub-1"

        # With sub-0 and sub-1 complete, sub-2 should be ready
        ready = result.get_dependent_subtasks({"sub-0", "sub-1"})
        assert len(ready) == 1
        assert ready[0].id == "sub-2"


# =============================================================================
# TaskDecomposer Tests
# =============================================================================


class TestTaskDecomposer:
    """Tests for TaskDecomposer class."""

    def test_initialization_defaults(self, decomposer):
        """TaskDecomposer MUST use default configuration."""
        assert decomposer.max_subtasks == 10
        assert decomposer.min_subtask_size == 1

    def test_initialization_custom(self):
        """TaskDecomposer MUST accept custom configuration."""
        decomposer = TaskDecomposer(max_subtasks=5, min_subtask_size=2)

        assert decomposer.max_subtasks == 5
        assert decomposer.min_subtask_size == 2

    @pytest.mark.asyncio
    async def test_decompose_test_run(self, decomposer, sample_task):
        """decompose MUST split test tasks by category."""
        result = await decomposer.decompose(sample_task)

        assert result.is_decomposed
        assert result.strategy == "parallel"
        assert result.subtask_count >= 2

        # Check test categories
        categories = {s.metadata.get("test_category") for s in result.subtasks}
        assert "unit" in categories
        assert "integration" in categories

    @pytest.mark.asyncio
    async def test_decompose_lint(self, decomposer):
        """decompose MUST split lint tasks by linter."""
        task = Task(
            id="lint-1",
            prompt="Lint the codebase",
            task_type=TaskType.LINT,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert result.is_decomposed
        assert result.strategy == "parallel"

        linters = {s.metadata.get("linter") for s in result.subtasks}
        assert "type-check" in linters
        assert "style" in linters

    @pytest.mark.asyncio
    async def test_decompose_build(self, decomposer):
        """decompose MUST split build tasks sequentially."""
        task = Task(
            id="build-1",
            prompt="Build the project",
            task_type=TaskType.BUILD,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert result.is_decomposed
        assert result.strategy == "sequential"

        # Check build steps have dependencies
        for i, subtask in enumerate(result.subtasks):
            if i > 0:
                assert len(subtask.depends_on) > 0

    @pytest.mark.asyncio
    async def test_decompose_interactive_stays_single(self, decomposer):
        """decompose MUST NOT split interactive tasks."""
        task = Task(
            id="interactive-1",
            prompt="Help me debug this",
            task_type=TaskType.INTERACTIVE,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert not result.is_decomposed
        assert result.strategy == "single"
        assert result.subtask_count == 1

    @pytest.mark.asyncio
    async def test_decompose_parallel_pattern(self, decomposer):
        """decompose MUST detect parallel patterns in prompt."""
        task = Task(
            id="task-1",
            prompt="Process all files in the directory",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert result.is_decomposed
        assert result.strategy == "parallel"

    @pytest.mark.asyncio
    async def test_decompose_sequential_pattern(self, decomposer):
        """decompose MUST detect sequential patterns in prompt."""
        task = Task(
            id="task-1",
            prompt="First compile the code, then run the tests",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert result.is_decomposed
        assert result.strategy == "sequential"

    @pytest.mark.asyncio
    async def test_decompose_respects_max_subtasks(self):
        """decompose MUST limit number of subtasks."""
        decomposer = TaskDecomposer(max_subtasks=2)

        task = Task(
            id="task-1",
            prompt="Run all tests",
            task_type=TaskType.TEST_RUN,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert result.subtask_count <= 2

    @pytest.mark.asyncio
    async def test_can_decompose_returns_false_for_interactive(self, decomposer):
        """can_decompose MUST return False for interactive tasks."""
        task = Task(
            id="task-1",
            prompt="Interactive task",
            task_type=TaskType.INTERACTIVE,
            requirements=TaskRequirements(),
        )

        assert not await decomposer.can_decompose(task)

    @pytest.mark.asyncio
    async def test_can_decompose_respects_no_decompose_flag(self, decomposer):
        """can_decompose MUST respect no_decompose metadata."""
        task = Task(
            id="task-1",
            prompt="Test task",
            task_type=TaskType.TEST_RUN,
            requirements=TaskRequirements(),
            metadata={"no_decompose": True},
        )

        assert not await decomposer.can_decompose(task)

    def test_create_subtask(self, decomposer, sample_task):
        """create_subtask MUST create subtask from parent."""
        subtask = decomposer.create_subtask(
            parent_task=sample_task,
            prompt="Run unit tests only",
            order=0,
        )

        assert subtask.parent_id == sample_task.id
        assert subtask.prompt == "Run unit tests only"
        assert subtask.task_type == sample_task.task_type
        assert subtask.id.startswith(f"{sample_task.id}-sub-")

    def test_create_subtask_with_overrides(self, decomposer, sample_task):
        """create_subtask MUST accept override values."""
        custom_reqs = TaskRequirements(needs_gpu=True, min_memory_gb=16)

        subtask = decomposer.create_subtask(
            parent_task=sample_task,
            prompt="GPU task",
            task_type=TaskType.BUILD,
            requirements=custom_reqs,
            order=1,
            depends_on=["prev-sub"],
        )

        assert subtask.task_type == TaskType.BUILD
        assert subtask.requirements.needs_gpu
        assert subtask.requirements.min_memory_gb == 16
        assert subtask.order == 1
        assert "prev-sub" in subtask.depends_on

    @pytest.mark.asyncio
    async def test_custom_analyzer(self):
        """TaskDecomposer MUST use custom analyzer."""
        async def custom_analyzer(task: Task) -> DecompositionResult:
            return DecompositionResult(
                task_id=task.id,
                subtasks=[],
                is_decomposed=False,
                strategy="custom",
                reasoning="Custom analyzer",
            )

        decomposer = TaskDecomposer(analyzer=custom_analyzer)
        task = Task(
            id="task-1",
            prompt="Test",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )

        result = await decomposer.decompose(task)

        assert result.strategy == "custom"


# =============================================================================
# DecompositionScheduler Tests
# =============================================================================


class TestDecompositionScheduler:
    """Tests for DecompositionScheduler class."""

    @pytest.fixture
    def sequential_result(self):
        """Create a sequential decomposition result."""
        subtasks = [
            Subtask(
                id="sub-0",
                parent_id="task-1",
                prompt="Step 1",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
                order=0,
            ),
            Subtask(
                id="sub-1",
                parent_id="task-1",
                prompt="Step 2",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
                order=1,
                depends_on=["sub-0"],
            ),
            Subtask(
                id="sub-2",
                parent_id="task-1",
                prompt="Step 3",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
                order=2,
                depends_on=["sub-1"],
            ),
        ]

        return DecompositionResult(
            task_id="task-1",
            subtasks=subtasks,
            is_decomposed=True,
            strategy="sequential",
        )

    @pytest.fixture
    def parallel_result(self):
        """Create a parallel decomposition result."""
        subtasks = [
            Subtask(
                id="sub-0",
                parent_id="task-1",
                prompt="Part 1",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
            Subtask(
                id="sub-1",
                parent_id="task-1",
                prompt="Part 2",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
            Subtask(
                id="sub-2",
                parent_id="task-1",
                prompt="Part 3",
                task_type=TaskType.GENERIC,
                requirements=TaskRequirements(),
            ),
        ]

        return DecompositionResult(
            task_id="task-1",
            subtasks=subtasks,
            is_decomposed=True,
            strategy="parallel",
        )

    def test_get_ready_subtasks_parallel(self, parallel_result):
        """get_ready_subtasks MUST return all for parallel tasks."""
        scheduler = DecompositionScheduler(parallel_result)

        ready = scheduler.get_ready_subtasks()

        assert len(ready) == 3

    def test_get_ready_subtasks_sequential(self, sequential_result):
        """get_ready_subtasks MUST respect dependencies."""
        scheduler = DecompositionScheduler(sequential_result)

        # Initially only first is ready
        ready = scheduler.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == "sub-0"

        # After first completes, second is ready
        scheduler.mark_completed("sub-0")
        ready = scheduler.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == "sub-1"

    def test_mark_running(self, parallel_result):
        """mark_running MUST prevent task from appearing in ready."""
        scheduler = DecompositionScheduler(parallel_result)

        scheduler.mark_running("sub-0")
        ready = scheduler.get_ready_subtasks()

        assert len(ready) == 2
        assert all(s.id != "sub-0" for s in ready)

    def test_mark_completed(self, sequential_result):
        """mark_completed MUST unblock dependent tasks."""
        scheduler = DecompositionScheduler(sequential_result)

        scheduler.mark_running("sub-0")
        scheduler.mark_completed("sub-0")

        ready = scheduler.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == "sub-1"

    def test_mark_failed(self, sequential_result):
        """mark_failed MUST prevent dependent tasks from running."""
        scheduler = DecompositionScheduler(sequential_result)

        scheduler.mark_failed("sub-0")

        # sub-1 depends on sub-0, so it's blocked
        ready = scheduler.get_ready_subtasks()
        assert len(ready) == 0

    def test_is_complete(self, parallel_result):
        """is_complete MUST return True when all done."""
        scheduler = DecompositionScheduler(parallel_result)

        assert not scheduler.is_complete

        scheduler.mark_completed("sub-0")
        scheduler.mark_completed("sub-1")
        assert not scheduler.is_complete

        scheduler.mark_completed("sub-2")
        assert scheduler.is_complete

    def test_is_complete_with_failures(self, parallel_result):
        """is_complete MUST be True even with failures."""
        scheduler = DecompositionScheduler(parallel_result)

        scheduler.mark_completed("sub-0")
        scheduler.mark_failed("sub-1")
        scheduler.mark_completed("sub-2")

        assert scheduler.is_complete
        assert not scheduler.all_succeeded
        assert scheduler.has_failures

    def test_all_succeeded(self, parallel_result):
        """all_succeeded MUST be True only if all completed."""
        scheduler = DecompositionScheduler(parallel_result)

        scheduler.mark_completed("sub-0")
        scheduler.mark_completed("sub-1")
        scheduler.mark_completed("sub-2")

        assert scheduler.all_succeeded

    def test_get_blocked_subtasks(self, sequential_result):
        """get_blocked_subtasks MUST return tasks with failed deps."""
        scheduler = DecompositionScheduler(sequential_result)

        scheduler.mark_failed("sub-0")

        blocked = scheduler.get_blocked_subtasks()
        assert len(blocked) == 2
        blocked_ids = {s.id for s in blocked}
        assert "sub-1" in blocked_ids
        assert "sub-2" in blocked_ids


# =============================================================================
# SubtaskRelation Tests
# =============================================================================


class TestSubtaskRelation:
    """Tests for SubtaskRelation enum."""

    def test_relation_values(self):
        """SubtaskRelation MUST define all relation types."""
        assert SubtaskRelation.INDEPENDENT.value == "independent"
        assert SubtaskRelation.SEQUENTIAL.value == "sequential"
        assert SubtaskRelation.DEPENDS_ON.value == "depends_on"
