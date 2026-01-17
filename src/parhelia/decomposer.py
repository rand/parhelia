"""Task decomposition for parallel execution.

Implements:
- [SPEC-05.11] Task Decomposition
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
import uuid

from parhelia.orchestrator import Task, TaskType, TaskRequirements


class SubtaskRelation(Enum):
    """Relationship between subtasks."""

    INDEPENDENT = "independent"  # Can run in parallel
    SEQUENTIAL = "sequential"  # Must run in order
    DEPENDS_ON = "depends_on"  # Depends on specific subtask


@dataclass
class Subtask:
    """A subtask derived from task decomposition.

    Implements [SPEC-05.11].
    """

    id: str
    parent_id: str
    prompt: str
    task_type: TaskType
    requirements: TaskRequirements
    order: int = 0  # Execution order for sequential tasks
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_task(self) -> Task:
        """Convert subtask to a Task for execution.

        Returns:
            Task instance ready for dispatch.
        """
        return Task(
            id=self.id,
            prompt=self.prompt,
            task_type=self.task_type,
            requirements=self.requirements,
            parent_id=self.parent_id,
            metadata={
                **self.metadata,
                "order": self.order,
                "depends_on": self.depends_on,
            },
        )


@dataclass
class DecompositionResult:
    """Result of task decomposition.

    Implements [SPEC-05.11].
    """

    task_id: str
    subtasks: list[Subtask]
    is_decomposed: bool
    strategy: str  # e.g., "parallel", "sequential", "single"
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def subtask_count(self) -> int:
        """Get number of subtasks."""
        return len(self.subtasks)

    def get_independent_subtasks(self) -> list[Subtask]:
        """Get subtasks that can run immediately (no dependencies)."""
        return [s for s in self.subtasks if not s.depends_on]

    def get_dependent_subtasks(self, completed_ids: set[str]) -> list[Subtask]:
        """Get subtasks whose dependencies are satisfied.

        Args:
            completed_ids: Set of completed subtask IDs.

        Returns:
            List of subtasks ready to run (excluding already completed).
        """
        result = []
        for subtask in self.subtasks:
            # Skip already completed subtasks
            if subtask.id in completed_ids:
                continue
            if subtask.depends_on:
                if all(dep_id in completed_ids for dep_id in subtask.depends_on):
                    result.append(subtask)
        return result


# Type for decomposition analyzers
DecompositionAnalyzer = Callable[[Task], Awaitable[DecompositionResult]]


class TaskDecomposer:
    """Decompose complex tasks into parallelizable subtasks.

    Implements [SPEC-05.11].

    The task decomposer uses a flat dispatch model where:
    - Only the orchestrator decomposes tasks
    - Subagents cannot spawn additional subtasks
    - Decomposition happens before dispatch
    """

    # Task types that are typically parallelizable
    PARALLELIZABLE_TYPES = {
        TaskType.TEST_RUN,
        TaskType.LINT,
        TaskType.BUILD,
    }

    # Task types that should remain single
    SINGLE_TASK_TYPES = {
        TaskType.INTERACTIVE,
    }

    def __init__(
        self,
        analyzer: DecompositionAnalyzer | None = None,
        max_subtasks: int = 10,
        min_subtask_size: int = 1,
    ):
        """Initialize the task decomposer.

        Args:
            analyzer: Custom analyzer function for decomposition.
            max_subtasks: Maximum number of subtasks to create.
            min_subtask_size: Minimum size threshold for creating subtasks.
        """
        self.analyzer = analyzer or self._default_analyzer
        self.max_subtasks = max_subtasks
        self.min_subtask_size = min_subtask_size

    async def decompose(self, task: Task) -> DecompositionResult:
        """Decompose a task into subtasks.

        Implements [SPEC-05.11].

        Args:
            task: The task to decompose.

        Returns:
            DecompositionResult with subtasks.
        """
        # Check if task should remain single
        if task.task_type in self.SINGLE_TASK_TYPES:
            return self._create_single_result(task, "Task type requires single execution")

        # Use analyzer for decomposition
        result = await self.analyzer(task)

        # Limit subtasks
        if result.subtask_count > self.max_subtasks:
            result.subtasks = result.subtasks[: self.max_subtasks]

        return result

    async def can_decompose(self, task: Task) -> bool:
        """Check if a task can be decomposed.

        Args:
            task: The task to check.

        Returns:
            True if task can be decomposed.
        """
        if task.task_type in self.SINGLE_TASK_TYPES:
            return False

        # Tasks with explicit "do not parallelize" flag
        if task.metadata.get("no_decompose"):
            return False

        return True

    def create_subtask(
        self,
        parent_task: Task,
        prompt: str,
        task_type: TaskType | None = None,
        order: int = 0,
        depends_on: list[str] | None = None,
        requirements: TaskRequirements | None = None,
    ) -> Subtask:
        """Create a subtask from a parent task.

        Args:
            parent_task: The parent task.
            prompt: Subtask prompt.
            task_type: Override task type.
            order: Execution order.
            depends_on: List of subtask IDs this depends on.
            requirements: Override requirements.

        Returns:
            New Subtask instance.
        """
        subtask_id = f"{parent_task.id}-sub-{uuid.uuid4().hex[:8]}"

        return Subtask(
            id=subtask_id,
            parent_id=parent_task.id,
            prompt=prompt,
            task_type=task_type or parent_task.task_type,
            requirements=requirements or parent_task.requirements,
            order=order,
            depends_on=depends_on or [],
        )

    def _create_single_result(self, task: Task, reasoning: str) -> DecompositionResult:
        """Create a result with the task as a single subtask."""
        subtask = Subtask(
            id=f"{task.id}-sub-0",
            parent_id=task.id,
            prompt=task.prompt,
            task_type=task.task_type,
            requirements=task.requirements,
            order=0,
        )

        return DecompositionResult(
            task_id=task.id,
            subtasks=[subtask],
            is_decomposed=False,
            strategy="single",
            reasoning=reasoning,
        )

    async def _default_analyzer(self, task: Task) -> DecompositionResult:
        """Default decomposition analyzer using heuristics.

        This analyzer uses pattern matching and task type heuristics.
        For production use, replace with Claude-based analysis.

        Args:
            task: The task to analyze.

        Returns:
            DecompositionResult with subtasks.
        """
        prompt = task.prompt.lower()

        # Check for explicit parallel indicators
        parallel_patterns = [
            r"for each (\w+)",
            r"all (\w+)s",
            r"every (\w+)",
            r"multiple (\w+)s",
            r"(\d+) files",
            r"(\d+) tests",
            r"(\d+) modules",
        ]

        for pattern in parallel_patterns:
            match = re.search(pattern, prompt)
            if match:
                return await self._decompose_parallel(task, match.group(1))

        # Check for sequential indicators
        sequential_patterns = [
            r"first.*then",
            r"step (\d+)",
            r"after.*do",
            r"once.*complete",
        ]

        for pattern in sequential_patterns:
            if re.search(pattern, prompt):
                return await self._decompose_sequential(task)

        # Check task type for parallelization hints
        if task.task_type in self.PARALLELIZABLE_TYPES:
            return await self._decompose_by_type(task)

        # Default: single task
        return self._create_single_result(task, "No decomposition pattern detected")

    async def _decompose_parallel(
        self, task: Task, target: str
    ) -> DecompositionResult:
        """Decompose task for parallel execution."""
        # In production, this would use Claude to identify actual targets
        # For now, create placeholder subtasks
        subtasks = []

        # Create a few parallel subtasks
        for i in range(min(3, self.max_subtasks)):
            subtask = self.create_subtask(
                parent_task=task,
                prompt=f"Process {target} {i + 1}: {task.prompt}",
                order=0,  # All same order = parallel
            )
            subtasks.append(subtask)

        return DecompositionResult(
            task_id=task.id,
            subtasks=subtasks,
            is_decomposed=True,
            strategy="parallel",
            reasoning=f"Detected parallel pattern for '{target}'",
        )

    async def _decompose_sequential(self, task: Task) -> DecompositionResult:
        """Decompose task for sequential execution."""
        subtasks = []

        # Create sequential subtasks with dependencies
        prev_id = None
        for i in range(min(3, self.max_subtasks)):
            subtask = self.create_subtask(
                parent_task=task,
                prompt=f"Step {i + 1}: {task.prompt}",
                order=i,
                depends_on=[prev_id] if prev_id else [],
            )
            subtasks.append(subtask)
            prev_id = subtask.id

        return DecompositionResult(
            task_id=task.id,
            subtasks=subtasks,
            is_decomposed=True,
            strategy="sequential",
            reasoning="Detected sequential execution pattern",
        )

    async def _decompose_by_type(self, task: Task) -> DecompositionResult:
        """Decompose based on task type."""
        if task.task_type == TaskType.TEST_RUN:
            return await self._decompose_test_run(task)
        elif task.task_type == TaskType.LINT:
            return await self._decompose_lint(task)
        elif task.task_type == TaskType.BUILD:
            return await self._decompose_build(task)

        return self._create_single_result(task, f"No decomposition for type {task.task_type}")

    async def _decompose_test_run(self, task: Task) -> DecompositionResult:
        """Decompose test run task."""
        subtasks = []

        # Split tests into unit, integration, e2e
        test_categories = ["unit", "integration", "e2e"]

        for i, category in enumerate(test_categories):
            subtask = self.create_subtask(
                parent_task=task,
                prompt=f"Run {category} tests: {task.prompt}",
                order=0,  # Can run in parallel
                requirements=TaskRequirements(
                    needs_gpu=task.requirements.needs_gpu,
                    min_memory_gb=task.requirements.min_memory_gb,
                ),
            )
            subtask.metadata["test_category"] = category
            subtasks.append(subtask)

        return DecompositionResult(
            task_id=task.id,
            subtasks=subtasks,
            is_decomposed=True,
            strategy="parallel",
            reasoning="Decomposed test run by category",
        )

    async def _decompose_lint(self, task: Task) -> DecompositionResult:
        """Decompose lint task."""
        subtasks = []

        # Run different linters in parallel
        linters = ["type-check", "style", "security"]

        for linter in linters:
            subtask = self.create_subtask(
                parent_task=task,
                prompt=f"Run {linter} linting: {task.prompt}",
                order=0,
            )
            subtask.metadata["linter"] = linter
            subtasks.append(subtask)

        return DecompositionResult(
            task_id=task.id,
            subtasks=subtasks,
            is_decomposed=True,
            strategy="parallel",
            reasoning="Decomposed linting by tool",
        )

    async def _decompose_build(self, task: Task) -> DecompositionResult:
        """Decompose build task."""
        # Builds typically have dependencies, so sequential
        subtasks = []

        build_steps = ["compile", "bundle", "optimize"]
        prev_id = None

        for i, step in enumerate(build_steps):
            subtask = self.create_subtask(
                parent_task=task,
                prompt=f"Build step - {step}: {task.prompt}",
                order=i,
                depends_on=[prev_id] if prev_id else [],
            )
            subtask.metadata["build_step"] = step
            subtasks.append(subtask)
            prev_id = subtask.id

        return DecompositionResult(
            task_id=task.id,
            subtasks=subtasks,
            is_decomposed=True,
            strategy="sequential",
            reasoning="Decomposed build into sequential steps",
        )


class DecompositionScheduler:
    """Schedule subtask execution based on dependencies.

    Implements [SPEC-05.11].
    """

    def __init__(self, result: DecompositionResult):
        """Initialize scheduler with decomposition result.

        Args:
            result: The decomposition result to schedule.
        """
        self.result = result
        self._completed: set[str] = set()
        self._running: set[str] = set()
        self._failed: set[str] = set()

    def get_ready_subtasks(self) -> list[Subtask]:
        """Get subtasks that are ready to execute.

        Returns:
            List of subtasks with satisfied dependencies.
        """
        ready = []
        for subtask in self.result.subtasks:
            if subtask.id in self._completed:
                continue
            if subtask.id in self._running:
                continue
            if subtask.id in self._failed:
                continue

            # Check dependencies
            if all(dep in self._completed for dep in subtask.depends_on):
                ready.append(subtask)

        return ready

    def mark_running(self, subtask_id: str) -> None:
        """Mark a subtask as running.

        Args:
            subtask_id: The subtask ID.
        """
        self._running.add(subtask_id)

    def mark_completed(self, subtask_id: str) -> None:
        """Mark a subtask as completed.

        Args:
            subtask_id: The subtask ID.
        """
        self._running.discard(subtask_id)
        self._completed.add(subtask_id)

    def mark_failed(self, subtask_id: str) -> None:
        """Mark a subtask as failed.

        Args:
            subtask_id: The subtask ID.
        """
        self._running.discard(subtask_id)
        self._failed.add(subtask_id)

    @property
    def is_complete(self) -> bool:
        """Check if all subtasks are done (completed or failed)."""
        done = self._completed | self._failed
        return len(done) == len(self.result.subtasks)

    @property
    def all_succeeded(self) -> bool:
        """Check if all subtasks completed successfully."""
        return len(self._completed) == len(self.result.subtasks)

    @property
    def has_failures(self) -> bool:
        """Check if any subtasks failed."""
        return len(self._failed) > 0

    def get_blocked_subtasks(self) -> list[Subtask]:
        """Get subtasks blocked by failed dependencies (including transitive)."""
        # First, compute all transitively blocked subtask IDs
        blocked_ids = set(self._failed)
        changed = True
        while changed:
            changed = False
            for subtask in self.result.subtasks:
                if subtask.id in blocked_ids:
                    continue
                # If any dependency is blocked/failed, this subtask is blocked
                if any(dep in blocked_ids for dep in subtask.depends_on):
                    blocked_ids.add(subtask.id)
                    changed = True

        # Return subtasks that are blocked but not directly failed
        blocked = []
        for subtask in self.result.subtasks:
            if subtask.id in self._completed:
                continue
            if subtask.id in self._failed:
                continue
            if subtask.id in blocked_ids:
                blocked.append(subtask)

        return blocked
