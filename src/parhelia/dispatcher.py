"""Dispatch logic and target selection.

Implements:
- [SPEC-05.11] Dispatch Logic
- [SPEC-05.12] Worker Lifecycle Management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from parhelia.orchestrator import Task, TaskRequirements


@dataclass
class ResourceCapacity:
    """Available resources on an execution target.

    Implements [SPEC-05.11].
    """

    available_cpu: int = 0
    available_memory_gb: int = 0
    cpu_percent: float = 0.0
    memory_percent: float = 0.0


@dataclass
class ExecutionTarget:
    """An execution target for task dispatch.

    Implements [SPEC-05.11].
    """

    id: str
    target_type: Literal["local", "parhelia-cpu", "parhelia-gpu"]
    region: str | None = None
    gpu_type: str | None = None  # "A10G", "A100", "H100"
    capacity: ResourceCapacity = field(default_factory=ResourceCapacity)


class Dispatcher:
    """Route tasks to optimal execution targets.

    Implements [SPEC-05.11].

    The dispatcher:
    - Maintains a registry of available targets
    - Filters targets by task requirements
    - Selects optimal target based on scoring
    """

    def __init__(self):
        """Initialize the dispatcher."""
        self.targets: dict[str, ExecutionTarget] = {}

    def register_target(self, target: ExecutionTarget) -> None:
        """Register an execution target.

        Args:
            target: The target to register.
        """
        self.targets[target.id] = target

    def unregister_target(self, target_id: str) -> ExecutionTarget | None:
        """Remove a target from the registry.

        Args:
            target_id: The target ID to remove.

        Returns:
            The removed target, or None if not found.
        """
        return self.targets.pop(target_id, None)

    def get_available_targets(self) -> list[ExecutionTarget]:
        """Get all registered targets.

        Returns:
            List of all execution targets.
        """
        return list(self.targets.values())

    def meets_requirements(
        self,
        target: ExecutionTarget,
        requirements: TaskRequirements,
    ) -> bool:
        """Check if target meets task requirements.

        Args:
            target: The execution target.
            requirements: Task requirements to check.

        Returns:
            True if target can satisfy requirements.
        """
        # Check GPU requirements
        if requirements.needs_gpu:
            if target.target_type != "parhelia-gpu":
                return False
            if requirements.gpu_type and target.gpu_type != requirements.gpu_type:
                return False

        # Check memory requirements
        if target.capacity.available_memory_gb < requirements.min_memory_gb:
            return False

        # Check CPU requirements
        if target.capacity.available_cpu < requirements.min_cpu:
            return False

        return True

    def select_optimal_target(self, task: Task) -> ExecutionTarget | None:
        """Select the best target for a task.

        Args:
            task: The task to dispatch.

        Returns:
            Best matching target, or None if no eligible target.
        """
        from parhelia.orchestrator import TaskType

        # Get eligible targets
        eligible = [
            t for t in self.targets.values()
            if self.meets_requirements(t, task.requirements)
        ]

        if not eligible:
            return None

        # Score and select
        def score(target: ExecutionTarget) -> float:
            s = 0.0

            # Prefer local for interactive/small tasks
            if task.task_type == TaskType.INTERACTIVE:
                if target.target_type == "local":
                    s += 100

            # Prefer less loaded targets
            utilization = target.capacity.cpu_percent / 100 if target.capacity.cpu_percent > 0 else 0
            s += (1 - utilization) * 50

            # Prefer cheaper targets (local > CPU > GPU)
            if target.target_type == "local":
                s += 30  # Free
            elif target.target_type == "parhelia-cpu":
                s += 20
            # GPU gets no bonus (most expensive)

            # Penalize if approaching capacity
            if target.capacity.memory_percent > 80:
                s -= 30

            return s

        return max(eligible, key=score)

    def get_target(self, target_id: str) -> ExecutionTarget | None:
        """Get target by ID.

        Args:
            target_id: The target ID.

        Returns:
            ExecutionTarget or None if not found.
        """
        return self.targets.get(target_id)

    def update_target_capacity(
        self,
        target_id: str,
        capacity: ResourceCapacity,
    ) -> bool:
        """Update capacity for a target.

        Args:
            target_id: The target to update.
            capacity: New capacity values.

        Returns:
            True if updated, False if target not found.
        """
        target = self.targets.get(target_id)
        if not target:
            return False
        target.capacity = capacity
        return True
