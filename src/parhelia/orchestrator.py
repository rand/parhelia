"""Local orchestrator for task coordination.

Implements:
- [SPEC-05.10] Task Decomposition
- [SPEC-05.11] Dispatch Logic
- [SPEC-05.12] Worker Lifecycle Management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from parhelia.permissions import TrustLevel


class TaskType(Enum):
    """Types of tasks the orchestrator handles.

    Implements [SPEC-05.10].
    """

    GENERIC = "generic"
    CODE_FIX = "code_fix"
    TEST_RUN = "test_run"
    BUILD = "build"
    LINT = "lint"
    REFACTOR = "refactor"
    INTERACTIVE = "interactive"


class WorkerState(Enum):
    """Worker lifecycle states.

    Implements [SPEC-05.12].
    """

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class TaskRequirements:
    """Requirements for task execution.

    Implements [SPEC-05.10].
    """

    needs_gpu: bool = False
    gpu_type: str | None = None  # "A10G", "A100", "H100"
    min_memory_gb: int = 4
    min_cpu: int = 1
    estimated_duration_minutes: int = 10
    needs_network: bool = True
    working_directory: str | None = None


@dataclass
class Task:
    """A task to be executed by the orchestrator.

    Implements [SPEC-05.10].
    """

    id: str
    prompt: str
    task_type: TaskType
    requirements: TaskRequirements
    parent_id: str | None = None  # For subtasks
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    trust_level: TrustLevel = TrustLevel.INTERACTIVE  # [SPEC-04.13]


@dataclass
class TaskResult:
    """Result from task execution.

    Implements [SPEC-05.10].
    """

    task_id: str
    status: Literal["success", "partial", "failed", "pending"]
    output: str = ""
    error: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    artifacts: list[str] = field(default_factory=list)


@dataclass
class WorkerInfo:
    """Information about a worker.

    Implements [SPEC-05.12] and [SPEC-21.13].
    """

    id: str
    task_id: str
    state: WorkerState
    target_type: Literal["local", "parhelia-cpu", "parhelia-gpu"]
    created_at: datetime = field(default_factory=datetime.now)
    gpu_type: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # SPEC-21.13 extensions for control plane state
    container_id: str | None = None  # Link to containers table
    session_id: str | None = None
    last_heartbeat_at: datetime | None = None
    health_status: str = "unknown"  # healthy, degraded, unhealthy, dead
    terminated_at: datetime | None = None
    exit_code: int | None = None


class LocalOrchestrator:
    """Central coordinator for task execution.

    Implements [SPEC-05.10].

    The orchestrator handles:
    - Task submission and tracking
    - Worker lifecycle management
    - Result collection and aggregation
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.pending_tasks: dict[str, Task] = {}
        self.active_workers: dict[str, WorkerInfo] = {}
        self.completed_results: dict[str, TaskResult] = {}

    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution.

        Args:
            task: The task to submit.

        Returns:
            The task ID.
        """
        self.pending_tasks[task.id] = task
        return task.id

    async def get_workers(self) -> list[WorkerInfo]:
        """Get list of all active workers.

        Returns:
            List of WorkerInfo for active workers.
        """
        return list(self.active_workers.values())

    async def collect_results(self, task_id: str) -> TaskResult | None:
        """Collect results for a completed task.

        Args:
            task_id: The task ID to get results for.

        Returns:
            TaskResult if available, None otherwise.
        """
        return self.completed_results.get(task_id)

    def register_worker(self, worker: WorkerInfo) -> None:
        """Register an active worker.

        Args:
            worker: The worker to register.
        """
        self.active_workers[worker.id] = worker

    def unregister_worker(self, worker_id: str) -> WorkerInfo | None:
        """Remove a worker from tracking.

        Args:
            worker_id: The worker ID to remove.

        Returns:
            The removed WorkerInfo, or None if not found.
        """
        return self.active_workers.pop(worker_id, None)

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """Get worker by ID.

        Args:
            worker_id: The worker ID.

        Returns:
            WorkerInfo or None if not found.
        """
        return self.active_workers.get(worker_id)

    def get_task(self, task_id: str) -> Task | None:
        """Get task by ID.

        Args:
            task_id: The task ID.

        Returns:
            Task or None if not found.
        """
        return self.pending_tasks.get(task_id)

    def mark_task_complete(
        self,
        task_id: str,
        result: TaskResult,
    ) -> None:
        """Mark a task as complete with its result.

        Args:
            task_id: The task ID.
            result: The task result.
        """
        self.completed_results[task_id] = result
        self.pending_tasks.pop(task_id, None)

    def get_pending_count(self) -> int:
        """Get count of pending tasks.

        Returns:
            Number of pending tasks.
        """
        return len(self.pending_tasks)

    def get_active_worker_count(self) -> int:
        """Get count of active workers.

        Returns:
            Number of active workers.
        """
        return len(self.active_workers)
