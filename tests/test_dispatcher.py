"""Tests for dispatch logic and target selection.

@trace SPEC-05.11 - Dispatch Logic
@trace SPEC-05.12 - Worker Lifecycle
"""

from typing import Literal

import pytest


class TestExecutionTarget:
    """Tests for ExecutionTarget dataclass - SPEC-05.11."""

    def test_execution_target_creation(self):
        """@trace SPEC-05.11 - ExecutionTarget MUST capture target info."""
        from parhelia.dispatcher import ExecutionTarget

        target = ExecutionTarget(
            id="target-1",
            target_type="parhelia-cpu",
            region="us-east",
        )

        assert target.id == "target-1"
        assert target.target_type == "parhelia-cpu"
        assert target.region == "us-east"

    def test_execution_target_types(self):
        """@trace SPEC-05.11 - ExecutionTarget MUST support local/cpu/gpu types."""
        from parhelia.dispatcher import ExecutionTarget

        local = ExecutionTarget(id="local", target_type="local")
        cpu = ExecutionTarget(id="cpu-1", target_type="parhelia-cpu")
        gpu = ExecutionTarget(id="gpu-1", target_type="parhelia-gpu", gpu_type="A100")

        assert local.target_type == "local"
        assert cpu.target_type == "parhelia-cpu"
        assert gpu.target_type == "parhelia-gpu"
        assert gpu.gpu_type == "A100"

    def test_execution_target_capacity(self):
        """@trace SPEC-05.11 - ExecutionTarget SHOULD track capacity."""
        from parhelia.dispatcher import ExecutionTarget, ResourceCapacity

        capacity = ResourceCapacity(
            available_cpu=4,
            available_memory_gb=16,
        )

        target = ExecutionTarget(
            id="target-1",
            target_type="parhelia-cpu",
            capacity=capacity,
        )

        assert target.capacity.available_cpu == 4
        assert target.capacity.available_memory_gb == 16


class TestResourceCapacity:
    """Tests for ResourceCapacity dataclass."""

    def test_resource_capacity_creation(self):
        """ResourceCapacity MUST capture CPU and memory."""
        from parhelia.dispatcher import ResourceCapacity

        capacity = ResourceCapacity(
            available_cpu=8,
            available_memory_gb=32,
        )

        assert capacity.available_cpu == 8
        assert capacity.available_memory_gb == 32

    def test_resource_capacity_defaults(self):
        """ResourceCapacity SHOULD have sensible defaults."""
        from parhelia.dispatcher import ResourceCapacity

        capacity = ResourceCapacity()

        assert capacity.available_cpu >= 0
        assert capacity.available_memory_gb >= 0


class TestDispatcher:
    """Tests for Dispatcher class - SPEC-05.11."""

    @pytest.fixture
    def dispatcher(self):
        """Create Dispatcher instance."""
        from parhelia.dispatcher import Dispatcher

        return Dispatcher()

    def test_dispatcher_initialization(self, dispatcher):
        """@trace SPEC-05.11 - Dispatcher MUST initialize with empty targets."""
        assert dispatcher is not None
        assert len(dispatcher.targets) == 0

    def test_register_target(self, dispatcher):
        """@trace SPEC-05.11 - Dispatcher MUST allow target registration."""
        from parhelia.dispatcher import ExecutionTarget

        target = ExecutionTarget(id="target-1", target_type="local")
        dispatcher.register_target(target)

        assert "target-1" in dispatcher.targets

    def test_get_available_targets(self, dispatcher):
        """@trace SPEC-05.11 - Dispatcher MUST list available targets."""
        from parhelia.dispatcher import ExecutionTarget

        dispatcher.register_target(ExecutionTarget(id="t1", target_type="local"))
        dispatcher.register_target(ExecutionTarget(id="t2", target_type="parhelia-cpu"))

        targets = dispatcher.get_available_targets()

        assert len(targets) == 2

    def test_meets_requirements_cpu_task(self, dispatcher):
        """@trace SPEC-05.11 - MUST check if target meets CPU requirements."""
        from parhelia.dispatcher import ExecutionTarget, ResourceCapacity
        from parhelia.orchestrator import TaskRequirements

        target = ExecutionTarget(
            id="cpu-target",
            target_type="parhelia-cpu",
            capacity=ResourceCapacity(available_cpu=4, available_memory_gb=16),
        )

        requirements = TaskRequirements(needs_gpu=False, min_cpu=2, min_memory_gb=8)

        assert dispatcher.meets_requirements(target, requirements) is True

    def test_meets_requirements_gpu_required(self, dispatcher):
        """@trace SPEC-05.11 - MUST reject CPU target for GPU task."""
        from parhelia.dispatcher import ExecutionTarget, ResourceCapacity
        from parhelia.orchestrator import TaskRequirements

        cpu_target = ExecutionTarget(
            id="cpu-target",
            target_type="parhelia-cpu",
            capacity=ResourceCapacity(available_cpu=4, available_memory_gb=16),
        )

        requirements = TaskRequirements(needs_gpu=True, gpu_type="A100")

        assert dispatcher.meets_requirements(cpu_target, requirements) is False

    def test_meets_requirements_specific_gpu(self, dispatcher):
        """@trace SPEC-05.11 - MUST check specific GPU type if required."""
        from parhelia.dispatcher import ExecutionTarget, ResourceCapacity
        from parhelia.orchestrator import TaskRequirements

        a10g_target = ExecutionTarget(
            id="gpu-target",
            target_type="parhelia-gpu",
            gpu_type="A10G",
            capacity=ResourceCapacity(available_cpu=4, available_memory_gb=16),
        )

        # Task requires A100
        requirements = TaskRequirements(needs_gpu=True, gpu_type="A100")

        assert dispatcher.meets_requirements(a10g_target, requirements) is False

    def test_meets_requirements_memory_insufficient(self, dispatcher):
        """@trace SPEC-05.11 - MUST reject target with insufficient memory."""
        from parhelia.dispatcher import ExecutionTarget, ResourceCapacity
        from parhelia.orchestrator import TaskRequirements

        target = ExecutionTarget(
            id="small-target",
            target_type="parhelia-cpu",
            capacity=ResourceCapacity(available_cpu=4, available_memory_gb=4),
        )

        requirements = TaskRequirements(min_memory_gb=16)

        assert dispatcher.meets_requirements(target, requirements) is False


class TestTargetSelection:
    """Tests for target selection logic - SPEC-05.11."""

    @pytest.fixture
    def dispatcher(self):
        """Create Dispatcher with targets."""
        from parhelia.dispatcher import Dispatcher, ExecutionTarget, ResourceCapacity

        d = Dispatcher()
        d.register_target(ExecutionTarget(
            id="local",
            target_type="local",
            capacity=ResourceCapacity(available_cpu=4, available_memory_gb=16),
        ))
        d.register_target(ExecutionTarget(
            id="cpu-1",
            target_type="parhelia-cpu",
            region="us-east",
            capacity=ResourceCapacity(available_cpu=4, available_memory_gb=16),
        ))
        d.register_target(ExecutionTarget(
            id="gpu-1",
            target_type="parhelia-gpu",
            region="us-east",
            gpu_type="A100",
            capacity=ResourceCapacity(available_cpu=8, available_memory_gb=32),
        ))
        return d

    def test_select_target_prefers_local_for_small_tasks(self, dispatcher):
        """@trace SPEC-05.11 - SHOULD prefer local for small tasks."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-1",
            prompt="Simple task",
            task_type=TaskType.INTERACTIVE,
            requirements=TaskRequirements(needs_gpu=False, min_cpu=1),
        )

        target = dispatcher.select_optimal_target(task)

        # Local should be preferred for interactive/small tasks
        assert target.target_type == "local"

    def test_select_target_uses_gpu_when_required(self, dispatcher):
        """@trace SPEC-05.11 - MUST use GPU target for GPU tasks."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-2",
            prompt="ML inference",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(needs_gpu=True),
        )

        target = dispatcher.select_optimal_target(task)

        assert target.target_type == "parhelia-gpu"

    def test_select_target_returns_none_if_no_eligible(self, dispatcher):
        """@trace SPEC-05.11 - MUST return None if no eligible target."""
        from parhelia.orchestrator import Task, TaskRequirements, TaskType

        task = Task(
            id="task-3",
            prompt="Need H100",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(needs_gpu=True, gpu_type="H100"),
        )

        target = dispatcher.select_optimal_target(task)

        # No H100 available
        assert target is None
