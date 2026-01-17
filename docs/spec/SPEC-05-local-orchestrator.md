# SPEC-05: Local Orchestrator and Dispatch Logic

**Status**: Draft
**Issue**: ph-ysa
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines the local orchestrator component that coordinates task dispatch to remote Modal containers, manages worker lifecycles, aggregates results, and enforces budget constraints.

## Goals

- [SPEC-05.01] Decompose complex tasks into parallelizable units
- [SPEC-05.02] Dispatch tasks to optimal execution targets (local vs remote, CPU vs GPU)
- [SPEC-05.03] Manage remote worker lifecycle (spawn, monitor, terminate)
- [SPEC-05.04] Aggregate results from parallel workers
- [SPEC-05.05] Enforce budget ceiling to prevent cost runaway
- [SPEC-05.06] Coordinate local Claude Code with remote workers

## Non-Goals

- Distributed consensus (single orchestrator model)
- Task queuing beyond in-memory (no external queue for v1)
- Multi-orchestrator coordination

---

## Claude Code Constraints

**Critical limitation** (verified January 2026): **Subagents cannot spawn other subagents.**

This affects Parhelia's architecture:

| Constraint | Impact | Design Response |
|------------|--------|-----------------|
| No nested subagents | Remote workers can't delegate further | Flat dispatch model - orchestrator does ALL decomposition |
| Single orchestrator | Can't have distributed coordination | Local-only orchestrator, remote workers are leaf nodes |
| Session isolation | Each remote session is independent | Results aggregated by orchestrator, not by workers |

### Architectural Implication

```
✅ SUPPORTED: Flat dispatch
┌─────────────────┐
│  Orchestrator   │  ← Does all task analysis and decomposition
│  (Local Claude) │
└────────┬────────┘
         │ dispatch atomic work units
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│Worker││Worker││Worker││Worker│  ← Execute only, no further delegation
└──────┘└──────┘└──────┘└──────┘

❌ NOT SUPPORTED: Hierarchical delegation
┌─────────────────┐
│  Orchestrator   │
└────────┬────────┘
         ▼
    ┌─────────┐
    │ Worker  │ ← Cannot spawn sub-workers
    └────┬────┘
         ✗
```

This means:
1. **Task decomposition MUST happen in the orchestrator**, not distributed
2. **Workers execute atomic units** - they cannot subdivide work
3. **All coordination flows through the orchestrator**

---

## Architecture

### Orchestrator Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LOCAL ORCHESTRATOR                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Task Coordinator                              │   │
│  │  - Receives tasks from local Claude Code                             │   │
│  │  - Decomposes into work units                                        │   │
│  │  - Routes to Dispatcher                                              │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                  │
│         │                       │                       │                  │
│         ▼                       ▼                       ▼                  │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐            │
│  │  Dispatcher │        │   Budget    │        │   Result    │            │
│  │             │◀──────▶│   Manager   │        │  Aggregator │            │
│  └──────┬──────┘        └─────────────┘        └──────▲──────┘            │
│         │                                             │                    │
│         │ spawn/dispatch                              │ results            │
│         ▼                                             │                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Worker Pool Manager                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │   │
│  │  │Worker 1 │  │Worker 2 │  │Worker 3 │  │   ...   │                │   │
│  │  │(remote) │  │(remote) │  │(local)  │  │         │                │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Task Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Local     │     │    Task     │     │  Dispatcher │     │   Workers   │
│ Claude Code │     │ Coordinator │     │             │     │             │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  complex task     │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │                   │                   │
       │                   │  decompose        │                   │
       │                   │─────────┐         │                   │
       │                   │         │         │                   │
       │                   │◀────────┘         │                   │
       │                   │                   │                   │
       │                   │  work units       │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │                   │  check budget     │
       │                   │                   │─────────┐         │
       │                   │                   │         │         │
       │                   │                   │◀────────┘         │
       │                   │                   │                   │
       │                   │                   │  spawn workers    │
       │                   │                   │──────────────────▶│
       │                   │                   │                   │
       │                   │                   │◀──────────────────│
       │                   │                   │  results          │
       │                   │                   │                   │
       │                   │◀──────────────────│                   │
       │                   │  aggregated       │                   │
       │◀──────────────────│  results          │                   │
       │                   │                   │                   │
```

---

## Requirements

### [SPEC-05.10] Task Decomposition

The orchestrator MUST decompose complex tasks into parallelizable work units:

```python
@dataclass
class Task:
    id: str
    prompt: str
    task_type: TaskType
    requirements: TaskRequirements
    parent_id: str | None = None  # For subtasks
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskRequirements:
    needs_gpu: bool = False
    gpu_type: str | None = None  # "A10G", "A100", "H100"
    min_memory_gb: int = 4
    min_cpu: int = 1
    estimated_duration_minutes: int = 10
    needs_network: bool = True
    working_directory: str | None = None

class TaskDecomposer:
    """Decompose complex tasks into work units."""

    async def decompose(self, task: Task) -> list[WorkUnit]:
        """Analyze task and create parallelizable work units."""

        # Analyze task to determine decomposition strategy
        strategy = await self.analyze_task(task)

        match strategy:
            case DecompositionStrategy.SINGLE:
                # Task runs as single unit
                return [WorkUnit(task=task, parallel_group=0)]

            case DecompositionStrategy.FILE_PARALLEL:
                # Split by files (e.g., "fix lint errors in all files")
                files = await self.identify_target_files(task)
                return [
                    WorkUnit(
                        task=self.create_file_subtask(task, f),
                        parallel_group=i % MAX_PARALLEL_WORKERS,
                    )
                    for i, f in enumerate(files)
                ]

            case DecompositionStrategy.TEST_PARALLEL:
                # Split by test files/suites
                test_files = await self.identify_test_files(task)
                return [
                    WorkUnit(
                        task=self.create_test_subtask(task, tf),
                        parallel_group=i % MAX_PARALLEL_WORKERS,
                    )
                    for i, tf in enumerate(test_files)
                ]

            case DecompositionStrategy.PIPELINE:
                # Sequential stages (e.g., build -> test -> deploy)
                stages = await self.identify_stages(task)
                return [
                    WorkUnit(
                        task=self.create_stage_subtask(task, stage),
                        parallel_group=0,  # Sequential
                        depends_on=[stages[i-1].id] if i > 0 else [],
                    )
                    for i, stage in enumerate(stages)
                ]

    async def analyze_task(self, task: Task) -> DecompositionStrategy:
        """Use Claude to analyze task and determine best decomposition."""
        analysis_prompt = f"""Analyze this task and determine the best parallelization strategy:

Task: {task.prompt}

Strategies:
1. SINGLE - Run as one unit (simple tasks, tightly coupled work)
2. FILE_PARALLEL - Split by files (lint, format, refactor across files)
3. TEST_PARALLEL - Split by test files (run tests in parallel)
4. PIPELINE - Sequential stages (build -> test -> deploy)

Return JSON: {{"strategy": "SINGLE|FILE_PARALLEL|TEST_PARALLEL|PIPELINE", "reason": "..."}}"""

        result = await self.claude.analyze(analysis_prompt)
        return DecompositionStrategy(result["strategy"])
```

### [SPEC-05.11] Dispatch Logic

The dispatcher MUST route work units to optimal execution targets:

```python
@dataclass
class ExecutionTarget:
    id: str
    target_type: Literal["local", "parhelia-cpu", "parhelia-gpu"]  # Matches SPEC-01 variants
    region: str | None
    capacity: ResourceCapacity
    current_load: ResourceUsage
    gpu_type: str | None = None  # "A10G", "A100", "H100" for GPU targets

class Dispatcher:
    """Route work units to execution targets."""

    async def dispatch(self, work_unit: WorkUnit) -> Worker:
        """Dispatch work unit to best available target."""

        # Get available targets
        targets = await self.get_available_targets()

        # Filter by requirements
        eligible = [
            t for t in targets
            if self.meets_requirements(t, work_unit.task.requirements)
        ]

        if not eligible:
            raise NoEligibleTargetError(
                f"No target meets requirements: {work_unit.task.requirements}"
            )

        # Select best target
        target = self.select_optimal_target(eligible, work_unit)

        # Spawn or reuse worker
        worker = await self.get_or_create_worker(target, work_unit)

        # Dispatch task
        await worker.execute(work_unit)

        return worker

    def meets_requirements(
        self,
        target: ExecutionTarget,
        requirements: TaskRequirements,
    ) -> bool:
        """Check if target can satisfy task requirements."""

        if requirements.needs_gpu:
            if target.target_type != "parhelia-gpu":
                return False
            if requirements.gpu_type and target.gpu_type != requirements.gpu_type:
                return False

        if target.capacity.available_memory_gb < requirements.min_memory_gb:
            return False

        if target.capacity.available_cpu < requirements.min_cpu:
            return False

        return True

    def select_optimal_target(
        self,
        targets: list[ExecutionTarget],
        work_unit: WorkUnit,
    ) -> ExecutionTarget:
        """Select best target based on scoring."""

        def score(target: ExecutionTarget) -> float:
            score = 0.0

            # Prefer local for interactive/small tasks
            if work_unit.task.task_type == TaskType.INTERACTIVE:
                if target.target_type == "local":
                    score += 100

            # Prefer less loaded targets
            utilization = target.current_load.cpu_percent / 100
            score += (1 - utilization) * 50

            # Prefer cheaper targets (CPU < GPU)
            if target.target_type == "parhelia-cpu":
                score += 20
            elif target.target_type == "local":
                score += 30  # Free!

            # Penalize if approaching capacity
            if target.current_load.memory_percent > 80:
                score -= 30

            return score

        return max(targets, key=score)
```

### [SPEC-05.12] Worker Lifecycle Management

The orchestrator MUST manage worker lifecycles:

```python
class WorkerPoolManager:
    """Manage pool of local and remote workers."""

    def __init__(self):
        self.workers: dict[str, Worker] = {}
        self.modal_client = modal.Client()

    async def spawn_worker(
        self,
        target: ExecutionTarget,
        task: Task,
    ) -> Worker:
        """Spawn new worker for task execution."""

        if target.target_type == "local":
            worker = LocalWorker(task=task)
        elif target.target_type == "parhelia-cpu":
            worker = await self.spawn_modal_worker(
                task=task,
                gpu=None,
            )
        elif target.target_type == "parhelia-gpu":
            worker = await self.spawn_modal_worker(
                task=task,
                gpu=task.requirements.gpu_type or "A10G",
            )

        self.workers[worker.id] = worker
        return worker

    async def spawn_modal_worker(
        self,
        task: Task,
        gpu: str | None,
    ) -> ModalWorker:
        """Spawn Modal container for task."""

        # Prepare secrets
        secrets = await self.secret_injector.prepare_secrets_for_task(task)

        # Spawn container
        if gpu:
            container = await modal.Function.lookup("parhelia", "run_claude_gpu").spawn(
                task=task.to_dict(),
            )
        else:
            container = await modal.Function.lookup("parhelia", "run_claude_cpu").spawn(
                task=task.to_dict(),
            )

        worker = ModalWorker(
            id=f"modal-{container.object_id}",
            container=container,
            task=task,
            spawned_at=datetime.now(),
        )

        # Start heartbeat monitoring
        asyncio.create_task(self.monitor_worker(worker))

        return worker

    async def monitor_worker(self, worker: Worker):
        """Monitor worker health and handle failures."""
        while worker.state not in (WorkerState.COMPLETED, WorkerState.FAILED):
            try:
                health = await worker.health_check()
                if not health.healthy:
                    await self.handle_unhealthy_worker(worker, health)
            except Exception as e:
                logger.error(f"Worker {worker.id} health check failed: {e}")
                await self.handle_worker_failure(worker, e)
                break

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def terminate_worker(self, worker_id: str):
        """Gracefully terminate worker."""
        worker = self.workers.get(worker_id)
        if not worker:
            return

        # Request graceful shutdown
        await worker.shutdown(timeout=30)

        # Force kill if needed
        if worker.state != WorkerState.TERMINATED:
            await worker.force_kill()

        del self.workers[worker_id]
```

### [SPEC-05.13] Result Aggregation

The orchestrator MUST aggregate results from parallel workers:

```python
@dataclass
class WorkResult:
    work_unit_id: str
    worker_id: str
    status: Literal["success", "partial", "failed"]
    output: str
    artifacts: list[Artifact]  # Files created/modified
    metrics: ExecutionMetrics
    error: str | None = None

class ResultAggregator:
    """Aggregate results from parallel work units."""

    async def aggregate(
        self,
        task: Task,
        results: list[WorkResult],
    ) -> AggregatedResult:
        """Combine results from all work units."""

        # Check for failures
        failed = [r for r in results if r.status == "failed"]
        if failed:
            return AggregatedResult(
                task_id=task.id,
                status="partial" if len(failed) < len(results) else "failed",
                summary=self.summarize_failures(failed),
                results=results,
            )

        # Merge artifacts (handle conflicts)
        merged_artifacts = await self.merge_artifacts(results)

        # Generate summary
        summary = await self.generate_summary(task, results)

        return AggregatedResult(
            task_id=task.id,
            status="success",
            summary=summary,
            results=results,
            artifacts=merged_artifacts,
            total_cost=sum(r.metrics.cost for r in results),
            total_duration=max(r.metrics.duration for r in results),  # Parallel
        )

    async def merge_artifacts(
        self,
        results: list[WorkResult],
    ) -> list[Artifact]:
        """Merge file artifacts, detecting conflicts."""
        artifacts_by_path: dict[str, list[Artifact]] = {}

        for result in results:
            for artifact in result.artifacts:
                if artifact.path not in artifacts_by_path:
                    artifacts_by_path[artifact.path] = []
                artifacts_by_path[artifact.path].append(artifact)

        merged = []
        for path, versions in artifacts_by_path.items():
            if len(versions) == 1:
                merged.append(versions[0])
            else:
                # Conflict - need resolution
                resolved = await self.resolve_conflict(path, versions)
                merged.append(resolved)

        return merged
```

### [SPEC-05.14] Budget Management

The orchestrator MUST enforce budget constraints:

```python
@dataclass
class BudgetConfig:
    max_cost_per_task: float = 10.0      # USD
    max_cost_per_hour: float = 50.0      # USD
    max_cost_per_day: float = 200.0      # USD
    max_parallel_workers: int = 10
    warn_threshold: float = 0.8          # Warn at 80% of limit

class BudgetManager:
    """Track and enforce budget constraints."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.current_hour_cost = 0.0
        self.current_day_cost = 0.0
        self.hour_start = datetime.now()
        self.day_start = datetime.now().date()

    async def check_budget(self, estimated_cost: float) -> BudgetCheckResult:
        """Check if task can proceed within budget."""

        # Reset counters if period elapsed
        self._maybe_reset_counters()

        # Check task limit
        if estimated_cost > self.config.max_cost_per_task:
            return BudgetCheckResult(
                allowed=False,
                reason=f"Task cost ${estimated_cost:.2f} exceeds limit ${self.config.max_cost_per_task:.2f}",
            )

        # Check hourly limit
        if self.current_hour_cost + estimated_cost > self.config.max_cost_per_hour:
            return BudgetCheckResult(
                allowed=False,
                reason=f"Would exceed hourly limit (${self.config.max_cost_per_hour:.2f})",
                retry_after=self._time_until_hour_reset(),
            )

        # Check daily limit
        if self.current_day_cost + estimated_cost > self.config.max_cost_per_day:
            return BudgetCheckResult(
                allowed=False,
                reason=f"Would exceed daily limit (${self.config.max_cost_per_day:.2f})",
                retry_after=self._time_until_day_reset(),
            )

        # Check warning threshold
        warnings = []
        if self.current_hour_cost / self.config.max_cost_per_hour > self.config.warn_threshold:
            warnings.append(f"Approaching hourly limit ({self.current_hour_cost:.2f}/{self.config.max_cost_per_hour:.2f})")

        return BudgetCheckResult(allowed=True, warnings=warnings)

    async def record_cost(self, cost: float, task_id: str):
        """Record cost after task completion."""
        self.current_hour_cost += cost
        self.current_day_cost += cost

        # Log for auditing
        await audit_logger.log(AuditEvent(
            event_type="budget.cost_recorded",
            action="record_cost",
            resource=task_id,
            outcome="success",
            details={"cost": cost, "hour_total": self.current_hour_cost},
        ))

        # Emit metrics
        metrics.budget_cost_total.inc(cost)
```

### [SPEC-05.15] Local-Remote Coordination Protocol

The orchestrator MUST coordinate between local Claude Code and remote workers:

```python
class CoordinationProtocol:
    """Protocol for local-remote coordination."""

    # Message types
    class MessageType(Enum):
        DISPATCH = "dispatch"           # Send task to remote
        PROGRESS = "progress"           # Progress update from remote
        RESULT = "result"               # Final result from remote
        CHECKPOINT = "checkpoint"       # Checkpoint notification
        NEEDS_HUMAN = "needs_human"     # Request human intervention
        CANCEL = "cancel"               # Cancel task
        SHUTDOWN = "shutdown"           # Shutdown worker

    @dataclass
    class Message:
        type: MessageType
        task_id: str
        worker_id: str
        timestamp: datetime
        payload: dict[str, Any]

    async def send_to_remote(self, worker: Worker, message: Message):
        """Send message to remote worker."""
        if isinstance(worker, ModalWorker):
            # Use Modal's built-in RPC
            await worker.container.call("receive_message", message.to_dict())
        elif isinstance(worker, LocalWorker):
            # Use local IPC
            await worker.send(message)

    async def receive_from_remote(self, message: Message):
        """Handle message from remote worker."""
        match message.type:
            case MessageType.PROGRESS:
                await self.handle_progress(message)
            case MessageType.RESULT:
                await self.handle_result(message)
            case MessageType.CHECKPOINT:
                await self.handle_checkpoint(message)
            case MessageType.NEEDS_HUMAN:
                await self.handle_needs_human(message)

    async def handle_needs_human(self, message: Message):
        """Handle request for human intervention."""
        task_id = message.task_id
        reason = message.payload.get("reason")

        # Notify user
        await notifications.send(
            title=f"Task {task_id} needs attention",
            body=reason,
            actions=[
                {"label": "Attach", "command": f"parhelia attach {task_id}"},
                {"label": "Cancel", "command": f"parhelia cancel {task_id}"},
            ]
        )

        # Update task state
        task = await self.task_store.get(task_id)
        task.state = TaskState.WAITING_HUMAN
        await self.task_store.update(task)
```

---

## CLI Integration

### `parhelia dispatch`

```bash
$ parhelia dispatch "Run all tests and fix any failures"

Analyzing task...
  Strategy: TEST_PARALLEL (47 test files)
  Estimated cost: $2.34
  Estimated time: 5 minutes (parallel)

Dispatching to 5 workers...
  ├── modal-abc123 (us-east, CPU): tests/unit/*
  ├── modal-def456 (us-east, CPU): tests/integration/*
  ├── modal-ghi789 (us-east, CPU): tests/api/*
  ├── modal-jkl012 (us-east, CPU): tests/e2e/*
  └── local: tests/smoke/*

Progress: ████████████░░░░░░░░ 60% (28/47 files)
```

### `parhelia workers`

```bash
$ parhelia workers

ID            TARGET      STATE     TASK                CPU   MEM    COST
modal-abc123  us-east     running   tests/unit/*        45%   2.1GB  $0.12
modal-def456  us-east     running   tests/integration/* 78%   3.4GB  $0.18
modal-ghi789  us-east     complete  tests/api/*         -     -      $0.08
local         local       idle      -                   12%   1.2GB  $0.00

Total active: 2 remote, 1 local
Session cost: $0.38 / $10.00 limit
```

### `parhelia budget`

```bash
$ parhelia budget

Current Period    Used        Limit       Remaining
─────────────────────────────────────────────────────
This hour         $12.34      $50.00      $37.66 (75%)
Today             $45.67      $200.00     $154.33 (77%)

Recent Tasks:
  fix-auth (2h ago)           $3.45
  run-tests (1h ago)          $8.89
  Total this session          $12.34
```

---

## Acceptance Criteria

- [ ] [SPEC-05.AC1] Tasks decomposed into parallel work units
- [ ] [SPEC-05.AC2] Work units dispatched to optimal targets
- [ ] [SPEC-05.AC3] Workers spawned and monitored correctly
- [ ] [SPEC-05.AC4] Results aggregated from parallel workers
- [ ] [SPEC-05.AC5] Budget limits enforced, tasks blocked when exceeded
- [ ] [SPEC-05.AC6] Local-remote coordination protocol functional
- [ ] [SPEC-05.AC7] CLI commands work as documented

---

## Open Questions

1. **Task decomposition accuracy**: How well can Claude analyze tasks for parallelization? Need testing.
2. **Conflict resolution**: What strategy for file conflicts from parallel workers?
3. **Cost estimation**: How accurate are pre-dispatch cost estimates? Need calibration data.

---

## References

- [Modal Function Spawning](https://modal.com/docs/guide/spawn)
- [Modal Sandboxes](https://modal.com/docs/guide/sandbox)
- [Modal Pricing](https://modal.com/pricing)
- ADR-001: System Architecture
- SPEC-01: Remote Environment Provisioning (container variants)
- SPEC-03: Checkpoint and Resume
- SPEC-04: Security Model
- SPEC-06: Resource Broadcasting (metrics for dispatch decisions)
