# SPEC-21: Control Plane State Management

**Status**: Draft
**Author**: Claude + rand
**Date**: 2026-01-21

## Overview

This specification defines the stateful control plane infrastructure for Parhelia, enabling proper introspection, recovery, and resource management. The control plane maintains an authoritative view of all running containers, their health status, and the mapping between logical tasks and physical Modal resources.

## Problem Statement

On 2026-01-21, 25 Modal containers ran for 11+ hours undetected, incurring $140 in costs. Root cause analysis revealed:

1. **No container registry**: Modal sandbox IDs were returned but never persisted
2. **No reconciliation**: Database state drifted from Modal reality with no detection mechanism
3. **No heartbeat persistence**: In-memory only; couldn't query historical health
4. **Reactive-only checks**: Pre-dispatch validation can't detect already-running orphans

The current architecture has state models but lacks the **reconciliation loop** that maintains consistency between Parhelia's internal state and Modal's actual container state.

## Goals

- [SPEC-21.01] Maintain authoritative registry of all container instances with Modal sandbox IDs
- [SPEC-21.02] Implement background reconciler that syncs DB state with Modal reality
- [SPEC-21.03] Persist heartbeat and event history for queryability and post-mortem analysis
- [SPEC-21.04] Detect stale/orphaned workers automatically and trigger cleanup or alerts
- [SPEC-21.05] Provide introspection APIs for "what's actually running right now"
- [SPEC-21.06] Enable recovery workflows when containers crash or drift from expected state
- [SPEC-21.07] Support both human operators and agent consumers with appropriate interfaces

## Non-Goals

- Distributed consensus (single control plane model)
- Real-time streaming to external systems (batch export sufficient for v1)
- Multi-region coordination (single Modal region for v1)
- Automatic container migration (manual recovery for v1)

---

## Architecture

### Control Plane Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTROL PLANE                                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     State Store (SQLite)                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ workers  │  │ events   │  │heartbeats│  │ containers│            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│         ┌───────────────────────┼───────────────────────┐                  │
│         │                       │                       │                  │
│         ▼                       ▼                       ▼                  │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐            │
│  │ Reconciler  │        │   Health    │        │   Event     │            │
│  │   Loop      │◀──────▶│   Monitor   │        │   Logger    │            │
│  └──────┬──────┘        └─────────────┘        └──────▲──────┘            │
│         │                                             │                    │
│         │ poll Modal API                              │ emit events        │
│         ▼                                             │                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Modal Interface                               │   │
│  │  - List sandboxes          - Get sandbox status                     │   │
│  │  - Terminate sandbox       - Query resource usage                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL CLOUD                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │Sandbox 1│  │Sandbox 2│  │Sandbox 3│  │   ...   │  │Sandbox N│          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### State Synchronization Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Modal     │     │ Reconciler  │     │   State     │     │   CLI/MCP   │
│   Cloud     │     │    Loop     │     │   Store     │     │   Queries   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │                   │ list sandboxes    │                   │
       │◀──────────────────│                   │                   │
       │                   │                   │                   │
       │  sandbox list     │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │                   │                   │
       │                   │  get db state     │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │  workers/containers                   │
       │                   │◀──────────────────│                   │
       │                   │                   │                   │
       │                   │  ┌─────────────┐  │                   │
       │                   │  │   DIFF &    │  │                   │
       │                   │  │ RECONCILE   │  │                   │
       │                   │  └─────────────┘  │                   │
       │                   │                   │                   │
       │                   │  update state     │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │  emit events      │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │                   │  query state      │
       │                   │                   │◀──────────────────│
       │                   │                   │                   │
       │                   │                   │  current view     │
       │                   │                   │──────────────────▶│
```

---

## Data Model

### [SPEC-21.10] Container Registry Schema

New `containers` table to track Modal sandbox instances:

```sql
CREATE TABLE containers (
    -- Identity
    id TEXT PRIMARY KEY,                    -- Parhelia container ID (uuid)
    modal_sandbox_id TEXT UNIQUE NOT NULL,  -- Modal's sandbox ID (sb-xxx)

    -- Relationships
    worker_id TEXT,                         -- FK to workers table (nullable for orphans)
    task_id TEXT,                           -- FK to tasks table
    session_id TEXT,                        -- FK to sessions (if applicable)

    -- Lifecycle
    state TEXT NOT NULL DEFAULT 'unknown',  -- running, stopped, terminated, orphaned
    created_at TEXT NOT NULL,               -- When Parhelia created it
    started_at TEXT,                        -- When Modal reported it running
    terminated_at TEXT,                     -- When it stopped/terminated
    exit_code INTEGER,                      -- Container exit code (null if running)
    termination_reason TEXT,                -- idle_timeout, manual, crash, oom, etc.

    -- Health
    last_heartbeat_at TEXT,                 -- Last successful heartbeat
    health_status TEXT DEFAULT 'unknown',   -- healthy, degraded, unhealthy, dead
    consecutive_failures INTEGER DEFAULT 0, -- Failed health checks in a row

    -- Resources
    cpu_cores INTEGER,                      -- Allocated CPU cores
    memory_mb INTEGER,                      -- Allocated memory
    gpu_type TEXT,                          -- GPU type if allocated
    region TEXT,                            -- Modal region

    -- Cost tracking
    cost_accrued_usd REAL DEFAULT 0.0,      -- Cost so far
    cost_rate_per_hour REAL,                -- $/hour rate

    -- Metadata
    metadata TEXT NOT NULL DEFAULT '{}',    -- JSON blob for extensibility
    updated_at TEXT NOT NULL,               -- Last state change

    FOREIGN KEY (worker_id) REFERENCES workers(id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX idx_containers_state ON containers(state);
CREATE INDEX idx_containers_modal_id ON containers(modal_sandbox_id);
CREATE INDEX idx_containers_health ON containers(health_status);
CREATE INDEX idx_containers_worker ON containers(worker_id);
CREATE INDEX idx_containers_updated ON containers(updated_at);
```

### [SPEC-21.11] Events Table Schema

Persistent event log for audit and debugging:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                -- ISO8601 UTC timestamp
    event_type TEXT NOT NULL,               -- state_change, heartbeat, error, etc.

    -- References (all nullable - event may relate to any/none)
    container_id TEXT,
    worker_id TEXT,
    task_id TEXT,
    session_id TEXT,

    -- Event data
    old_value TEXT,                         -- Previous state (for changes)
    new_value TEXT,                         -- New state (for changes)
    message TEXT,                           -- Human-readable description
    details TEXT NOT NULL DEFAULT '{}',     -- JSON blob with full context

    -- Source
    source TEXT NOT NULL DEFAULT 'system',  -- system, reconciler, user, container

    FOREIGN KEY (container_id) REFERENCES containers(id)
);

CREATE INDEX idx_events_timestamp ON events(timestamp);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_container ON events(container_id);
CREATE INDEX idx_events_task ON events(task_id);

-- For efficient time-range queries
CREATE INDEX idx_events_type_timestamp ON events(event_type, timestamp);
```

### [SPEC-21.12] Heartbeat History Schema

Persistent heartbeat tracking:

```sql
CREATE TABLE heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,                -- When heartbeat received

    -- Health metrics
    cpu_percent REAL,
    memory_percent REAL,
    memory_mb INTEGER,
    disk_percent REAL,

    -- Container state
    uptime_seconds INTEGER,
    tmux_active INTEGER,                    -- 1 if tmux session active
    claude_responsive INTEGER,              -- 1 if Claude responding

    -- Metadata
    metadata TEXT NOT NULL DEFAULT '{}',

    FOREIGN KEY (container_id) REFERENCES containers(id)
);

CREATE INDEX idx_heartbeats_container ON heartbeats(container_id);
CREATE INDEX idx_heartbeats_timestamp ON heartbeats(timestamp);

-- Retention: Keep last 24 hours of heartbeats, then aggregate
CREATE INDEX idx_heartbeats_container_time ON heartbeats(container_id, timestamp);
```

### [SPEC-21.13] Workers Table Additions

Extend existing workers table:

```sql
-- Add columns to existing workers table
ALTER TABLE workers ADD COLUMN container_id TEXT REFERENCES containers(id);
ALTER TABLE workers ADD COLUMN session_id TEXT;
ALTER TABLE workers ADD COLUMN last_heartbeat_at TEXT;
ALTER TABLE workers ADD COLUMN health_status TEXT DEFAULT 'unknown';
ALTER TABLE workers ADD COLUMN terminated_at TEXT;
ALTER TABLE workers ADD COLUMN exit_code INTEGER;

CREATE INDEX idx_workers_container ON workers(container_id);
CREATE INDEX idx_workers_session ON workers(session_id);
CREATE INDEX idx_workers_health ON workers(health_status);
```

---

## Reconciliation Loop

### [SPEC-21.20] Reconciler Implementation

```python
@dataclass
class ReconcilerConfig:
    """Configuration for the reconciliation loop."""
    poll_interval_seconds: int = 60         # How often to poll Modal
    stale_threshold_seconds: int = 300      # 5 min without heartbeat = stale
    orphan_grace_period_seconds: int = 120  # Wait before marking orphan
    auto_terminate_orphans: bool = False    # Require manual cleanup by default
    max_reconcile_batch: int = 100          # Max containers per reconcile cycle


class ContainerReconciler:
    """Reconciles Parhelia state with Modal reality."""

    def __init__(
        self,
        store: StateStore,
        modal_client: ModalClient,
        config: ReconcilerConfig,
        event_logger: EventLogger,
    ):
        self.store = store
        self.modal = modal_client
        self.config = config
        self.events = event_logger
        self._running = False

    async def start(self):
        """Start the reconciliation loop."""
        self._running = True
        while self._running:
            try:
                await self.reconcile()
            except Exception as e:
                await self.events.log_error("reconcile_failed", str(e))
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def reconcile(self):
        """Single reconciliation cycle."""
        # 1. Get Modal's view of reality
        modal_sandboxes = await self.modal.list_sandboxes(app="parhelia")
        modal_by_id = {s.id: s for s in modal_sandboxes}

        # 2. Get our view of reality
        db_containers = await self.store.get_active_containers()
        db_by_modal_id = {c.modal_sandbox_id: c for c in db_containers}

        # 3. Find discrepancies
        modal_ids = set(modal_by_id.keys())
        db_modal_ids = set(db_by_modal_id.keys())

        # Containers we know about but Modal doesn't (terminated externally)
        terminated = db_modal_ids - modal_ids

        # Containers Modal has but we don't know about (orphans)
        orphans = modal_ids - db_modal_ids

        # Containers in both (check for state drift)
        common = modal_ids & db_modal_ids

        # 4. Handle terminated containers
        for modal_id in terminated:
            container = db_by_modal_id[modal_id]
            await self._handle_terminated(container)

        # 5. Handle orphans
        for modal_id in orphans:
            sandbox = modal_by_id[modal_id]
            await self._handle_orphan(sandbox)

        # 6. Reconcile common containers
        for modal_id in common:
            container = db_by_modal_id[modal_id]
            sandbox = modal_by_id[modal_id]
            await self._reconcile_container(container, sandbox)

        # 7. Detect stale containers (no heartbeat)
        await self._detect_stale_containers()

    async def _handle_terminated(self, container: Container):
        """Handle container that Modal no longer reports."""
        old_state = container.state
        container.state = "terminated"
        container.terminated_at = datetime.utcnow().isoformat()
        container.termination_reason = "external"  # Terminated outside Parhelia

        await self.store.update_container(container)
        await self.events.log(
            event_type="container_terminated",
            container_id=container.id,
            old_value=old_state,
            new_value="terminated",
            message=f"Container {container.modal_sandbox_id} no longer running in Modal",
            source="reconciler",
        )

        # Update associated worker
        if container.worker_id:
            await self.store.update_worker_state(
                container.worker_id,
                WorkerState.TERMINATED,
            )

    async def _handle_orphan(self, sandbox: ModalSandbox):
        """Handle container running in Modal but unknown to Parhelia."""
        # Check if it's a Parhelia container (has PARHELIA_TASK_ID env var)
        if not self._is_parhelia_container(sandbox):
            return  # Not ours, ignore

        # Create orphan record
        container = Container(
            id=str(uuid.uuid4()),
            modal_sandbox_id=sandbox.id,
            state="orphaned",
            created_at=sandbox.created_at or datetime.utcnow().isoformat(),
            health_status="unknown",
            region=sandbox.region,
        )

        # Try to extract task_id from sandbox environment
        task_id = self._extract_task_id(sandbox)
        if task_id:
            container.task_id = task_id

        await self.store.create_container(container)
        await self.events.log(
            event_type="orphan_detected",
            container_id=container.id,
            message=f"Orphan container detected: {sandbox.id}",
            details={"modal_sandbox_id": sandbox.id, "task_id": task_id},
            source="reconciler",
        )

        # Auto-terminate if configured
        if self.config.auto_terminate_orphans:
            await self._terminate_orphan(container, sandbox)

    async def _reconcile_container(self, container: Container, sandbox: ModalSandbox):
        """Reconcile known container with Modal state."""
        changes = []

        # Check state
        modal_state = self._map_modal_state(sandbox.status)
        if container.state != modal_state:
            changes.append(("state", container.state, modal_state))
            container.state = modal_state

        # Update resource usage if available
        if sandbox.resources:
            container.cost_accrued_usd = sandbox.resources.cost_usd

        container.updated_at = datetime.utcnow().isoformat()

        if changes:
            await self.store.update_container(container)
            for field, old, new in changes:
                await self.events.log(
                    event_type="state_drift_corrected",
                    container_id=container.id,
                    old_value=old,
                    new_value=new,
                    message=f"Container {field} corrected: {old} -> {new}",
                    source="reconciler",
                )

    async def _detect_stale_containers(self):
        """Detect containers without recent heartbeats."""
        threshold = datetime.utcnow() - timedelta(
            seconds=self.config.stale_threshold_seconds
        )

        stale = await self.store.get_containers_without_heartbeat_since(threshold)

        for container in stale:
            if container.health_status != "unhealthy":
                container.health_status = "unhealthy"
                container.consecutive_failures += 1
                await self.store.update_container(container)
                await self.events.log(
                    event_type="container_unhealthy",
                    container_id=container.id,
                    message=f"No heartbeat for {self.config.stale_threshold_seconds}s",
                    details={"consecutive_failures": container.consecutive_failures},
                    source="reconciler",
                )
```

### [SPEC-21.21] Health Monitor

```python
class HealthMonitor:
    """Monitors container health via heartbeats."""

    THRESHOLDS = {
        "healthy": 0,           # 0 missed heartbeats
        "degraded": 2,          # 2 missed heartbeats
        "unhealthy": 5,         # 5 missed heartbeats
        "dead": 10,             # 10 missed heartbeats
    }

    async def record_heartbeat(self, heartbeat: Heartbeat):
        """Record heartbeat and update container health."""
        container = await self.store.get_container(heartbeat.container_id)
        if not container:
            return  # Unknown container, ignore

        # Store heartbeat
        await self.store.create_heartbeat(heartbeat)

        # Update container health
        old_status = container.health_status
        container.health_status = "healthy"
        container.consecutive_failures = 0
        container.last_heartbeat_at = heartbeat.timestamp

        await self.store.update_container(container)

        if old_status != "healthy":
            await self.events.log(
                event_type="container_recovered",
                container_id=container.id,
                old_value=old_status,
                new_value="healthy",
                message="Container health recovered",
                source="health_monitor",
            )

    def calculate_health_status(self, missed_heartbeats: int) -> str:
        """Determine health status from missed heartbeat count."""
        if missed_heartbeats >= self.THRESHOLDS["dead"]:
            return "dead"
        elif missed_heartbeats >= self.THRESHOLDS["unhealthy"]:
            return "unhealthy"
        elif missed_heartbeats >= self.THRESHOLDS["degraded"]:
            return "degraded"
        return "healthy"
```

---

## Introspection APIs

### [SPEC-21.30] StateStore Query Interface

```python
class StateStore:
    """Unified interface for control plane state queries."""

    # Container queries
    async def get_container(self, id: str) -> Container | None: ...
    async def get_container_by_modal_id(self, modal_id: str) -> Container | None: ...
    async def get_active_containers(self) -> list[Container]: ...
    async def get_containers_by_state(self, state: str) -> list[Container]: ...
    async def get_containers_by_health(self, health: str) -> list[Container]: ...
    async def get_orphaned_containers(self) -> list[Container]: ...
    async def get_containers_for_task(self, task_id: str) -> list[Container]: ...

    # Aggregations
    async def get_container_stats(self) -> ContainerStats:
        """Get aggregate container statistics."""
        return ContainerStats(
            total=await self._count_containers(),
            by_state=await self._count_by_state(),
            by_health=await self._count_by_health(),
            total_cost_usd=await self._sum_cost(),
            oldest_running=await self._oldest_running(),
        )

    # Event queries
    async def get_events(
        self,
        container_id: str | None = None,
        task_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Event]: ...

    # Heartbeat queries
    async def get_heartbeat_history(
        self,
        container_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Heartbeat]: ...

    async def get_containers_without_heartbeat_since(
        self,
        threshold: datetime,
    ) -> list[Container]: ...
```

### [SPEC-21.31] CLI Commands

```bash
# List all containers with current state
$ parhelia containers
ID          MODAL ID     STATE      HEALTH     TASK         AGE      COST
c-abc123    sb-xyz789    running    healthy    ta-def456    2h 15m   $1.23
c-def456    sb-uvw123    running    degraded   ta-ghi789    45m      $0.45
c-ghi789    sb-rst456    orphaned   unknown    -            3h 20m   $2.10

Total: 3 containers | Running: 2 | Orphaned: 1 | Cost: $3.78/hr

# Show detailed container info
$ parhelia containers show c-abc123
Container: c-abc123
  Modal ID:     sb-xyz789
  State:        running
  Health:       healthy (last heartbeat: 30s ago)
  Task:         ta-def456 (Run tests for auth module)
  Worker:       w-mno345
  Session:      sess-pqr678

  Resources:
    CPU:        4 cores (45% used)
    Memory:     8192 MB (62% used)
    GPU:        None
    Region:     us-east

  Lifecycle:
    Created:    2026-01-21 14:30:00 UTC
    Started:    2026-01-21 14:30:15 UTC
    Uptime:     2h 15m

  Cost:
    Rate:       $0.54/hr
    Accrued:    $1.23

# Show container events
$ parhelia containers events c-abc123 --limit 10
TIMESTAMP                EVENT              MESSAGE
2026-01-21 14:30:00     created            Container created for task ta-def456
2026-01-21 14:30:15     started            Modal sandbox sb-xyz789 started
2026-01-21 14:30:20     healthy            Initial health check passed
2026-01-21 15:45:00     degraded           Missed 2 heartbeats
2026-01-21 15:45:30     healthy            Health recovered

# Show orphaned containers
$ parhelia containers orphans
ID          MODAL ID     AGE      COST     TASK (if known)
c-ghi789    sb-rst456    3h 20m   $2.10    -

1 orphan detected. Run 'parhelia cleanup --orphans' to terminate.

# Show container health overview
$ parhelia containers health
HEALTH      COUNT    CONTAINERS
healthy     2        c-abc123, c-def456
degraded    1        c-jkl012
unhealthy   0        -
dead        0        -
orphaned    1        c-ghi789

# Real-time container status
$ parhelia containers watch
[14:32:15] c-abc123  healthy   CPU: 45%  MEM: 62%  ↑ 2h 15m
[14:32:15] c-def456  degraded  CPU: 78%  MEM: 85%  ↑ 45m
[14:32:16] c-abc123  healthy   CPU: 42%  MEM: 61%  ↑ 2h 15m
^C

# Terminate specific container
$ parhelia containers terminate c-ghi789
Terminating container c-ghi789 (sb-rst456)...
Container terminated.

# Show reconciliation status
$ parhelia reconciler status
Reconciler: running
  Last run:       30 seconds ago
  Next run:       in 30 seconds
  Poll interval:  60 seconds

  Last cycle:
    Modal containers:   5
    DB containers:      4
    Orphans detected:   1
    Terminated:         0
    State corrections:  0
```

### [SPEC-21.32] MCP Tools

```python
# Tool: parhelia_containers
{
    "name": "parhelia_containers",
    "description": "List and manage running containers",
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "show", "terminate", "events"],
                "default": "list"
            },
            "container_id": {
                "type": "string",
                "description": "Container ID (required for show/terminate/events)"
            },
            "state_filter": {
                "type": "string",
                "enum": ["all", "running", "orphaned", "terminated"],
                "default": "running"
            },
            "health_filter": {
                "type": "string",
                "enum": ["all", "healthy", "degraded", "unhealthy", "dead"],
                "default": "all"
            },
            "limit": {
                "type": "integer",
                "default": 20
            }
        }
    }
}

# Tool: parhelia_health
{
    "name": "parhelia_health",
    "description": "Get control plane health status",
    "inputSchema": {
        "type": "object",
        "properties": {
            "include_containers": {
                "type": "boolean",
                "default": true
            },
            "include_reconciler": {
                "type": "boolean",
                "default": true
            }
        }
    }
}

# Tool: parhelia_events
{
    "name": "parhelia_events",
    "description": "Query control plane events",
    "inputSchema": {
        "type": "object",
        "properties": {
            "container_id": {"type": "string"},
            "task_id": {"type": "string"},
            "event_type": {"type": "string"},
            "since_minutes": {"type": "integer", "default": 60},
            "limit": {"type": "integer", "default": 50}
        }
    }
}
```

---

## Container Lifecycle

### [SPEC-21.40] Lifecycle States

```
                    ┌─────────────┐
                    │   created   │
                    └──────┬──────┘
                           │ Modal sandbox starts
                           ▼
                    ┌─────────────┐
         ┌─────────│   running   │─────────┐
         │         └──────┬──────┘         │
         │                │                │
         │ healthy        │ heartbeat      │ missed heartbeats
         │ heartbeats     │ timeout        │
         ▼                ▼                ▼
┌─────────────┐    ┌─────────────┐   ┌─────────────┐
│   healthy   │◀──▶│  degraded   │──▶│  unhealthy  │
└─────────────┘    └─────────────┘   └──────┬──────┘
                                            │
                                            │ no recovery
                                            ▼
                                     ┌─────────────┐
                                     │    dead     │
                                     └──────┬──────┘
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         │                                  │                                  │
         ▼                                  ▼                                  ▼
┌─────────────┐                     ┌─────────────┐                    ┌─────────────┐
│ terminated  │◀────────────────────│   stopped   │                    │  orphaned   │
│  (normal)   │   manual/timeout    └─────────────┘                    │ (no record) │
└─────────────┘                                                        └─────────────┘
```

### [SPEC-21.41] State Transitions

| From | To | Trigger | Action |
|------|----|---------|--------|
| created | running | Modal reports started | Update started_at |
| running | healthy | Heartbeat received | Reset failure count |
| healthy | degraded | 2 missed heartbeats | Log warning |
| degraded | healthy | Heartbeat received | Log recovery |
| degraded | unhealthy | 5 missed heartbeats | Log error, notify |
| unhealthy | dead | 10 missed heartbeats | Mark for cleanup |
| running | stopped | Manual stop or idle timeout | Record reason |
| stopped | terminated | Cleanup | Record terminated_at |
| * | orphaned | Reconciler finds unknown container | Create record |
| orphaned | terminated | Manual cleanup or auto-terminate | Record reason |

---

## Implementation Phases

### Phase 1: Data Model & Storage
- [ ] [SPEC-21.P1.1] Create containers table with schema
- [ ] [SPEC-21.P1.2] Create events table with schema
- [ ] [SPEC-21.P1.3] Create heartbeats table with schema
- [ ] [SPEC-21.P1.4] Add columns to workers table
- [ ] [SPEC-21.P1.5] Implement StateStore class with basic CRUD
- [ ] [SPEC-21.P1.6] Add database migrations

### Phase 2: Container Registration
- [ ] [SPEC-21.P2.1] Modify dispatch to register containers with Modal sandbox IDs
- [ ] [SPEC-21.P2.2] Update container state on lifecycle events
- [ ] [SPEC-21.P2.3] Link containers to workers and tasks
- [ ] [SPEC-21.P2.4] Emit events on state changes

### Phase 3: Heartbeat Persistence
- [ ] [SPEC-21.P3.1] Modify HeartbeatMonitor to persist to database
- [ ] [SPEC-21.P3.2] Update container health on heartbeat receipt
- [ ] [SPEC-21.P3.3] Add heartbeat history queries
- [ ] [SPEC-21.P3.4] Add heartbeat retention/cleanup job

### Phase 4: Reconciliation Loop
- [ ] [SPEC-21.P4.1] Implement Modal API client for sandbox listing
- [ ] [SPEC-21.P4.2] Implement ContainerReconciler class
- [ ] [SPEC-21.P4.3] Add orphan detection and registration
- [ ] [SPEC-21.P4.4] Add state drift correction
- [ ] [SPEC-21.P4.5] Add stale container detection
- [ ] [SPEC-21.P4.6] Add background reconciler process

### Phase 5: CLI Introspection
- [ ] [SPEC-21.P5.1] Add `parhelia containers` command
- [ ] [SPEC-21.P5.2] Add `parhelia containers show <id>` subcommand
- [ ] [SPEC-21.P5.3] Add `parhelia containers events` subcommand
- [ ] [SPEC-21.P5.4] Add `parhelia containers health` subcommand
- [ ] [SPEC-21.P5.5] Add `parhelia containers watch` real-time view
- [ ] [SPEC-21.P5.6] Add `parhelia reconciler status` command

### Phase 6: MCP Integration
- [ ] [SPEC-21.P6.1] Add parhelia_containers MCP tool
- [ ] [SPEC-21.P6.2] Add parhelia_health MCP tool
- [ ] [SPEC-21.P6.3] Add parhelia_events MCP tool
- [ ] [SPEC-21.P6.4] Update existing tools to include container info

---

## Acceptance Criteria

- [ ] [SPEC-21.AC1] All Modal sandbox IDs persisted in containers table
- [ ] [SPEC-21.AC2] Reconciler detects orphan containers within 2 minutes
- [ ] [SPEC-21.AC3] Stale containers (no heartbeat >5min) flagged as unhealthy
- [ ] [SPEC-21.AC4] `parhelia containers` shows accurate real-time state
- [ ] [SPEC-21.AC5] Events queryable by container, task, time range
- [ ] [SPEC-21.AC6] MCP tools return consistent data with CLI
- [ ] [SPEC-21.AC7] Cleanup command removes orphans safely
- [ ] [SPEC-21.AC8] No container can run >4 hours without heartbeat persistence

---

## Cost Analysis

This infrastructure prevents incidents like the $140 cost overrun by:

1. **Detection**: Reconciler catches orphans within 60 seconds
2. **Visibility**: CLI/MCP show all running containers and costs
3. **Automation**: Optional auto-terminate for orphans and stale containers
4. **Audit**: Event history for post-mortem analysis

Estimated overhead:
- SQLite storage: ~1KB per container, ~100 bytes per heartbeat, ~200 bytes per event
- Modal API calls: 1 list call per minute (~$0.00 - free tier)
- CPU: Negligible (async reconciliation)

---

## References

- SPEC-05: Local Orchestrator (worker lifecycle)
- SPEC-01: Remote Environment (container provisioning)
- SPEC-03: Checkpoint and Resume (session recovery)
- [Modal Sandbox API](https://modal.com/docs/reference/modal.Sandbox)
- [Modal Container Lifecycle](https://modal.com/docs/guide/lifecycle)
