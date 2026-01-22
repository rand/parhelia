"""Control plane state management.

Provides stateful tracking of containers, events, and heartbeats for
orchestrator introspection, recovery, and resource management.

Implements:
- [SPEC-21.10] Container Registry Schema
- [SPEC-21.11] Events Table Schema
- [SPEC-21.12] Heartbeat History Schema
- [SPEC-21.13] Workers Table Extensions
- [SPEC-21.30] StateStore Query Interface
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generator


class ContainerState(str, Enum):
    """Container lifecycle states."""

    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ORPHANED = "orphaned"
    UNKNOWN = "unknown"


class HealthStatus(str, Enum):
    """Container health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DEAD = "dead"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    """Control plane event types."""

    # Lifecycle events
    CONTAINER_CREATED = "container_created"
    CONTAINER_STARTED = "container_started"
    CONTAINER_STOPPED = "container_stopped"
    CONTAINER_TERMINATED = "container_terminated"

    # Health events
    CONTAINER_HEALTHY = "container_healthy"
    CONTAINER_DEGRADED = "container_degraded"
    CONTAINER_UNHEALTHY = "container_unhealthy"
    CONTAINER_DEAD = "container_dead"
    CONTAINER_RECOVERED = "container_recovered"

    # Reconciliation events
    ORPHAN_DETECTED = "orphan_detected"
    STATE_DRIFT_CORRECTED = "state_drift_corrected"
    RECONCILE_FAILED = "reconcile_failed"

    # Heartbeat events
    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_MISSED = "heartbeat_missed"

    # Error events
    ERROR = "error"


@dataclass
class Container:
    """Container instance tracked by the control plane.

    Maps Parhelia's internal IDs to Modal sandbox IDs and tracks
    lifecycle, health, and resource usage.
    """

    id: str
    modal_sandbox_id: str
    state: ContainerState = ContainerState.UNKNOWN
    health_status: HealthStatus = HealthStatus.UNKNOWN

    # Relationships
    worker_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None

    # Lifecycle timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    terminated_at: datetime | None = None

    # Termination info
    exit_code: int | None = None
    termination_reason: str | None = None

    # Health tracking
    last_heartbeat_at: datetime | None = None
    consecutive_failures: int = 0

    # Resources
    cpu_cores: int | None = None
    memory_mb: int | None = None
    gpu_type: str | None = None
    region: str | None = None

    # Cost tracking
    cost_accrued_usd: float = 0.0
    cost_rate_per_hour: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        modal_sandbox_id: str,
        task_id: str | None = None,
        worker_id: str | None = None,
        **kwargs,
    ) -> Container:
        """Create a new container with generated ID."""
        return cls(
            id=f"c-{uuid.uuid4().hex[:8]}",
            modal_sandbox_id=modal_sandbox_id,
            task_id=task_id,
            worker_id=worker_id,
            state=ContainerState.CREATED,
            **kwargs,
        )


@dataclass
class Event:
    """Control plane event for audit and debugging.

    Events are immutable once created and provide a complete
    history of state changes and system activity.
    """

    id: int | None  # Auto-assigned by database
    timestamp: datetime
    event_type: EventType

    # References (all optional)
    container_id: str | None = None
    worker_id: str | None = None
    task_id: str | None = None
    session_id: str | None = None

    # Event data
    old_value: str | None = None
    new_value: str | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    # Source
    source: str = "system"

    @classmethod
    def create(
        cls,
        event_type: EventType,
        message: str | None = None,
        **kwargs,
    ) -> Event:
        """Create a new event with current timestamp."""
        return cls(
            id=None,
            timestamp=datetime.utcnow(),
            event_type=event_type,
            message=message,
            **kwargs,
        )


@dataclass
class Heartbeat:
    """Container heartbeat record.

    Heartbeats are received periodically from running containers
    and used to track health and detect failures.
    """

    id: int | None  # Auto-assigned by database
    container_id: str
    timestamp: datetime

    # Health metrics
    cpu_percent: float | None = None
    memory_percent: float | None = None
    memory_mb: int | None = None
    disk_percent: float | None = None

    # Container state
    uptime_seconds: int | None = None
    tmux_active: bool = False
    claude_responsive: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, container_id: str, **kwargs) -> Heartbeat:
        """Create a new heartbeat with current timestamp."""
        return cls(
            id=None,
            container_id=container_id,
            timestamp=datetime.utcnow(),
            **kwargs,
        )


@dataclass
class ContainerStats:
    """Aggregate container statistics."""

    total: int
    by_state: dict[str, int]
    by_health: dict[str, int]
    total_cost_usd: float
    oldest_running: datetime | None


class ContainerStore:
    """SQLite-backed container storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS containers (
                    id TEXT PRIMARY KEY,
                    modal_sandbox_id TEXT UNIQUE NOT NULL,
                    state TEXT NOT NULL DEFAULT 'unknown',
                    health_status TEXT NOT NULL DEFAULT 'unknown',

                    worker_id TEXT,
                    task_id TEXT,
                    session_id TEXT,

                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    terminated_at TEXT,

                    exit_code INTEGER,
                    termination_reason TEXT,

                    last_heartbeat_at TEXT,
                    consecutive_failures INTEGER DEFAULT 0,

                    cpu_cores INTEGER,
                    memory_mb INTEGER,
                    gpu_type TEXT,
                    region TEXT,

                    cost_accrued_usd REAL DEFAULT 0.0,
                    cost_rate_per_hour REAL,

                    metadata TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_containers_state
                ON containers(state)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_containers_modal_id
                ON containers(modal_sandbox_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_containers_health
                ON containers(health_status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_containers_worker
                ON containers(worker_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_containers_task
                ON containers(task_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_containers_updated
                ON containers(updated_at)
            """)

    def save(self, container: Container) -> None:
        """Save or update a container."""
        container.updated_at = datetime.utcnow()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO containers (
                    id, modal_sandbox_id, state, health_status,
                    worker_id, task_id, session_id,
                    created_at, started_at, terminated_at,
                    exit_code, termination_reason,
                    last_heartbeat_at, consecutive_failures,
                    cpu_cores, memory_mb, gpu_type, region,
                    cost_accrued_usd, cost_rate_per_hour,
                    metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    container.id,
                    container.modal_sandbox_id,
                    container.state.value,
                    container.health_status.value,
                    container.worker_id,
                    container.task_id,
                    container.session_id,
                    container.created_at.isoformat(),
                    container.started_at.isoformat() if container.started_at else None,
                    container.terminated_at.isoformat() if container.terminated_at else None,
                    container.exit_code,
                    container.termination_reason,
                    container.last_heartbeat_at.isoformat() if container.last_heartbeat_at else None,
                    container.consecutive_failures,
                    container.cpu_cores,
                    container.memory_mb,
                    container.gpu_type,
                    container.region,
                    container.cost_accrued_usd,
                    container.cost_rate_per_hour,
                    json.dumps(container.metadata),
                    container.updated_at.isoformat(),
                ),
            )

    def get(self, container_id: str) -> Container | None:
        """Get container by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM containers WHERE id = ?", (container_id,)
            ).fetchone()
            return self._row_to_container(row) if row else None

    def get_by_modal_id(self, modal_sandbox_id: str) -> Container | None:
        """Get container by Modal sandbox ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM containers WHERE modal_sandbox_id = ?",
                (modal_sandbox_id,),
            ).fetchone()
            return self._row_to_container(row) if row else None

    def get_by_task(self, task_id: str) -> list[Container]:
        """Get all containers for a task."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM containers WHERE task_id = ? ORDER BY created_at DESC",
                (task_id,),
            ).fetchall()
            return [self._row_to_container(row) for row in rows]

    def get_by_worker(self, worker_id: str) -> Container | None:
        """Get container by worker ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM containers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
            return self._row_to_container(row) if row else None

    def list_active(self, limit: int = 100) -> list[Container]:
        """List active (running/created) containers."""
        active_states = [ContainerState.RUNNING.value, ContainerState.CREATED.value]
        with self._connection() as conn:
            placeholders = ",".join("?" * len(active_states))
            rows = conn.execute(
                f"""
                SELECT * FROM containers
                WHERE state IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (*active_states, limit),
            ).fetchall()
            return [self._row_to_container(row) for row in rows]

    def list_by_state(self, state: ContainerState, limit: int = 100) -> list[Container]:
        """List containers by state."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM containers
                WHERE state = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (state.value, limit),
            ).fetchall()
            return [self._row_to_container(row) for row in rows]

    def list_by_health(self, health: HealthStatus, limit: int = 100) -> list[Container]:
        """List containers by health status."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM containers
                WHERE health_status = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (health.value, limit),
            ).fetchall()
            return [self._row_to_container(row) for row in rows]

    def list_orphaned(self, limit: int = 100) -> list[Container]:
        """List orphaned containers."""
        return self.list_by_state(ContainerState.ORPHANED, limit)

    def list_without_heartbeat_since(
        self, threshold: datetime, limit: int = 100
    ) -> list[Container]:
        """List active containers without heartbeat since threshold."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM containers
                WHERE state IN ('running', 'created')
                AND (last_heartbeat_at IS NULL OR last_heartbeat_at < ?)
                ORDER BY last_heartbeat_at ASC
                LIMIT ?
                """,
                (threshold.isoformat(), limit),
            ).fetchall()
            return [self._row_to_container(row) for row in rows]

    def delete(self, container_id: str) -> bool:
        """Delete a container."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM containers WHERE id = ?", (container_id,)
            )
            return cursor.rowcount > 0

    def get_stats(self) -> ContainerStats:
        """Get aggregate container statistics."""
        with self._connection() as conn:
            # Count by state
            state_rows = conn.execute(
                "SELECT state, COUNT(*) as count FROM containers GROUP BY state"
            ).fetchall()
            by_state = {row["state"]: row["count"] for row in state_rows}

            # Count by health
            health_rows = conn.execute(
                "SELECT health_status, COUNT(*) as count FROM containers GROUP BY health_status"
            ).fetchall()
            by_health = {row["health_status"]: row["count"] for row in health_rows}

            # Total cost
            cost_row = conn.execute(
                "SELECT SUM(cost_accrued_usd) as total FROM containers"
            ).fetchone()
            total_cost = cost_row["total"] or 0.0

            # Oldest running
            oldest_row = conn.execute(
                """
                SELECT MIN(created_at) as oldest FROM containers
                WHERE state = 'running'
                """
            ).fetchone()
            oldest_running = (
                datetime.fromisoformat(oldest_row["oldest"])
                if oldest_row["oldest"]
                else None
            )

            return ContainerStats(
                total=sum(by_state.values()),
                by_state=by_state,
                by_health=by_health,
                total_cost_usd=total_cost,
                oldest_running=oldest_running,
            )

    def _row_to_container(self, row: sqlite3.Row) -> Container:
        """Convert database row to Container."""
        return Container(
            id=row["id"],
            modal_sandbox_id=row["modal_sandbox_id"],
            state=ContainerState(row["state"]),
            health_status=HealthStatus(row["health_status"]),
            worker_id=row["worker_id"],
            task_id=row["task_id"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            terminated_at=datetime.fromisoformat(row["terminated_at"]) if row["terminated_at"] else None,
            exit_code=row["exit_code"],
            termination_reason=row["termination_reason"],
            last_heartbeat_at=datetime.fromisoformat(row["last_heartbeat_at"]) if row["last_heartbeat_at"] else None,
            consecutive_failures=row["consecutive_failures"],
            cpu_cores=row["cpu_cores"],
            memory_mb=row["memory_mb"],
            gpu_type=row["gpu_type"],
            region=row["region"],
            cost_accrued_usd=row["cost_accrued_usd"],
            cost_rate_per_hour=row["cost_rate_per_hour"],
            metadata=json.loads(row["metadata"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


class EventStore:
    """SQLite-backed event storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,

                    container_id TEXT,
                    worker_id TEXT,
                    task_id TEXT,
                    session_id TEXT,

                    old_value TEXT,
                    new_value TEXT,
                    message TEXT,
                    details TEXT NOT NULL DEFAULT '{}',

                    source TEXT NOT NULL DEFAULT 'system'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_container
                ON events(container_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_task
                ON events(task_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp
                ON events(event_type, timestamp)
            """)

    def save(self, event: Event) -> int:
        """Save an event and return its ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    timestamp, event_type,
                    container_id, worker_id, task_id, session_id,
                    old_value, new_value, message, details, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.container_id,
                    event.worker_id,
                    event.task_id,
                    event.session_id,
                    event.old_value,
                    event.new_value,
                    event.message,
                    json.dumps(event.details),
                    event.source,
                ),
            )
            return cursor.lastrowid

    def get(self, event_id: int) -> Event | None:
        """Get event by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            return self._row_to_event(row) if row else None

    def list(
        self,
        container_id: str | None = None,
        task_id: str | None = None,
        event_type: EventType | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        """List events with optional filters."""
        conditions = []
        params: list[Any] = []

        if container_id:
            conditions.append("container_id = ?")
            params.append(container_id)
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM events
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [self._row_to_event(row) for row in rows]

    def count(
        self,
        event_type: EventType | None = None,
        since: datetime | None = None,
    ) -> int:
        """Count events with optional filters."""
        conditions = []
        params: list[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as count FROM events WHERE {where_clause}",
                params,
            ).fetchone()
            return row["count"]

    def delete_before(self, before: datetime) -> int:
        """Delete events before a timestamp (for retention)."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (before.isoformat(),),
            )
            return cursor.rowcount

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert database row to Event."""
        return Event(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=EventType(row["event_type"]),
            container_id=row["container_id"],
            worker_id=row["worker_id"],
            task_id=row["task_id"],
            session_id=row["session_id"],
            old_value=row["old_value"],
            new_value=row["new_value"],
            message=row["message"],
            details=json.loads(row["details"]),
            source=row["source"],
        )


class HeartbeatStore:
    """SQLite-backed heartbeat storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS heartbeats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    container_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,

                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_mb INTEGER,
                    disk_percent REAL,

                    uptime_seconds INTEGER,
                    tmux_active INTEGER,
                    claude_responsive INTEGER,

                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_heartbeats_container
                ON heartbeats(container_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_heartbeats_timestamp
                ON heartbeats(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_heartbeats_container_time
                ON heartbeats(container_id, timestamp)
            """)

    def save(self, heartbeat: Heartbeat) -> int:
        """Save a heartbeat and return its ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO heartbeats (
                    container_id, timestamp,
                    cpu_percent, memory_percent, memory_mb, disk_percent,
                    uptime_seconds, tmux_active, claude_responsive,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    heartbeat.container_id,
                    heartbeat.timestamp.isoformat(),
                    heartbeat.cpu_percent,
                    heartbeat.memory_percent,
                    heartbeat.memory_mb,
                    heartbeat.disk_percent,
                    heartbeat.uptime_seconds,
                    1 if heartbeat.tmux_active else 0,
                    1 if heartbeat.claude_responsive else 0,
                    json.dumps(heartbeat.metadata),
                ),
            )
            return cursor.lastrowid

    def get_latest(self, container_id: str) -> Heartbeat | None:
        """Get most recent heartbeat for a container."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM heartbeats
                WHERE container_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (container_id,),
            ).fetchone()
            return self._row_to_heartbeat(row) if row else None

    def list(
        self,
        container_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Heartbeat]:
        """List heartbeats for a container."""
        with self._connection() as conn:
            if since:
                rows = conn.execute(
                    """
                    SELECT * FROM heartbeats
                    WHERE container_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (container_id, since.isoformat(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM heartbeats
                    WHERE container_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (container_id, limit),
                ).fetchall()
            return [self._row_to_heartbeat(row) for row in rows]

    def delete_before(self, before: datetime) -> int:
        """Delete heartbeats before a timestamp (for retention)."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM heartbeats WHERE timestamp < ?",
                (before.isoformat(),),
            )
            return cursor.rowcount

    def _row_to_heartbeat(self, row: sqlite3.Row) -> Heartbeat:
        """Convert database row to Heartbeat."""
        return Heartbeat(
            id=row["id"],
            container_id=row["container_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            cpu_percent=row["cpu_percent"],
            memory_percent=row["memory_percent"],
            memory_mb=row["memory_mb"],
            disk_percent=row["disk_percent"],
            uptime_seconds=row["uptime_seconds"],
            tmux_active=bool(row["tmux_active"]),
            claude_responsive=bool(row["claude_responsive"]),
            metadata=json.loads(row["metadata"]),
        )


class StateStore:
    """Unified interface for control plane state.

    Provides a single access point for containers, events, and heartbeats
    with convenience methods for common queries.

    Implements [SPEC-21.30] StateStore Query Interface.
    """

    DEFAULT_DB_PATH = ".parhelia/state.db"

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the state store.

        Args:
            db_path: Path to database file. Defaults to .parhelia/state.db
        """
        if db_path is None:
            db_path = Path.cwd() / self.DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.containers = ContainerStore(self.db_path)
        self.events = EventStore(self.db_path)
        self.heartbeats = HeartbeatStore(self.db_path)

    # Container operations

    def create_container(self, container: Container) -> None:
        """Create a new container and emit creation event."""
        self.containers.save(container)
        self.events.save(
            Event.create(
                EventType.CONTAINER_CREATED,
                container_id=container.id,
                task_id=container.task_id,
                message=f"Container created: {container.modal_sandbox_id}",
                details={
                    "modal_sandbox_id": container.modal_sandbox_id,
                    "task_id": container.task_id,
                    "worker_id": container.worker_id,
                },
            )
        )

    def update_container(self, container: Container) -> None:
        """Update an existing container."""
        self.containers.save(container)

    def update_container_state(
        self,
        container_id: str,
        new_state: ContainerState,
        reason: str | None = None,
    ) -> None:
        """Update container state and emit event."""
        container = self.containers.get(container_id)
        if not container:
            return

        old_state = container.state
        container.state = new_state

        if new_state == ContainerState.RUNNING:
            container.started_at = datetime.utcnow()
        elif new_state in (ContainerState.STOPPED, ContainerState.TERMINATED):
            container.terminated_at = datetime.utcnow()
            container.termination_reason = reason

        self.containers.save(container)

        # Emit appropriate event
        event_type_map = {
            ContainerState.RUNNING: EventType.CONTAINER_STARTED,
            ContainerState.STOPPED: EventType.CONTAINER_STOPPED,
            ContainerState.TERMINATED: EventType.CONTAINER_TERMINATED,
            ContainerState.ORPHANED: EventType.ORPHAN_DETECTED,
        }
        event_type = event_type_map.get(new_state, EventType.CONTAINER_CREATED)

        self.events.save(
            Event.create(
                event_type,
                container_id=container_id,
                task_id=container.task_id,
                old_value=old_state.value,
                new_value=new_state.value,
                message=f"Container state: {old_state.value} -> {new_state.value}",
                details={"reason": reason} if reason else {},
            )
        )

    def update_container_health(
        self,
        container_id: str,
        new_health: HealthStatus,
    ) -> None:
        """Update container health and emit event if changed."""
        container = self.containers.get(container_id)
        if not container:
            return

        old_health = container.health_status
        if old_health == new_health:
            return

        container.health_status = new_health

        # Update consecutive failures
        if new_health in (HealthStatus.UNHEALTHY, HealthStatus.DEAD):
            container.consecutive_failures += 1
        elif new_health == HealthStatus.HEALTHY:
            container.consecutive_failures = 0

        self.containers.save(container)

        # Emit event
        event_type_map = {
            HealthStatus.HEALTHY: EventType.CONTAINER_HEALTHY,
            HealthStatus.DEGRADED: EventType.CONTAINER_DEGRADED,
            HealthStatus.UNHEALTHY: EventType.CONTAINER_UNHEALTHY,
            HealthStatus.DEAD: EventType.CONTAINER_DEAD,
        }
        event_type = event_type_map.get(new_health, EventType.CONTAINER_HEALTHY)

        # Check for recovery
        if (
            old_health in (HealthStatus.UNHEALTHY, HealthStatus.DEAD, HealthStatus.DEGRADED)
            and new_health == HealthStatus.HEALTHY
        ):
            event_type = EventType.CONTAINER_RECOVERED

        self.events.save(
            Event.create(
                event_type,
                container_id=container_id,
                task_id=container.task_id,
                old_value=old_health.value,
                new_value=new_health.value,
                message=f"Container health: {old_health.value} -> {new_health.value}",
            )
        )

    def get_container(self, container_id: str) -> Container | None:
        """Get container by ID."""
        return self.containers.get(container_id)

    def get_container_by_modal_id(self, modal_sandbox_id: str) -> Container | None:
        """Get container by Modal sandbox ID."""
        return self.containers.get_by_modal_id(modal_sandbox_id)

    def get_active_containers(self) -> list[Container]:
        """Get all active (running/created) containers."""
        return self.containers.list_active()

    def get_containers_by_state(self, state: ContainerState) -> list[Container]:
        """Get containers by state."""
        return self.containers.list_by_state(state)

    def get_containers_by_health(self, health: HealthStatus) -> list[Container]:
        """Get containers by health status."""
        return self.containers.list_by_health(health)

    def get_orphaned_containers(self) -> list[Container]:
        """Get orphaned containers."""
        return self.containers.list_orphaned()

    def get_containers_for_task(self, task_id: str) -> list[Container]:
        """Get all containers for a task."""
        return self.containers.get_by_task(task_id)

    def get_containers_without_heartbeat_since(
        self, threshold: datetime
    ) -> list[Container]:
        """Get containers without recent heartbeat."""
        return self.containers.list_without_heartbeat_since(threshold)

    def get_container_stats(self) -> ContainerStats:
        """Get aggregate container statistics."""
        return self.containers.get_stats()

    # Heartbeat operations

    def record_heartbeat(self, heartbeat: Heartbeat) -> None:
        """Record a heartbeat and update container health."""
        self.heartbeats.save(heartbeat)

        # Update container
        container = self.containers.get(heartbeat.container_id)
        if container:
            container.last_heartbeat_at = heartbeat.timestamp
            self.containers.save(container)

            # Update health to healthy
            if container.health_status != HealthStatus.HEALTHY:
                self.update_container_health(container.id, HealthStatus.HEALTHY)

    def get_heartbeat_history(
        self,
        container_id: str,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Heartbeat]:
        """Get heartbeat history for a container."""
        return self.heartbeats.list(container_id, since, limit)

    # Event operations

    def log_event(
        self,
        event_type: EventType,
        message: str | None = None,
        container_id: str | None = None,
        task_id: str | None = None,
        source: str = "system",
        **details,
    ) -> int:
        """Log an event."""
        return self.events.save(
            Event.create(
                event_type=event_type,
                message=message,
                container_id=container_id,
                task_id=task_id,
                source=source,
                details=details,
            )
        )

    def log_error(self, message: str, source: str = "system", **details) -> int:
        """Log an error event."""
        return self.log_event(
            EventType.ERROR,
            message=message,
            source=source,
            **details,
        )

    def get_events(
        self,
        container_id: str | None = None,
        task_id: str | None = None,
        event_type: EventType | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get events with filters."""
        return self.events.list(
            container_id=container_id,
            task_id=task_id,
            event_type=event_type,
            since=since,
            until=until,
            limit=limit,
        )

    # Maintenance operations

    def cleanup_old_data(self, retention_days: int = 7) -> dict[str, int]:
        """Clean up old heartbeats and events."""
        threshold = datetime.utcnow() - timedelta(days=retention_days)
        return {
            "heartbeats_deleted": self.heartbeats.delete_before(threshold),
            "events_deleted": self.events.delete_before(threshold),
        }
