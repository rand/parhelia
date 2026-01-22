"""Persistence layer for tasks and workers.

Provides SQLite-backed storage for orchestrator state that survives
between CLI invocations.

Implements:
- [SPEC-05.10] Task persistence
- [SPEC-05.12] Worker persistence
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Generator

from parhelia.orchestrator import (
    Task,
    TaskRequirements,
    TaskResult,
    TaskType,
    WorkerInfo,
    WorkerState,
)


class PersistenceError(Exception):
    """Base exception for persistence errors."""


class TaskStore:
    """SQLite-backed task storage.

    Persists tasks across CLI invocations.
    """

    def __init__(self, db_path: str | Path):
        """Initialize the task store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    requirements TEXT NOT NULL,
                    parent_id TEXT,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_results (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    output TEXT NOT NULL,
                    error TEXT,
                    cost_usd REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    artifacts TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
            """)

    def save(self, task: Task, status: str = "pending") -> None:
        """Save a task to the store.

        Args:
            task: The task to save.
            status: Task status (pending, running, completed, failed).
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks
                (id, prompt, task_type, requirements, parent_id, metadata, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.id,
                    task.prompt,
                    task.task_type.value,
                    json.dumps(asdict(task.requirements)),
                    task.parent_id,
                    json.dumps(task.metadata),
                    task.created_at.isoformat(),
                    status,
                ),
            )

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task ID.

        Returns:
            Task if found, None otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_task(row)

    def get_status(self, task_id: str) -> str | None:
        """Get task status.

        Args:
            task_id: The task ID.

        Returns:
            Status string or None if not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return row["status"] if row else None

    def update_status(self, task_id: str, status: str) -> None:
        """Update task status.

        Args:
            task_id: The task ID.
            status: New status.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (status, task_id),
            )

    def list_by_status(self, status: str | None = None, limit: int = 100) -> list[Task]:
        """List tasks, optionally filtered by status.

        Args:
            status: Filter by status (None for all).
            limit: Maximum number to return.

        Returns:
            List of tasks.
        """
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            return [self._row_to_task(row) for row in rows]

    def list_pending(self, limit: int = 100) -> list[Task]:
        """List pending tasks."""
        return self.list_by_status("pending", limit)

    def list_running(self, limit: int = 100) -> list[Task]:
        """List running tasks."""
        return self.list_by_status("running", limit)

    def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: The task ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            return cursor.rowcount > 0

    def save_result(self, result: TaskResult) -> None:
        """Save a task result.

        Args:
            result: The task result.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_results
                (task_id, status, output, error, cost_usd, duration_seconds, artifacts, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.task_id,
                    result.status,
                    result.output,
                    result.error,
                    result.cost_usd,
                    result.duration_seconds,
                    json.dumps(result.artifacts),
                    datetime.now().isoformat(),
                ),
            )
            # Also update task status
            conn.execute(
                "UPDATE tasks SET status = ? WHERE id = ?",
                (result.status, result.task_id),
            )

    def get_result(self, task_id: str) -> TaskResult | None:
        """Get task result.

        Args:
            task_id: The task ID.

        Returns:
            TaskResult if found, None otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM task_results WHERE task_id = ?", (task_id,)
            ).fetchone()

            if not row:
                return None

            return TaskResult(
                task_id=row["task_id"],
                status=row["status"],
                output=row["output"],
                error=row["error"],
                cost_usd=row["cost_usd"],
                duration_seconds=row["duration_seconds"],
                artifacts=json.loads(row["artifacts"]),
            )

    def count_by_status(self) -> dict[str, int]:
        """Get count of tasks by status.

        Returns:
            Dict mapping status to count.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM tasks GROUP BY status"
            ).fetchall()
            return {row["status"]: row["count"] for row in rows}

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert database row to Task."""
        requirements_data = json.loads(row["requirements"])
        return Task(
            id=row["id"],
            prompt=row["prompt"],
            task_type=TaskType(row["task_type"]),
            requirements=TaskRequirements(**requirements_data),
            parent_id=row["parent_id"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )


class WorkerStore:
    """SQLite-backed worker storage.

    Persists worker state across CLI invocations.
    """

    def __init__(self, db_path: str | Path):
        """Initialize the worker store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    gpu_type TEXT,
                    metrics TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(state)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workers_task ON workers(task_id)
            """)
            # Run migrations for new columns [SPEC-21.13]
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Apply schema migrations for SPEC-21 control plane columns."""
        # Get existing columns
        cursor = conn.execute("PRAGMA table_info(workers)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # New columns for SPEC-21.13
        migrations = [
            ("container_id", "TEXT"),
            ("session_id", "TEXT"),
            ("last_heartbeat_at", "TEXT"),
            ("health_status", "TEXT DEFAULT 'unknown'"),
            ("terminated_at", "TEXT"),
            ("exit_code", "INTEGER"),
        ]

        for column_name, column_type in migrations:
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE workers ADD COLUMN {column_name} {column_type}"
                )

        # Add new indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workers_container
            ON workers(container_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workers_session
            ON workers(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workers_health
            ON workers(health_status)
        """)

    def save(self, worker: WorkerInfo) -> None:
        """Save a worker to the store.

        Args:
            worker: The worker to save.
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workers
                (id, task_id, state, target_type, created_at, gpu_type, metrics, updated_at,
                 container_id, session_id, last_heartbeat_at, health_status, terminated_at, exit_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    worker.id,
                    worker.task_id,
                    worker.state.value,
                    worker.target_type,
                    worker.created_at.isoformat(),
                    worker.gpu_type,
                    json.dumps(worker.metrics),
                    datetime.now().isoformat(),
                    worker.container_id,
                    worker.session_id,
                    worker.last_heartbeat_at.isoformat() if worker.last_heartbeat_at else None,
                    worker.health_status,
                    worker.terminated_at.isoformat() if worker.terminated_at else None,
                    worker.exit_code,
                ),
            )

    def get(self, worker_id: str) -> WorkerInfo | None:
        """Get a worker by ID.

        Args:
            worker_id: The worker ID.

        Returns:
            WorkerInfo if found, None otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE id = ?", (worker_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_worker(row)

    def get_by_task(self, task_id: str) -> WorkerInfo | None:
        """Get worker by task ID.

        Args:
            task_id: The task ID.

        Returns:
            WorkerInfo if found, None otherwise.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE task_id = ?", (task_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_worker(row)

    def update_state(self, worker_id: str, state: WorkerState) -> None:
        """Update worker state.

        Args:
            worker_id: The worker ID.
            state: New state.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE workers SET state = ?, updated_at = ? WHERE id = ?",
                (state.value, datetime.now().isoformat(), worker_id),
            )

    def update_metrics(self, worker_id: str, metrics: dict) -> None:
        """Update worker metrics.

        Args:
            worker_id: The worker ID.
            metrics: New metrics dict.
        """
        with self._connection() as conn:
            conn.execute(
                "UPDATE workers SET metrics = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metrics), datetime.now().isoformat(), worker_id),
            )

    def list_active(self, limit: int = 100) -> list[WorkerInfo]:
        """List active workers (not completed/terminated/failed).

        Args:
            limit: Maximum number to return.

        Returns:
            List of active workers.
        """
        active_states = [WorkerState.IDLE.value, WorkerState.RUNNING.value]
        with self._connection() as conn:
            placeholders = ",".join("?" * len(active_states))
            rows = conn.execute(
                f"SELECT * FROM workers WHERE state IN ({placeholders}) ORDER BY created_at DESC LIMIT ?",
                (*active_states, limit),
            ).fetchall()

            return [self._row_to_worker(row) for row in rows]

    def list_by_state(self, state: WorkerState, limit: int = 100) -> list[WorkerInfo]:
        """List workers by state.

        Args:
            state: The state to filter by.
            limit: Maximum number to return.

        Returns:
            List of workers.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM workers WHERE state = ? ORDER BY created_at DESC LIMIT ?",
                (state.value, limit),
            ).fetchall()

            return [self._row_to_worker(row) for row in rows]

    def list_all(self, limit: int = 100) -> list[WorkerInfo]:
        """List all workers.

        Args:
            limit: Maximum number to return.

        Returns:
            List of workers.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM workers ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

            return [self._row_to_worker(row) for row in rows]

    def delete(self, worker_id: str) -> bool:
        """Delete a worker.

        Args:
            worker_id: The worker ID.

        Returns:
            True if deleted, False if not found.
        """
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM workers WHERE id = ?", (worker_id,))
            return cursor.rowcount > 0

    def count_by_state(self) -> dict[str, int]:
        """Get count of workers by state.

        Returns:
            Dict mapping state to count.
        """
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) as count FROM workers GROUP BY state"
            ).fetchall()
            return {row["state"]: row["count"] for row in rows}

    def _row_to_worker(self, row: sqlite3.Row) -> WorkerInfo:
        """Convert database row to WorkerInfo."""
        # Check which columns exist (for backwards compatibility with old DBs)
        columns = row.keys()

        # Helper to safely get nullable datetime fields
        def get_datetime(col: str) -> datetime | None:
            if col in columns and row[col]:
                return datetime.fromisoformat(row[col])
            return None

        return WorkerInfo(
            id=row["id"],
            task_id=row["task_id"],
            state=WorkerState(row["state"]),
            target_type=row["target_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            gpu_type=row["gpu_type"],
            metrics=json.loads(row["metrics"]),
            # SPEC-21.13 extensions
            container_id=row["container_id"] if "container_id" in columns else None,
            session_id=row["session_id"] if "session_id" in columns else None,
            last_heartbeat_at=get_datetime("last_heartbeat_at"),
            health_status=row["health_status"] if "health_status" in columns else "unknown",
            terminated_at=get_datetime("terminated_at"),
            exit_code=row["exit_code"] if "exit_code" in columns else None,
        )


class PersistentOrchestrator:
    """Orchestrator with persistent storage.

    Wraps TaskStore and WorkerStore to provide a unified interface
    for task and worker management with persistence.
    """

    DEFAULT_DB_PATH = ".parhelia/orchestrator.db"

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the persistent orchestrator.

        Args:
            db_path: Path to database file. Defaults to .parhelia/orchestrator.db
        """
        if db_path is None:
            db_path = Path.cwd() / self.DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.task_store = TaskStore(self.db_path)
        self.worker_store = WorkerStore(self.db_path)

    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution.

        Args:
            task: The task to submit.

        Returns:
            The task ID.
        """
        self.task_store.save(task, status="pending")
        return task.id

    async def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self.task_store.get(task_id)

    async def get_workers(self) -> list[WorkerInfo]:
        """Get list of active workers."""
        return self.worker_store.list_active()

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Get list of all workers."""
        return self.worker_store.list_all()

    async def get_pending_tasks(self) -> list[Task]:
        """Get list of pending tasks."""
        return self.task_store.list_pending()

    async def get_running_tasks(self) -> list[Task]:
        """Get list of running tasks."""
        return self.task_store.list_running()

    async def get_all_tasks(self, limit: int = 100) -> list[Task]:
        """Get all tasks."""
        return self.task_store.list_by_status(None, limit)

    def register_worker(self, worker: WorkerInfo) -> None:
        """Register a worker."""
        self.worker_store.save(worker)
        self.task_store.update_status(worker.task_id, "running")

    def unregister_worker(self, worker_id: str) -> WorkerInfo | None:
        """Unregister a worker."""
        worker = self.worker_store.get(worker_id)
        if worker:
            self.worker_store.update_state(worker_id, WorkerState.TERMINATED)
        return worker

    def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """Get worker by ID."""
        return self.worker_store.get(worker_id)

    def mark_task_complete(self, task_id: str, result: TaskResult) -> None:
        """Mark a task as complete."""
        self.task_store.save_result(result)
        worker = self.worker_store.get_by_task(task_id)
        if worker:
            self.worker_store.update_state(worker.id, WorkerState.COMPLETED)

    def mark_task_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        result = TaskResult(
            task_id=task_id,
            status="failed",
            error=error,
        )
        self.task_store.save_result(result)
        worker = self.worker_store.get_by_task(task_id)
        if worker:
            self.worker_store.update_state(worker.id, WorkerState.FAILED)

    async def collect_results(self, task_id: str) -> TaskResult | None:
        """Get results for a task."""
        return self.task_store.get_result(task_id)

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        counts = self.task_store.count_by_status()
        return counts.get("pending", 0)

    def get_active_worker_count(self) -> int:
        """Get count of active workers."""
        counts = self.worker_store.count_by_state()
        return counts.get(WorkerState.IDLE.value, 0) + counts.get(WorkerState.RUNNING.value, 0)

    def get_stats(self) -> dict:
        """Get orchestrator statistics."""
        task_counts = self.task_store.count_by_status()
        worker_counts = self.worker_store.count_by_state()

        return {
            "tasks": {
                "pending": task_counts.get("pending", 0),
                "running": task_counts.get("running", 0),
                "completed": task_counts.get("completed", 0),
                "failed": task_counts.get("failed", 0),
                "total": sum(task_counts.values()),
            },
            "workers": {
                "idle": worker_counts.get("idle", 0),
                "running": worker_counts.get("running", 0),
                "completed": worker_counts.get("completed", 0),
                "failed": worker_counts.get("failed", 0),
                "total": sum(worker_counts.values()),
            },
        }
