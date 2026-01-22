"""Container reconciliation loop.

Syncs database state with Modal reality to detect and handle:
- Orphan containers (in Modal but not in DB)
- Stale containers (in DB but not in Modal)
- State drift (mismatched state between DB and Modal)

Implements:
- [SPEC-21.20] Reconciler Implementation
- [SPEC-21.40] Container Lifecycle States
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

from parhelia.state import (
    Container,
    ContainerState,
    EventType,
    HealthStatus,
    StateStore,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModalSandboxStatus(str, Enum):
    """Modal sandbox status values."""

    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass
class ModalSandboxInfo:
    """Information about a Modal sandbox.

    Represents the data returned by Modal's sandbox listing API.
    """

    sandbox_id: str
    status: ModalSandboxStatus
    created_at: datetime | None = None
    region: str | None = None
    app_name: str | None = None

    # Resource info (if available)
    cpu_count: int | None = None
    memory_mb: int | None = None
    gpu_type: str | None = None

    # Cost info (if available)
    cost_usd: float | None = None

    # Metadata from sandbox labels/environment
    task_id: str | None = None
    parhelia_managed: bool = False


@dataclass
class ReconcileResult:
    """Result of a reconciliation cycle.

    Tracks what changes were detected and applied during reconciliation.
    """

    orphans_detected: int = 0
    stale_marked: int = 0
    drift_corrected: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    # Detailed tracking
    orphan_ids: list[str] = field(default_factory=list)
    stale_ids: list[str] = field(default_factory=list)
    drift_ids: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable summary."""
        parts = []
        if self.orphans_detected:
            parts.append(f"{self.orphans_detected} orphans")
        if self.stale_marked:
            parts.append(f"{self.stale_marked} stale")
        if self.drift_corrected:
            parts.append(f"{self.drift_corrected} drift corrections")
        if self.errors:
            parts.append(f"{len(self.errors)} errors")

        summary = ", ".join(parts) if parts else "no changes"
        return f"ReconcileResult({summary}, {self.duration_seconds:.2f}s)"


@dataclass
class ReconcilerConfig:
    """Configuration for the reconciliation loop."""

    poll_interval_seconds: int = 60
    stale_threshold_seconds: int = 300  # 5 min without heartbeat = stale
    orphan_grace_period_seconds: int = 120  # Wait before marking orphan
    auto_terminate_orphans: bool = False  # Require manual cleanup by default
    max_reconcile_batch: int = 100


class ModalClient(ABC):
    """Abstract interface for Modal API operations.

    This abstraction allows for:
    - Testing with mock implementations
    - Swapping Modal API versions
    - Implementing retry/fallback logic
    """

    @abstractmethod
    async def list_sandboxes(self, app_name: str | None = None) -> list[ModalSandboxInfo]:
        """List all sandboxes, optionally filtered by app name.

        Args:
            app_name: Optional app name to filter by (e.g., "parhelia").

        Returns:
            List of sandbox information objects.
        """
        ...

    @abstractmethod
    async def get_sandbox(self, sandbox_id: str) -> ModalSandboxInfo | None:
        """Get information about a specific sandbox.

        Args:
            sandbox_id: The Modal sandbox ID.

        Returns:
            Sandbox info if found, None otherwise.
        """
        ...

    @abstractmethod
    async def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Terminate a sandbox.

        Args:
            sandbox_id: The Modal sandbox ID to terminate.

        Returns:
            True if termination was successful, False otherwise.
        """
        ...


class MockModalClient(ModalClient):
    """Mock Modal client for testing.

    Allows tests to inject sandbox data and verify termination calls.
    """

    def __init__(self, sandboxes: list[ModalSandboxInfo] | None = None):
        """Initialize with optional list of sandboxes.

        Args:
            sandboxes: Initial list of sandboxes to return from list_sandboxes.
        """
        self._sandboxes: dict[str, ModalSandboxInfo] = {}
        if sandboxes:
            for sb in sandboxes:
                self._sandboxes[sb.sandbox_id] = sb

        self.terminated_ids: list[str] = []
        self.list_calls: int = 0
        self.get_calls: int = 0

    def add_sandbox(self, sandbox: ModalSandboxInfo) -> None:
        """Add a sandbox to the mock."""
        self._sandboxes[sandbox.sandbox_id] = sandbox

    def remove_sandbox(self, sandbox_id: str) -> None:
        """Remove a sandbox from the mock."""
        self._sandboxes.pop(sandbox_id, None)

    def clear(self) -> None:
        """Clear all sandboxes."""
        self._sandboxes.clear()

    async def list_sandboxes(self, app_name: str | None = None) -> list[ModalSandboxInfo]:
        """Return all sandboxes, optionally filtered by app name."""
        self.list_calls += 1
        sandboxes = list(self._sandboxes.values())
        if app_name:
            sandboxes = [s for s in sandboxes if s.app_name == app_name]
        return sandboxes

    async def get_sandbox(self, sandbox_id: str) -> ModalSandboxInfo | None:
        """Get a specific sandbox."""
        self.get_calls += 1
        return self._sandboxes.get(sandbox_id)

    async def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Mark a sandbox as terminated."""
        if sandbox_id in self._sandboxes:
            self._sandboxes[sandbox_id].status = ModalSandboxStatus.TERMINATED
            self.terminated_ids.append(sandbox_id)
            return True
        return False


class RealModalClient(ModalClient):
    """Real Modal client using the Modal SDK.

    This implementation wraps the actual Modal API calls.
    """

    def __init__(self, app_name: str = "parhelia"):
        """Initialize the real Modal client.

        Args:
            app_name: The Modal app name to use for filtering.
        """
        self.app_name = app_name

    async def list_sandboxes(self, app_name: str | None = None) -> list[ModalSandboxInfo]:
        """List sandboxes using Modal SDK.

        Note: This is a best-effort implementation. The actual Modal API
        may have different methods or return types.
        """
        try:
            import modal

            sandboxes = []
            # Modal SDK provides Sandbox.list() for listing sandboxes
            # This is an async generator
            async for sb in modal.Sandbox.list():
                # Extract info from sandbox
                info = ModalSandboxInfo(
                    sandbox_id=str(sb.object_id) if hasattr(sb, "object_id") else str(sb),
                    status=ModalSandboxStatus.RUNNING,  # Assume running if listed
                    app_name=app_name or self.app_name,
                    parhelia_managed=True,  # Assume managed for now
                )
                sandboxes.append(info)

            return sandboxes

        except ImportError:
            logger.warning("Modal SDK not available")
            return []
        except Exception as e:
            logger.error(f"Failed to list Modal sandboxes: {e}")
            return []

    async def get_sandbox(self, sandbox_id: str) -> ModalSandboxInfo | None:
        """Get a specific sandbox by ID."""
        try:
            import modal

            sb = await modal.Sandbox.from_id(sandbox_id)
            if sb:
                return ModalSandboxInfo(
                    sandbox_id=sandbox_id,
                    status=ModalSandboxStatus.RUNNING,
                    app_name=self.app_name,
                    parhelia_managed=True,
                )
            return None

        except ImportError:
            logger.warning("Modal SDK not available")
            return None
        except Exception as e:
            logger.debug(f"Failed to get sandbox {sandbox_id}: {e}")
            return None

    async def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Terminate a sandbox."""
        try:
            import modal

            sb = await modal.Sandbox.from_id(sandbox_id)
            if sb:
                await sb.terminate()
                return True
            return False

        except ImportError:
            logger.warning("Modal SDK not available")
            return False
        except Exception as e:
            logger.error(f"Failed to terminate sandbox {sandbox_id}: {e}")
            return False


class ContainerReconciler:
    """Syncs database state with Modal reality.

    Implements [SPEC-21.40] - detects and handles:
    - Orphans: Containers in Modal but not tracked in DB
    - Stale: Containers in DB but gone from Modal
    - Drift: State mismatches between DB and Modal

    The reconciler is stateless - all state lives in StateStore.
    """

    def __init__(
        self,
        state_store: StateStore,
        modal_client: ModalClient,
        config: ReconcilerConfig | None = None,
    ):
        """Initialize the reconciler.

        Args:
            state_store: The state store for container/event persistence.
            modal_client: Client for Modal API operations.
            config: Optional configuration (uses defaults if not provided).
        """
        self.state_store = state_store
        self.modal_client = modal_client
        self.config = config or ReconcilerConfig()
        self._running = False
        self._stop_event: asyncio.Event | None = None

    async def reconcile(self) -> ReconcileResult:
        """Run one reconciliation cycle.

        This is the core reconciliation logic:
        1. Get containers from DB (state=RUNNING or CREATED)
        2. Get sandboxes from Modal
        3. Detect orphans (in Modal, not in DB) -> create ORPHANED record
        4. Detect stale (in DB, not in Modal) -> mark TERMINATED
        5. Detect drift (state mismatch) -> update DB to match Modal
        6. Emit events for all changes

        Returns:
            ReconcileResult with counts and details of changes made.
        """
        start_time = time.time()
        result = ReconcileResult()

        try:
            # Step 1: Get Modal's view of reality
            modal_sandboxes = await self.modal_client.list_sandboxes(app_name="parhelia")
            modal_by_id = {s.sandbox_id: s for s in modal_sandboxes}

            # Step 2: Get our view of reality (active containers)
            db_containers = self.state_store.get_active_containers()
            db_by_modal_id = {c.modal_sandbox_id: c for c in db_containers}

            # Step 3: Compute set differences
            modal_ids = set(modal_by_id.keys())
            db_modal_ids = set(db_by_modal_id.keys())

            # Containers we know about but Modal doesn't have (terminated externally)
            stale_ids = db_modal_ids - modal_ids

            # Containers Modal has but we don't know about (orphans)
            orphan_ids = modal_ids - db_modal_ids

            # Containers in both (check for state drift)
            common_ids = modal_ids & db_modal_ids

            # Step 4: Handle stale containers (in DB but not in Modal)
            for modal_id in stale_ids:
                container = db_by_modal_id[modal_id]
                await self._handle_stale(container, result)

            # Step 5: Handle orphans (in Modal but not in DB)
            for modal_id in orphan_ids:
                sandbox = modal_by_id[modal_id]
                await self._handle_orphan(sandbox, result)

            # Step 6: Check for state drift on common containers
            for modal_id in common_ids:
                container = db_by_modal_id[modal_id]
                sandbox = modal_by_id[modal_id]
                await self._check_drift(container, sandbox, result)

        except Exception as e:
            error_msg = f"Reconciliation failed: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg, exc_info=True)

            # Log the error event
            self.state_store.log_event(
                EventType.RECONCILE_FAILED,
                message=error_msg,
                source="reconciler",
            )

        result.duration_seconds = time.time() - start_time
        logger.info(str(result))

        return result

    async def _handle_stale(self, container: Container, result: ReconcileResult) -> None:
        """Handle a container that exists in DB but not in Modal.

        This means the container was terminated externally (timeout, crash, etc.)
        We update the DB state to TERMINATED and emit an event.
        """
        old_state = container.state

        # Update container state
        self.state_store.update_container_state(
            container.id,
            ContainerState.TERMINATED,
            reason="External termination (not found in Modal)",
        )

        result.stale_marked += 1
        result.stale_ids.append(container.id)

        logger.info(
            f"Marked stale container {container.id} as TERMINATED "
            f"(modal_id={container.modal_sandbox_id}, was {old_state.value})"
        )

    async def _handle_orphan(
        self, sandbox: ModalSandboxInfo, result: ReconcileResult
    ) -> None:
        """Handle a sandbox in Modal that we don't have a record for.

        Creates an ORPHANED container record and emits an event.
        Optionally terminates the orphan if auto_terminate_orphans is enabled.
        """
        # Check if this is a Parhelia-managed container
        # (In real implementation, would check sandbox labels/environment)
        if not sandbox.parhelia_managed:
            return

        # Check if we already have a record for this sandbox (e.g., an existing orphan)
        existing = self.state_store.get_container_by_modal_id(sandbox.sandbox_id)
        if existing:
            # Already tracked (possibly as ORPHANED from previous cycle)
            # Don't re-detect, but we could update cost here
            if sandbox.cost_usd is not None:
                existing.cost_accrued_usd = sandbox.cost_usd
                self.state_store.update_container(existing)
            return

        # Create orphan container record
        container = Container(
            id=f"c-{uuid.uuid4().hex[:8]}",
            modal_sandbox_id=sandbox.sandbox_id,
            state=ContainerState.ORPHANED,
            health_status=HealthStatus.UNKNOWN,
            created_at=sandbox.created_at or datetime.utcnow(),
            task_id=sandbox.task_id,
            region=sandbox.region,
            cpu_cores=sandbox.cpu_count,
            memory_mb=sandbox.memory_mb,
            gpu_type=sandbox.gpu_type,
            cost_accrued_usd=sandbox.cost_usd or 0.0,
        )

        self.state_store.containers.save(container)

        # Emit orphan detected event
        self.state_store.log_event(
            EventType.ORPHAN_DETECTED,
            container_id=container.id,
            message=f"Orphan container detected: {sandbox.sandbox_id}",
            source="reconciler",
            modal_sandbox_id=sandbox.sandbox_id,
            task_id=sandbox.task_id,
        )

        result.orphans_detected += 1
        result.orphan_ids.append(container.id)

        logger.warning(
            f"Detected orphan container: {container.id} (modal_id={sandbox.sandbox_id})"
        )

        # Auto-terminate if configured
        if self.config.auto_terminate_orphans:
            await self._terminate_orphan(container)

    async def _terminate_orphan(self, container: Container) -> None:
        """Terminate an orphaned container."""
        success = await self.modal_client.terminate_sandbox(container.modal_sandbox_id)

        if success:
            self.state_store.update_container_state(
                container.id,
                ContainerState.TERMINATED,
                reason="Auto-terminated orphan",
            )
            logger.info(f"Auto-terminated orphan container {container.id}")
        else:
            logger.error(
                f"Failed to auto-terminate orphan container {container.id}"
            )

    async def _check_drift(
        self,
        container: Container,
        sandbox: ModalSandboxInfo,
        result: ReconcileResult,
    ) -> None:
        """Check for and correct state drift between DB and Modal.

        If the Modal state doesn't match our DB state, we update the DB
        to match Modal (Modal is the source of truth for running state).
        """
        # Map Modal status to our ContainerState
        modal_state = self._map_modal_status(sandbox.status)

        # Check if state has drifted
        if container.state != modal_state:
            old_state = container.state

            # Emit drift correction event
            self.state_store.log_event(
                EventType.STATE_DRIFT_CORRECTED,
                container_id=container.id,
                old_value=old_state.value,
                new_value=modal_state.value,
                message=f"State drift corrected: {old_state.value} -> {modal_state.value}",
                source="reconciler",
            )

            # Update to match Modal
            self.state_store.update_container_state(
                container.id,
                modal_state,
                reason=f"State drift correction (Modal reports {sandbox.status.value})",
            )

            result.drift_corrected += 1
            result.drift_ids.append(container.id)

            logger.info(
                f"Corrected state drift for {container.id}: "
                f"{old_state.value} -> {modal_state.value}"
            )

        # Update cost if available
        if sandbox.cost_usd is not None and sandbox.cost_usd != container.cost_accrued_usd:
            container.cost_accrued_usd = sandbox.cost_usd
            self.state_store.update_container(container)

    def _map_modal_status(self, status: ModalSandboxStatus) -> ContainerState:
        """Map Modal sandbox status to our ContainerState."""
        mapping = {
            ModalSandboxStatus.RUNNING: ContainerState.RUNNING,
            ModalSandboxStatus.STOPPED: ContainerState.STOPPED,
            ModalSandboxStatus.TERMINATED: ContainerState.TERMINATED,
            ModalSandboxStatus.UNKNOWN: ContainerState.UNKNOWN,
        }
        return mapping.get(status, ContainerState.UNKNOWN)

    async def run_background(self, interval_seconds: int | None = None) -> None:
        """Run reconciliation loop in background.

        This method runs until stop() is called or the task is cancelled.

        Args:
            interval_seconds: Override the config poll interval.
        """
        interval = interval_seconds or self.config.poll_interval_seconds
        self._running = True
        self._stop_event = asyncio.Event()

        logger.info(f"Starting reconciliation loop (interval={interval}s)")

        while self._running:
            try:
                await self.reconcile()
            except Exception as e:
                logger.error(f"Reconciliation cycle failed: {e}", exc_info=True)

            # Wait for interval or stop signal
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval,
                )
                # If we get here, stop was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                pass

        logger.info("Reconciliation loop stopped")

    def stop(self) -> None:
        """Stop the background reconciliation loop."""
        self._running = False
        if self._stop_event:
            self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """Check if the reconciler is running."""
        return self._running
