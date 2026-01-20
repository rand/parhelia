"""Multi-session per container support for Parhelia.

Implements:
- [SPEC-02.16] Multi-Session Support
"""

from __future__ import annotations

import asyncio
import os
import psutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from parhelia.session import Session, SessionState


class ResourceType(Enum):
    """Types of resources to track."""

    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    SESSIONS = "sessions"


@dataclass
class ResourceStatus:
    """Current resource availability status.

    Implements [SPEC-02.16].
    """

    sessions_available: int
    sessions_total: int
    sessions_active: int
    cpu_available_percent: float
    cpu_total_cores: int
    memory_available_mb: float
    memory_total_mb: float
    gpu_available: bool = False
    gpu_memory_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_capacity(self) -> bool:
        """Check if container has capacity for new sessions."""
        return (
            self.sessions_available > 0
            and self.cpu_available_percent > 10.0
            and self.memory_available_mb > 512.0
        )

    @property
    def utilization_percent(self) -> float:
        """Get overall utilization percentage."""
        session_util = (self.sessions_active / self.sessions_total) * 100 if self.sessions_total > 0 else 0
        cpu_util = 100 - self.cpu_available_percent
        mem_util = ((self.memory_total_mb - self.memory_available_mb) / self.memory_total_mb) * 100 if self.memory_total_mb > 0 else 0
        return max(session_util, cpu_util, mem_util)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sessions_available": self.sessions_available,
            "sessions_total": self.sessions_total,
            "sessions_active": self.sessions_active,
            "cpu_available_percent": self.cpu_available_percent,
            "cpu_total_cores": self.cpu_total_cores,
            "memory_available_mb": self.memory_available_mb,
            "memory_total_mb": self.memory_total_mb,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
            "has_capacity": self.has_capacity,
            "utilization_percent": self.utilization_percent,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SessionSlot:
    """A slot for a session in the container.

    Tracks resource allocation for a single session.
    """

    session_id: str
    session: Session | None = None
    allocated_at: datetime = field(default_factory=datetime.now)
    cpu_allocation: float = 1.0  # Cores
    memory_allocation_mb: float = 4096.0  # MB
    gpu_allocated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerConfig:
    """Configuration for multi-session container.

    Implements [SPEC-02.16].
    """

    max_sessions: int = 4
    min_cpu_per_session: float = 0.5  # Cores
    min_memory_per_session_mb: float = 2048.0  # MB
    reserved_cpu_percent: float = 10.0  # Reserved for system
    reserved_memory_mb: float = 512.0  # Reserved for system
    enable_gpu_sharing: bool = False


# Type alias for capacity callbacks
CapacityCallback = Callable[[ResourceStatus], Awaitable[None]]


class SessionRegistry:
    """Registry for tracking active sessions in a container.

    Implements [SPEC-02.16].

    Manages:
    - Session slot allocation
    - Resource tracking
    - Capacity reporting
    """

    def __init__(self, config: ContainerConfig | None = None):
        """Initialize the session registry.

        Args:
            config: Container configuration.
        """
        self.config = config or ContainerConfig()
        self._slots: dict[str, SessionSlot] = {}
        self._on_capacity_change: list[CapacityCallback] = []

    async def register_session(
        self,
        session: Session,
        cpu_allocation: float | None = None,
        memory_allocation_mb: float | None = None,
    ) -> SessionSlot:
        """Register a new session in the container.

        Args:
            session: The session to register.
            cpu_allocation: CPU cores to allocate.
            memory_allocation_mb: Memory to allocate in MB.

        Returns:
            SessionSlot for the registered session.

        Raises:
            RuntimeError: If no capacity available.
        """
        if not await self.can_accept_session():
            raise RuntimeError("Container has no capacity for new sessions")

        slot = SessionSlot(
            session_id=session.id,
            session=session,
            cpu_allocation=cpu_allocation or self.config.min_cpu_per_session,
            memory_allocation_mb=memory_allocation_mb or self.config.min_memory_per_session_mb,
        )

        self._slots[session.id] = slot

        # Notify capacity change
        await self._notify_capacity_change()

        return slot

    async def unregister_session(self, session_id: str) -> None:
        """Unregister a session from the container.

        Args:
            session_id: The session to unregister.
        """
        if session_id in self._slots:
            del self._slots[session_id]
            await self._notify_capacity_change()

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session ID.

        Returns:
            Session if found, None otherwise.
        """
        slot = self._slots.get(session_id)
        return slot.session if slot else None

    async def get_slot(self, session_id: str) -> SessionSlot | None:
        """Get a session slot by ID.

        Args:
            session_id: The session ID.

        Returns:
            SessionSlot if found, None otherwise.
        """
        return self._slots.get(session_id)

    async def list_sessions(self) -> list[Session]:
        """List all registered sessions.

        Returns:
            List of Session instances.
        """
        return [slot.session for slot in self._slots.values() if slot.session]

    async def list_slots(self) -> list[SessionSlot]:
        """List all session slots.

        Returns:
            List of SessionSlot instances.
        """
        return list(self._slots.values())

    async def can_accept_session(self) -> bool:
        """Check if container can accept another session.

        Implements [SPEC-02.16].

        Returns:
            True if container has capacity.
        """
        if len(self._slots) >= self.config.max_sessions:
            return False

        status = await self.get_resource_status()
        return status.has_capacity

    async def get_active_count(self) -> int:
        """Get count of active sessions.

        Returns:
            Number of active (running/attached) sessions.
        """
        active_states = {SessionState.RUNNING}
        count = 0
        for slot in self._slots.values():
            if slot.session and slot.session.state in active_states:
                count += 1
        return count

    async def get_resource_status(self) -> ResourceStatus:
        """Get current resource availability.

        Implements [SPEC-02.16].

        Returns:
            ResourceStatus with current availability.
        """
        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Calculate availability
        cpu_available = 100.0 - cpu_percent - self.config.reserved_cpu_percent
        memory_available = (memory.available / (1024 * 1024)) - self.config.reserved_memory_mb

        active_count = await self.get_active_count()

        return ResourceStatus(
            sessions_available=self.config.max_sessions - len(self._slots),
            sessions_total=self.config.max_sessions,
            sessions_active=active_count,
            cpu_available_percent=max(0, cpu_available),
            cpu_total_cores=psutil.cpu_count() or 1,
            memory_available_mb=max(0, memory_available),
            memory_total_mb=memory.total / (1024 * 1024),
        )

    async def get_allocated_resources(self) -> dict[str, float]:
        """Get total allocated resources.

        Returns:
            Dict with cpu and memory allocations.
        """
        total_cpu = sum(slot.cpu_allocation for slot in self._slots.values())
        total_memory = sum(slot.memory_allocation_mb for slot in self._slots.values())

        return {
            "cpu_cores": total_cpu,
            "memory_mb": total_memory,
        }

    def on_capacity_change(self, callback: CapacityCallback) -> None:
        """Register callback for capacity changes."""
        self._on_capacity_change.append(callback)

    async def _notify_capacity_change(self) -> None:
        """Notify listeners of capacity change."""
        status = await self.get_resource_status()
        for callback in self._on_capacity_change:
            await callback(status)


class ResourceMonitor:
    """Monitor container resources for multi-session management.

    Provides continuous resource monitoring and alerts.
    """

    DEFAULT_POLL_INTERVAL = 10.0  # seconds
    WARNING_THRESHOLD_PERCENT = 80.0
    CRITICAL_THRESHOLD_PERCENT = 95.0

    def __init__(
        self,
        registry: SessionRegistry,
        poll_interval: float | None = None,
    ):
        """Initialize the resource monitor.

        Args:
            registry: Session registry to monitor.
            poll_interval: Seconds between polls.
        """
        self.registry = registry
        self.poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL

        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._history: list[ResourceStatus] = []
        self._max_history = 100

        # Callbacks
        self._on_warning: list[CapacityCallback] = []
        self._on_critical: list[CapacityCallback] = []

    async def start(self) -> None:
        """Start resource monitoring."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def get_current_status(self) -> ResourceStatus:
        """Get current resource status.

        Returns:
            Current ResourceStatus.
        """
        return await self.registry.get_resource_status()

    async def get_history(self, limit: int = 10) -> list[ResourceStatus]:
        """Get resource history.

        Args:
            limit: Max entries to return.

        Returns:
            List of ResourceStatus history.
        """
        return self._history[-limit:]

    def on_warning(self, callback: CapacityCallback) -> None:
        """Register callback for warning threshold."""
        self._on_warning.append(callback)

    def on_critical(self, callback: CapacityCallback) -> None:
        """Register callback for critical threshold."""
        self._on_critical.append(callback)

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            status = await self.registry.get_resource_status()

            # Store in history
            self._history.append(status)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Check thresholds
            utilization = status.utilization_percent

            if utilization >= self.CRITICAL_THRESHOLD_PERCENT:
                for callback in self._on_critical:
                    await callback(status)
            elif utilization >= self.WARNING_THRESHOLD_PERCENT:
                for callback in self._on_warning:
                    await callback(status)

            await asyncio.sleep(self.poll_interval)


class SessionScheduler:
    """Schedule sessions across container resources.

    Provides load balancing and resource allocation.
    """

    def __init__(self, registry: SessionRegistry):
        """Initialize the session scheduler.

        Args:
            registry: Session registry to schedule for.
        """
        self.registry = registry

    async def find_best_slot(self) -> dict[str, float] | None:
        """Find optimal resource allocation for a new session.

        Returns:
            Dict with cpu and memory allocation, or None if no capacity.
        """
        if not await self.registry.can_accept_session():
            return None

        status = await self.registry.get_resource_status()
        allocated = await self.registry.get_allocated_resources()

        # Calculate fair share
        max_sessions = self.registry.config.max_sessions
        current_sessions = len(await self.registry.list_slots())
        remaining_slots = max_sessions - current_sessions

        if remaining_slots <= 0:
            return None

        # Allocate fair share of remaining resources
        available_cpu = status.cpu_total_cores - allocated["cpu_cores"]
        available_memory = status.memory_total_mb - allocated["memory_mb"] - self.registry.config.reserved_memory_mb

        cpu_per_session = max(
            self.registry.config.min_cpu_per_session,
            available_cpu / remaining_slots,
        )
        memory_per_session = max(
            self.registry.config.min_memory_per_session_mb,
            available_memory / remaining_slots,
        )

        return {
            "cpu_cores": cpu_per_session,
            "memory_mb": memory_per_session,
        }

    async def rebalance(self) -> list[SessionSlot]:
        """Rebalance resource allocations across sessions.

        Returns:
            List of modified SessionSlots.
        """
        slots = await self.registry.list_slots()
        if not slots:
            return []

        status = await self.registry.get_resource_status()

        # Calculate equal share
        available_cpu = status.cpu_total_cores * (1 - self.registry.config.reserved_cpu_percent / 100)
        available_memory = status.memory_total_mb - self.registry.config.reserved_memory_mb

        cpu_per_session = available_cpu / len(slots)
        memory_per_session = available_memory / len(slots)

        modified = []
        for slot in slots:
            if slot.cpu_allocation != cpu_per_session or slot.memory_allocation_mb != memory_per_session:
                slot.cpu_allocation = cpu_per_session
                slot.memory_allocation_mb = memory_per_session
                modified.append(slot)

        return modified
