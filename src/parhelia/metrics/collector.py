"""Metrics collection for container resources.

Implements:
- [SPEC-06.12] Remote Metrics Collector
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import psutil


# GPU cost lookup table (Modal pricing, approximate as of 2026)
GPU_COSTS = {
    "T4": 0.59,
    "A10G": 1.10,
    "A100-40GB": 3.20,
    "A100-80GB": 4.50,
    "H100": 5.50,
}

# Base CPU-only cost
CPU_BASE_COST = 0.15

# Default max sessions per container
DEFAULT_MAX_SESSIONS = 4


@dataclass
class GPUMetrics:
    """Metrics for a single GPU.

    Implements [SPEC-06.12].
    """

    gpu_index: int
    gpu_type: str
    memory_total_bytes: int
    memory_used_bytes: int
    utilization_percent: float


@dataclass
class ContainerMetrics:
    """Resource metrics for a container.

    Implements [SPEC-06.12].
    """

    # CPU metrics
    cpu_total_cores: int
    cpu_available_cores: float
    cpu_usage_percent: float

    # Memory metrics
    memory_total_bytes: int
    memory_available_bytes: int
    memory_usage_percent: float

    # Session metrics
    sessions_active: int
    sessions_capacity: int

    # Cost metrics
    cost_per_hour_usd: float

    # GPU metrics (optional)
    gpu_total_count: int = 0
    gpu_available_count: int = 0
    gpu_metrics: list[GPUMetrics] = field(default_factory=list)

    def to_prometheus_dict(self) -> dict[str, float]:
        """Convert to Prometheus metric format.

        Returns:
            Dictionary mapping metric names to values.
        """
        metrics = {
            "parhelia_cpu_total_cores": self.cpu_total_cores,
            "parhelia_cpu_available_cores": self.cpu_available_cores,
            "parhelia_cpu_usage_percent": self.cpu_usage_percent,
            "parhelia_memory_total_bytes": self.memory_total_bytes,
            "parhelia_memory_available_bytes": self.memory_available_bytes,
            "parhelia_memory_usage_percent": self.memory_usage_percent,
            "parhelia_sessions_active": self.sessions_active,
            "parhelia_sessions_capacity": self.sessions_capacity,
            "parhelia_cost_per_hour_usd": self.cost_per_hour_usd,
            "parhelia_gpu_total_count": self.gpu_total_count,
            "parhelia_gpu_available_count": self.gpu_available_count,
        }

        # Add per-GPU metrics
        for gpu in self.gpu_metrics:
            idx = gpu.gpu_index
            metrics[f"parhelia_gpu_memory_total_bytes_gpu{idx}"] = gpu.memory_total_bytes
            metrics[f"parhelia_gpu_memory_used_bytes_gpu{idx}"] = gpu.memory_used_bytes
            metrics[f"parhelia_gpu_utilization_percent_gpu{idx}"] = gpu.utilization_percent

        return metrics


class MetricsCollector:
    """Collect resource metrics from container.

    Implements [SPEC-06.12].
    """

    def __init__(
        self,
        environment: str = "local",
        gpu_type: str | None = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ):
        """Initialize the metrics collector.

        Args:
            environment: Environment type ('local' or 'modal').
            gpu_type: GPU type if present (e.g., 'A10G', 'T4').
            max_sessions: Maximum sessions this container can handle.
        """
        self.environment = environment
        self.gpu_type = gpu_type
        self.max_sessions = max_sessions
        self._active_sessions: set[str] = set()

        # Get container ID from environment
        self.container_id = os.environ.get("MODAL_TASK_ID", "local")

    @property
    def active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._active_sessions)

    def register_session(self, session_id: str) -> None:
        """Register an active session.

        Args:
            session_id: Session identifier.
        """
        self._active_sessions.add(session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session.

        Args:
            session_id: Session identifier.
        """
        self._active_sessions.discard(session_id)

    def has_gpu(self) -> bool:
        """Check if this collector has GPU support.

        Returns:
            True if GPU is configured.
        """
        return self.gpu_type is not None

    def get_hourly_cost(self) -> float:
        """Get estimated hourly cost for this container.

        Implements [SPEC-06.12].

        Returns:
            Cost in USD per hour.
        """
        if self.environment == "local":
            return 0.0

        if self.gpu_type:
            return GPU_COSTS.get(self.gpu_type, 1.10)

        return CPU_BASE_COST

    async def collect(self) -> ContainerMetrics:
        """Collect current container metrics.

        Implements [SPEC-06.12].

        Returns:
            ContainerMetrics with current resource usage.
        """
        # CPU metrics
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_available = cpu_count * (1 - cpu_percent / 100)

        # Memory metrics
        memory = psutil.virtual_memory()

        # GPU metrics
        gpu_total = 0
        gpu_available = 0
        gpu_metrics_list: list[GPUMetrics] = []

        if self.has_gpu():
            gpu_data = await self._collect_gpu_metrics()
            gpu_total = len(gpu_data)
            gpu_available = gpu_total  # Simplified: assume all available
            gpu_metrics_list = [
                GPUMetrics(
                    gpu_index=g["gpu_index"],
                    gpu_type=g["gpu_type"],
                    memory_total_bytes=g["memory_total_bytes"],
                    memory_used_bytes=g["memory_used_bytes"],
                    utilization_percent=g["utilization_percent"],
                )
                for g in gpu_data
            ]

        return ContainerMetrics(
            cpu_total_cores=cpu_count,
            cpu_available_cores=cpu_available,
            cpu_usage_percent=cpu_percent,
            memory_total_bytes=memory.total,
            memory_available_bytes=memory.available,
            memory_usage_percent=memory.percent,
            sessions_active=self.active_session_count,
            sessions_capacity=self.max_sessions,
            cost_per_hour_usd=self.get_hourly_cost(),
            gpu_total_count=gpu_total,
            gpu_available_count=gpu_available,
            gpu_metrics=gpu_metrics_list,
        )

    async def _collect_gpu_metrics(self) -> list[dict]:
        """Collect GPU metrics using pynvml.

        Returns:
            List of GPU metric dictionaries.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            results = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                results.append({
                    "gpu_index": i,
                    "gpu_type": self.gpu_type or "unknown",
                    "memory_total_bytes": memory.total,
                    "memory_used_bytes": memory.used,
                    "utilization_percent": float(utilization.gpu),
                })

            pynvml.nvmlShutdown()
            return results

        except Exception:
            # GPU not available or pynvml not installed
            return []
