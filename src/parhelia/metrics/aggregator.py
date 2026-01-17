"""Metrics aggregation from local and Pushgateway sources.

Implements:
- [SPEC-06.14] Metrics Aggregator
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import aiohttp

from parhelia.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PUSHGATEWAY_PORT = 9091
DEFAULT_CACHE_TTL = 5  # seconds
DEFAULT_STALE_THRESHOLD = 30  # seconds


@dataclass
class ResourceMetrics:
    """Aggregated resource metrics for an environment.

    Implements [SPEC-06.14].
    """

    environment_id: str
    cpu_total: int
    cpu_available: float
    cpu_usage_percent: float
    memory_total: int
    memory_available: int
    memory_usage_percent: float
    gpu_available: int
    sessions_active: int
    sessions_capacity: int
    cost_per_hour: float
    last_updated: float

    @classmethod
    def from_dict(cls, data: dict[str, float], environment_id: str) -> ResourceMetrics:
        """Create ResourceMetrics from Prometheus metric dictionary.

        Args:
            data: Dictionary of metric name -> value.
            environment_id: Environment/container identifier.

        Returns:
            ResourceMetrics instance.
        """
        return cls(
            environment_id=environment_id,
            cpu_total=int(data.get("parhelia_cpu_total_cores", 0)),
            cpu_available=data.get("parhelia_cpu_available_cores", 0.0),
            cpu_usage_percent=data.get("parhelia_cpu_usage_percent", 0.0),
            memory_total=int(data.get("parhelia_memory_total_bytes", 0)),
            memory_available=int(data.get("parhelia_memory_available_bytes", 0)),
            memory_usage_percent=data.get("parhelia_memory_usage_percent", 0.0),
            gpu_available=int(data.get("parhelia_gpu_available_count", 0)),
            sessions_active=int(data.get("parhelia_sessions_active", 0)),
            sessions_capacity=int(data.get("parhelia_sessions_capacity", 0)),
            cost_per_hour=data.get("parhelia_cost_per_hour_usd", 0.0),
            last_updated=time.time(),
        )


@dataclass
class CapacitySummary:
    """Summary of total system capacity.

    Implements [SPEC-06.14].
    """

    total_cpu_cores: float
    total_memory_gb: float
    total_gpu_count: int
    available_session_slots: int
    environments: list[str] = field(default_factory=list)


class MetricsAggregator:
    """Aggregate metrics from local machine and Pushgateway.

    Implements [SPEC-06.14].
    """

    def __init__(
        self,
        pushgateway_port: int = DEFAULT_PUSHGATEWAY_PORT,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        stale_threshold: int = DEFAULT_STALE_THRESHOLD,
    ):
        """Initialize the metrics aggregator.

        Args:
            pushgateway_port: Port where Pushgateway is running.
            cache_ttl: How long to cache metrics (seconds).
            stale_threshold: When to consider metrics stale (seconds).
        """
        self.pushgateway_url = f"http://localhost:{pushgateway_port}"
        self.cache_ttl = cache_ttl
        self.stale_threshold = stale_threshold

        # Local metrics collector
        self.local_collector = MetricsCollector(environment="local")

        # Metrics cache
        self._cached_metrics: dict[str, ResourceMetrics] = {}
        self._cache_timestamp: float = 0

    async def get_all_metrics(self) -> dict[str, ResourceMetrics]:
        """Get current metrics from local and Pushgateway.

        Implements [SPEC-06.14].

        Returns:
            Dictionary mapping environment_id to ResourceMetrics.
        """
        results: dict[str, ResourceMetrics] = {}

        # Collect local metrics directly
        local_container_metrics = await self.local_collector.collect()
        local_dict = local_container_metrics.to_prometheus_dict()
        results["local"] = ResourceMetrics.from_dict(local_dict, "local")

        # Fetch remote metrics from Pushgateway
        remote_metrics = await self._fetch_from_pushgateway()
        results.update(remote_metrics)

        return results

    async def _fetch_from_pushgateway(self) -> dict[str, ResourceMetrics]:
        """Fetch and parse metrics from Pushgateway.

        Implements [SPEC-06.14].

        Returns:
            Dictionary mapping container_id to ResourceMetrics.
        """
        results: dict[str, ResourceMetrics] = {}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.pushgateway_url}/metrics",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Pushgateway returned {response.status}")
                        return results

                    text = await response.text()

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to fetch from Pushgateway: {e}")
            return results

        if not text.strip():
            return results

        # Parse metrics grouped by container_id
        container_metrics: dict[str, dict[str, float]] = {}
        container_timestamps: dict[str, float] = {}

        for line in text.strip().split("\n"):
            if line.startswith("#") or not line:
                continue

            # Parse: metric_name{labels} value timestamp?
            match = re.match(r'(\w+)\{([^}]+)\}\s+(\S+)(?:\s+(\d+))?', line)
            if not match:
                continue

            name, labels_str, value, timestamp = match.groups()

            # Extract container_id from labels
            labels = {}
            for label_pair in labels_str.split(","):
                if "=" in label_pair:
                    key, val = label_pair.split("=", 1)
                    labels[key] = val.strip('"')

            container_id = labels.get("container_id", "")

            if not container_id or container_id == "local":
                continue

            if container_id not in container_metrics:
                container_metrics[container_id] = {}

            try:
                container_metrics[container_id][name] = float(value)
            except ValueError:
                continue

            # Track push timestamp for staleness detection
            if timestamp:
                try:
                    container_timestamps[container_id] = float(timestamp)
                except ValueError:
                    pass

        # Convert to ResourceMetrics, filtering stale entries
        now = time.time()
        for container_id, metrics in container_metrics.items():
            push_time = container_timestamps.get(container_id, now)
            age = now - push_time

            if age > self.stale_threshold:
                logger.debug(
                    f"Skipping stale metrics for {container_id} (age: {age:.1f}s)"
                )
                continue

            resource_metrics = ResourceMetrics.from_dict(metrics, container_id)
            results[container_id] = resource_metrics

        return results

    def _is_stale(self, metrics: ResourceMetrics) -> bool:
        """Check if metrics are stale.

        Args:
            metrics: ResourceMetrics to check.

        Returns:
            True if metrics are older than stale_threshold.
        """
        age = time.time() - metrics.last_updated
        return age > self.stale_threshold

    async def get_capacity_summary(self) -> CapacitySummary:
        """Get summary of total system capacity.

        Implements [SPEC-06.14].

        Returns:
            CapacitySummary with aggregated capacity info.
        """
        all_metrics = await self.get_all_metrics()

        total_cpu = sum(m.cpu_available for m in all_metrics.values())
        total_memory = sum(m.memory_available for m in all_metrics.values())
        total_gpu = sum(m.gpu_available for m in all_metrics.values())
        total_sessions = sum(
            m.sessions_capacity - m.sessions_active for m in all_metrics.values()
        )

        return CapacitySummary(
            total_cpu_cores=total_cpu,
            total_memory_gb=total_memory / (1024**3),
            total_gpu_count=total_gpu,
            available_session_slots=total_sessions,
            environments=list(all_metrics.keys()),
        )

    async def cleanup_stale_metrics(self) -> None:
        """Remove metrics from containers that have stopped pushing.

        Implements [SPEC-06.14].
        """
        try:
            async with aiohttp.ClientSession() as session:
                # List all groups from Pushgateway API
                async with session.get(
                    f"{self.pushgateway_url}/api/v1/metrics",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        return
                    groups = await resp.json()

                # Delete stale groups
                for group in groups.get("data", []):
                    container_id = group.get("labels", {}).get("container_id")
                    if not container_id:
                        continue

                    # Delete the metric group
                    delete_url = (
                        f"{self.pushgateway_url}/metrics/job/parhelia_remote"
                        f"/container_id/{container_id}"
                    )
                    async with session.delete(delete_url) as delete_resp:
                        if delete_resp.status == 202:
                            logger.info(f"Cleaned up stale metrics for {container_id}")

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to cleanup stale metrics: {e}")
