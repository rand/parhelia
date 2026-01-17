"""Metrics pushing to Prometheus Pushgateway.

Implements:
- [SPEC-06.13] Metrics Push to Pushgateway
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

if TYPE_CHECKING:
    from parhelia.metrics.collector import ContainerMetrics, MetricsCollector

logger = logging.getLogger(__name__)

# Metric definitions for Prometheus
METRIC_DEFINITIONS = {
    "parhelia_cpu_total_cores": "Total CPU cores available",
    "parhelia_cpu_available_cores": "CPU cores available for new work",
    "parhelia_cpu_usage_percent": "Current CPU usage percentage",
    "parhelia_memory_total_bytes": "Total memory in bytes",
    "parhelia_memory_available_bytes": "Available memory in bytes",
    "parhelia_memory_usage_percent": "Current memory usage percentage",
    "parhelia_gpu_total_count": "Total GPUs available",
    "parhelia_gpu_available_count": "GPUs available for new work",
    "parhelia_sessions_active": "Number of active Claude Code sessions",
    "parhelia_sessions_capacity": "Maximum sessions this environment can handle",
    "parhelia_cost_per_hour_usd": "Estimated cost per hour for this environment",
}

# Default push interval in seconds
DEFAULT_PUSH_INTERVAL = 10


class MetricsPusher:
    """Push metrics from Modal container to Pushgateway.

    Implements [SPEC-06.13].
    """

    def __init__(
        self,
        pushgateway_url: str,
        push_interval: int = DEFAULT_PUSH_INTERVAL,
        environment: str = "modal",
    ):
        """Initialize the metrics pusher.

        Args:
            pushgateway_url: URL of the Prometheus Pushgateway.
            push_interval: Seconds between metric pushes.
            environment: Environment identifier ('local' or 'modal').
        """
        self.pushgateway_url = pushgateway_url
        self.push_interval = push_interval
        self.environment = environment

        # Get container ID from environment
        self.container_id = os.environ.get("MODAL_TASK_ID", "unknown")

        # Create registry and gauges
        self.registry = CollectorRegistry()
        self.gauges: dict[str, Gauge] = {}

        for metric_name, description in METRIC_DEFINITIONS.items():
            self.gauges[metric_name] = Gauge(
                metric_name,
                description,
                ["environment", "container_id"],
                registry=self.registry,
            )

        # Control flag for push loop
        self._running = False

    async def push_once(self, metrics: ContainerMetrics) -> None:
        """Push metrics once to Pushgateway.

        Args:
            metrics: Container metrics to push.
        """
        try:
            # Update all gauges from metrics
            metrics_dict = metrics.to_prometheus_dict()

            for name, gauge in self.gauges.items():
                if name in metrics_dict:
                    gauge.labels(
                        environment=self.environment,
                        container_id=self.container_id,
                    ).set(metrics_dict[name])

            # Push to gateway
            push_to_gateway(
                gateway=self.pushgateway_url,
                job="parhelia_remote",
                grouping_key={"container_id": self.container_id},
                registry=self.registry,
            )

        except Exception as e:
            logger.warning(f"Failed to push metrics: {e}")

    async def push_loop(
        self,
        collector: MetricsCollector,
        max_iterations: int | None = None,
    ) -> None:
        """Continuously push metrics at interval.

        Implements [SPEC-06.13].

        Args:
            collector: MetricsCollector to get metrics from.
            max_iterations: Optional max iterations (for testing).
        """
        self._running = True
        iteration = 0

        while self._running:
            if max_iterations is not None and iteration >= max_iterations:
                break

            try:
                metrics = await collector.collect()
                await self.push_once(metrics)
            except Exception as e:
                logger.warning(f"Error in push loop: {e}")

            iteration += 1

            if self._running:
                await asyncio.sleep(self.push_interval)

    def stop(self) -> None:
        """Stop the push loop gracefully."""
        self._running = False

    def push_final(self) -> None:
        """Push final metrics before container shutdown.

        Implements [SPEC-06.13].
        """
        try:
            # Set sessions to 0 to indicate shutdown
            self.gauges["parhelia_sessions_active"].labels(
                environment=self.environment,
                container_id=self.container_id,
            ).set(0)

            push_to_gateway(
                gateway=self.pushgateway_url,
                job="parhelia_remote",
                grouping_key={"container_id": self.container_id},
                registry=self.registry,
            )
        except Exception:
            pass  # Best effort on shutdown
