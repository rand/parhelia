# SPEC-06: Resource Capacity Broadcasting

**Status**: Draft
**Issue**: ph-8aq
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines how execution environments (local and remote) broadcast their resource capacity and utilization, enabling the orchestrator to make optimal dispatch decisions.

## Goals

- [SPEC-06.01] Collect resource metrics from all execution environments
- [SPEC-06.02] Expose metrics in Prometheus format for monitoring
- [SPEC-06.03] Enable capacity-aware task dispatch decisions
- [SPEC-06.04] Provide real-time visibility into system utilization
- [SPEC-06.05] Support heterogeneous resources (CPU, memory, GPU)

## Non-Goals

- Historical metrics storage (use external Prometheus/Grafana)
- Alerting rules (configure in monitoring stack)
- Detailed per-process profiling

---

## Architecture

### Metrics Flow (Pushgateway Pattern)

**Design Decision**: Modal containers are ephemeral and may not be reachable via traditional Prometheus scraping. We use the **Pushgateway pattern** where containers push metrics to a central gateway.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL CONTAINERS                                   │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Container 1     │  │  Container 2     │  │  Container 3     │          │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │          │
│  │  │  Metrics   │  │  │  │  Metrics   │  │  │  │  Metrics   │  │          │
│  │  │  Pusher    │  │  │  │  Pusher    │  │  │  │  Pusher    │  │          │
│  │  └─────┬──────┘  │  │  └─────┬──────┘  │  │  └─────┬──────┘  │          │
│  └────────┼─────────┘  └────────┼─────────┘  └────────┼─────────┘          │
│           │                     │                     │                     │
│           │  PUSH (every 10s)   │                     │                     │
│           └─────────────────────┼─────────────────────┘                     │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL ENVIRONMENT                                  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     Prometheus Pushgateway                          │    │
│  │  - Receives pushed metrics from Modal containers                    │    │
│  │  - Groups metrics by job/instance labels                            │    │
│  │  - Exposes /metrics for Prometheus scraping                         │    │
│  └───────────────────────────────┬────────────────────────────────────┘    │
│                                  │                                          │
│                    ┌─────────────┴─────────────┐                           │
│                    │ SCRAPE (by Prometheus)    │                           │
│                    ▼                           ▼                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                     Metrics Aggregator                              │    │
│  │  - Reads from local Pushgateway                                     │    │
│  │  - Collects local machine metrics directly                          │    │
│  │  - Handles stale metric cleanup                                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                 │                                           │
│                                 ▼                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  Orchestrator  │  │  Prometheus    │  │  Grafana       │               │
│  │  (dispatch)    │  │  (optional)    │  │  (optional)    │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Pushgateway?**
1. Modal containers are ephemeral and can scale to zero
2. Container IPs are not predictable or directly reachable
3. Push model survives container restarts/replacements
4. Centralized collection point simplifies monitoring setup

---

## Requirements

### [SPEC-06.10] Metrics Schema

All environments MUST expose these metrics:

```python
# Resource Capacity Metrics
METRICS = {
    # CPU
    "parhelia_cpu_total_cores": Gauge(
        "Total CPU cores available",
        labels=["environment", "container_id"]
    ),
    "parhelia_cpu_available_cores": Gauge(
        "CPU cores available for new work",
        labels=["environment", "container_id"]
    ),
    "parhelia_cpu_usage_percent": Gauge(
        "Current CPU usage percentage",
        labels=["environment", "container_id"]
    ),

    # Memory
    "parhelia_memory_total_bytes": Gauge(
        "Total memory in bytes",
        labels=["environment", "container_id"]
    ),
    "parhelia_memory_available_bytes": Gauge(
        "Available memory in bytes",
        labels=["environment", "container_id"]
    ),
    "parhelia_memory_usage_percent": Gauge(
        "Current memory usage percentage",
        labels=["environment", "container_id"]
    ),

    # GPU (when available)
    "parhelia_gpu_total_count": Gauge(
        "Total GPUs available",
        labels=["environment", "container_id", "gpu_type"]
    ),
    "parhelia_gpu_available_count": Gauge(
        "GPUs available for new work",
        labels=["environment", "container_id", "gpu_type"]
    ),
    "parhelia_gpu_memory_total_bytes": Gauge(
        "Total GPU memory in bytes",
        labels=["environment", "container_id", "gpu_type", "gpu_index"]
    ),
    "parhelia_gpu_memory_used_bytes": Gauge(
        "Used GPU memory in bytes",
        labels=["environment", "container_id", "gpu_type", "gpu_index"]
    ),
    "parhelia_gpu_utilization_percent": Gauge(
        "GPU compute utilization",
        labels=["environment", "container_id", "gpu_type", "gpu_index"]
    ),

    # Sessions
    "parhelia_sessions_active": Gauge(
        "Number of active Claude Code sessions",
        labels=["environment", "container_id"]
    ),
    "parhelia_sessions_capacity": Gauge(
        "Maximum sessions this environment can handle",
        labels=["environment", "container_id"]
    ),

    # Cost
    "parhelia_cost_per_hour_usd": Gauge(
        "Estimated cost per hour for this environment",
        labels=["environment", "container_id"]
    ),
}
```

### [SPEC-06.11] Local Metrics Collector

The local environment MUST collect and expose its metrics:

```python
class LocalMetricsCollector:
    """Collect metrics from local machine."""

    def __init__(self):
        import psutil
        self.psutil = psutil

    async def collect(self) -> dict[str, float]:
        """Collect current local metrics."""
        cpu_percent = self.psutil.cpu_percent(interval=0.1)
        memory = self.psutil.virtual_memory()

        metrics = {
            "parhelia_cpu_total_cores": self.psutil.cpu_count(),
            "parhelia_cpu_available_cores": self.psutil.cpu_count() * (1 - cpu_percent / 100),
            "parhelia_cpu_usage_percent": cpu_percent,
            "parhelia_memory_total_bytes": memory.total,
            "parhelia_memory_available_bytes": memory.available,
            "parhelia_memory_usage_percent": memory.percent,
            "parhelia_sessions_active": len(self.active_sessions),
            "parhelia_sessions_capacity": MAX_LOCAL_SESSIONS,
            "parhelia_cost_per_hour_usd": 0.0,  # Local is free
        }

        # GPU metrics if available
        if self._has_gpu():
            gpu_metrics = await self._collect_gpu_metrics()
            metrics.update(gpu_metrics)

        return metrics

    async def _collect_gpu_metrics(self) -> dict[str, float]:
        """Collect GPU metrics using nvidia-smi or similar."""
        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            metrics = {
                "parhelia_gpu_total_count": device_count,
                "parhelia_gpu_available_count": device_count,  # Simplified
            }

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                metrics[f"parhelia_gpu_memory_total_bytes{{gpu_index=\"{i}\"}}"] = memory.total
                metrics[f"parhelia_gpu_memory_used_bytes{{gpu_index=\"{i}\"}}"] = memory.used
                metrics[f"parhelia_gpu_utilization_percent{{gpu_index=\"{i}\"}}"] = utilization.gpu

            return metrics
        except Exception:
            return {}
```

### [SPEC-06.12] Remote Metrics Collector

Modal containers MUST collect and expose metrics:

```python
class RemoteMetricsCollector:
    """Collect metrics inside Modal container."""

    async def collect(self) -> dict[str, float]:
        """Collect current container metrics."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        metrics = {
            "parhelia_cpu_total_cores": psutil.cpu_count(),
            "parhelia_cpu_available_cores": psutil.cpu_count() * (1 - cpu_percent / 100),
            "parhelia_cpu_usage_percent": cpu_percent,
            "parhelia_memory_total_bytes": memory.total,
            "parhelia_memory_available_bytes": memory.available,
            "parhelia_memory_usage_percent": memory.percent,
            "parhelia_sessions_active": len(session_manager.sessions),
            "parhelia_sessions_capacity": MAX_SESSIONS_PER_CONTAINER,
            "parhelia_cost_per_hour_usd": self._get_hourly_cost(),
        }

        # GPU metrics
        if self._has_gpu():
            gpu_metrics = await self._collect_gpu_metrics()
            metrics.update(gpu_metrics)

        return metrics

    def _get_hourly_cost(self) -> float:
        """Get estimated hourly cost for this container type."""
        # Modal pricing (approximate, as of 2026)
        if self._has_gpu():
            gpu_type = os.environ.get("MODAL_GPU_TYPE", "A10G")
            costs = {
                "T4": 0.59,
                "A10G": 1.10,
                "A100-40GB": 3.20,
                "A100-80GB": 4.50,
                "H100": 5.50,
            }
            return costs.get(gpu_type, 1.10)
        else:
            # CPU pricing based on cores/memory
            return 0.15  # Base rate
```

### [SPEC-06.13] Metrics Push to Pushgateway

Modal containers MUST push metrics to a Prometheus Pushgateway running on the local orchestrator. This replaces direct endpoint exposure since Modal containers are ephemeral and not directly scrapable.

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import os
import asyncio

class MetricsPusher:
    """Push metrics from Modal container to Pushgateway."""

    def __init__(self, pushgateway_url: str):
        self.pushgateway_url = pushgateway_url
        self.registry = CollectorRegistry()
        self.container_id = os.environ.get("MODAL_TASK_ID", "unknown")
        self.environment = "modal"

        # Register gauges
        self.gauges = {}
        for metric_name in METRICS.keys():
            self.gauges[metric_name] = Gauge(
                metric_name,
                METRICS[metric_name].description,
                ["environment", "container_id"],
                registry=self.registry
            )

    async def push_loop(self, interval_seconds: int = 10):
        """Continuously push metrics at interval."""
        collector = RemoteMetricsCollector()

        while True:
            try:
                metrics = await collector.collect()

                # Update all gauges
                for name, value in metrics.items():
                    if name in self.gauges:
                        self.gauges[name].labels(
                            environment=self.environment,
                            container_id=self.container_id
                        ).set(value)

                # Push to gateway
                push_to_gateway(
                    self.pushgateway_url,
                    job='parhelia_remote',
                    grouping_key={'container_id': self.container_id},
                    registry=self.registry,
                )

            except Exception as e:
                logger.warning(f"Failed to push metrics: {e}")

            await asyncio.sleep(interval_seconds)

    def push_final(self):
        """Push final metrics before container shutdown."""
        try:
            # Set sessions to 0 to indicate shutdown
            self.gauges["parhelia_sessions_active"].labels(
                environment=self.environment,
                container_id=self.container_id
            ).set(0)

            push_to_gateway(
                self.pushgateway_url,
                job='parhelia_remote',
                grouping_key={'container_id': self.container_id},
                registry=self.registry,
            )
        except Exception:
            pass  # Best effort on shutdown
```

### [SPEC-06.13a] Pushgateway Configuration

The local orchestrator MUST run a Prometheus Pushgateway:

```python
# parhelia/metrics/pushgateway.py

class PushgatewayManager:
    """Manage local Pushgateway instance."""

    def __init__(self, port: int = 9091):
        self.port = port
        self.process: subprocess.Popen | None = None

    async def start(self):
        """Start Pushgateway process."""
        self.process = subprocess.Popen(
            ["pushgateway", f"--web.listen-address=:{self.port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await self._wait_ready()

    async def _wait_ready(self, timeout: int = 10):
        """Wait for Pushgateway to be ready."""
        async with aiohttp.ClientSession() as session:
            for _ in range(timeout * 10):
                try:
                    async with session.get(f"http://localhost:{self.port}/-/ready") as resp:
                        if resp.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.1)
            raise TimeoutError("Pushgateway failed to start")

    def get_url(self) -> str:
        """Get Pushgateway URL for containers."""
        # Return the URL that containers should push to
        # This must be accessible from Modal containers
        return os.environ.get(
            "PARHELIA_PUSHGATEWAY_URL",
            f"http://{get_public_ip()}:{self.port}"
        )
```

### [SPEC-06.13b] Container Startup Integration

Containers MUST start the metrics pusher on initialization:

```python
# In container entrypoint
async def container_main():
    # Get Pushgateway URL from environment (set by orchestrator)
    pushgateway_url = os.environ["PARHELIA_PUSHGATEWAY_URL"]

    # Start metrics push loop in background
    pusher = MetricsPusher(pushgateway_url)
    metrics_task = asyncio.create_task(pusher.push_loop())

    try:
        # Main container logic
        await run_claude_session(...)
    finally:
        # Push final metrics before shutdown
        pusher.push_final()
        metrics_task.cancel()
```

### [SPEC-06.14] Metrics Aggregator

The orchestrator MUST aggregate metrics from local collection and the Pushgateway:

```python
class MetricsAggregator:
    """Aggregate metrics from local machine and Pushgateway."""

    def __init__(self, pushgateway_port: int = 9091):
        self.local_collector = LocalMetricsCollector()
        self.pushgateway_url = f"http://localhost:{pushgateway_port}"
        self.cached_metrics: dict[str, ResourceMetrics] = {}
        self.cache_ttl = 5  # seconds
        self.stale_threshold = 30  # seconds - mark metrics stale after this

    async def get_all_metrics(self) -> dict[str, ResourceMetrics]:
        """Get current metrics from local and Pushgateway."""
        results = {}

        # Collect local metrics directly
        local = await self.local_collector.collect()
        results["local"] = ResourceMetrics.from_dict(local, "local")

        # Fetch remote metrics from Pushgateway
        remote_metrics = await self._fetch_from_pushgateway()
        results.update(remote_metrics)

        return results

    async def _fetch_from_pushgateway(self) -> dict[str, ResourceMetrics]:
        """Fetch and parse metrics from Pushgateway."""
        results = {}

        async with aiohttp.ClientSession() as session:
            # Query Pushgateway's metrics endpoint
            async with session.get(f"{self.pushgateway_url}/metrics", timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"Pushgateway returned {response.status}")
                    return results

                text = await response.text()

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
            labels = dict(l.split("=") for l in labels_str.split(","))
            container_id = labels.get("container_id", "").strip('"')

            if not container_id or container_id == "local":
                continue

            if container_id not in container_metrics:
                container_metrics[container_id] = {}

            container_metrics[container_id][name] = float(value)

            # Track push timestamp for staleness detection
            if timestamp:
                container_timestamps[container_id] = float(timestamp)

        # Convert to ResourceMetrics, filtering stale entries
        now = time.time()
        for container_id, metrics in container_metrics.items():
            push_time = container_timestamps.get(container_id, now)
            age = now - push_time

            if age > self.stale_threshold:
                logger.debug(f"Skipping stale metrics for {container_id} (age: {age:.1f}s)")
                continue

            results[container_id] = ResourceMetrics.from_dict(metrics, container_id)

        return results

    async def cleanup_stale_metrics(self):
        """Remove metrics from containers that have stopped pushing.

        Called periodically to clean up the Pushgateway.
        """
        async with aiohttp.ClientSession() as session:
            # List all groups
            async with session.get(f"{self.pushgateway_url}/api/v1/metrics") as resp:
                if resp.status != 200:
                    return
                groups = await resp.json()

            # Delete stale groups
            for group in groups.get("data", []):
                container_id = group.get("labels", {}).get("container_id")
                # Delete if no recent push (Pushgateway tracks push_time_seconds)
                delete_url = f"{self.pushgateway_url}/metrics/job/parhelia_remote/container_id/{container_id}"
                async with session.delete(delete_url) as delete_resp:
                    if delete_resp.status == 202:
                        logger.info(f"Cleaned up stale metrics for {container_id}")

    async def get_capacity_summary(self) -> CapacitySummary:
        """Get summary of total system capacity."""
        all_metrics = await self.get_all_metrics()

        total_cpu = sum(m.cpu_available for m in all_metrics.values())
        total_memory = sum(m.memory_available for m in all_metrics.values())
        total_gpu = sum(m.gpu_available for m in all_metrics.values())
        total_sessions = sum(m.sessions_capacity - m.sessions_active for m in all_metrics.values())

        return CapacitySummary(
            total_cpu_cores=total_cpu,
            total_memory_gb=total_memory / (1024**3),
            total_gpu_count=total_gpu,
            available_session_slots=total_sessions,
            environments=list(all_metrics.keys()),
        )
```

### [SPEC-06.15] Capacity-Aware Dispatch

The orchestrator MUST use metrics for dispatch decisions:

```python
class CapacityAwareDispatcher:
    """Dispatch tasks based on real-time capacity."""

    def __init__(self, aggregator: MetricsAggregator):
        self.aggregator = aggregator

    async def select_target(self, task: Task) -> ExecutionTarget:
        """Select best target based on current capacity."""
        metrics = await self.aggregator.get_all_metrics()

        # Filter by task requirements
        candidates = []
        for env_id, env_metrics in metrics.items():
            if self._meets_requirements(env_metrics, task.requirements):
                candidates.append((env_id, env_metrics))

        if not candidates:
            raise NoCapacityError(
                f"No environment has capacity for task: {task.requirements}"
            )

        # Score candidates
        scored = [
            (env_id, self._score_target(env_metrics, task))
            for env_id, env_metrics in candidates
        ]

        # Select best
        best_env_id, _ = max(scored, key=lambda x: x[1])
        return ExecutionTarget(
            id=best_env_id,
            metrics=metrics[best_env_id],
        )

    def _score_target(self, metrics: ResourceMetrics, task: Task) -> float:
        """Score an execution target for a task."""
        score = 0.0

        # Prefer less loaded environments
        cpu_headroom = 1 - (metrics.cpu_usage_percent / 100)
        memory_headroom = 1 - (metrics.memory_usage_percent / 100)
        score += (cpu_headroom + memory_headroom) * 25

        # Prefer cheaper environments
        score += (5 - metrics.cost_per_hour) * 10

        # Prefer local for small tasks
        if task.estimated_duration_minutes < 5 and metrics.environment == "local":
            score += 50

        # Prefer GPU environments for GPU tasks
        if task.requirements.needs_gpu and metrics.gpu_available > 0:
            score += 100

        return score
```

---

## CLI Commands

### `parhelia capacity`

```bash
$ parhelia capacity

Environment      CPU (avail/total)   Memory (avail/total)   GPU    Sessions   Cost/hr
─────────────────────────────────────────────────────────────────────────────────────
local            6.2 / 8 cores       12.4 / 16 GB           -      1/4        $0.00
modal-abc123     3.1 / 4 cores       14.2 / 16 GB           -      2/4        $0.15
modal-def456     2.0 / 4 cores       8.0 / 16 GB            A10G   1/2        $1.10
─────────────────────────────────────────────────────────────────────────────────────
Total            11.3 / 16 cores     34.6 / 48 GB           1      4/10       $1.25
```

### `parhelia metrics`

```bash
$ parhelia metrics --format prometheus

# HELP parhelia_cpu_usage_percent Current CPU usage percentage
# TYPE parhelia_cpu_usage_percent gauge
parhelia_cpu_usage_percent{environment="local",container_id="local"} 23.5
parhelia_cpu_usage_percent{environment="modal",container_id="abc123"} 45.2

# HELP parhelia_memory_available_bytes Available memory in bytes
# TYPE parhelia_memory_available_bytes gauge
parhelia_memory_available_bytes{environment="local",container_id="local"} 13312000000
...
```

---

## Acceptance Criteria

- [ ] [SPEC-06.AC1] Local metrics collected (CPU, memory, GPU if present)
- [ ] [SPEC-06.AC2] Pushgateway starts and accepts pushed metrics
- [ ] [SPEC-06.AC3] Modal containers push metrics every 10 seconds
- [ ] [SPEC-06.AC4] Metrics aggregated from local + Pushgateway
- [ ] [SPEC-06.AC5] Stale metrics (>30s) filtered from dispatch decisions
- [ ] [SPEC-06.AC6] Dispatch decisions use real-time capacity
- [ ] [SPEC-06.AC7] Prometheus format exposed for monitoring tools
- [ ] [SPEC-06.AC8] CLI shows capacity summary

---

## Resolved Questions

1. **Metrics push vs pull**: **Resolved** - Use Pushgateway pattern (push). Modal containers are ephemeral and not directly scrapable. Containers push to local Pushgateway every 10 seconds.
2. **Metric staleness**: **Resolved** - Metrics older than 30 seconds are filtered out. Periodic cleanup removes stale entries from Pushgateway.

## Open Questions

1. **GPU sharing**: Can multiple sessions share a GPU? How to track utilization per-session?
2. **Pushgateway availability**: What happens if local machine (running Pushgateway) is offline? Should we use a hosted Pushgateway?

---

## References

- [Prometheus Exposition Format](https://prometheus.io/docs/instrumenting/exposition_formats/)
- [Prometheus Pushgateway](https://prometheus.io/docs/instrumenting/pushing/)
- [Modal Monitoring](https://modal.com/docs/guide/observability)
- [psutil Documentation](https://psutil.readthedocs.io/)
- SPEC-01: Remote Environment Provisioning (container image includes prometheus-client)
- SPEC-05: Local Orchestrator (uses metrics for dispatch decisions)
