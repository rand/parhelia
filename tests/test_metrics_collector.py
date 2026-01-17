"""Tests for MetricsCollector - SPEC-06.12."""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestContainerMetrics:
    """Tests for ContainerMetrics dataclass - SPEC-06.12."""

    def test_container_metrics_creation(self):
        """@trace SPEC-06.12 - ContainerMetrics MUST store resource metrics."""
        from parhelia.metrics.collector import ContainerMetrics

        metrics = ContainerMetrics(
            cpu_total_cores=8,
            cpu_available_cores=6.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=17179869184,
            memory_available_bytes=12884901888,
            memory_usage_percent=25.0,
            sessions_active=2,
            sessions_capacity=4,
            cost_per_hour_usd=1.10,
        )

        assert metrics.cpu_total_cores == 8
        assert metrics.cpu_usage_percent == 25.0
        assert metrics.memory_total_bytes == 17179869184
        assert metrics.sessions_active == 2
        assert metrics.cost_per_hour_usd == 1.10

    def test_container_metrics_with_gpu(self):
        """@trace SPEC-06.12 - ContainerMetrics SHOULD support GPU metrics."""
        from parhelia.metrics.collector import ContainerMetrics, GPUMetrics

        gpu_metrics = GPUMetrics(
            gpu_index=0,
            gpu_type="A10G",
            memory_total_bytes=24576000000,
            memory_used_bytes=8192000000,
            utilization_percent=45.0,
        )

        metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=2.5,
            cpu_usage_percent=37.5,
            memory_total_bytes=17179869184,
            memory_available_bytes=10000000000,
            memory_usage_percent=42.0,
            sessions_active=1,
            sessions_capacity=2,
            cost_per_hour_usd=1.10,
            gpu_total_count=1,
            gpu_available_count=1,
            gpu_metrics=[gpu_metrics],
        )

        assert metrics.gpu_total_count == 1
        assert len(metrics.gpu_metrics) == 1
        assert metrics.gpu_metrics[0].utilization_percent == 45.0


class TestMetricsCollector:
    """Tests for MetricsCollector class - SPEC-06.12."""

    @pytest.fixture
    def collector(self):
        """Create MetricsCollector for testing."""
        from parhelia.metrics.collector import MetricsCollector

        return MetricsCollector()

    def test_collector_initialization(self, collector):
        """@trace SPEC-06.12 - MetricsCollector MUST initialize properly."""
        assert collector is not None

    @pytest.mark.asyncio
    async def test_collect_returns_container_metrics(self, collector):
        """@trace SPEC-06.12 - collect() MUST return ContainerMetrics."""
        from parhelia.metrics.collector import ContainerMetrics

        with patch("psutil.cpu_count", return_value=8):
            with patch("psutil.cpu_percent", return_value=25.0):
                mock_memory = MagicMock()
                mock_memory.total = 17179869184
                mock_memory.available = 12884901888
                mock_memory.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        assert isinstance(metrics, ContainerMetrics)

    @pytest.mark.asyncio
    async def test_collect_cpu_metrics(self, collector):
        """@trace SPEC-06.12 - collect() MUST include CPU metrics."""
        with patch("psutil.cpu_count", return_value=8):
            with patch("psutil.cpu_percent", return_value=50.0):
                mock_memory = MagicMock()
                mock_memory.total = 17179869184
                mock_memory.available = 12884901888
                mock_memory.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        assert metrics.cpu_total_cores == 8
        assert metrics.cpu_usage_percent == 50.0
        assert metrics.cpu_available_cores == 4.0  # 8 * (1 - 50/100)

    @pytest.mark.asyncio
    async def test_collect_memory_metrics(self, collector):
        """@trace SPEC-06.12 - collect() MUST include memory metrics."""
        with patch("psutil.cpu_count", return_value=4):
            with patch("psutil.cpu_percent", return_value=10.0):
                mock_memory = MagicMock()
                mock_memory.total = 16000000000
                mock_memory.available = 12000000000
                mock_memory.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        assert metrics.memory_total_bytes == 16000000000
        assert metrics.memory_available_bytes == 12000000000
        assert metrics.memory_usage_percent == 25.0


class TestMetricsCollectorSessions:
    """Tests for session tracking in MetricsCollector - SPEC-06.12."""

    @pytest.fixture
    def collector(self):
        """Create MetricsCollector for testing."""
        from parhelia.metrics.collector import MetricsCollector

        return MetricsCollector(max_sessions=4)

    def test_default_max_sessions(self):
        """@trace SPEC-06.12 - MetricsCollector SHOULD have configurable max sessions."""
        from parhelia.metrics.collector import MetricsCollector

        collector = MetricsCollector()
        assert collector.max_sessions > 0

    @pytest.mark.asyncio
    async def test_collect_includes_session_capacity(self, collector):
        """@trace SPEC-06.12 - collect() MUST include session capacity."""
        with patch("psutil.cpu_count", return_value=4):
            with patch("psutil.cpu_percent", return_value=10.0):
                mock_memory = MagicMock()
                mock_memory.total = 16000000000
                mock_memory.available = 12000000000
                mock_memory.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        assert metrics.sessions_capacity == 4

    def test_register_active_session(self, collector):
        """@trace SPEC-06.12 - MetricsCollector MUST track active sessions."""
        collector.register_session("session-1")
        assert collector.active_session_count == 1

        collector.register_session("session-2")
        assert collector.active_session_count == 2

    def test_unregister_session(self, collector):
        """@trace SPEC-06.12 - MetricsCollector MUST unregister sessions."""
        collector.register_session("session-1")
        collector.register_session("session-2")
        collector.unregister_session("session-1")

        assert collector.active_session_count == 1

    @pytest.mark.asyncio
    async def test_collect_includes_active_sessions(self, collector):
        """@trace SPEC-06.12 - collect() MUST include active session count."""
        collector.register_session("session-1")
        collector.register_session("session-2")

        with patch("psutil.cpu_count", return_value=4):
            with patch("psutil.cpu_percent", return_value=10.0):
                mock_memory = MagicMock()
                mock_memory.total = 16000000000
                mock_memory.available = 12000000000
                mock_memory.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_memory):
                    metrics = await collector.collect()

        assert metrics.sessions_active == 2


class TestMetricsCollectorCost:
    """Tests for cost calculation in MetricsCollector - SPEC-06.12."""

    def test_local_environment_cost_zero(self):
        """@trace SPEC-06.12 - Local environment cost MUST be zero."""
        from parhelia.metrics.collector import MetricsCollector

        collector = MetricsCollector(environment="local")
        assert collector.get_hourly_cost() == 0.0

    def test_modal_cpu_environment_cost(self):
        """@trace SPEC-06.12 - Modal CPU environment SHOULD have base rate cost."""
        from parhelia.metrics.collector import MetricsCollector

        collector = MetricsCollector(environment="modal", gpu_type=None)
        cost = collector.get_hourly_cost()
        assert cost > 0.0

    def test_modal_gpu_environment_cost(self):
        """@trace SPEC-06.12 - Modal GPU environment MUST have GPU-based cost."""
        from parhelia.metrics.collector import MetricsCollector

        collector_t4 = MetricsCollector(environment="modal", gpu_type="T4")
        collector_a10g = MetricsCollector(environment="modal", gpu_type="A10G")
        collector_h100 = MetricsCollector(environment="modal", gpu_type="H100")

        # GPU costs should be higher than CPU and vary by type
        assert collector_t4.get_hourly_cost() > 0.15
        assert collector_a10g.get_hourly_cost() > collector_t4.get_hourly_cost()
        assert collector_h100.get_hourly_cost() > collector_a10g.get_hourly_cost()


class TestMetricsCollectorGPU:
    """Tests for GPU metrics collection - SPEC-06.12."""

    @pytest.fixture
    def collector_with_gpu(self):
        """Create MetricsCollector with GPU support for testing."""
        from parhelia.metrics.collector import MetricsCollector

        return MetricsCollector(environment="modal", gpu_type="A10G")

    def test_has_gpu_returns_false_without_gpu(self):
        """@trace SPEC-06.12 - has_gpu() MUST return False without GPU."""
        from parhelia.metrics.collector import MetricsCollector

        collector = MetricsCollector(environment="local", gpu_type=None)
        assert collector.has_gpu() is False

    def test_has_gpu_returns_true_with_gpu(self, collector_with_gpu):
        """@trace SPEC-06.12 - has_gpu() MUST return True with GPU configured."""
        assert collector_with_gpu.has_gpu() is True

    @pytest.mark.asyncio
    async def test_collect_gpu_metrics_when_available(self, collector_with_gpu):
        """@trace SPEC-06.12 - collect() SHOULD include GPU metrics when available."""
        # Mock pynvml
        mock_handle = MagicMock()

        mock_memory = MagicMock()
        mock_memory.total = 24576000000
        mock_memory.used = 8192000000

        mock_utilization = MagicMock()
        mock_utilization.gpu = 45

        with patch("psutil.cpu_count", return_value=4):
            with patch("psutil.cpu_percent", return_value=10.0):
                mock_mem = MagicMock()
                mock_mem.total = 16000000000
                mock_mem.available = 12000000000
                mock_mem.percent = 25.0
                with patch("psutil.virtual_memory", return_value=mock_mem):
                    with patch.object(
                        collector_with_gpu,
                        "_collect_gpu_metrics",
                        new_callable=AsyncMock,
                        return_value=[
                            {
                                "gpu_index": 0,
                                "gpu_type": "A10G",
                                "memory_total_bytes": 24576000000,
                                "memory_used_bytes": 8192000000,
                                "utilization_percent": 45.0,
                            }
                        ],
                    ):
                        metrics = await collector_with_gpu.collect()

        assert metrics.gpu_total_count == 1
        assert len(metrics.gpu_metrics) == 1
        assert metrics.gpu_metrics[0].utilization_percent == 45.0


class TestMetricsCollectorContainerID:
    """Tests for container ID handling - SPEC-06.12."""

    def test_container_id_from_environment(self):
        """@trace SPEC-06.12 - Container ID SHOULD come from MODAL_TASK_ID env."""
        from parhelia.metrics.collector import MetricsCollector

        with patch.dict(os.environ, {"MODAL_TASK_ID": "modal-abc123"}):
            collector = MetricsCollector()
            assert collector.container_id == "modal-abc123"

    def test_container_id_default_local(self):
        """@trace SPEC-06.12 - Container ID SHOULD default to 'local' without env."""
        from parhelia.metrics.collector import MetricsCollector

        with patch.dict(os.environ, {}, clear=True):
            # Remove MODAL_TASK_ID if present
            env = dict(os.environ)
            env.pop("MODAL_TASK_ID", None)
            with patch.dict(os.environ, env, clear=True):
                collector = MetricsCollector()
                assert collector.container_id == "local"


class TestMetricsToDict:
    """Tests for metrics serialization - SPEC-06.12."""

    def test_container_metrics_to_dict(self):
        """@trace SPEC-06.12 - ContainerMetrics MUST serialize to dict for Prometheus."""
        from parhelia.metrics.collector import ContainerMetrics

        metrics = ContainerMetrics(
            cpu_total_cores=8,
            cpu_available_cores=6.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=17179869184,
            memory_available_bytes=12884901888,
            memory_usage_percent=25.0,
            sessions_active=2,
            sessions_capacity=4,
            cost_per_hour_usd=1.10,
        )

        d = metrics.to_prometheus_dict()

        assert "parhelia_cpu_total_cores" in d
        assert d["parhelia_cpu_total_cores"] == 8
        assert "parhelia_cpu_usage_percent" in d
        assert d["parhelia_cpu_usage_percent"] == 25.0
        assert "parhelia_memory_total_bytes" in d
        assert "parhelia_sessions_active" in d
        assert "parhelia_cost_per_hour_usd" in d
