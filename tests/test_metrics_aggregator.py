"""Tests for MetricsAggregator - SPEC-06.14."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestResourceMetrics:
    """Tests for ResourceMetrics dataclass - SPEC-06.14."""

    def test_resource_metrics_creation(self):
        """@trace SPEC-06.14 - ResourceMetrics MUST store aggregated metrics."""
        from parhelia.metrics.aggregator import ResourceMetrics

        metrics = ResourceMetrics(
            environment_id="modal-abc123",
            cpu_total=4,
            cpu_available=2.5,
            cpu_usage_percent=37.5,
            memory_total=16000000000,
            memory_available=10000000000,
            memory_usage_percent=37.5,
            gpu_available=1,
            sessions_active=2,
            sessions_capacity=4,
            cost_per_hour=1.10,
            last_updated=time.time(),
        )

        assert metrics.environment_id == "modal-abc123"
        assert metrics.cpu_total == 4
        assert metrics.sessions_active == 2

    def test_resource_metrics_from_dict(self):
        """@trace SPEC-06.14 - ResourceMetrics MUST be creatable from Prometheus dict."""
        from parhelia.metrics.aggregator import ResourceMetrics

        data = {
            "parhelia_cpu_total_cores": 8,
            "parhelia_cpu_available_cores": 6.0,
            "parhelia_cpu_usage_percent": 25.0,
            "parhelia_memory_total_bytes": 17179869184,
            "parhelia_memory_available_bytes": 12884901888,
            "parhelia_memory_usage_percent": 25.0,
            "parhelia_gpu_available_count": 0,
            "parhelia_sessions_active": 1,
            "parhelia_sessions_capacity": 4,
            "parhelia_cost_per_hour_usd": 0.15,
        }

        metrics = ResourceMetrics.from_dict(data, "container-1")

        assert metrics.environment_id == "container-1"
        assert metrics.cpu_total == 8
        assert metrics.memory_total == 17179869184


class TestCapacitySummary:
    """Tests for CapacitySummary dataclass - SPEC-06.14."""

    def test_capacity_summary_creation(self):
        """@trace SPEC-06.14 - CapacitySummary MUST aggregate system capacity."""
        from parhelia.metrics.aggregator import CapacitySummary

        summary = CapacitySummary(
            total_cpu_cores=16.0,
            total_memory_gb=48.0,
            total_gpu_count=2,
            available_session_slots=8,
            environments=["local", "modal-abc123", "modal-def456"],
        )

        assert summary.total_cpu_cores == 16.0
        assert summary.total_gpu_count == 2
        assert len(summary.environments) == 3


class TestMetricsAggregatorInit:
    """Tests for MetricsAggregator initialization - SPEC-06.14."""

    def test_aggregator_initialization(self):
        """@trace SPEC-06.14 - MetricsAggregator MUST initialize with Pushgateway port."""
        from parhelia.metrics.aggregator import MetricsAggregator

        aggregator = MetricsAggregator(pushgateway_port=9091)
        assert aggregator.pushgateway_url == "http://localhost:9091"

    def test_aggregator_default_port(self):
        """@trace SPEC-06.14 - Default Pushgateway port SHOULD be 9091."""
        from parhelia.metrics.aggregator import MetricsAggregator

        aggregator = MetricsAggregator()
        assert "9091" in aggregator.pushgateway_url

    def test_aggregator_cache_ttl(self):
        """@trace SPEC-06.14 - Cache TTL SHOULD be 5 seconds."""
        from parhelia.metrics.aggregator import MetricsAggregator

        aggregator = MetricsAggregator()
        assert aggregator.cache_ttl == 5

    def test_aggregator_stale_threshold(self):
        """@trace SPEC-06.14 - Stale threshold SHOULD be 30 seconds."""
        from parhelia.metrics.aggregator import MetricsAggregator

        aggregator = MetricsAggregator()
        assert aggregator.stale_threshold == 30


class TestMetricsAggregatorLocal:
    """Tests for local metrics collection - SPEC-06.14."""

    @pytest.fixture
    def aggregator(self):
        """Create MetricsAggregator for testing."""
        from parhelia.metrics.aggregator import MetricsAggregator

        return MetricsAggregator()

    @pytest.mark.asyncio
    async def test_get_all_metrics_includes_local(self, aggregator):
        """@trace SPEC-06.14 - get_all_metrics MUST include local metrics."""
        from parhelia.metrics.collector import ContainerMetrics

        mock_local_metrics = ContainerMetrics(
            cpu_total_cores=8,
            cpu_available_cores=6.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=17179869184,
            memory_available_bytes=12884901888,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.0,
        )

        with patch.object(
            aggregator.local_collector,
            "collect",
            new_callable=AsyncMock,
            return_value=mock_local_metrics,
        ):
            with patch.object(
                aggregator,
                "_fetch_from_pushgateway",
                new_callable=AsyncMock,
                return_value={},
            ):
                metrics = await aggregator.get_all_metrics()

        assert "local" in metrics
        assert metrics["local"].cpu_total == 8


class TestMetricsAggregatorPushgateway:
    """Tests for Pushgateway metrics fetching - SPEC-06.14."""

    @pytest.fixture
    def aggregator(self):
        """Create MetricsAggregator for testing."""
        from parhelia.metrics.aggregator import MetricsAggregator

        return MetricsAggregator()

    @pytest.mark.asyncio
    async def test_fetch_from_pushgateway_parses_metrics(self, aggregator):
        """@trace SPEC-06.14 - _fetch_from_pushgateway MUST parse Prometheus format."""
        prometheus_output = '''# HELP parhelia_cpu_total_cores Total CPU cores
# TYPE parhelia_cpu_total_cores gauge
parhelia_cpu_total_cores{environment="modal",container_id="abc123"} 4
parhelia_cpu_usage_percent{environment="modal",container_id="abc123"} 50.0
parhelia_memory_total_bytes{environment="modal",container_id="abc123"} 16000000000
parhelia_memory_available_bytes{environment="modal",container_id="abc123"} 8000000000
parhelia_memory_usage_percent{environment="modal",container_id="abc123"} 50.0
parhelia_sessions_active{environment="modal",container_id="abc123"} 2
parhelia_sessions_capacity{environment="modal",container_id="abc123"} 4
parhelia_cost_per_hour_usd{environment="modal",container_id="abc123"} 0.15
'''

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=prometheus_output)

        mock_session = MagicMock()
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            metrics = await aggregator._fetch_from_pushgateway()

        assert "abc123" in metrics
        assert metrics["abc123"].cpu_total == 4

    @pytest.mark.asyncio
    async def test_fetch_handles_empty_response(self, aggregator):
        """@trace SPEC-06.14 - _fetch_from_pushgateway MUST handle empty response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="")

        mock_session = MagicMock()
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_get_cm)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            metrics = await aggregator._fetch_from_pushgateway()

        assert metrics == {}

    @pytest.mark.asyncio
    async def test_fetch_handles_connection_error(self, aggregator):
        """@trace SPEC-06.14 - _fetch_from_pushgateway MUST handle connection errors."""
        import aiohttp

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            metrics = await aggregator._fetch_from_pushgateway()

        assert metrics == {}


class TestMetricsAggregatorStaleness:
    """Tests for stale metrics handling - SPEC-06.14."""

    @pytest.fixture
    def aggregator(self):
        """Create MetricsAggregator for testing."""
        from parhelia.metrics.aggregator import MetricsAggregator

        return MetricsAggregator(stale_threshold=30)

    @pytest.mark.asyncio
    async def test_stale_metrics_filtered(self, aggregator):
        """@trace SPEC-06.14 - Stale metrics (>30s) MUST be filtered."""
        from parhelia.metrics.aggregator import ResourceMetrics

        old_time = time.time() - 60  # 60 seconds ago

        stale_metrics = ResourceMetrics(
            environment_id="modal-stale",
            cpu_total=4,
            cpu_available=2.0,
            cpu_usage_percent=50.0,
            memory_total=16000000000,
            memory_available=8000000000,
            memory_usage_percent=50.0,
            gpu_available=0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour=0.15,
            last_updated=old_time,
        )

        assert aggregator._is_stale(stale_metrics) is True

    @pytest.mark.asyncio
    async def test_fresh_metrics_not_filtered(self, aggregator):
        """@trace SPEC-06.14 - Fresh metrics MUST NOT be filtered."""
        from parhelia.metrics.aggregator import ResourceMetrics

        fresh_metrics = ResourceMetrics(
            environment_id="modal-fresh",
            cpu_total=4,
            cpu_available=2.0,
            cpu_usage_percent=50.0,
            memory_total=16000000000,
            memory_available=8000000000,
            memory_usage_percent=50.0,
            gpu_available=0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour=0.15,
            last_updated=time.time(),
        )

        assert aggregator._is_stale(fresh_metrics) is False


class TestMetricsAggregatorCapacitySummary:
    """Tests for capacity summary - SPEC-06.14."""

    @pytest.fixture
    def aggregator(self):
        """Create MetricsAggregator for testing."""
        from parhelia.metrics.aggregator import MetricsAggregator

        return MetricsAggregator()

    @pytest.mark.asyncio
    async def test_get_capacity_summary_aggregates(self, aggregator):
        """@trace SPEC-06.14 - get_capacity_summary MUST aggregate all environments."""
        from parhelia.metrics.aggregator import ResourceMetrics

        mock_metrics = {
            "local": ResourceMetrics(
                environment_id="local",
                cpu_total=8,
                cpu_available=6.0,
                cpu_usage_percent=25.0,
                memory_total=17179869184,
                memory_available=12884901888,
                memory_usage_percent=25.0,
                gpu_available=0,
                sessions_active=1,
                sessions_capacity=4,
                cost_per_hour=0.0,
                last_updated=time.time(),
            ),
            "modal-abc": ResourceMetrics(
                environment_id="modal-abc",
                cpu_total=4,
                cpu_available=2.0,
                cpu_usage_percent=50.0,
                memory_total=16000000000,
                memory_available=8000000000,
                memory_usage_percent=50.0,
                gpu_available=1,
                sessions_active=2,
                sessions_capacity=4,
                cost_per_hour=1.10,
                last_updated=time.time(),
            ),
        }

        with patch.object(
            aggregator,
            "get_all_metrics",
            new_callable=AsyncMock,
            return_value=mock_metrics,
        ):
            summary = await aggregator.get_capacity_summary()

        # Total CPU available = 6.0 + 2.0 = 8.0
        assert summary.total_cpu_cores == 8.0
        # Total GPU = 0 + 1 = 1
        assert summary.total_gpu_count == 1
        # Available sessions = (4-1) + (4-2) = 3 + 2 = 5
        assert summary.available_session_slots == 5
        assert len(summary.environments) == 2
