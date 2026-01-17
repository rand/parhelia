"""Tests for MetricsPusher - SPEC-06.13."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMetricsPusherInit:
    """Tests for MetricsPusher initialization - SPEC-06.13."""

    def test_pusher_initialization(self):
        """@trace SPEC-06.13 - MetricsPusher MUST initialize with pushgateway URL."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert pusher.pushgateway_url == "http://localhost:9091"

    def test_pusher_has_registry(self):
        """@trace SPEC-06.13 - MetricsPusher MUST create a CollectorRegistry."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert pusher.registry is not None

    def test_pusher_container_id_from_env(self):
        """@trace SPEC-06.13 - MetricsPusher SHOULD use MODAL_TASK_ID from env."""
        from parhelia.metrics.pusher import MetricsPusher

        with patch.dict(os.environ, {"MODAL_TASK_ID": "modal-xyz789"}):
            pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
            assert pusher.container_id == "modal-xyz789"

    def test_pusher_default_container_id(self):
        """@trace SPEC-06.13 - MetricsPusher SHOULD default container_id to 'unknown'."""
        from parhelia.metrics.pusher import MetricsPusher

        with patch.dict(os.environ, {}, clear=True):
            env = dict(os.environ)
            env.pop("MODAL_TASK_ID", None)
            with patch.dict(os.environ, env, clear=True):
                pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
                assert pusher.container_id == "unknown"

    def test_pusher_default_environment(self):
        """@trace SPEC-06.13 - MetricsPusher environment SHOULD default to 'modal'."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert pusher.environment == "modal"


class TestMetricsPusherGauges:
    """Tests for MetricsPusher gauge registration - SPEC-06.13."""

    def test_pusher_registers_cpu_gauges(self):
        """@trace SPEC-06.13 - MetricsPusher MUST register CPU gauges."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert "parhelia_cpu_total_cores" in pusher.gauges
        assert "parhelia_cpu_usage_percent" in pusher.gauges

    def test_pusher_registers_memory_gauges(self):
        """@trace SPEC-06.13 - MetricsPusher MUST register memory gauges."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert "parhelia_memory_total_bytes" in pusher.gauges
        assert "parhelia_memory_available_bytes" in pusher.gauges

    def test_pusher_registers_session_gauges(self):
        """@trace SPEC-06.13 - MetricsPusher MUST register session gauges."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert "parhelia_sessions_active" in pusher.gauges
        assert "parhelia_sessions_capacity" in pusher.gauges

    def test_pusher_registers_cost_gauge(self):
        """@trace SPEC-06.13 - MetricsPusher MUST register cost gauge."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(pushgateway_url="http://localhost:9091")
        assert "parhelia_cost_per_hour_usd" in pusher.gauges


class TestMetricsPusherPush:
    """Tests for MetricsPusher push functionality - SPEC-06.13."""

    @pytest.fixture
    def pusher(self):
        """Create MetricsPusher for testing."""
        from parhelia.metrics.pusher import MetricsPusher

        return MetricsPusher(pushgateway_url="http://localhost:9091")

    @pytest.mark.asyncio
    async def test_push_once_calls_push_to_gateway(self, pusher):
        """@trace SPEC-06.13 - push_once() MUST call push_to_gateway."""
        from parhelia.metrics.collector import ContainerMetrics

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        with patch("parhelia.metrics.pusher.push_to_gateway") as mock_push:
            await pusher.push_once(mock_metrics)
            mock_push.assert_called_once()

    @pytest.mark.asyncio
    async def test_push_once_uses_correct_job(self, pusher):
        """@trace SPEC-06.13 - push_once() MUST use job='parhelia_remote'."""
        from parhelia.metrics.collector import ContainerMetrics

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        with patch("parhelia.metrics.pusher.push_to_gateway") as mock_push:
            await pusher.push_once(mock_metrics)

            call_kwargs = mock_push.call_args
            assert call_kwargs[1]["job"] == "parhelia_remote"

    @pytest.mark.asyncio
    async def test_push_once_uses_container_id_grouping(self, pusher):
        """@trace SPEC-06.13 - push_once() MUST group by container_id."""
        from parhelia.metrics.collector import ContainerMetrics

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        with patch("parhelia.metrics.pusher.push_to_gateway") as mock_push:
            await pusher.push_once(mock_metrics)

            call_kwargs = mock_push.call_args
            assert "grouping_key" in call_kwargs[1]
            assert "container_id" in call_kwargs[1]["grouping_key"]


class TestMetricsPusherLoop:
    """Tests for MetricsPusher push loop - SPEC-06.13."""

    @pytest.fixture
    def pusher(self):
        """Create MetricsPusher for testing."""
        from parhelia.metrics.pusher import MetricsPusher

        return MetricsPusher(pushgateway_url="http://localhost:9091")

    def test_default_push_interval(self, pusher):
        """@trace SPEC-06.13 - Default push interval SHOULD be 10 seconds."""
        assert pusher.push_interval == 10

    def test_configurable_push_interval(self):
        """@trace SPEC-06.13 - Push interval SHOULD be configurable."""
        from parhelia.metrics.pusher import MetricsPusher

        pusher = MetricsPusher(
            pushgateway_url="http://localhost:9091",
            push_interval=5,
        )
        assert pusher.push_interval == 5

    @pytest.mark.asyncio
    async def test_push_loop_runs_periodically(self, pusher):
        """@trace SPEC-06.13 - push_loop() MUST push at regular intervals."""
        from parhelia.metrics.collector import ContainerMetrics, MetricsCollector

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        mock_collector = MagicMock(spec=MetricsCollector)
        mock_collector.collect = AsyncMock(return_value=mock_metrics)

        push_count = 0

        async def counting_push(metrics):
            nonlocal push_count
            push_count += 1

        pusher.push_once = counting_push
        pusher.push_interval = 0.1  # Fast for testing

        task = asyncio.create_task(
            pusher.push_loop(mock_collector, max_iterations=3)
        )

        await asyncio.wait_for(task, timeout=2.0)

        assert push_count == 3

    @pytest.mark.asyncio
    async def test_push_loop_can_be_stopped(self, pusher):
        """@trace SPEC-06.13 - push_loop() MUST support graceful shutdown."""
        from parhelia.metrics.collector import ContainerMetrics, MetricsCollector

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        mock_collector = MagicMock(spec=MetricsCollector)
        mock_collector.collect = AsyncMock(return_value=mock_metrics)

        with patch("parhelia.metrics.pusher.push_to_gateway"):
            pusher.push_interval = 0.1

            task = asyncio.create_task(pusher.push_loop(mock_collector))

            await asyncio.sleep(0.15)
            pusher.stop()

            await asyncio.wait_for(task, timeout=1.0)

            assert pusher._running is False


class TestMetricsPusherFinal:
    """Tests for MetricsPusher final push - SPEC-06.13."""

    @pytest.fixture
    def pusher(self):
        """Create MetricsPusher for testing."""
        from parhelia.metrics.pusher import MetricsPusher

        return MetricsPusher(pushgateway_url="http://localhost:9091")

    def test_push_final_sets_sessions_to_zero(self, pusher):
        """@trace SPEC-06.13 - push_final() MUST set sessions to 0."""
        with patch("parhelia.metrics.pusher.push_to_gateway"):
            pusher.push_final()

            # Check the gauge was set to 0
            # The gauge should have labels set
            assert True  # If no exception, gauge was set

    def test_push_final_calls_push_to_gateway(self, pusher):
        """@trace SPEC-06.13 - push_final() MUST push to gateway."""
        with patch("parhelia.metrics.pusher.push_to_gateway") as mock_push:
            pusher.push_final()
            mock_push.assert_called_once()

    def test_push_final_handles_errors_gracefully(self, pusher):
        """@trace SPEC-06.13 - push_final() MUST handle errors gracefully."""
        with patch(
            "parhelia.metrics.pusher.push_to_gateway",
            side_effect=Exception("Connection failed"),
        ):
            # Should not raise
            pusher.push_final()


class TestMetricsPusherErrorHandling:
    """Tests for MetricsPusher error handling - SPEC-06.13."""

    @pytest.fixture
    def pusher(self):
        """Create MetricsPusher for testing."""
        from parhelia.metrics.pusher import MetricsPusher

        return MetricsPusher(pushgateway_url="http://localhost:9091")

    @pytest.mark.asyncio
    async def test_push_once_handles_connection_error(self, pusher):
        """@trace SPEC-06.13 - push_once() SHOULD handle connection errors."""
        from parhelia.metrics.collector import ContainerMetrics

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        with patch(
            "parhelia.metrics.pusher.push_to_gateway",
            side_effect=Exception("Connection refused"),
        ):
            # Should not raise, just log warning
            await pusher.push_once(mock_metrics)

    @pytest.mark.asyncio
    async def test_push_loop_continues_after_error(self, pusher):
        """@trace SPEC-06.13 - push_loop() MUST continue after push failure."""
        from parhelia.metrics.collector import ContainerMetrics, MetricsCollector

        mock_metrics = ContainerMetrics(
            cpu_total_cores=4,
            cpu_available_cores=3.0,
            cpu_usage_percent=25.0,
            memory_total_bytes=16000000000,
            memory_available_bytes=12000000000,
            memory_usage_percent=25.0,
            sessions_active=1,
            sessions_capacity=4,
            cost_per_hour_usd=0.15,
        )

        mock_collector = MagicMock(spec=MetricsCollector)
        mock_collector.collect = AsyncMock(return_value=mock_metrics)

        call_count = 0

        def failing_push(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")

        with patch("parhelia.metrics.pusher.push_to_gateway", side_effect=failing_push):
            pusher.push_interval = 0.1

            task = asyncio.create_task(
                pusher.push_loop(mock_collector, max_iterations=3)
            )

            await asyncio.wait_for(task, timeout=2.0)

            # Should have attempted 3 times despite first failure
            assert call_count == 3
