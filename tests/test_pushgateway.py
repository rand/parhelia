"""Tests for Prometheus Pushgateway management - SPEC-06.13a."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPushgatewayConfig:
    """Tests for Pushgateway configuration - SPEC-06.13a."""

    def test_default_port(self):
        """@trace SPEC-06.13a - Pushgateway SHOULD use default port 9091."""
        from parhelia.metrics.pushgateway import PushgatewayConfig

        config = PushgatewayConfig()
        assert config.port == 9091

    def test_configurable_port(self):
        """@trace SPEC-06.13a - Pushgateway port MUST be configurable."""
        from parhelia.metrics.pushgateway import PushgatewayConfig

        config = PushgatewayConfig(port=9092)
        assert config.port == 9092

    def test_default_group_ttl(self):
        """@trace SPEC-06.13a - Group TTL SHOULD default to 60 seconds."""
        from parhelia.metrics.pushgateway import PushgatewayConfig

        config = PushgatewayConfig()
        assert config.group_ttl_seconds == 60

    def test_local_url(self):
        """@trace SPEC-06.13a - Config MUST provide local URL."""
        from parhelia.metrics.pushgateway import PushgatewayConfig

        config = PushgatewayConfig(port=9091)
        assert config.local_url == "http://localhost:9091"


class TestPushgatewayManager:
    """Tests for PushgatewayManager class - SPEC-06.13a."""

    @pytest.fixture
    def manager(self):
        """Create PushgatewayManager for testing."""
        from parhelia.metrics.pushgateway import PushgatewayConfig, PushgatewayManager

        config = PushgatewayConfig(port=9091)
        return PushgatewayManager(config)

    def test_manager_initialization(self, manager):
        """@trace SPEC-06.13a - Manager MUST initialize with config."""
        assert manager.config.port == 9091
        assert manager.process is None
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_start_sets_running(self, manager):
        """@trace SPEC-06.13a - Start MUST set running state."""
        with patch.object(manager, "_spawn_process", new_callable=AsyncMock):
            with patch.object(manager, "_wait_ready", new_callable=AsyncMock):
                await manager.start()

                assert manager._running is True

    @pytest.mark.asyncio
    async def test_start_spawns_process(self, manager):
        """@trace SPEC-06.13a - Start MUST spawn Pushgateway process."""
        mock_spawn = AsyncMock()
        with patch.object(manager, "_spawn_process", mock_spawn):
            with patch.object(manager, "_wait_ready", new_callable=AsyncMock):
                await manager.start()

                mock_spawn.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_waits_for_ready(self, manager):
        """@trace SPEC-06.13a - Start MUST wait for Pushgateway to be ready."""
        mock_wait = AsyncMock()
        with patch.object(manager, "_spawn_process", new_callable=AsyncMock):
            with patch.object(manager, "_wait_ready", mock_wait):
                await manager.start()

                mock_wait.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self, manager):
        """@trace SPEC-06.13a - Stop MUST terminate Pushgateway process."""
        mock_process = MagicMock()
        manager.process = mock_process
        manager._running = True

        await manager.stop()

        mock_process.terminate.assert_called_once()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_stop_handles_no_process(self, manager):
        """@trace SPEC-06.13a - Stop MUST handle case when process not running."""
        manager.process = None
        manager._running = False

        # Should not raise
        await manager.stop()

        assert manager._running is False

    def test_get_url_returns_local_url(self, manager):
        """@trace SPEC-06.13a - get_url MUST return URL for containers."""
        url = manager.get_url()
        assert url == "http://localhost:9091"

    def test_get_url_with_custom_host(self):
        """@trace SPEC-06.13a - get_url MUST support custom host for remote access."""
        from parhelia.metrics.pushgateway import PushgatewayConfig, PushgatewayManager

        config = PushgatewayConfig(port=9091, public_host="192.168.1.100")
        manager = PushgatewayManager(config)

        url = manager.get_url()
        assert url == "http://192.168.1.100:9091"

    def test_get_url_from_env(self, manager):
        """@trace SPEC-06.13a - get_url MUST check PARHELIA_PUSHGATEWAY_URL env."""
        with patch.dict("os.environ", {"PARHELIA_PUSHGATEWAY_URL": "http://custom:9091"}):
            url = manager.get_url()
            assert url == "http://custom:9091"


class TestPushgatewayHealth:
    """Tests for Pushgateway health checking - SPEC-06.13a."""

    @pytest.fixture
    def manager(self):
        """Create PushgatewayManager for testing."""
        from parhelia.metrics.pushgateway import PushgatewayConfig, PushgatewayManager

        return PushgatewayManager(PushgatewayConfig())

    @pytest.mark.asyncio
    async def test_is_healthy_returns_true_when_ready(self, manager):
        """@trace SPEC-06.13a - is_healthy MUST return True when Pushgateway responds."""
        mock_response = MagicMock()
        mock_response.status = 200

        # Create proper async context manager for response
        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)

        # Create mock session with get method
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_cm)

        # Create proper async context manager for session
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await manager.is_healthy()
            assert result is True

    @pytest.mark.asyncio
    async def test_is_healthy_returns_false_on_error(self, manager):
        """@trace SPEC-06.13a - is_healthy MUST return False on connection error."""
        import aiohttp

        # Create mock session that raises on get
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientError())

        # Create proper async context manager for session
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await manager.is_healthy()
            assert result is False


class TestPushgatewayDocker:
    """Tests for Docker-based Pushgateway deployment - SPEC-06.13a."""

    def test_docker_compose_config_generated(self, tmp_path):
        """@trace SPEC-06.13a - Manager MUST generate Docker compose config."""
        from parhelia.metrics.pushgateway import PushgatewayConfig, generate_docker_compose

        config = PushgatewayConfig(port=9091)
        compose_content = generate_docker_compose(config)

        assert "pushgateway" in compose_content
        assert "9091:9091" in compose_content
        assert "prom/pushgateway" in compose_content

    def test_docker_compose_includes_persistence(self, tmp_path):
        """@trace SPEC-06.13a - Docker compose SHOULD include persistence config."""
        from parhelia.metrics.pushgateway import PushgatewayConfig, generate_docker_compose

        config = PushgatewayConfig(port=9091, persistence_file="/data/metrics")
        compose_content = generate_docker_compose(config)

        assert "--persistence.file" in compose_content


class TestMetricsEndpoint:
    """Tests for metrics endpoint access - SPEC-06.13a."""

    @pytest.fixture
    def manager(self):
        """Create PushgatewayManager for testing."""
        from parhelia.metrics.pushgateway import PushgatewayConfig, PushgatewayManager

        return PushgatewayManager(PushgatewayConfig())

    def test_metrics_endpoint_url(self, manager):
        """@trace SPEC-06.13a - Manager MUST provide metrics endpoint URL."""
        url = manager.get_metrics_url()
        assert url == "http://localhost:9091/metrics"

    def test_api_endpoint_url(self, manager):
        """@trace SPEC-06.13a - Manager MUST provide API endpoint URL."""
        url = manager.get_api_url()
        assert url == "http://localhost:9091/api/v1/metrics"
