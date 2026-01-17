"""Tests for secret rotation support - SPEC-04.20-23."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVersionedSecretName:
    """Tests for versioned secret naming - SPEC-04.20."""

    def test_versioned_secret_name_format(self):
        """@trace SPEC-04.20 - Secrets MUST use versioned names: {name}-v{N}."""
        from parhelia.secrets import versioned_secret_name

        assert versioned_secret_name("anthropic-api-key", 1) == "anthropic-api-key-v1"
        assert versioned_secret_name("github-token", 2) == "github-token-v2"
        assert versioned_secret_name("db-password", 10) == "db-password-v10"

    def test_parse_versioned_secret_name(self):
        """@trace SPEC-04.20 - Parser MUST extract name and version."""
        from parhelia.secrets import parse_versioned_secret_name

        name, version = parse_versioned_secret_name("anthropic-api-key-v1")
        assert name == "anthropic-api-key"
        assert version == 1

        name, version = parse_versioned_secret_name("github-token-v5")
        assert name == "github-token"
        assert version == 5

    def test_parse_unversioned_secret_name(self):
        """@trace SPEC-04.20 - Unversioned secrets default to version 0."""
        from parhelia.secrets import parse_versioned_secret_name

        name, version = parse_versioned_secret_name("legacy-secret")
        assert name == "legacy-secret"
        assert version == 0


class TestDualKeyOverlap:
    """Tests for dual-key overlap period - SPEC-04.21."""

    def test_overlap_period_exceeds_max_session(self):
        """@trace SPEC-04.21 - Overlap period MUST exceed max session duration (24h + 1h buffer)."""
        from parhelia.secrets import SecretRefreshManager

        # Overlap period should be at least 25 hours (24h max session + 1h buffer)
        assert SecretRefreshManager.OVERLAP_PERIOD_HOURS >= 25

    def test_secret_version_manager_tracks_active_versions(self):
        """@trace SPEC-04.21 - Manager MUST track which versions are active."""
        from parhelia.secrets import SecretVersionManager

        manager = SecretVersionManager()
        manager.activate_version("anthropic-api-key", 2)

        assert manager.is_active("anthropic-api-key", 2) is True
        assert manager.is_active("anthropic-api-key", 1) is False

    def test_secret_version_manager_dual_active(self):
        """@trace SPEC-04.21 - Manager MUST support dual active versions during overlap."""
        from parhelia.secrets import SecretVersionManager

        manager = SecretVersionManager()
        manager.activate_version("anthropic-api-key", 1)
        manager.activate_version("anthropic-api-key", 2)  # Start overlap

        assert manager.is_active("anthropic-api-key", 1) is True
        assert manager.is_active("anthropic-api-key", 2) is True

    def test_secret_version_manager_deactivate_old(self):
        """@trace SPEC-04.21 - Manager MUST deactivate old versions after overlap."""
        from parhelia.secrets import SecretVersionManager

        manager = SecretVersionManager()
        manager.activate_version("anthropic-api-key", 1)
        manager.activate_version("anthropic-api-key", 2)
        manager.deactivate_version("anthropic-api-key", 1)

        assert manager.is_active("anthropic-api-key", 1) is False
        assert manager.is_active("anthropic-api-key", 2) is True


class TestSecretRefreshManager:
    """Tests for SecretRefreshManager - SPEC-04.22."""

    @pytest.fixture
    def refresh_manager(self):
        """Create SecretRefreshManager for testing."""
        from parhelia.secrets import SecretRefreshManager

        return SecretRefreshManager(refresh_interval_seconds=1)

    def test_refresh_interval_configurable(self, refresh_manager):
        """@trace SPEC-04.22 - Refresh interval SHOULD be configurable."""
        assert refresh_manager.refresh_interval == 1

    def test_default_refresh_interval(self):
        """@trace SPEC-04.22 - Default refresh interval SHOULD be 5 minutes."""
        from parhelia.secrets import SecretRefreshManager

        manager = SecretRefreshManager()
        assert manager.refresh_interval == 300  # 5 minutes

    @pytest.mark.asyncio
    async def test_register_refresh_callback(self, refresh_manager):
        """@trace SPEC-04.22 - Manager MUST support registering refresh callbacks."""
        callback = AsyncMock()
        refresh_manager.register_callback("anthropic-api-key", callback)

        assert "anthropic-api-key" in refresh_manager._callbacks
        assert callback in refresh_manager._callbacks["anthropic-api-key"]

    @pytest.mark.asyncio
    async def test_notify_callbacks_on_rotation(self, refresh_manager):
        """@trace SPEC-04.22 - Manager MUST notify callbacks when secret rotates."""
        callback = AsyncMock()
        refresh_manager.register_callback("anthropic-api-key", callback)

        await refresh_manager.notify_rotation("anthropic-api-key", "new-value")

        callback.assert_called_once_with("anthropic-api-key", "new-value")

    @pytest.mark.asyncio
    async def test_multiple_callbacks_per_secret(self, refresh_manager):
        """@trace SPEC-04.22 - Manager MUST support multiple callbacks per secret."""
        callback1 = AsyncMock()
        callback2 = AsyncMock()

        refresh_manager.register_callback("anthropic-api-key", callback1)
        refresh_manager.register_callback("anthropic-api-key", callback2)

        await refresh_manager.notify_rotation("anthropic-api-key", "new-value")

        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_callback(self, refresh_manager):
        """@trace SPEC-04.22 - Manager MUST support unregistering callbacks."""
        callback = AsyncMock()
        refresh_manager.register_callback("anthropic-api-key", callback)
        refresh_manager.unregister_callback("anthropic-api-key", callback)

        await refresh_manager.notify_rotation("anthropic-api-key", "new-value")

        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_for_rotation_detects_change(self, refresh_manager):
        """@trace SPEC-04.22 - Manager MUST detect when secret value changes."""
        # Mock the secret fetcher
        refresh_manager._cached_values = {"anthropic-api-key": "old-value"}
        refresh_manager._fetch_secret = AsyncMock(return_value="new-value")

        changed = await refresh_manager.check_for_rotation("anthropic-api-key")

        assert changed is True
        assert refresh_manager._cached_values["anthropic-api-key"] == "new-value"

    @pytest.mark.asyncio
    async def test_check_for_rotation_no_change(self, refresh_manager):
        """@trace SPEC-04.22 - Manager MUST not notify if value unchanged."""
        refresh_manager._cached_values = {"anthropic-api-key": "same-value"}
        refresh_manager._fetch_secret = AsyncMock(return_value="same-value")

        changed = await refresh_manager.check_for_rotation("anthropic-api-key")

        assert changed is False


class TestRotationLogging:
    """Tests for rotation event logging - SPEC-04.23."""

    @pytest.fixture
    def refresh_manager(self, tmp_path):
        """Create SecretRefreshManager with audit logger."""
        from parhelia.audit import AuditLogger
        from parhelia.secrets import SecretRefreshManager

        audit_logger = AuditLogger(audit_root=str(tmp_path / "audit"))
        return SecretRefreshManager(audit_logger=audit_logger)

    @pytest.mark.asyncio
    async def test_rotation_logged_with_correlation_id(self, refresh_manager, tmp_path):
        """@trace SPEC-04.23 - Rotation events MUST be logged with correlation IDs."""
        import json

        refresh_manager._cached_values = {"test-secret": "old-value"}
        refresh_manager._fetch_secret = AsyncMock(return_value="new-value")

        await refresh_manager.check_for_rotation(
            "test-secret",
            session_id="sess-123",
            correlation_id="corr-456",
        )

        log_file = tmp_path / "audit" / "audit.jsonl"
        assert log_file.exists()

        entry = json.loads(log_file.read_text().strip())
        assert entry["type"] == "secret.rotation"
        assert entry["details"]["correlation_id"] == "corr-456"
        assert entry["session"] == "sess-123"

    @pytest.mark.asyncio
    async def test_rotation_log_does_not_contain_secret_value(
        self, refresh_manager, tmp_path
    ):
        """@trace SPEC-04.23 - Rotation logs MUST NOT contain secret values."""
        import json

        refresh_manager._cached_values = {"api-key": "sk-secret123"}
        refresh_manager._fetch_secret = AsyncMock(return_value="sk-newsecret456")

        await refresh_manager.check_for_rotation("api-key", correlation_id="corr-789")

        log_file = tmp_path / "audit" / "audit.jsonl"
        content = log_file.read_text()

        assert "sk-secret123" not in content
        assert "sk-newsecret456" not in content


class TestSecretRefreshLoop:
    """Tests for the refresh loop functionality - SPEC-04.22."""

    @pytest.mark.asyncio
    async def test_refresh_loop_runs_periodically(self):
        """@trace SPEC-04.22 - Long-running sessions MUST implement refresh callbacks."""
        from parhelia.secrets import SecretRefreshManager

        manager = SecretRefreshManager(refresh_interval_seconds=0.1)
        manager._fetch_secret = AsyncMock(return_value="value")

        check_count = 0
        original_check = manager.check_for_rotation

        async def counting_check(*args, **kwargs):
            nonlocal check_count
            check_count += 1
            return await original_check(*args, **kwargs)

        manager.check_for_rotation = counting_check
        manager._cached_values = {"test-secret": "value"}

        # Start loop and let it run briefly
        task = asyncio.create_task(
            manager.start_refresh_loop(["test-secret"], max_iterations=3)
        )

        await asyncio.wait_for(task, timeout=2.0)

        assert check_count >= 2

    @pytest.mark.asyncio
    async def test_refresh_loop_can_be_stopped(self):
        """@trace SPEC-04.22 - Refresh loop MUST support graceful shutdown."""
        from parhelia.secrets import SecretRefreshManager

        manager = SecretRefreshManager(refresh_interval_seconds=0.1)
        manager._fetch_secret = AsyncMock(return_value="value")
        manager._cached_values = {"test-secret": "value"}

        task = asyncio.create_task(manager.start_refresh_loop(["test-secret"]))

        await asyncio.sleep(0.15)
        manager.stop_refresh_loop()

        await asyncio.wait_for(task, timeout=1.0)

        assert manager._running is False
