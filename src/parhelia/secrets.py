"""Secret rotation support for long-running sessions.

Implements:
- [SPEC-04.20] Versioned secret names
- [SPEC-04.21] Dual-key overlap period
- [SPEC-04.22] Secret refresh callbacks
- [SPEC-04.23] Rotation event logging
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from parhelia.audit import AuditLogger


# =============================================================================
# [SPEC-04.20] Versioned Secret Names
# =============================================================================


def versioned_secret_name(name: str, version: int) -> str:
    """Create versioned secret name.

    Implements [SPEC-04.20].

    Args:
        name: Base secret name.
        version: Version number.

    Returns:
        Versioned name in format: {name}-v{version}
    """
    return f"{name}-v{version}"


def parse_versioned_secret_name(versioned_name: str) -> tuple[str, int]:
    """Parse versioned secret name into name and version.

    Implements [SPEC-04.20].

    Args:
        versioned_name: Versioned secret name.

    Returns:
        Tuple of (base_name, version). Unversioned names return version 0.
    """
    match = re.match(r"^(.+)-v(\d+)$", versioned_name)
    if match:
        return match.group(1), int(match.group(2))
    return versioned_name, 0


# =============================================================================
# [SPEC-04.21] Dual-Key Overlap Period
# =============================================================================


class SecretVersionManager:
    """Track active secret versions for dual-key overlap.

    Implements [SPEC-04.21].
    """

    def __init__(self):
        # Maps secret name -> set of active versions
        self._active_versions: dict[str, set[int]] = defaultdict(set)

    def activate_version(self, secret_name: str, version: int) -> None:
        """Activate a secret version.

        Args:
            secret_name: Base secret name.
            version: Version to activate.
        """
        self._active_versions[secret_name].add(version)

    def deactivate_version(self, secret_name: str, version: int) -> None:
        """Deactivate a secret version.

        Args:
            secret_name: Base secret name.
            version: Version to deactivate.
        """
        self._active_versions[secret_name].discard(version)

    def is_active(self, secret_name: str, version: int) -> bool:
        """Check if a secret version is active.

        Args:
            secret_name: Base secret name.
            version: Version to check.

        Returns:
            True if version is active.
        """
        return version in self._active_versions[secret_name]

    def get_active_versions(self, secret_name: str) -> set[int]:
        """Get all active versions for a secret.

        Args:
            secret_name: Base secret name.

        Returns:
            Set of active version numbers.
        """
        return self._active_versions[secret_name].copy()


# =============================================================================
# [SPEC-04.22] Secret Refresh Callbacks
# =============================================================================

# Type alias for refresh callbacks
RefreshCallback = Callable[[str, str], Coroutine[Any, Any, None]]


class SecretRefreshManager:
    """Manage secret refresh for long-running sessions.

    Implements [SPEC-04.22], [SPEC-04.23].
    """

    # [SPEC-04.21] Overlap period must exceed max session duration (24h + 1h buffer)
    OVERLAP_PERIOD_HOURS = 25

    # Default refresh interval: 5 minutes
    DEFAULT_REFRESH_INTERVAL = 300

    def __init__(
        self,
        refresh_interval_seconds: int = DEFAULT_REFRESH_INTERVAL,
        audit_logger: AuditLogger | None = None,
    ):
        """Initialize the secret refresh manager.

        Args:
            refresh_interval_seconds: How often to check for rotations.
            audit_logger: Optional audit logger for rotation events.
        """
        self.refresh_interval = refresh_interval_seconds
        self._audit_logger = audit_logger

        # Callbacks registered per secret
        self._callbacks: dict[str, list[RefreshCallback]] = defaultdict(list)

        # Cached secret values to detect changes
        self._cached_values: dict[str, str] = {}

        # Control flag for refresh loop
        self._running = False

    def register_callback(
        self,
        secret_name: str,
        callback: RefreshCallback,
    ) -> None:
        """Register a callback for secret rotation.

        Implements [SPEC-04.22].

        Args:
            secret_name: Secret to watch.
            callback: Async callback(secret_name, new_value).
        """
        self._callbacks[secret_name].append(callback)

    def unregister_callback(
        self,
        secret_name: str,
        callback: RefreshCallback,
    ) -> None:
        """Unregister a refresh callback.

        Args:
            secret_name: Secret name.
            callback: Callback to remove.
        """
        if secret_name in self._callbacks:
            try:
                self._callbacks[secret_name].remove(callback)
            except ValueError:
                pass

    async def notify_rotation(self, secret_name: str, new_value: str) -> None:
        """Notify all callbacks of a secret rotation.

        Args:
            secret_name: Rotated secret name.
            new_value: New secret value.
        """
        callbacks = self._callbacks.get(secret_name, [])
        for callback in callbacks:
            try:
                await callback(secret_name, new_value)
            except Exception:
                # Log but don't fail other callbacks
                pass

    async def check_for_rotation(
        self,
        secret_name: str,
        session_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Check if a secret has been rotated.

        Implements [SPEC-04.22], [SPEC-04.23].

        Args:
            secret_name: Secret to check.
            session_id: Optional session context for logging.
            correlation_id: Optional correlation ID for logging.

        Returns:
            True if secret was rotated.
        """
        new_value = await self._fetch_secret(secret_name)
        old_value = self._cached_values.get(secret_name)

        if old_value is not None and new_value != old_value:
            # Secret was rotated
            self._cached_values[secret_name] = new_value

            # Log rotation event [SPEC-04.23]
            if self._audit_logger:
                await self._log_rotation(
                    secret_name=secret_name,
                    session_id=session_id,
                    correlation_id=correlation_id or str(uuid.uuid4()),
                )

            # Notify callbacks
            await self.notify_rotation(secret_name, new_value)

            return True

        # Update cache even if no change
        self._cached_values[secret_name] = new_value
        return False

    async def _fetch_secret(self, secret_name: str) -> str:
        """Fetch current secret value.

        This is a placeholder - in production this would fetch from Modal Secrets.

        Args:
            secret_name: Secret to fetch.

        Returns:
            Current secret value.
        """
        # Override this method or mock it in tests
        raise NotImplementedError("Must be implemented or mocked")

    async def _log_rotation(
        self,
        secret_name: str,
        session_id: str | None,
        correlation_id: str,
    ) -> None:
        """Log secret rotation event.

        Implements [SPEC-04.23].

        Args:
            secret_name: Rotated secret name.
            session_id: Optional session context.
            correlation_id: Correlation ID for tracking.
        """
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="secret.rotation",
            session_id=session_id,
            user="system",
            action="rotate",
            resource=secret_name,
            outcome="success",
            details={
                "correlation_id": correlation_id,
            },
            source_ip=None,
        )
        await self._audit_logger.log(event)

    async def start_refresh_loop(
        self,
        secrets: list[str],
        session_id: str | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Start the refresh loop for secrets.

        Implements [SPEC-04.22].

        Args:
            secrets: List of secrets to monitor.
            session_id: Optional session context for logging.
            max_iterations: Optional max iterations (for testing).
        """
        self._running = True
        iteration = 0

        while self._running:
            if max_iterations is not None and iteration >= max_iterations:
                break

            for secret_name in secrets:
                if not self._running:
                    break
                try:
                    await self.check_for_rotation(
                        secret_name,
                        session_id=session_id,
                        correlation_id=str(uuid.uuid4()),
                    )
                except Exception:
                    # Continue checking other secrets
                    pass

            iteration += 1
            await asyncio.sleep(self.refresh_interval)

    def stop_refresh_loop(self) -> None:
        """Stop the refresh loop gracefully."""
        self._running = False
