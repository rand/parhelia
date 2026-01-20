"""Audit logging for security compliance.

Implements:
- [SPEC-04.15] Audit Logging
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os


@dataclass
class AuditEvent:
    """Security audit event.

    Implements [SPEC-04.15].
    """

    timestamp: datetime
    event_type: str
    session_id: str | None
    user: str
    action: str
    resource: str
    outcome: str  # "success", "denied", "error"
    details: dict[str, Any]
    source_ip: str | None


class AuditLogger:
    """Log security-relevant events.

    Implements [SPEC-04.15].
    """

    AUDITED_EVENTS = [
        "auth.login",
        "auth.logout",
        "secret.access",
        "secret.inject",
        "session.create",
        "session.attach",
        "session.detach",
        "permission.denied",
        "bash.dangerous_command",
        "network.egress_blocked",
        "checkpoint.create",
        "checkpoint.restore",
        "checkpoint.approve",  # [SPEC-07.20.04]
        "checkpoint.reject",  # [SPEC-07.20.04]
    ]

    # Keys containing these substrings will be redacted
    SECRET_PATTERNS = ["key", "token", "secret", "password", "credential"]

    def __init__(self, audit_root: str = "/vol/parhelia/audit"):
        """Initialize the audit logger.

        Args:
            audit_root: Root directory for audit logs.
        """
        self.audit_root = Path(audit_root)
        self.log_file = self.audit_root / "audit.jsonl"

    async def log(self, event: AuditEvent) -> None:
        """Log audit event to secure audit log.

        Implements [SPEC-04.15].

        Args:
            event: The audit event to log.
        """
        # Build structured log entry
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type,
            "session": event.session_id,
            "user": event.user,
            "action": event.action,
            "resource": event.resource,
            "outcome": event.outcome,
            "details": self._redact_secrets(event.details),
            "source_ip": event.source_ip,
        }

        # Ensure directory exists
        await aiofiles.os.makedirs(str(self.audit_root), exist_ok=True)

        # Write to append-only audit log
        async with aiofiles.open(str(self.log_file), mode="a") as f:
            await f.write(json.dumps(log_entry) + "\n")

    def _redact_secrets(self, details: dict[str, Any]) -> dict[str, Any]:
        """Redact any secrets from audit log details.

        Implements [SPEC-04.15].

        Args:
            details: The details dictionary to redact.

        Returns:
            Redacted copy of details.
        """
        redacted = {}
        for key, value in details.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in self.SECRET_PATTERNS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    def is_auditable(self, event_type: str) -> bool:
        """Check if event type should be audited.

        Args:
            event_type: The event type to check.

        Returns:
            True if event should be audited.
        """
        return event_type in self.AUDITED_EVENTS

    # ==========================================================================
    # Convenience methods for common audit events
    # ==========================================================================

    async def log_session_create(
        self,
        session_id: str,
        user: str,
        task_id: str,
        source_ip: str | None = None,
    ) -> None:
        """Log session creation event.

        Args:
            session_id: The created session ID.
            user: The user who created the session.
            task_id: The associated task ID.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="session.create",
            session_id=session_id,
            user=user,
            action="create",
            resource="session",
            outcome="success",
            details={"task_id": task_id},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_session_attach(
        self,
        session_id: str,
        user: str,
        source_ip: str | None = None,
    ) -> None:
        """Log session attachment event.

        Args:
            session_id: The session being attached to.
            user: The user attaching.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="session.attach",
            session_id=session_id,
            user=user,
            action="attach",
            resource="session",
            outcome="success",
            details={},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_session_detach(
        self,
        session_id: str,
        user: str,
        source_ip: str | None = None,
    ) -> None:
        """Log session detachment event.

        Args:
            session_id: The session being detached from.
            user: The user detaching.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="session.detach",
            session_id=session_id,
            user=user,
            action="detach",
            resource="session",
            outcome="success",
            details={},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_secret_access(
        self,
        session_id: str | None,
        user: str,
        secret_name: str,
        outcome: str = "success",
        source_ip: str | None = None,
    ) -> None:
        """Log secret access event.

        Args:
            session_id: Optional session context.
            user: The user/system accessing the secret.
            secret_name: Name of the secret being accessed.
            outcome: Access outcome (success/denied/error).
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="secret.access",
            session_id=session_id,
            user=user,
            action="access",
            resource=secret_name,
            outcome=outcome,
            details={},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_secret_inject(
        self,
        session_id: str,
        user: str,
        secret_names: list[str],
        container_id: str,
        source_ip: str | None = None,
    ) -> None:
        """Log secret injection event.

        Args:
            session_id: The session context.
            user: The user/system injecting secrets.
            secret_names: Names of secrets being injected.
            container_id: Target container ID.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="secret.inject",
            session_id=session_id,
            user=user,
            action="inject",
            resource="container",
            outcome="success",
            details={
                "secret_count": len(secret_names),
                "container_id": container_id,
            },
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
        action: str,  # "create" or "restore"
        user: str = "system",
        source_ip: str | None = None,
    ) -> None:
        """Log checkpoint creation or restoration.

        Args:
            session_id: The session context.
            checkpoint_id: The checkpoint ID.
            action: Either "create" or "restore".
            user: The user/system performing the action.
            source_ip: Optional source IP address.
        """
        event_type = f"checkpoint.{action}"
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            session_id=session_id,
            user=user,
            action=action,
            resource="checkpoint",
            outcome="success",
            details={"checkpoint_id": checkpoint_id},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_permission_denied(
        self,
        session_id: str | None,
        user: str,
        action: str,
        resource: str,
        reason: str,
        source_ip: str | None = None,
    ) -> None:
        """Log permission denied event.

        Args:
            session_id: Optional session context.
            user: The user whose permission was denied.
            action: The action that was denied.
            resource: The resource access was denied to.
            reason: Reason for denial.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="permission.denied",
            session_id=session_id,
            user=user,
            action=action,
            resource=resource,
            outcome="denied",
            details={"reason": reason},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_dangerous_command(
        self,
        session_id: str,
        user: str,
        command: str,
        blocked: bool = True,
        source_ip: str | None = None,
    ) -> None:
        """Log dangerous bash command attempt.

        Args:
            session_id: The session context.
            user: The user who attempted the command.
            command: The dangerous command (will be truncated).
            blocked: Whether the command was blocked.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="bash.dangerous_command",
            session_id=session_id,
            user=user,
            action="execute",
            resource="bash",
            outcome="denied" if blocked else "success",
            details={"command": command[:200]},  # Truncate for safety
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_network_blocked(
        self,
        session_id: str,
        user: str,
        destination: str,
        reason: str,
        source_ip: str | None = None,
    ) -> None:
        """Log blocked network egress attempt.

        Args:
            session_id: The session context.
            user: The user/process attempting egress.
            destination: The blocked destination.
            reason: Reason for blocking.
            source_ip: Optional source IP address.
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="network.egress_blocked",
            session_id=session_id,
            user=user,
            action="egress",
            resource=destination,
            outcome="denied",
            details={"reason": reason},
            source_ip=source_ip,
        )
        await self.log(event)

    async def log_auth(
        self,
        user: str,
        action: str,  # "login" or "logout"
        outcome: str = "success",
        details: dict[str, Any] | None = None,
        source_ip: str | None = None,
    ) -> None:
        """Log authentication event.

        Args:
            user: The user authenticating.
            action: Either "login" or "logout".
            outcome: Authentication outcome.
            details: Optional additional details.
            source_ip: Optional source IP address.
        """
        event_type = f"auth.{action}"
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            session_id=None,
            user=user,
            action=action,
            resource="auth",
            outcome=outcome,
            details=details or {},
            source_ip=source_ip,
        )
        await self.log(event)
