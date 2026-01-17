"""Tests for audit logging - SPEC-04.15."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


class TestAuditEvent:
    """Tests for AuditEvent dataclass - SPEC-04.15."""

    def test_audit_event_creation(self):
        """@trace SPEC-04.15 - AuditEvent MUST capture required fields."""
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="auth.login",
            session_id="sess-123",
            user="testuser",
            action="login",
            resource="session",
            outcome="success",
            details={"method": "api_key"},
            source_ip="192.168.1.1",
        )

        assert event.event_type == "auth.login"
        assert event.session_id == "sess-123"
        assert event.user == "testuser"
        assert event.outcome == "success"

    def test_audit_event_optional_fields(self):
        """@trace SPEC-04.15 - AuditEvent MUST support optional session_id and source_ip."""
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="secret.access",
            session_id=None,
            user="system",
            action="read",
            resource="anthropic-api-key",
            outcome="success",
            details={},
            source_ip=None,
        )

        assert event.session_id is None
        assert event.source_ip is None


class TestAuditLogger:
    """Tests for AuditLogger class - SPEC-04.15."""

    @pytest.fixture
    def audit_dir(self, tmp_path: Path) -> Path:
        """Create temporary audit directory."""
        audit_path = tmp_path / "audit"
        audit_path.mkdir()
        return audit_path

    @pytest.fixture
    def audit_logger(self, audit_dir: Path):
        """Create AuditLogger with temporary directory."""
        from parhelia.audit import AuditLogger

        return AuditLogger(audit_root=str(audit_dir))

    @pytest.mark.asyncio
    async def test_log_writes_jsonl(self, audit_logger, audit_dir: Path):
        """@trace SPEC-04.15 - AuditLogger MUST write to append-only JSONL."""
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="session.create",
            session_id="sess-456",
            user="testuser",
            action="create",
            resource="session",
            outcome="success",
            details={"task_id": "task-789"},
            source_ip="10.0.0.1",
        )

        await audit_logger.log(event)

        log_file = audit_dir / "audit.jsonl"
        assert log_file.exists()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["type"] == "session.create"
        assert entry["session"] == "sess-456"
        assert entry["outcome"] == "success"

    @pytest.mark.asyncio
    async def test_log_appends_multiple_events(self, audit_logger, audit_dir: Path):
        """@trace SPEC-04.15 - AuditLogger MUST append events (not overwrite)."""
        from parhelia.audit import AuditEvent

        for i in range(3):
            event = AuditEvent(
                timestamp=datetime.now(),
                event_type="auth.login",
                session_id=f"sess-{i}",
                user="testuser",
                action="login",
                resource="session",
                outcome="success",
                details={},
                source_ip=None,
            )
            await audit_logger.log(event)

        log_file = audit_dir / "audit.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_log_redacts_secrets_in_details(self, audit_logger, audit_dir: Path):
        """@trace SPEC-04.15 - AuditLogger MUST redact secrets from details."""
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="secret.inject",
            session_id="sess-123",
            user="orchestrator",
            action="inject",
            resource="container",
            outcome="success",
            details={
                "secret_name": "anthropic-api-key",
                "api_key": "sk-ant-secret123",
                "token": "ghp_sensitive",
                "normal_field": "visible",
            },
            source_ip=None,
        )

        await audit_logger.log(event)

        log_file = audit_dir / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())

        assert entry["details"]["api_key"] == "[REDACTED]"
        assert entry["details"]["token"] == "[REDACTED]"
        assert entry["details"]["normal_field"] == "visible"

    @pytest.mark.asyncio
    async def test_log_redacts_password_fields(self, audit_logger, audit_dir: Path):
        """@trace SPEC-04.15 - AuditLogger MUST redact password fields."""
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="auth.login",
            session_id=None,
            user="testuser",
            action="login",
            resource="auth",
            outcome="success",
            details={
                "password": "hunter2",
                "db_password": "secret123",
                "password_hash": "should_redact",
            },
            source_ip="127.0.0.1",
        )

        await audit_logger.log(event)

        log_file = audit_dir / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())

        assert entry["details"]["password"] == "[REDACTED]"
        assert entry["details"]["db_password"] == "[REDACTED]"
        assert entry["details"]["password_hash"] == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_log_includes_timestamp_iso_format(self, audit_logger, audit_dir: Path):
        """@trace SPEC-04.15 - AuditLogger MUST include ISO timestamp."""
        from parhelia.audit import AuditEvent

        now = datetime(2026, 1, 17, 12, 0, 0)
        event = AuditEvent(
            timestamp=now,
            event_type="checkpoint.create",
            session_id="sess-123",
            user="system",
            action="create",
            resource="checkpoint",
            outcome="success",
            details={},
            source_ip=None,
        )

        await audit_logger.log(event)

        log_file = audit_dir / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())

        assert entry["timestamp"] == "2026-01-17T12:00:00"


class TestAuditedEvents:
    """Tests for audited event types - SPEC-04.15."""

    def test_audited_events_list(self):
        """@trace SPEC-04.15 - AuditLogger MUST define audited event types."""
        from parhelia.audit import AuditLogger

        expected_events = [
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
        ]

        for event in expected_events:
            assert event in AuditLogger.AUDITED_EVENTS


class TestAuditHelpers:
    """Tests for audit helper functions - SPEC-04.15."""

    @pytest.fixture
    def audit_logger(self, tmp_path: Path):
        """Create AuditLogger with temporary directory."""
        from parhelia.audit import AuditLogger

        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        return AuditLogger(audit_root=str(audit_dir))

    def test_is_auditable_event(self, audit_logger):
        """@trace SPEC-04.15 - AuditLogger MUST identify auditable events."""
        assert audit_logger.is_auditable("auth.login") is True
        assert audit_logger.is_auditable("session.create") is True
        assert audit_logger.is_auditable("random.event") is False

    @pytest.mark.asyncio
    async def test_log_denied_event(self, audit_logger, tmp_path: Path):
        """@trace SPEC-04.15 - Denied events SHOULD be tracked for metrics."""
        from parhelia.audit import AuditEvent

        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="permission.denied",
            session_id="sess-123",
            user="attacker",
            action="access",
            resource="/etc/shadow",
            outcome="denied",
            details={"reason": "path not allowed"},
            source_ip="1.2.3.4",
        )

        await audit_logger.log(event)

        # Verify event logged
        log_file = tmp_path / "audit" / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())
        assert entry["outcome"] == "denied"

    @pytest.mark.asyncio
    async def test_convenience_log_methods(self, audit_logger, tmp_path: Path):
        """@trace SPEC-04.15 - AuditLogger SHOULD provide convenience methods."""
        await audit_logger.log_session_create(
            session_id="sess-abc",
            user="testuser",
            task_id="task-123",
        )

        log_file = tmp_path / "audit" / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())

        assert entry["type"] == "session.create"
        assert entry["session"] == "sess-abc"
        assert entry["details"]["task_id"] == "task-123"

    @pytest.mark.asyncio
    async def test_log_secret_access(self, audit_logger, tmp_path: Path):
        """@trace SPEC-04.15 - Secret access MUST be logged."""
        await audit_logger.log_secret_access(
            session_id="sess-xyz",
            user="orchestrator",
            secret_name="anthropic-api-key",
            outcome="success",
        )

        log_file = tmp_path / "audit" / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())

        assert entry["type"] == "secret.access"
        assert entry["resource"] == "anthropic-api-key"

    @pytest.mark.asyncio
    async def test_log_checkpoint_create(self, audit_logger, tmp_path: Path):
        """@trace SPEC-04.15 - Checkpoint creation MUST be logged."""
        await audit_logger.log_checkpoint(
            session_id="sess-cp",
            checkpoint_id="cp-123",
            action="create",
            user="system",
        )

        log_file = tmp_path / "audit" / "audit.jsonl"
        entry = json.loads(log_file.read_text().strip())

        assert entry["type"] == "checkpoint.create"
        assert entry["details"]["checkpoint_id"] == "cp-123"
