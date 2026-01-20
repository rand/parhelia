"""Tests for approval workflow.

@trace SPEC-07.20.01 - Approval States
@trace SPEC-07.20.02 - Escalation Policies
@trace SPEC-07.20.04 - Approval Audit
@trace SPEC-07.20.05 - Resume Strategy
"""

from datetime import datetime

import pytest

from parhelia.approval import (
    ApprovalConfig,
    ApprovalManager,
    ApprovalPolicy,
    EscalationConfig,
    SessionMetrics,
    parse_approval_config,
)
from parhelia.session import (
    ApprovalStatus,
    Checkpoint,
    CheckpointApproval,
    CheckpointTrigger,
)


class TestApprovalConfig:
    """Tests for ApprovalConfig."""

    def test_default_config(self):
        """@trace SPEC-07.20.02 - Default config MUST have auto/review/strict policies."""
        config = ApprovalConfig.default()

        assert config.default_policy == "auto"
        assert "auto" in config.policies
        assert "review" in config.policies
        assert "strict" in config.policies

    def test_auto_policy(self):
        """@trace SPEC-07.20.02 - Auto policy MUST auto-approve periodic/detach."""
        config = ApprovalConfig.default()
        auto = config.policies["auto"]

        assert "periodic" in auto.auto_approve
        assert "detach" in auto.auto_approve
        assert "complete" in auto.require_review
        assert "error" in auto.require_review

    def test_strict_policy(self):
        """@trace SPEC-07.20.02 - Strict policy MUST require review for everything."""
        config = ApprovalConfig.default()
        strict = config.policies["strict"]

        assert strict.auto_approve == []
        assert len(strict.require_review) >= 4  # All trigger types

    def test_escalation_defaults(self):
        """@trace SPEC-07.20.02 - Escalation MUST have default thresholds."""
        config = ApprovalConfig.default()

        assert config.escalation.cost_threshold_usd == 5.0
        assert config.escalation.duration_threshold_hours == 4.0
        assert config.escalation.error_count_threshold == 3


class TestApprovalManager:
    """Tests for ApprovalManager."""

    @pytest.fixture
    def manager(self) -> ApprovalManager:
        """Create ApprovalManager with default config."""
        return ApprovalManager()

    @pytest.fixture
    def checkpoint_periodic(self) -> Checkpoint:
        """Create checkpoint with periodic trigger."""
        return Checkpoint(
            id="cp-periodic-123",
            session_id="session-test",
            trigger=CheckpointTrigger.PERIODIC,
            working_directory="/tmp/workspace",
        )

    @pytest.fixture
    def checkpoint_complete(self) -> Checkpoint:
        """Create checkpoint with complete trigger."""
        return Checkpoint(
            id="cp-complete-123",
            session_id="session-test",
            trigger=CheckpointTrigger.COMPLETE,
            working_directory="/tmp/workspace",
        )

    @pytest.fixture
    def checkpoint_error(self) -> Checkpoint:
        """Create checkpoint with error trigger."""
        return Checkpoint(
            id="cp-error-123",
            session_id="session-test",
            trigger=CheckpointTrigger.ERROR,
            working_directory="/tmp/workspace",
        )

    def test_evaluate_auto_approve_periodic(
        self, manager: ApprovalManager, checkpoint_periodic: Checkpoint
    ):
        """@trace SPEC-07.20.02 - Periodic checkpoints MUST be auto-approved in auto policy."""
        approval = manager.evaluate_checkpoint(checkpoint_periodic)

        assert approval.status == ApprovalStatus.AUTO_APPROVED
        assert approval.policy == "auto"

    def test_evaluate_require_review_complete(
        self, manager: ApprovalManager, checkpoint_complete: Checkpoint
    ):
        """@trace SPEC-07.20.02 - Complete checkpoints MUST require review in auto policy."""
        approval = manager.evaluate_checkpoint(checkpoint_complete)

        assert approval.status == ApprovalStatus.PENDING
        assert approval.policy == "auto"

    def test_evaluate_require_review_error(
        self, manager: ApprovalManager, checkpoint_error: Checkpoint
    ):
        """@trace SPEC-07.20.02 - Error checkpoints MUST require review."""
        approval = manager.evaluate_checkpoint(checkpoint_error)

        assert approval.status == ApprovalStatus.PENDING

    def test_evaluate_with_policy_override(
        self, manager: ApprovalManager, checkpoint_periodic: Checkpoint
    ):
        """@trace SPEC-07.20.02 - Policy override MUST be respected."""
        approval = manager.evaluate_checkpoint(
            checkpoint_periodic, policy_override="strict"
        )

        # Strict policy requires review for everything
        assert approval.status == ApprovalStatus.PENDING
        assert approval.policy == "strict"

    def test_evaluate_escalation_cost(
        self, manager: ApprovalManager, checkpoint_periodic: Checkpoint
    ):
        """@trace SPEC-07.20.02 - High cost MUST escalate to review."""
        metrics = SessionMetrics(cost_usd=10.0)  # Above threshold

        approval = manager.evaluate_checkpoint(checkpoint_periodic, session_metrics=metrics)

        assert approval.status == ApprovalStatus.PENDING
        assert "threshold" in (approval.reason or "").lower()

    def test_evaluate_escalation_duration(
        self, manager: ApprovalManager, checkpoint_periodic: Checkpoint
    ):
        """@trace SPEC-07.20.02 - Long duration MUST escalate to review."""
        metrics = SessionMetrics(duration_hours=5.0)  # Above threshold

        approval = manager.evaluate_checkpoint(checkpoint_periodic, session_metrics=metrics)

        assert approval.status == ApprovalStatus.PENDING

    def test_evaluate_escalation_errors(
        self, manager: ApprovalManager, checkpoint_periodic: Checkpoint
    ):
        """@trace SPEC-07.20.02 - Many errors MUST escalate to review."""
        metrics = SessionMetrics(error_count=5)  # Above threshold

        approval = manager.evaluate_checkpoint(checkpoint_periodic, session_metrics=metrics)

        assert approval.status == ApprovalStatus.PENDING


class TestApprovalManagerActions:
    """Tests for approve/reject actions."""

    @pytest.fixture
    def manager(self) -> ApprovalManager:
        """Create ApprovalManager."""
        return ApprovalManager()

    @pytest.fixture
    def pending_checkpoint(self) -> Checkpoint:
        """Create checkpoint with pending approval."""
        return Checkpoint(
            id="cp-pending-123",
            session_id="session-test",
            trigger=CheckpointTrigger.COMPLETE,
            working_directory="/tmp/workspace",
            approval=CheckpointApproval(
                status=ApprovalStatus.PENDING,
                policy="auto",
            ),
        )

    @pytest.mark.asyncio
    async def test_approve_checkpoint(
        self, manager: ApprovalManager, pending_checkpoint: Checkpoint
    ):
        """@trace SPEC-07.20.01 - approve MUST set status to APPROVED."""
        approval = await manager.approve(
            pending_checkpoint,
            user="test-reviewer",
            reason="Looks good",
        )

        assert approval.status == ApprovalStatus.APPROVED
        assert approval.user == "test-reviewer"
        assert approval.reason == "Looks good"
        assert approval.timestamp is not None

    @pytest.mark.asyncio
    async def test_reject_checkpoint(
        self, manager: ApprovalManager, pending_checkpoint: Checkpoint
    ):
        """@trace SPEC-07.20.01 - reject MUST set status to REJECTED."""
        approval = await manager.reject(
            pending_checkpoint,
            user="test-reviewer",
            reason="Needs more tests",
        )

        assert approval.status == ApprovalStatus.REJECTED
        assert approval.user == "test-reviewer"
        assert approval.reason == "Needs more tests"
        assert approval.timestamp is not None


class TestResumeStrategy:
    """Tests for checkpoint resume strategy."""

    @pytest.fixture
    def manager(self) -> ApprovalManager:
        """Create ApprovalManager."""
        return ApprovalManager()

    @pytest.fixture
    def checkpoints(self) -> list[Checkpoint]:
        """Create list of checkpoints with different approval states."""
        base_time = datetime(2026, 1, 20, 10, 0, 0)

        return [
            # Oldest - approved
            Checkpoint(
                id="cp-1",
                session_id="session-test",
                trigger=CheckpointTrigger.PERIODIC,
                working_directory="/tmp/workspace",
                created_at=datetime(2026, 1, 20, 10, 0, 0),
                approval=CheckpointApproval(
                    status=ApprovalStatus.APPROVED,
                    user="reviewer",
                    timestamp=datetime(2026, 1, 20, 10, 5, 0),
                ),
            ),
            # Auto-approved
            Checkpoint(
                id="cp-2",
                session_id="session-test",
                trigger=CheckpointTrigger.PERIODIC,
                working_directory="/tmp/workspace",
                created_at=datetime(2026, 1, 20, 11, 0, 0),
                approval=CheckpointApproval(
                    status=ApprovalStatus.AUTO_APPROVED,
                    policy="auto",
                ),
            ),
            # Pending
            Checkpoint(
                id="cp-3",
                session_id="session-test",
                trigger=CheckpointTrigger.COMPLETE,
                working_directory="/tmp/workspace",
                created_at=datetime(2026, 1, 20, 12, 0, 0),
                approval=CheckpointApproval(status=ApprovalStatus.PENDING),
            ),
            # Rejected
            Checkpoint(
                id="cp-4",
                session_id="session-test",
                trigger=CheckpointTrigger.COMPLETE,
                working_directory="/tmp/workspace",
                created_at=datetime(2026, 1, 20, 13, 0, 0),
                approval=CheckpointApproval(
                    status=ApprovalStatus.REJECTED,
                    user="reviewer",
                    reason="Bad changes",
                ),
            ),
            # Latest approved
            Checkpoint(
                id="cp-5",
                session_id="session-test",
                trigger=CheckpointTrigger.MANUAL,
                working_directory="/tmp/workspace",
                created_at=datetime(2026, 1, 20, 14, 0, 0),
                approval=CheckpointApproval(
                    status=ApprovalStatus.APPROVED,
                    user="reviewer",
                ),
            ),
        ]

    def test_resume_prefers_approved(
        self, manager: ApprovalManager, checkpoints: list[Checkpoint]
    ):
        """@trace SPEC-07.20.05 - Resume MUST prefer approved checkpoints."""
        result = manager.select_resume_checkpoint(checkpoints)

        assert result is not None
        assert result.id == "cp-5"  # Latest approved
        assert result.approval.status == ApprovalStatus.APPROVED

    def test_resume_falls_back_to_auto_approved(
        self, manager: ApprovalManager, checkpoints: list[Checkpoint]
    ):
        """@trace SPEC-07.20.05 - Resume MUST fall back to auto_approved."""
        # Remove approved checkpoints
        filtered = [cp for cp in checkpoints if cp.approval.status != ApprovalStatus.APPROVED]

        result = manager.select_resume_checkpoint(filtered)

        assert result is not None
        assert result.id == "cp-2"  # Auto-approved
        assert result.approval.status == ApprovalStatus.AUTO_APPROVED

    def test_resume_allows_pending_by_default(
        self, manager: ApprovalManager, checkpoints: list[Checkpoint]
    ):
        """@trace SPEC-07.20.05 - Resume SHOULD allow pending with warning."""
        # Remove approved and auto_approved
        filtered = [
            cp
            for cp in checkpoints
            if cp.approval.status not in (ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED)
        ]

        result = manager.select_resume_checkpoint(filtered, allow_pending=True)

        assert result is not None
        assert result.id == "cp-3"  # Pending
        assert result.approval.status == ApprovalStatus.PENDING

    def test_resume_rejects_rejected_by_default(
        self, manager: ApprovalManager, checkpoints: list[Checkpoint]
    ):
        """@trace SPEC-07.20.05 - Resume MUST NOT use rejected by default."""
        # Only keep rejected
        filtered = [cp for cp in checkpoints if cp.approval.status == ApprovalStatus.REJECTED]

        result = manager.select_resume_checkpoint(filtered, allow_pending=False)

        assert result is None

    def test_resume_allows_rejected_with_override(
        self, manager: ApprovalManager, checkpoints: list[Checkpoint]
    ):
        """@trace SPEC-07.20.05 - Resume MAY use rejected with explicit override."""
        # Only keep rejected
        filtered = [cp for cp in checkpoints if cp.approval.status == ApprovalStatus.REJECTED]

        result = manager.select_resume_checkpoint(
            filtered, allow_pending=False, allow_rejected=True
        )

        assert result is not None
        assert result.id == "cp-4"
        assert result.approval.status == ApprovalStatus.REJECTED

    def test_resume_empty_list(self, manager: ApprovalManager):
        """@trace SPEC-07.20.05 - Resume with empty list MUST return None."""
        result = manager.select_resume_checkpoint([])
        assert result is None


class TestParseApprovalConfig:
    """Tests for parsing approval config from TOML."""

    def test_parse_empty_config(self):
        """Parse empty config should use defaults."""
        config = parse_approval_config({})

        assert config.default_policy == "auto"
        assert "auto" in config.policies
        assert "review" in config.policies
        assert "strict" in config.policies

    def test_parse_custom_policy(self):
        """Parse custom policy configuration."""
        data = {
            "default_policy": "review",
            "policies": {
                "custom": {
                    "auto_approve": ["periodic"],
                    "require_review": ["complete", "error", "manual"],
                }
            },
            "escalation": {
                "cost_threshold_usd": 10.0,
                "duration_threshold_hours": 8.0,
                "error_count_threshold": 5,
            },
        }

        config = parse_approval_config(data)

        assert config.default_policy == "review"
        assert "custom" in config.policies
        assert config.policies["custom"].auto_approve == ["periodic"]
        assert config.escalation.cost_threshold_usd == 10.0
        assert config.escalation.duration_threshold_hours == 8.0
        assert config.escalation.error_count_threshold == 5


class TestGetPendingCheckpoints:
    """Tests for getting pending checkpoints."""

    @pytest.fixture
    def manager(self) -> ApprovalManager:
        """Create ApprovalManager."""
        return ApprovalManager()

    def test_get_pending_includes_no_approval(self, manager: ApprovalManager):
        """Checkpoints with no approval should be considered pending."""
        checkpoints = [
            Checkpoint(
                id="cp-1",
                session_id="session-test",
                trigger=CheckpointTrigger.PERIODIC,
                working_directory="/tmp/workspace",
                approval=None,  # No approval
            ),
            Checkpoint(
                id="cp-2",
                session_id="session-test",
                trigger=CheckpointTrigger.PERIODIC,
                working_directory="/tmp/workspace",
                approval=CheckpointApproval(status=ApprovalStatus.APPROVED),
            ),
        ]

        pending = manager.get_pending_checkpoints(checkpoints)

        assert len(pending) == 1
        assert pending[0].id == "cp-1"

    def test_get_pending_includes_pending_status(self, manager: ApprovalManager):
        """Checkpoints with PENDING status should be included."""
        checkpoints = [
            Checkpoint(
                id="cp-1",
                session_id="session-test",
                trigger=CheckpointTrigger.PERIODIC,
                working_directory="/tmp/workspace",
                approval=CheckpointApproval(status=ApprovalStatus.PENDING),
            ),
            Checkpoint(
                id="cp-2",
                session_id="session-test",
                trigger=CheckpointTrigger.PERIODIC,
                working_directory="/tmp/workspace",
                approval=CheckpointApproval(status=ApprovalStatus.AUTO_APPROVED),
            ),
        ]

        pending = manager.get_pending_checkpoints(checkpoints)

        assert len(pending) == 1
        assert pending[0].id == "cp-1"
