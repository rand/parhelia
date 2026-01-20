"""Tests for recovery workflows.

@trace SPEC-07.42.01 - Resume from Failure
@trace SPEC-07.42.02 - Resume after Rejection
@trace SPEC-07.42.03 - Manual Recovery
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from parhelia.checkpoint import Checkpoint, CheckpointTrigger
from parhelia.recovery import (
    RecoveryAction,
    RecoveryConfig,
    RecoveryManager,
    RecoveryOption,
    RecoveryPlan,
    RecoveryPolicy,
    RecoveryResult,
    RecoveryScenario,
    parse_recovery_config,
)
from parhelia.session import ApprovalStatus, CheckpointApproval


class TestRecoveryOption:
    """Tests for RecoveryOption dataclass."""

    def test_creation(self):
        """RecoveryOption MUST capture action and checkpoint."""
        option = RecoveryOption(
            action=RecoveryAction.RESUME,
            checkpoint_id="cp-abc123",
            description="Resume from checkpoint",
            recommended=True,
        )

        assert option.action == RecoveryAction.RESUME
        assert option.checkpoint_id == "cp-abc123"
        assert option.recommended is True

    def test_to_dict(self):
        """RecoveryOption MUST serialize to dict."""
        option = RecoveryOption(
            action=RecoveryAction.ABANDON,
            checkpoint_id=None,
            description="Abandon session",
        )

        data = option.to_dict()
        assert data["action"] == "abandon"
        assert data["checkpoint_id"] is None


class TestRecoveryPlan:
    """Tests for RecoveryPlan dataclass."""

    def test_creation(self):
        """RecoveryPlan MUST capture scenario and options."""
        plan = RecoveryPlan(
            scenario=RecoveryScenario.FAILURE,
            session_id="test-session",
            options=[
                RecoveryOption(RecoveryAction.RESUME, "cp-1", "Resume"),
            ],
        )

        assert plan.scenario == RecoveryScenario.FAILURE
        assert plan.session_id == "test-session"
        assert len(plan.options) == 1

    def test_to_dict(self):
        """RecoveryPlan MUST serialize to dict."""
        plan = RecoveryPlan(
            scenario=RecoveryScenario.REJECTION,
            session_id="test",
            checkpoint_age_minutes=10.5,
        )

        data = plan.to_dict()
        assert data["scenario"] == "rejection"
        assert data["checkpoint_age_minutes"] == 10.5


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_success(self):
        """RecoveryResult MUST capture success state."""
        result = RecoveryResult(
            success=True,
            action_taken=RecoveryAction.RESUME,
            checkpoint_id="cp-abc123",
        )

        assert result.success is True
        assert result.checkpoint_id == "cp-abc123"

    def test_failure(self):
        """RecoveryResult MUST capture failure details."""
        result = RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RESUME,
            error_message="No checkpoint found",
        )

        assert result.success is False
        assert "No checkpoint" in result.error_message

    def test_to_dict(self):
        """RecoveryResult MUST serialize to dict."""
        result = RecoveryResult(
            success=True,
            action_taken=RecoveryAction.NEW_SESSION,
            new_session_id="new-session-123",
        )

        data = result.to_dict()
        assert data["action_taken"] == "new_session"
        assert data["new_session_id"] == "new-session-123"


class TestRecoveryConfig:
    """Tests for RecoveryConfig dataclass."""

    def test_defaults(self):
        """RecoveryConfig MUST have sensible defaults."""
        config = RecoveryConfig()

        assert config.policy == RecoveryPolicy.NOTIFY_AND_RESUME
        assert config.stale_checkpoint_threshold_minutes == 5
        assert config.auto_resume_max_age_minutes == 30

    def test_custom_values(self):
        """RecoveryConfig MUST accept custom values."""
        config = RecoveryConfig(
            policy=RecoveryPolicy.WAIT_FOR_USER,
            stale_checkpoint_threshold_minutes=10,
        )

        assert config.policy == RecoveryPolicy.WAIT_FOR_USER
        assert config.stale_checkpoint_threshold_minutes == 10

    def test_serialization(self):
        """RecoveryConfig MUST serialize to/from dict."""
        config = RecoveryConfig(
            policy=RecoveryPolicy.AUTO_RESUME,
            auto_resume_max_age_minutes=60,
        )

        data = config.to_dict()
        restored = RecoveryConfig.from_dict(data)

        assert restored.policy == config.policy
        assert restored.auto_resume_max_age_minutes == 60


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Create mock checkpoint manager."""
        manager = MagicMock()
        manager.list_checkpoints = AsyncMock(return_value=[])
        return manager

    @pytest.fixture
    def recovery_manager(self, mock_checkpoint_manager) -> RecoveryManager:
        """Create RecoveryManager instance."""
        return RecoveryManager(
            checkpoint_manager=mock_checkpoint_manager,
            config=RecoveryConfig(),
        )

    def _create_checkpoint(
        self,
        cp_id: str,
        created_at: datetime,
        approval_status: ApprovalStatus | None = None,
    ) -> Checkpoint:
        """Helper to create test checkpoint."""
        cp = Checkpoint(
            id=cp_id,
            session_id="test-session",
            created_at=created_at,
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/workspace",
        )
        if approval_status:
            cp.approval = CheckpointApproval(
                status=approval_status,
                user="test-user",
                timestamp=created_at,
            )
        return cp

    @pytest.mark.asyncio
    async def test_find_approved_checkpoints(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Find MUST return approved checkpoints."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(hours=2), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-2", now - timedelta(hours=1), ApprovalStatus.PENDING),
            self._create_checkpoint("cp-3", now, ApprovalStatus.AUTO_APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        approved = await recovery_manager.find_approved_checkpoints("test-session")

        assert len(approved) == 2
        # Newest first
        assert approved[0].id == "cp-3"
        assert approved[1].id == "cp-1"

    @pytest.mark.asyncio
    async def test_find_latest_approved(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Find MUST return latest approved."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(hours=1), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-2", now, ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        latest = await recovery_manager.find_latest_approved("test-session")

        assert latest is not None
        assert latest.id == "cp-2"

    @pytest.mark.asyncio
    async def test_find_latest_approved_none(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Find MUST return None if no approved."""
        checkpoints = [
            self._create_checkpoint("cp-1", datetime.now(), ApprovalStatus.PENDING),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        latest = await recovery_manager.find_latest_approved("test-session")

        assert latest is None

    @pytest.mark.asyncio
    async def test_find_previous_approved(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.02 - Find MUST return previous approved."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(hours=2), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-2", now - timedelta(hours=1), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-3", now, ApprovalStatus.REJECTED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        previous = await recovery_manager.find_previous_approved("test-session", "cp-3")

        assert previous is not None
        assert previous.id == "cp-2"

    @pytest.mark.asyncio
    async def test_plan_failure_recovery(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Plan MUST include recovery options."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(minutes=2), ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_failure_recovery("test-session")

        assert plan.scenario == RecoveryScenario.FAILURE
        assert plan.recommended_checkpoint_id == "cp-1"
        assert len(plan.options) >= 2

        # Should have resume option
        resume_options = [o for o in plan.options if o.action == RecoveryAction.RESUME]
        assert len(resume_options) == 1
        assert resume_options[0].recommended is True

    @pytest.mark.asyncio
    async def test_plan_failure_recovery_no_checkpoints(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Plan MUST handle no approved checkpoints."""
        mock_checkpoint_manager.list_checkpoints.return_value = []

        plan = await recovery_manager.plan_failure_recovery("test-session")

        assert len(plan.options) == 1
        assert plan.options[0].action == RecoveryAction.ABANDON

    @pytest.mark.asyncio
    async def test_plan_failure_recovery_stale_notification(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Plan MUST flag stale checkpoints."""
        # Checkpoint older than threshold (5 min default)
        old_time = datetime.now() - timedelta(minutes=10)
        checkpoints = [
            self._create_checkpoint("cp-1", old_time, ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_failure_recovery("test-session")

        assert plan.requires_notification is True
        assert plan.checkpoint_age_minutes is not None
        assert plan.checkpoint_age_minutes > 5

    @pytest.mark.asyncio
    async def test_execute_failure_recovery_resume(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Execute MUST resume from checkpoint."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now, ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_failure_recovery("test-session")
        result = await recovery_manager.execute_failure_recovery(
            plan, RecoveryAction.RESUME
        )

        assert result.success is True
        assert result.action_taken == RecoveryAction.RESUME
        assert result.checkpoint_id == "cp-1"

    @pytest.mark.asyncio
    async def test_execute_failure_recovery_policy_wait(
        self, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.01 - Execute MUST respect wait policy."""
        config = RecoveryConfig(policy=RecoveryPolicy.WAIT_FOR_USER)
        manager = RecoveryManager(mock_checkpoint_manager, config)

        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now, ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await manager.plan_failure_recovery("test-session")
        result = await manager.execute_failure_recovery(plan)

        assert result.action_taken == RecoveryAction.WAIT

    @pytest.mark.asyncio
    async def test_plan_rejection_recovery(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.02 - Plan MUST find previous approved."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(hours=1), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-2", now, ApprovalStatus.REJECTED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_rejection_recovery("test-session", "cp-2")

        assert plan.scenario == RecoveryScenario.REJECTION
        assert plan.recommended_checkpoint_id == "cp-1"
        assert plan.current_checkpoint_id == "cp-2"

        # Should have options
        actions = [o.action for o in plan.options]
        assert RecoveryAction.RESUME in actions
        assert RecoveryAction.ABANDON in actions

    @pytest.mark.asyncio
    async def test_plan_rejection_recovery_no_previous(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.02 - Plan MUST handle no previous checkpoint."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now, ApprovalStatus.REJECTED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_rejection_recovery("test-session", "cp-1")

        # Only abandon option
        assert len(plan.options) == 1
        assert plan.options[0].action == RecoveryAction.ABANDON
        assert plan.options[0].recommended is True

    @pytest.mark.asyncio
    async def test_plan_manual_recovery_with_checkpoint(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.03 - Plan MUST use specified checkpoint."""
        plan = await recovery_manager.plan_manual_recovery(
            "test-session", from_checkpoint_id="cp-specified"
        )

        assert plan.scenario == RecoveryScenario.MANUAL
        assert plan.recommended_checkpoint_id == "cp-specified"

        resume_options = [o for o in plan.options if o.action == RecoveryAction.RESUME]
        assert len(resume_options) == 1
        assert resume_options[0].checkpoint_id == "cp-specified"

    @pytest.mark.asyncio
    async def test_plan_manual_recovery_list_all(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.03 - Plan MUST list available checkpoints."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(hours=2), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-2", now - timedelta(hours=1), ApprovalStatus.APPROVED),
            self._create_checkpoint("cp-3", now, ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_manual_recovery("test-session")

        # Should show multiple resume options
        resume_options = [o for o in plan.options if o.action == RecoveryAction.RESUME]
        assert len(resume_options) >= 3

    @pytest.mark.asyncio
    async def test_get_detailed_checkpoint_list(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.03 - List MUST show detailed info."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now - timedelta(minutes=30), ApprovalStatus.APPROVED),
        ]
        checkpoints[0].tokens_used = 50000
        checkpoints[0].cost_estimate = 0.50
        checkpoints[0].uncommitted_changes = ["file1.py", "file2.py"]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        details = await recovery_manager.get_detailed_checkpoint_list("test-session")

        assert len(details) == 1
        detail = details[0]
        assert detail["id"] == "cp-1"
        assert detail["approval_status"] == "approved"
        assert detail["tokens_used"] == 50000
        assert detail["files_changed"] == 2

    @pytest.mark.asyncio
    async def test_execute_new_session(
        self, recovery_manager, mock_checkpoint_manager
    ):
        """@trace SPEC-07.42.02 - Execute MUST create new session ID."""
        now = datetime.now()
        checkpoints = [
            self._create_checkpoint("cp-1", now, ApprovalStatus.APPROVED),
        ]
        mock_checkpoint_manager.list_checkpoints.return_value = checkpoints

        plan = await recovery_manager.plan_manual_recovery(
            "test-session", from_checkpoint_id="cp-1"
        )
        result = await recovery_manager._execute_action(plan, RecoveryAction.NEW_SESSION)

        assert result.success is True
        assert result.action_taken == RecoveryAction.NEW_SESSION
        assert result.new_session_id is not None
        assert "test-session" in result.new_session_id
        assert "recovered" in result.new_session_id

    def test_format_recovery_plan(self, recovery_manager):
        """@trace SPEC-07.42.03 - Format MUST show options."""
        plan = RecoveryPlan(
            scenario=RecoveryScenario.FAILURE,
            session_id="test-session",
            checkpoint_age_minutes=15.5,
            requires_notification=True,
            notification_reason="Checkpoint is stale",
            options=[
                RecoveryOption(RecoveryAction.RESUME, "cp-1", "Resume from cp-1", True),
                RecoveryOption(RecoveryAction.ABANDON, None, "Abandon", False),
            ],
        )

        output = recovery_manager.format_recovery_plan(plan)

        assert "test-session" in output
        assert "failure" in output
        assert "15.5" in output
        assert "Resume from cp-1" in output
        assert "[recommended]" in output

    def test_format_recovery_result_success(self, recovery_manager):
        """@trace SPEC-07.42.03 - Format MUST show success."""
        result = RecoveryResult(
            success=True,
            action_taken=RecoveryAction.RESUME,
            checkpoint_id="cp-abc123",
        )

        output = recovery_manager.format_recovery_result(result)

        assert "successful" in output.lower()
        assert "resume" in output
        assert "cp-abc123" in output

    def test_format_recovery_result_failure(self, recovery_manager):
        """@trace SPEC-07.42.03 - Format MUST show failure."""
        result = RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RESUME,
            error_message="Checkpoint corrupted",
        )

        output = recovery_manager.format_recovery_result(result)

        assert "failed" in output.lower()
        assert "Checkpoint corrupted" in output

    def test_notify_callback(self, recovery_manager):
        """@trace SPEC-07.42.01 - Manager MUST use notify callback."""
        notifications: list[tuple[str, str]] = []

        def callback(title: str, message: str) -> None:
            notifications.append((title, message))

        recovery_manager.set_notify_callback(callback)
        recovery_manager._notify("Test Title", "Test Message")

        assert len(notifications) == 1
        assert notifications[0] == ("Test Title", "Test Message")


class TestParseRecoveryConfig:
    """Tests for parse_recovery_config function."""

    def test_parse_empty(self):
        """Parse MUST handle empty config."""
        config = parse_recovery_config({})

        assert config.policy == RecoveryPolicy.NOTIFY_AND_RESUME

    def test_parse_full(self):
        """Parse MUST handle full config."""
        config = parse_recovery_config({
            "recovery": {
                "policy": "wait_for_user",
                "stale_checkpoint_threshold_minutes": 10,
                "auto_resume_max_age_minutes": 60,
            }
        })

        assert config.policy == RecoveryPolicy.WAIT_FOR_USER
        assert config.stale_checkpoint_threshold_minutes == 10
        assert config.auto_resume_max_age_minutes == 60
