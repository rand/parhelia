"""Recovery workflows for common scenarios.

Implements:
- [SPEC-07.42.01] Resume from Failure
- [SPEC-07.42.02] Resume after Rejection
- [SPEC-07.42.03] Manual Recovery
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from parhelia.checkpoint import Checkpoint, CheckpointManager
from parhelia.session import ApprovalStatus


class RecoveryScenario(Enum):
    """Type of recovery scenario."""

    FAILURE = "failure"  # Container/session failed
    REJECTION = "rejection"  # User rejected checkpoint
    MANUAL = "manual"  # User-initiated recovery


class RecoveryAction(Enum):
    """Action to take during recovery."""

    RESUME = "resume"  # Resume from checkpoint
    NEW_SESSION = "new_session"  # Start fresh session from checkpoint
    ABANDON = "abandon"  # Abandon the session
    WAIT = "wait"  # Wait for user input


@dataclass
class RecoveryOption:
    """A recovery option to present to user."""

    action: RecoveryAction
    checkpoint_id: str | None
    description: str
    recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action": self.action.value,
            "checkpoint_id": self.checkpoint_id,
            "description": self.description,
            "recommended": self.recommended,
        }


@dataclass
class RecoveryPlan:
    """Plan for recovering a session."""

    scenario: RecoveryScenario
    session_id: str
    options: list[RecoveryOption] = field(default_factory=list)
    current_checkpoint_id: str | None = None
    recommended_checkpoint_id: str | None = None
    checkpoint_age_minutes: float | None = None
    requires_notification: bool = False
    notification_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scenario": self.scenario.value,
            "session_id": self.session_id,
            "options": [o.to_dict() for o in self.options],
            "current_checkpoint_id": self.current_checkpoint_id,
            "recommended_checkpoint_id": self.recommended_checkpoint_id,
            "checkpoint_age_minutes": self.checkpoint_age_minutes,
            "requires_notification": self.requires_notification,
            "notification_reason": self.notification_reason,
        }


@dataclass
class RecoveryResult:
    """Result of executing a recovery plan."""

    success: bool
    action_taken: RecoveryAction
    checkpoint_id: str | None = None
    new_session_id: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "action_taken": self.action_taken.value,
            "checkpoint_id": self.checkpoint_id,
            "new_session_id": self.new_session_id,
            "error_message": self.error_message,
        }


class RecoveryPolicy(Enum):
    """Policy for automatic recovery behavior."""

    AUTO_RESUME = "auto_resume"  # Automatically resume from latest approved
    WAIT_FOR_USER = "wait_for_user"  # Always wait for user decision
    NOTIFY_AND_RESUME = "notify_and_resume"  # Notify user, then resume


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    policy: RecoveryPolicy = RecoveryPolicy.NOTIFY_AND_RESUME
    stale_checkpoint_threshold_minutes: int = 5  # Notify if checkpoint older than this
    auto_resume_max_age_minutes: int = 30  # Don't auto-resume if older than this

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "policy": self.policy.value,
            "stale_checkpoint_threshold_minutes": self.stale_checkpoint_threshold_minutes,
            "auto_resume_max_age_minutes": self.auto_resume_max_age_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryConfig:
        """Deserialize from dictionary."""
        policy = RecoveryPolicy.NOTIFY_AND_RESUME
        if "policy" in data:
            policy = RecoveryPolicy(data["policy"])

        return cls(
            policy=policy,
            stale_checkpoint_threshold_minutes=data.get(
                "stale_checkpoint_threshold_minutes", 5
            ),
            auto_resume_max_age_minutes=data.get("auto_resume_max_age_minutes", 30),
        )


class RecoveryManager:
    """Manage session recovery workflows.

    Implements [SPEC-07.42].
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        config: RecoveryConfig | None = None,
    ):
        """Initialize the recovery manager.

        Args:
            checkpoint_manager: Manager for checkpoint operations.
            config: Recovery configuration. Uses defaults if not provided.
        """
        self.checkpoint_manager = checkpoint_manager
        self.config = config or RecoveryConfig()
        self._notify_callback: Callable[[str, str], None] | None = None

    def set_notify_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Set callback for notifications.

        Args:
            callback: Function called with (title, message) for notifications.
        """
        self._notify_callback = callback

    def _notify(self, title: str, message: str) -> None:
        """Send notification if callback set."""
        if self._notify_callback:
            self._notify_callback(title, message)

    # =========================================================================
    # Checkpoint Analysis
    # =========================================================================

    async def find_approved_checkpoints(
        self,
        session_id: str,
    ) -> list[Checkpoint]:
        """Find all approved/auto-approved checkpoints for session.

        Args:
            session_id: Session to search.

        Returns:
            List of approved checkpoints, newest first.
        """
        all_checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)

        approved = []
        for cp in all_checkpoints:
            if cp.approval and cp.approval.status in (
                ApprovalStatus.APPROVED,
                ApprovalStatus.AUTO_APPROVED,
            ):
                approved.append(cp)

        # Sort by creation time, newest first
        return sorted(approved, key=lambda cp: cp.created_at, reverse=True)

    async def find_latest_approved(
        self,
        session_id: str,
    ) -> Checkpoint | None:
        """Find the latest approved checkpoint for session.

        Args:
            session_id: Session to search.

        Returns:
            Latest approved checkpoint, or None.
        """
        approved = await self.find_approved_checkpoints(session_id)
        return approved[0] if approved else None

    async def find_previous_approved(
        self,
        session_id: str,
        before_checkpoint_id: str,
    ) -> Checkpoint | None:
        """Find approved checkpoint before a given checkpoint.

        Args:
            session_id: Session to search.
            before_checkpoint_id: Find checkpoint before this one.

        Returns:
            Previous approved checkpoint, or None.
        """
        approved = await self.find_approved_checkpoints(session_id)

        # Find the target checkpoint's creation time
        all_checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)
        target_time = None
        for cp in all_checkpoints:
            if cp.id == before_checkpoint_id:
                target_time = cp.created_at
                break

        if not target_time:
            return None

        # Find latest approved before target
        for cp in approved:
            if cp.created_at < target_time:
                return cp

        return None

    def calculate_checkpoint_age(self, checkpoint: Checkpoint) -> timedelta:
        """Calculate age of checkpoint from now.

        Args:
            checkpoint: The checkpoint.

        Returns:
            Age as timedelta.
        """
        return datetime.now() - checkpoint.created_at

    # =========================================================================
    # Recovery from Failure [SPEC-07.42.01]
    # =========================================================================

    async def plan_failure_recovery(
        self,
        session_id: str,
    ) -> RecoveryPlan:
        """Create recovery plan for session failure.

        Implements [SPEC-07.42.01].

        Args:
            session_id: Failed session ID.

        Returns:
            RecoveryPlan with options.
        """
        plan = RecoveryPlan(
            scenario=RecoveryScenario.FAILURE,
            session_id=session_id,
        )

        # Find latest approved checkpoint
        latest_approved = await self.find_latest_approved(session_id)

        if not latest_approved:
            # No approved checkpoint - can only abandon
            plan.options.append(
                RecoveryOption(
                    action=RecoveryAction.ABANDON,
                    checkpoint_id=None,
                    description="No approved checkpoints found. Abandon session.",
                    recommended=True,
                )
            )
            return plan

        # Calculate age
        age = self.calculate_checkpoint_age(latest_approved)
        age_minutes = age.total_seconds() / 60
        plan.checkpoint_age_minutes = age_minutes
        plan.recommended_checkpoint_id = latest_approved.id

        # Check if notification needed
        if age_minutes > self.config.stale_checkpoint_threshold_minutes:
            plan.requires_notification = True
            plan.notification_reason = (
                f"Latest checkpoint is {age_minutes:.1f} minutes old"
            )

        # Build options
        plan.options.append(
            RecoveryOption(
                action=RecoveryAction.RESUME,
                checkpoint_id=latest_approved.id,
                description=f"Resume from {latest_approved.id} ({age_minutes:.1f} min old)",
                recommended=True,
            )
        )

        plan.options.append(
            RecoveryOption(
                action=RecoveryAction.NEW_SESSION,
                checkpoint_id=latest_approved.id,
                description=f"Start new session from {latest_approved.id}",
                recommended=False,
            )
        )

        plan.options.append(
            RecoveryOption(
                action=RecoveryAction.ABANDON,
                checkpoint_id=None,
                description="Abandon session",
                recommended=False,
            )
        )

        return plan

    async def execute_failure_recovery(
        self,
        plan: RecoveryPlan,
        action: RecoveryAction | None = None,
    ) -> RecoveryResult:
        """Execute failure recovery based on policy or user choice.

        Implements [SPEC-07.42.01].

        Args:
            plan: The recovery plan.
            action: Specific action to take, or None for policy-based decision.

        Returns:
            RecoveryResult with outcome.
        """
        # Determine action
        if action is None:
            action = self._determine_action_from_policy(plan)

        # Notify if needed
        if plan.requires_notification:
            self._notify(
                "Session Recovery",
                f"Recovering session {plan.session_id}. {plan.notification_reason}",
            )

        return await self._execute_action(plan, action)

    def _determine_action_from_policy(self, plan: RecoveryPlan) -> RecoveryAction:
        """Determine action based on policy and plan.

        Args:
            plan: The recovery plan.

        Returns:
            Action to take.
        """
        if self.config.policy == RecoveryPolicy.WAIT_FOR_USER:
            return RecoveryAction.WAIT

        # Check if checkpoint is too old for auto-resume
        if plan.checkpoint_age_minutes is not None:
            if plan.checkpoint_age_minutes > self.config.auto_resume_max_age_minutes:
                return RecoveryAction.WAIT

        return RecoveryAction.RESUME

    # =========================================================================
    # Recovery from Rejection [SPEC-07.42.02]
    # =========================================================================

    async def plan_rejection_recovery(
        self,
        session_id: str,
        rejected_checkpoint_id: str,
    ) -> RecoveryPlan:
        """Create recovery plan after checkpoint rejection.

        Implements [SPEC-07.42.02].

        Args:
            session_id: Session ID.
            rejected_checkpoint_id: ID of rejected checkpoint.

        Returns:
            RecoveryPlan with options.
        """
        plan = RecoveryPlan(
            scenario=RecoveryScenario.REJECTION,
            session_id=session_id,
            current_checkpoint_id=rejected_checkpoint_id,
        )

        # Find previous approved checkpoint
        previous_approved = await self.find_previous_approved(
            session_id, rejected_checkpoint_id
        )

        if previous_approved:
            plan.recommended_checkpoint_id = previous_approved.id
            age = self.calculate_checkpoint_age(previous_approved)
            age_minutes = age.total_seconds() / 60
            plan.checkpoint_age_minutes = age_minutes

            # Option 1: Resume from previous
            plan.options.append(
                RecoveryOption(
                    action=RecoveryAction.RESUME,
                    checkpoint_id=previous_approved.id,
                    description=f"Resume from previous approved: {previous_approved.id}",
                    recommended=True,
                )
            )

            # Option 2: New session from previous
            plan.options.append(
                RecoveryOption(
                    action=RecoveryAction.NEW_SESSION,
                    checkpoint_id=previous_approved.id,
                    description=f"Start new session from {previous_approved.id}",
                    recommended=False,
                )
            )

        # Option 3: Abandon
        plan.options.append(
            RecoveryOption(
                action=RecoveryAction.ABANDON,
                checkpoint_id=None,
                description="Abandon session",
                recommended=not bool(previous_approved),
            )
        )

        return plan

    # =========================================================================
    # Manual Recovery [SPEC-07.42.03]
    # =========================================================================

    async def plan_manual_recovery(
        self,
        session_id: str,
        from_checkpoint_id: str | None = None,
    ) -> RecoveryPlan:
        """Create recovery plan for manual recovery.

        Implements [SPEC-07.42.03].

        Args:
            session_id: Session to recover.
            from_checkpoint_id: Specific checkpoint to use, or None for choice.

        Returns:
            RecoveryPlan with options.
        """
        plan = RecoveryPlan(
            scenario=RecoveryScenario.MANUAL,
            session_id=session_id,
        )

        if from_checkpoint_id:
            # User specified checkpoint
            plan.recommended_checkpoint_id = from_checkpoint_id

            plan.options.append(
                RecoveryOption(
                    action=RecoveryAction.RESUME,
                    checkpoint_id=from_checkpoint_id,
                    description=f"Resume from specified checkpoint: {from_checkpoint_id}",
                    recommended=True,
                )
            )

            plan.options.append(
                RecoveryOption(
                    action=RecoveryAction.NEW_SESSION,
                    checkpoint_id=from_checkpoint_id,
                    description=f"Start new session from {from_checkpoint_id}",
                    recommended=False,
                )
            )

        else:
            # Show all available checkpoints
            all_checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)
            approved = await self.find_approved_checkpoints(session_id)

            # Add resume options for each approved checkpoint
            for i, cp in enumerate(approved[:5]):  # Top 5
                age = self.calculate_checkpoint_age(cp)
                age_minutes = age.total_seconds() / 60

                plan.options.append(
                    RecoveryOption(
                        action=RecoveryAction.RESUME,
                        checkpoint_id=cp.id,
                        description=f"Resume from {cp.id} ({age_minutes:.1f} min old, {cp.approval.status.value if cp.approval else 'unknown'})",
                        recommended=(i == 0),
                    )
                )

            if approved:
                plan.recommended_checkpoint_id = approved[0].id

        plan.options.append(
            RecoveryOption(
                action=RecoveryAction.ABANDON,
                checkpoint_id=None,
                description="Abandon session",
                recommended=False,
            )
        )

        return plan

    async def get_detailed_checkpoint_list(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Get detailed list of checkpoints for display.

        Implements `parhelia checkpoint list --detailed`.

        Args:
            session_id: Session to list.

        Returns:
            List of checkpoint details.
        """
        checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)
        details = []

        for cp in checkpoints:
            age = self.calculate_checkpoint_age(cp)
            status = "none"
            if cp.approval:
                status = cp.approval.status.value

            details.append({
                "id": cp.id,
                "created_at": cp.created_at.isoformat(),
                "trigger": cp.trigger.value,
                "approval_status": status,
                "age_minutes": age.total_seconds() / 60,
                "tokens_used": cp.tokens_used,
                "cost_estimate": cp.cost_estimate,
                "files_changed": len(cp.uncommitted_changes or []),
            })

        return details

    # =========================================================================
    # Action Execution
    # =========================================================================

    async def _execute_action(
        self,
        plan: RecoveryPlan,
        action: RecoveryAction,
    ) -> RecoveryResult:
        """Execute a recovery action.

        Args:
            plan: The recovery plan.
            action: Action to execute.

        Returns:
            RecoveryResult with outcome.
        """
        if action == RecoveryAction.ABANDON:
            return RecoveryResult(
                success=True,
                action_taken=action,
                checkpoint_id=None,
            )

        if action == RecoveryAction.WAIT:
            return RecoveryResult(
                success=True,
                action_taken=action,
                checkpoint_id=None,
            )

        # Find the checkpoint to use
        checkpoint_id = plan.recommended_checkpoint_id
        for option in plan.options:
            if option.action == action and option.checkpoint_id:
                checkpoint_id = option.checkpoint_id
                break

        if not checkpoint_id:
            return RecoveryResult(
                success=False,
                action_taken=action,
                error_message="No checkpoint available for this action",
            )

        if action == RecoveryAction.RESUME:
            return RecoveryResult(
                success=True,
                action_taken=action,
                checkpoint_id=checkpoint_id,
            )

        if action == RecoveryAction.NEW_SESSION:
            # Generate new session ID
            new_session_id = f"{plan.session_id}-recovered-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            return RecoveryResult(
                success=True,
                action_taken=action,
                checkpoint_id=checkpoint_id,
                new_session_id=new_session_id,
            )

        return RecoveryResult(
            success=False,
            action_taken=action,
            error_message=f"Unknown action: {action}",
        )

    # =========================================================================
    # Formatting
    # =========================================================================

    def format_recovery_plan(self, plan: RecoveryPlan) -> str:
        """Format recovery plan for CLI display.

        Args:
            plan: The recovery plan.

        Returns:
            Formatted string.
        """
        lines: list[str] = []

        lines.append(f"Recovery Plan for Session: {plan.session_id}")
        lines.append(f"Scenario: {plan.scenario.value}")
        lines.append("")

        if plan.checkpoint_age_minutes is not None:
            lines.append(f"Latest checkpoint age: {plan.checkpoint_age_minutes:.1f} minutes")

        if plan.requires_notification:
            lines.append(f"Note: {plan.notification_reason}")

        lines.append("")
        lines.append("Options:")

        for i, option in enumerate(plan.options):
            marker = "[recommended]" if option.recommended else ""
            lines.append(f"  {i + 1}. {option.description} {marker}")

        return "\n".join(lines)

    def format_recovery_result(self, result: RecoveryResult) -> str:
        """Format recovery result for CLI display.

        Args:
            result: The recovery result.

        Returns:
            Formatted string.
        """
        lines: list[str] = []

        if result.success:
            lines.append("Recovery successful")
            lines.append(f"Action: {result.action_taken.value}")
            if result.checkpoint_id:
                lines.append(f"Checkpoint: {result.checkpoint_id}")
            if result.new_session_id:
                lines.append(f"New session: {result.new_session_id}")
        else:
            lines.append("Recovery failed")
            if result.error_message:
                lines.append(f"Error: {result.error_message}")

        return "\n".join(lines)


def parse_recovery_config(data: dict[str, Any]) -> RecoveryConfig:
    """Parse recovery configuration from TOML data.

    Args:
        data: Configuration dictionary.

    Returns:
        RecoveryConfig instance.
    """
    return RecoveryConfig.from_dict(data.get("recovery", {}))
