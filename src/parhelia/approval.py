"""Approval workflow for human oversight of checkpoints.

Implements:
- [SPEC-07.20.01] Approval States
- [SPEC-07.20.02] Escalation Policies
- [SPEC-07.20.04] Approval Audit
- [SPEC-07.20.05] Resume Strategy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from parhelia.session import (
    ApprovalStatus,
    Checkpoint,
    CheckpointApproval,
    CheckpointTrigger,
)

if TYPE_CHECKING:
    from parhelia.audit import AuditLogger


# Policy names
PolicyName = Literal["auto", "review", "strict"]


@dataclass
class ApprovalPolicy:
    """Configuration for when to auto-approve vs require review.

    Implements [SPEC-07.20.02].
    """

    name: str
    auto_approve: list[str] = field(default_factory=list)  # Trigger types to auto-approve
    require_review: list[str] = field(default_factory=list)  # Trigger types requiring review


@dataclass
class EscalationConfig:
    """Thresholds that escalate to review regardless of policy.

    Implements [SPEC-07.20.02].
    """

    cost_threshold_usd: float = 5.0  # Session cost exceeds this
    duration_threshold_hours: float = 4.0  # Session runs longer than this
    error_count_threshold: int = 3  # Number of errors in session


@dataclass
class ApprovalConfig:
    """Full approval configuration.

    Implements [SPEC-07.20.02].
    """

    default_policy: PolicyName = "auto"
    policies: dict[str, ApprovalPolicy] = field(default_factory=dict)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)

    @classmethod
    def default(cls) -> "ApprovalConfig":
        """Create default approval configuration."""
        return cls(
            default_policy="auto",
            policies={
                "auto": ApprovalPolicy(
                    name="auto",
                    auto_approve=["periodic", "detach"],
                    require_review=["complete", "error"],
                ),
                "review": ApprovalPolicy(
                    name="review",
                    auto_approve=["periodic"],
                    require_review=["complete", "error", "detach"],
                ),
                "strict": ApprovalPolicy(
                    name="strict",
                    auto_approve=[],
                    require_review=["periodic", "detach", "complete", "error", "shutdown", "manual"],
                ),
            },
            escalation=EscalationConfig(),
        )


@dataclass
class SessionMetrics:
    """Session metrics for escalation evaluation."""

    cost_usd: float = 0.0
    duration_hours: float = 0.0
    error_count: int = 0


class ApprovalManager:
    """Manage checkpoint approval workflow.

    Implements [SPEC-07.20].
    """

    def __init__(
        self,
        config: ApprovalConfig | None = None,
        audit_logger: "AuditLogger | None" = None,
    ):
        """Initialize the approval manager.

        Args:
            config: Approval configuration. Uses defaults if None.
            audit_logger: Optional audit logger for logging decisions.
        """
        self.config = config or ApprovalConfig.default()
        self.audit_logger = audit_logger

    def evaluate_checkpoint(
        self,
        checkpoint: Checkpoint,
        session_metrics: SessionMetrics | None = None,
        policy_override: PolicyName | None = None,
    ) -> CheckpointApproval:
        """Evaluate a checkpoint and determine its approval status.

        Implements [SPEC-07.20.01], [SPEC-07.20.02].

        Args:
            checkpoint: The checkpoint to evaluate.
            session_metrics: Optional session metrics for escalation.
            policy_override: Override the default policy for this checkpoint.

        Returns:
            CheckpointApproval with the determined status.
        """
        policy_name = policy_override or self.config.default_policy
        policy = self.config.policies.get(policy_name)

        if policy is None:
            # Fall back to auto policy
            policy = self.config.policies.get("auto", ApprovalPolicy(name="auto"))

        trigger_value = checkpoint.trigger.value

        # Check escalation triggers first
        if session_metrics and self._should_escalate(session_metrics):
            return CheckpointApproval(
                status=ApprovalStatus.PENDING,
                policy=policy_name,
                reason="Escalated due to session thresholds",
            )

        # Check if trigger requires review
        if trigger_value in policy.require_review:
            return CheckpointApproval(
                status=ApprovalStatus.PENDING,
                policy=policy_name,
            )

        # Check if trigger can be auto-approved
        if trigger_value in policy.auto_approve:
            return CheckpointApproval(
                status=ApprovalStatus.AUTO_APPROVED,
                policy=policy_name,
                timestamp=datetime.now(),
            )

        # Default to pending for unknown triggers
        return CheckpointApproval(
            status=ApprovalStatus.PENDING,
            policy=policy_name,
        )

    def _should_escalate(self, metrics: SessionMetrics) -> bool:
        """Check if session metrics trigger escalation.

        Implements [SPEC-07.20.02].
        """
        escalation = self.config.escalation

        if metrics.cost_usd >= escalation.cost_threshold_usd:
            return True
        if metrics.duration_hours >= escalation.duration_threshold_hours:
            return True
        if metrics.error_count >= escalation.error_count_threshold:
            return True

        return False

    async def approve(
        self,
        checkpoint: Checkpoint,
        user: str,
        reason: str | None = None,
    ) -> CheckpointApproval:
        """Approve a checkpoint.

        Implements [SPEC-07.20.01], [SPEC-07.20.04].

        Args:
            checkpoint: The checkpoint to approve.
            user: Username of the approver.
            reason: Optional reason for approval.

        Returns:
            Updated CheckpointApproval.
        """
        approval = CheckpointApproval(
            status=ApprovalStatus.APPROVED,
            user=user,
            timestamp=datetime.now(),
            reason=reason,
            policy=checkpoint.approval.policy if checkpoint.approval else None,
        )

        # Log to audit
        if self.audit_logger:
            from parhelia.audit import AuditEvent

            await self.audit_logger.log(
                AuditEvent(
                    timestamp=datetime.now(),
                    event_type="checkpoint.approve",
                    session_id=checkpoint.session_id,
                    user=user,
                    action="approve",
                    resource=f"checkpoint:{checkpoint.id}",
                    outcome="success",
                    details={
                        "checkpoint_id": checkpoint.id,
                        "reason": reason,
                        "policy": approval.policy,
                    },
                    source_ip=None,
                )
            )

        return approval

    async def reject(
        self,
        checkpoint: Checkpoint,
        user: str,
        reason: str,
    ) -> CheckpointApproval:
        """Reject a checkpoint.

        Implements [SPEC-07.20.01], [SPEC-07.20.04].

        Args:
            checkpoint: The checkpoint to reject.
            user: Username of the rejector.
            reason: Reason for rejection (required).

        Returns:
            Updated CheckpointApproval.
        """
        approval = CheckpointApproval(
            status=ApprovalStatus.REJECTED,
            user=user,
            timestamp=datetime.now(),
            reason=reason,
            policy=checkpoint.approval.policy if checkpoint.approval else None,
        )

        # Log to audit
        if self.audit_logger:
            from parhelia.audit import AuditEvent

            await self.audit_logger.log(
                AuditEvent(
                    timestamp=datetime.now(),
                    event_type="checkpoint.reject",
                    session_id=checkpoint.session_id,
                    user=user,
                    action="reject",
                    resource=f"checkpoint:{checkpoint.id}",
                    outcome="success",
                    details={
                        "checkpoint_id": checkpoint.id,
                        "reason": reason,
                        "policy": approval.policy,
                    },
                    source_ip=None,
                )
            )

        return approval

    def select_resume_checkpoint(
        self,
        checkpoints: list[Checkpoint],
        allow_pending: bool = True,
        allow_rejected: bool = False,
    ) -> Checkpoint | None:
        """Select the best checkpoint for resuming a session.

        Implements [SPEC-07.20.05].

        Priority order:
        1. Latest approved checkpoint
        2. Latest auto_approved checkpoint
        3. Latest pending checkpoint (if allow_pending)
        4. Never rejected (unless allow_rejected)

        Args:
            checkpoints: List of checkpoints to choose from.
            allow_pending: Whether to allow resuming from pending checkpoints.
            allow_rejected: Whether to allow resuming from rejected checkpoints.

        Returns:
            Best checkpoint for resume, or None if none suitable.
        """
        if not checkpoints:
            return None

        # Sort by created_at descending (most recent first)
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda c: c.created_at,
            reverse=True,
        )

        # Group by approval status
        approved: list[Checkpoint] = []
        auto_approved: list[Checkpoint] = []
        pending: list[Checkpoint] = []
        rejected: list[Checkpoint] = []

        for cp in sorted_checkpoints:
            if cp.approval is None:
                # No approval = pending
                pending.append(cp)
            elif cp.approval.status == ApprovalStatus.APPROVED:
                approved.append(cp)
            elif cp.approval.status == ApprovalStatus.AUTO_APPROVED:
                auto_approved.append(cp)
            elif cp.approval.status == ApprovalStatus.PENDING:
                pending.append(cp)
            elif cp.approval.status == ApprovalStatus.REJECTED:
                rejected.append(cp)

        # Return in priority order
        if approved:
            return approved[0]
        if auto_approved:
            return auto_approved[0]
        if allow_pending and pending:
            return pending[0]
        if allow_rejected and rejected:
            return rejected[0]

        return None

    def get_pending_checkpoints(
        self,
        checkpoints: list[Checkpoint],
    ) -> list[Checkpoint]:
        """Get all checkpoints awaiting review.

        Args:
            checkpoints: List of checkpoints to filter.

        Returns:
            List of checkpoints with pending approval.
        """
        return [
            cp
            for cp in checkpoints
            if cp.approval is None or cp.approval.status == ApprovalStatus.PENDING
        ]


def parse_approval_config(data: dict) -> ApprovalConfig:
    """Parse approval configuration from TOML data.

    Args:
        data: The 'approval' section from parhelia.toml.

    Returns:
        ApprovalConfig parsed from data.
    """
    default_policy = data.get("default_policy", "auto")

    # Parse policies
    policies: dict[str, ApprovalPolicy] = {}
    policies_data = data.get("policies", {})

    for policy_name, policy_data in policies_data.items():
        policies[policy_name] = ApprovalPolicy(
            name=policy_name,
            auto_approve=policy_data.get("auto_approve", []),
            require_review=policy_data.get("require_review", []),
        )

    # Add default policies if not defined
    defaults = ApprovalConfig.default()
    for name, policy in defaults.policies.items():
        if name not in policies:
            policies[name] = policy

    # Parse escalation
    escalation_data = data.get("escalation", {})
    escalation = EscalationConfig(
        cost_threshold_usd=escalation_data.get("cost_threshold_usd", 5.0),
        duration_threshold_hours=escalation_data.get("duration_threshold_hours", 4.0),
        error_count_threshold=escalation_data.get("error_count_threshold", 3),
    )

    return ApprovalConfig(
        default_policy=default_policy,
        policies=policies,
        escalation=escalation,
    )
