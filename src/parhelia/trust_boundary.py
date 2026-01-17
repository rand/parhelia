"""Trust boundary enforcement for local-remote execution.

Implements:
- [SPEC-04.12] Trust Boundary Enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


class TrustBoundary(IntEnum):
    """Trust zones in the Parhelia architecture.

    Implements [SPEC-04.12].

    Lower values indicate higher trust:
    - LOCAL: Full trust, user's machine
    - REMOTE_CONTAINER: Sandboxed trust, ephemeral Modal container
    - EXTERNAL: Untrusted, external network
    """

    LOCAL = 0
    REMOTE_CONTAINER = 1
    EXTERNAL = 2


@dataclass
class BoundaryContext:
    """Context for a cross-boundary operation.

    Implements [SPEC-04.12].
    """

    source: TrustBoundary
    target: TrustBoundary
    session_id: str | None = None
    action: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def is_cross_boundary(self) -> bool:
        """Check if this is a cross-boundary call."""
        return self.source != self.target

    def is_trusted_direction(self) -> bool:
        """Check if call direction maintains trust.

        Trusted direction is from higher trust to lower trust,
        i.e., LOCAL -> REMOTE_CONTAINER is trusted (orchestrator controls container),
        but REMOTE_CONTAINER -> LOCAL is untrusted (callback from sandbox).
        """
        return self.source.value <= self.target.value


@dataclass
class ValidationResult:
    """Result of boundary validation.

    Implements [SPEC-04.12].
    """

    allowed: bool
    reason: str | None = None


@dataclass
class SecurityViolation:
    """Record of a security violation.

    Implements [SPEC-04.12].
    """

    context: BoundaryContext
    action: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def session_id(self) -> str | None:
        """Get session ID from context."""
        return self.context.session_id


class BoundaryValidator:
    """Validate cross-boundary operations.

    Implements [SPEC-04.12].

    The validator enforces trust boundaries:
    - LOCAL can call REMOTE_CONTAINER (dispatch tasks)
    - REMOTE_CONTAINER can only call LOCAL for whitelisted callbacks
    - EXTERNAL cannot call LOCAL or REMOTE_CONTAINER
    """

    # Actions allowed from remote container back to local
    ALLOWED_CALLBACKS = {
        "report_status",
        "report_progress",
        "request_checkpoint",
        "report_completion",
        "report_error",
        "request_secret",  # Via secure channel only
    }

    def __init__(self):
        """Initialize the validator."""
        self.violations: list[SecurityViolation] = []

    def validate(self, context: BoundaryContext, action: str) -> ValidationResult:
        """Validate a cross-boundary operation.

        Args:
            context: The boundary context.
            action: The action being performed.

        Returns:
            ValidationResult indicating if operation is allowed.
        """
        # Same boundary is always allowed
        if not context.is_cross_boundary():
            return ValidationResult(allowed=True)

        # Check trusted direction
        if context.is_trusted_direction():
            # LOCAL -> REMOTE_CONTAINER: always allowed
            # LOCAL -> EXTERNAL: allowed with caveats
            # REMOTE_CONTAINER -> EXTERNAL: allowed for egress
            return ValidationResult(allowed=True)

        # Untrusted direction - need specific whitelist
        if context.source == TrustBoundary.REMOTE_CONTAINER:
            if context.target == TrustBoundary.LOCAL:
                # Check if action is in allowed callbacks
                if action in self.ALLOWED_CALLBACKS:
                    return ValidationResult(allowed=True)
                else:
                    violation = SecurityViolation(
                        context=context,
                        action=action,
                        reason=f"Action '{action}' not allowed from remote to local",
                    )
                    self.violations.append(violation)
                    return ValidationResult(
                        allowed=False,
                        reason=f"Action '{action}' not in allowed callbacks",
                    )

        # EXTERNAL -> anything is denied
        if context.source == TrustBoundary.EXTERNAL:
            violation = SecurityViolation(
                context=context,
                action=action,
                reason="External source not permitted",
            )
            self.violations.append(violation)
            return ValidationResult(
                allowed=False,
                reason="Cross-boundary call from external source not permitted",
            )

        # Default deny
        violation = SecurityViolation(
            context=context,
            action=action,
            reason="Unrecognized boundary crossing",
        )
        self.violations.append(violation)
        return ValidationResult(
            allowed=False,
            reason="Cross-boundary call not permitted",
        )
