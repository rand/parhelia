"""Tests for trust boundary enforcement.

@trace SPEC-04.12 - Trust Boundary Enforcement
"""

import pytest


class TestTrustBoundary:
    """Tests for TrustBoundary enum - SPEC-04.12."""

    def test_trust_boundary_values(self):
        """@trace SPEC-04.12 - TrustBoundary MUST define all zones."""
        from parhelia.trust_boundary import TrustBoundary

        assert TrustBoundary.LOCAL is not None
        assert TrustBoundary.REMOTE_CONTAINER is not None
        assert TrustBoundary.EXTERNAL is not None

    def test_trust_boundary_ordering(self):
        """@trace SPEC-04.12 - TrustBoundary SHOULD have trust ordering."""
        from parhelia.trust_boundary import TrustBoundary

        # LOCAL is most trusted
        assert TrustBoundary.LOCAL.value < TrustBoundary.REMOTE_CONTAINER.value
        assert TrustBoundary.REMOTE_CONTAINER.value < TrustBoundary.EXTERNAL.value


class TestBoundaryContext:
    """Tests for BoundaryContext dataclass - SPEC-04.12."""

    def test_context_creation(self):
        """@trace SPEC-04.12 - BoundaryContext MUST capture execution context."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.LOCAL,
            target=TrustBoundary.REMOTE_CONTAINER,
            session_id="session-123",
        )

        assert ctx.source == TrustBoundary.LOCAL
        assert ctx.target == TrustBoundary.REMOTE_CONTAINER
        assert ctx.session_id == "session-123"

    def test_context_is_cross_boundary(self):
        """@trace SPEC-04.12 - BoundaryContext MUST detect cross-boundary calls."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        cross = BoundaryContext(
            source=TrustBoundary.LOCAL,
            target=TrustBoundary.REMOTE_CONTAINER,
        )
        same = BoundaryContext(
            source=TrustBoundary.LOCAL,
            target=TrustBoundary.LOCAL,
        )

        assert cross.is_cross_boundary() is True
        assert same.is_cross_boundary() is False

    def test_context_is_trusted_direction(self):
        """@trace SPEC-04.12 - BoundaryContext MUST validate trust direction."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        # Local to remote is trusted (orchestrator controls container)
        local_to_remote = BoundaryContext(
            source=TrustBoundary.LOCAL,
            target=TrustBoundary.REMOTE_CONTAINER,
        )

        # Remote to local is less trusted (container calling back)
        remote_to_local = BoundaryContext(
            source=TrustBoundary.REMOTE_CONTAINER,
            target=TrustBoundary.LOCAL,
        )

        assert local_to_remote.is_trusted_direction() is True
        assert remote_to_local.is_trusted_direction() is False


class TestBoundaryValidator:
    """Tests for BoundaryValidator - SPEC-04.12."""

    @pytest.fixture
    def validator(self):
        """Create BoundaryValidator instance."""
        from parhelia.trust_boundary import BoundaryValidator

        return BoundaryValidator()

    def test_validator_allows_same_boundary(self, validator):
        """@trace SPEC-04.12 - Validator MUST allow same-boundary calls."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.LOCAL,
            target=TrustBoundary.LOCAL,
        )

        result = validator.validate(ctx, action="read_file")
        assert result.allowed is True

    def test_validator_allows_local_to_remote(self, validator):
        """@trace SPEC-04.12 - Validator MUST allow local to remote calls."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.LOCAL,
            target=TrustBoundary.REMOTE_CONTAINER,
        )

        result = validator.validate(ctx, action="dispatch_task")
        assert result.allowed is True

    def test_validator_restricts_remote_to_local(self, validator):
        """@trace SPEC-04.12 - Validator MUST restrict remote to local calls."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.REMOTE_CONTAINER,
            target=TrustBoundary.LOCAL,
        )

        # Arbitrary actions from remote should be denied
        result = validator.validate(ctx, action="execute_command")
        assert result.allowed is False

    def test_validator_allows_whitelisted_callbacks(self, validator):
        """@trace SPEC-04.12 - Validator SHOULD allow whitelisted callbacks."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.REMOTE_CONTAINER,
            target=TrustBoundary.LOCAL,
        )

        # Status updates are allowed callbacks
        result = validator.validate(ctx, action="report_status")
        assert result.allowed is True

    def test_validator_denies_external_to_local(self, validator):
        """@trace SPEC-04.12 - Validator MUST deny external to local."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.EXTERNAL,
            target=TrustBoundary.LOCAL,
        )

        result = validator.validate(ctx, action="any_action")
        assert result.allowed is False


class TestValidationResult:
    """Tests for ValidationResult - SPEC-04.12."""

    def test_result_success(self):
        """@trace SPEC-04.12 - ValidationResult MUST indicate success."""
        from parhelia.trust_boundary import ValidationResult

        result = ValidationResult(allowed=True)

        assert result.allowed is True
        assert result.reason is None

    def test_result_failure_with_reason(self):
        """@trace SPEC-04.12 - ValidationResult MUST include denial reason."""
        from parhelia.trust_boundary import ValidationResult

        result = ValidationResult(
            allowed=False,
            reason="Cross-boundary call not permitted",
        )

        assert result.allowed is False
        assert "not permitted" in result.reason


class TestSecurityViolation:
    """Tests for security violation handling - SPEC-04.12."""

    @pytest.fixture
    def validator(self):
        """Create BoundaryValidator instance."""
        from parhelia.trust_boundary import BoundaryValidator

        return BoundaryValidator()

    def test_violation_logged(self, validator):
        """@trace SPEC-04.12 - Violations SHOULD be logged."""
        from parhelia.trust_boundary import BoundaryContext, TrustBoundary

        ctx = BoundaryContext(
            source=TrustBoundary.EXTERNAL,
            target=TrustBoundary.LOCAL,
            session_id="session-456",
        )

        result = validator.validate(ctx, action="malicious_action")

        assert result.allowed is False
        # Violation should be recorded
        assert len(validator.violations) > 0
        assert validator.violations[-1].session_id == "session-456"
