"""Tests for error handling and retry logic."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestRetryPolicy:
    """Tests for RetryPolicy with exponential backoff."""

    def test_retry_policy_creation(self):
        """RetryPolicy MUST be configurable with max_attempts and base_delay."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(max_attempts=5, base_delay=1.0)
        assert policy.max_attempts == 5
        assert policy.base_delay == 1.0

    def test_retry_policy_defaults(self):
        """RetryPolicy SHOULD have sensible defaults."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0

    def test_calculate_delay_exponential_backoff(self):
        """RetryPolicy MUST calculate delay with exponential backoff."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0)

        # attempt 1: 1.0 * 2^0 = 1.0
        assert policy.calculate_delay(1) == 1.0
        # attempt 2: 1.0 * 2^1 = 2.0
        assert policy.calculate_delay(2) == 2.0
        # attempt 3: 1.0 * 2^2 = 4.0
        assert policy.calculate_delay(3) == 4.0

    def test_calculate_delay_respects_max_delay(self):
        """RetryPolicy MUST cap delay at max_delay."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(base_delay=1.0, max_delay=5.0, exponential_base=2.0)

        # attempt 4: 1.0 * 2^3 = 8.0, capped to 5.0
        assert policy.calculate_delay(4) == 5.0

    def test_calculate_delay_with_jitter(self):
        """RetryPolicy SHOULD support jitter to prevent thundering herd."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(base_delay=1.0, jitter=0.1)

        delays = [policy.calculate_delay(1) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1  # Not all same
        # But within jitter range
        assert all(0.9 <= d <= 1.1 for d in delays)


class TestRetryPolicyExecution:
    """Tests for RetryPolicy execution."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_attempt(self):
        """RetryPolicy MUST return result on first success."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(max_attempts=3)
        func = AsyncMock(return_value="success")

        result = await policy.execute(func)

        assert result == "success"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        """RetryPolicy MUST retry on failure and return on success."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        func = AsyncMock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])

        result = await policy.execute(func)

        assert result == "success"
        assert func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_raises_after_max_attempts(self):
        """RetryPolicy MUST raise after max_attempts exhausted."""
        from parhelia.resilience import RetryPolicy, RetriesExhaustedError

        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        func = AsyncMock(side_effect=Exception("always fails"))

        with pytest.raises(RetriesExhaustedError) as exc_info:
            await policy.execute(func)

        assert exc_info.value.attempts == 3
        assert func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_only_on_specified_exceptions(self):
        """RetryPolicy SHOULD only retry on specified exception types."""
        from parhelia.resilience import RetryPolicy

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            retry_on=(ValueError,),
        )

        # ValueError should retry
        func_value_error = AsyncMock(
            side_effect=[ValueError("retry me"), "success"]
        )
        result = await policy.execute(func_value_error)
        assert result == "success"

        # TypeError should not retry
        func_type_error = AsyncMock(side_effect=TypeError("don't retry"))
        with pytest.raises(TypeError):
            await policy.execute(func_type_error)


class TestCircuitBreaker:
    """Tests for CircuitBreaker pattern."""

    def test_circuit_breaker_creation(self):
        """CircuitBreaker MUST be configurable."""
        from parhelia.resilience import CircuitBreaker

        cb = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=3,
        )

        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.half_open_max_calls == 3

    def test_circuit_breaker_defaults(self):
        """CircuitBreaker SHOULD have sensible defaults."""
        from parhelia.resilience import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.failure_threshold == 5
        assert cb.recovery_timeout == 30.0
        assert cb.half_open_max_calls == 1

    def test_circuit_breaker_starts_closed(self):
        """CircuitBreaker MUST start in CLOSED state."""
        from parhelia.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_state_enum(self):
        """CircuitState enum MUST have CLOSED, OPEN, HALF_OPEN."""
        from parhelia.resilience import CircuitState

        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerExecution:
    """Tests for CircuitBreaker execution."""

    @pytest.mark.asyncio
    async def test_circuit_stays_closed_on_success(self):
        """CircuitBreaker MUST stay CLOSED on successful calls."""
        from parhelia.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)
        func = AsyncMock(return_value="success")

        for _ in range(5):
            await cb.execute(func)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """CircuitBreaker MUST open after failure_threshold failures."""
        from parhelia.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)
        func = AsyncMock(side_effect=Exception("fail"))

        for _ in range(3):
            try:
                await cb.execute(func)
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self):
        """CircuitBreaker MUST reject calls when OPEN."""
        from parhelia.resilience import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker(failure_threshold=1)
        func = AsyncMock(side_effect=Exception("fail"))

        # Trigger open
        try:
            await cb.execute(func)
        except Exception:
            pass

        # Should reject without calling func
        func.reset_mock()
        with pytest.raises(CircuitOpenError):
            await cb.execute(func)

        assert func.call_count == 0

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self):
        """CircuitBreaker MUST transition to HALF_OPEN after recovery_timeout."""
        from parhelia.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        func = AsyncMock(side_effect=Exception("fail"))

        # Trigger open
        try:
            await cb.execute(func)
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Check state - it transitions on next call attempt
        func.reset_mock()
        func.return_value = "success"
        func.side_effect = None

        result = await cb.execute(func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_closes_on_half_open_success(self):
        """CircuitBreaker MUST close on successful call in HALF_OPEN."""
        from parhelia.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        fail_func = AsyncMock(side_effect=Exception("fail"))

        # Trigger open
        try:
            await cb.execute(fail_func)
        except Exception:
            pass

        await asyncio.sleep(0.02)

        # Success in half-open should close
        success_func = AsyncMock(return_value="success")
        await cb.execute(success_func)

        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_half_open_failure(self):
        """CircuitBreaker MUST reopen on failure in HALF_OPEN."""
        from parhelia.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        func = AsyncMock(side_effect=Exception("fail"))

        # Trigger open
        try:
            await cb.execute(func)
        except Exception:
            pass

        await asyncio.sleep(0.02)

        # Failure in half-open should reopen
        try:
            await cb.execute(func)
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerMetrics:
    """Tests for CircuitBreaker metrics."""

    def test_circuit_tracks_failure_count(self):
        """CircuitBreaker MUST track consecutive failure count."""
        from parhelia.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)
        cb._record_failure()
        cb._record_failure()

        assert cb.failure_count == 2

    def test_circuit_resets_failure_count_on_success(self):
        """CircuitBreaker MUST reset failure count on success."""
        from parhelia.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)
        cb._record_failure()
        cb._record_failure()
        cb._record_success()

        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_tracks_call_statistics(self):
        """CircuitBreaker SHOULD track call statistics."""
        from parhelia.resilience import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=5)

        success_func = AsyncMock(return_value="ok")
        fail_func = AsyncMock(side_effect=Exception("fail"))

        await cb.execute(success_func)
        await cb.execute(success_func)
        try:
            await cb.execute(fail_func)
        except Exception:
            pass

        stats = cb.get_stats()
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1


class TestRetryWithCircuitBreaker:
    """Tests for combining RetryPolicy with CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """RetryPolicy SHOULD work with CircuitBreaker."""
        from parhelia.resilience import CircuitBreaker, RetryPolicy

        cb = CircuitBreaker(failure_threshold=5)
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)

        func = AsyncMock(side_effect=[Exception("fail"), "success"])

        result = await policy.execute(func, circuit_breaker=cb)

        assert result == "success"
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_stops_when_circuit_opens(self):
        """RetryPolicy MUST stop retrying when CircuitBreaker opens."""
        from parhelia.resilience import (
            CircuitBreaker,
            CircuitOpenError,
            RetryPolicy,
        )

        cb = CircuitBreaker(failure_threshold=2)
        policy = RetryPolicy(max_attempts=5, base_delay=0.01)

        func = AsyncMock(side_effect=Exception("fail"))

        with pytest.raises(CircuitOpenError):
            await policy.execute(func, circuit_breaker=cb)

        # Should have stopped at circuit open, not max_attempts
        assert func.call_count == 2
