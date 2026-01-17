"""Error handling and retry logic.

Implements robust error handling with:
- RetryPolicy: Exponential backoff for transient failures
- CircuitBreaker: Protection against cascading failures
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class RetriesExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, attempts: int, last_exception: Exception | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


# =============================================================================
# CircuitBreaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Calls rejected, waiting for recovery timeout
    - HALF_OPEN: Testing if service recovered
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    # Statistics
    _total_calls: int = field(default=0, init=False)
    _successful_calls: int = field(default=0, init=False)
    _failed_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    async def execute(
        self,
        func: Callable[[], Coroutine[Any, Any, T]],
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute.

        Returns:
            Function result.

        Raises:
            CircuitOpenError: If circuit is open.
            Exception: If function raises and circuit doesn't absorb it.
        """
        state = self.state  # Triggers timeout check

        if state == CircuitState.OPEN:
            raise CircuitOpenError()

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitOpenError("Half-open call limit reached")
            self._half_open_calls += 1

        self._total_calls += 1

        try:
            result = await func()
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def _record_success(self) -> None:
        """Record a successful call."""
        self._successful_calls += 1
        self._failure_count = 0

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            logger.info("Circuit breaker closed after successful recovery")

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._failed_calls += 1
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened after half-open failure")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0


# =============================================================================
# RetryPolicy
# =============================================================================


@dataclass
class RetryPolicy:
    """Retry policy with exponential backoff.

    Calculates delay as: base_delay * exponential_base^(attempt-1)
    With optional jitter to prevent thundering herd.
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.0
    retry_on: tuple[type[Exception], ...] = field(default=(Exception,))

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (1-based).

        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return delay

    async def execute(
        self,
        func: Callable[[], Coroutine[Any, Any, T]],
        circuit_breaker: CircuitBreaker | None = None,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Async function to execute.
            circuit_breaker: Optional circuit breaker to use.

        Returns:
            Function result.

        Raises:
            RetriesExhaustedError: If all retries fail.
            CircuitOpenError: If circuit breaker opens.
            Exception: If non-retryable exception occurs.
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                if circuit_breaker:
                    return await circuit_breaker.execute(func)
                else:
                    return await func()

            except CircuitOpenError:
                # Don't retry if circuit is open
                raise

            except self.retry_on as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed: {e}"
                )

                if attempt < self.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.debug(f"Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)

            except Exception:
                # Non-retryable exception
                raise

        raise RetriesExhaustedError(
            f"All {self.max_attempts} retry attempts exhausted",
            attempts=self.max_attempts,
            last_exception=last_exception,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def with_retry(
    func: Callable[[], Coroutine[Any, Any, T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute function with retry logic.

    Convenience wrapper around RetryPolicy.

    Args:
        func: Async function to execute.
        max_attempts: Maximum number of attempts.
        base_delay: Base delay between retries.
        retry_on: Exception types to retry on.

    Returns:
        Function result.
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retry_on=retry_on,
    )
    return await policy.execute(func)
