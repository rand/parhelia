"""Budget management for task execution.

Implements:
- [SPEC-05.14] Budget Ceiling Enforcement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parhelia.orchestrator import Task


class BudgetExceededError(Exception):
    """Raised when budget ceiling is exceeded."""

    def __init__(
        self,
        message: str,
        current_cost: float,
        ceiling: float,
        task_id: str | None = None,
    ):
        """Initialize the error.

        Args:
            message: Error message.
            current_cost: Current accumulated cost.
            ceiling: Budget ceiling that was exceeded.
            task_id: Optional task ID that triggered the error.
        """
        super().__init__(message)
        self.current_cost = current_cost
        self.ceiling = ceiling
        self.task_id = task_id


@dataclass
class UsageRecord:
    """Record of token/cost usage for a task.

    Implements [SPEC-05.14].
    """

    task_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "claude-sonnet-4-20250514"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class BudgetStatus:
    """Current budget status.

    Implements [SPEC-05.14].
    """

    ceiling_usd: float
    used_usd: float
    remaining_usd: float
    usage_percent: float
    total_input_tokens: int
    total_output_tokens: int
    task_count: int
    is_exceeded: bool
    warning_threshold_reached: bool


class BudgetManager:
    """Manage budget ceiling enforcement for task execution.

    Implements [SPEC-05.14].

    The budget manager tracks:
    - Token usage per task
    - Cost estimates based on model pricing
    - Budget ceiling enforcement
    - Usage warnings at configurable thresholds
    """

    # Default pricing per 1M tokens (as of 2025)
    DEFAULT_PRICING = {
        "claude-sonnet-4-20250514": {
            "input": 3.00,  # $3 per 1M input tokens
            "output": 15.00,  # $15 per 1M output tokens
        },
        "claude-opus-4-20250514": {
            "input": 15.00,  # $15 per 1M input tokens
            "output": 75.00,  # $75 per 1M output tokens
        },
        "claude-haiku-3-20250307": {
            "input": 0.25,  # $0.25 per 1M input tokens
            "output": 1.25,  # $1.25 per 1M output tokens
        },
    }

    def __init__(
        self,
        ceiling_usd: float = 10.0,
        warning_threshold: float = 0.8,
        pricing: dict[str, dict[str, float]] | None = None,
    ):
        """Initialize the budget manager.

        Args:
            ceiling_usd: Maximum budget in USD.
            warning_threshold: Percentage of budget that triggers warning (0.0-1.0).
            pricing: Optional custom pricing per model.
        """
        self.ceiling_usd = ceiling_usd
        self.warning_threshold = warning_threshold
        self.pricing = pricing or self.DEFAULT_PRICING

        self._usage_records: list[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def track_usage(
        self,
        task_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-sonnet-4-20250514",
    ) -> UsageRecord:
        """Track token usage for a task.

        Implements [SPEC-05.14].

        Args:
            task_id: The task ID.
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
            model: The model used.

        Returns:
            UsageRecord for this usage.
        """
        # Calculate cost
        cost = self._calculate_cost(input_tokens, output_tokens, model)

        # Create record
        record = UsageRecord(
            task_id=task_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=model,
        )

        # Update totals
        self._usage_records.append(record)
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        return record

    def check_budget(self, raise_on_exceeded: bool = True) -> BudgetStatus:
        """Check if budget is within ceiling.

        Implements [SPEC-05.14].

        Args:
            raise_on_exceeded: If True, raise BudgetExceededError when exceeded.

        Returns:
            Current BudgetStatus.

        Raises:
            BudgetExceededError: If budget exceeded and raise_on_exceeded is True.
        """
        remaining = self.ceiling_usd - self._total_cost
        usage_percent = (self._total_cost / self.ceiling_usd * 100) if self.ceiling_usd > 0 else 0

        status = BudgetStatus(
            ceiling_usd=self.ceiling_usd,
            used_usd=self._total_cost,
            remaining_usd=max(0, remaining),
            usage_percent=usage_percent,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            task_count=len(self._usage_records),
            is_exceeded=self._total_cost > self.ceiling_usd,
            warning_threshold_reached=usage_percent >= self.warning_threshold * 100,
        )

        if status.is_exceeded and raise_on_exceeded:
            raise BudgetExceededError(
                f"Budget exceeded: ${self._total_cost:.2f} used of ${self.ceiling_usd:.2f} ceiling",
                current_cost=self._total_cost,
                ceiling=self.ceiling_usd,
            )

        return status

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if an operation can be afforded within budget.

        Args:
            estimated_cost: Estimated cost of the operation.

        Returns:
            True if operation fits within remaining budget.
        """
        return (self._total_cost + estimated_cost) <= self.ceiling_usd

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "claude-sonnet-4-20250514",
    ) -> float:
        """Estimate cost for a given token count.

        Args:
            input_tokens: Expected input tokens.
            output_tokens: Expected output tokens.
            model: The model to use.

        Returns:
            Estimated cost in USD.
        """
        return self._calculate_cost(input_tokens, output_tokens, model)

    def get_usage_by_task(self, task_id: str) -> list[UsageRecord]:
        """Get all usage records for a task.

        Args:
            task_id: The task ID.

        Returns:
            List of UsageRecord for the task.
        """
        return [r for r in self._usage_records if r.task_id == task_id]

    def get_task_cost(self, task_id: str) -> float:
        """Get total cost for a task.

        Args:
            task_id: The task ID.

        Returns:
            Total cost in USD.
        """
        return sum(r.cost_usd for r in self.get_usage_by_task(task_id))

    def set_ceiling(self, ceiling_usd: float) -> None:
        """Update the budget ceiling.

        Args:
            ceiling_usd: New ceiling in USD.
        """
        self.ceiling_usd = ceiling_usd

    def reset(self) -> None:
        """Reset all usage tracking."""
        self._usage_records.clear()
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            model: The model used.

        Returns:
            Cost in USD.
        """
        # Get pricing for model, fallback to default sonnet pricing
        default_pricing = {"input": 3.00, "output": 15.00}
        pricing = self.pricing.get(model, default_pricing)

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost
