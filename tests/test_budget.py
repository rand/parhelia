"""Tests for budget management.

Tests [SPEC-05.14] Budget Ceiling Enforcement.
"""

from __future__ import annotations

import pytest

from parhelia.budget import (
    BudgetExceededError,
    BudgetManager,
    BudgetStatus,
    UsageRecord,
)


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_usage_record_creation(self):
        """UsageRecord MUST track token usage."""
        record = UsageRecord(
            task_id="task-1",
            input_tokens=1000,
            output_tokens=500,
            cost_usd=0.01,
            model="claude-sonnet-4-20250514",
        )

        assert record.task_id == "task-1"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd == 0.01
        assert record.model == "claude-sonnet-4-20250514"

    def test_usage_record_total_tokens(self):
        """UsageRecord MUST calculate total tokens."""
        record = UsageRecord(
            task_id="task-1",
            input_tokens=1000,
            output_tokens=500,
        )

        assert record.total_tokens == 1500


class TestBudgetManager:
    """Tests for BudgetManager class."""

    def test_initialization_with_defaults(self):
        """BudgetManager MUST initialize with default values."""
        manager = BudgetManager()

        assert manager.ceiling_usd == 10.0
        assert manager.warning_threshold == 0.8
        assert manager._total_cost == 0.0

    def test_initialization_with_custom_ceiling(self):
        """BudgetManager MUST accept custom ceiling."""
        manager = BudgetManager(ceiling_usd=50.0)

        assert manager.ceiling_usd == 50.0

    def test_track_usage(self):
        """track_usage MUST record token usage and calculate cost."""
        manager = BudgetManager()

        record = manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,  # 1M tokens
            output_tokens=100_000,  # 100K tokens
            model="claude-sonnet-4-20250514",
        )

        # Sonnet pricing: $3/1M input, $15/1M output
        # Cost = (1M * $3/1M) + (0.1M * $15/1M) = $3 + $1.5 = $4.5
        assert record.cost_usd == pytest.approx(4.5, rel=0.01)
        assert record.task_id == "task-1"
        assert manager._total_cost == pytest.approx(4.5, rel=0.01)

    def test_track_usage_opus_model(self):
        """track_usage MUST use correct pricing for Opus."""
        manager = BudgetManager()

        record = manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="claude-opus-4-20250514",
        )

        # Opus pricing: $15/1M input, $75/1M output
        # Cost = (1M * $15/1M) + (0.1M * $75/1M) = $15 + $7.5 = $22.5
        assert record.cost_usd == pytest.approx(22.5, rel=0.01)

    def test_track_usage_haiku_model(self):
        """track_usage MUST use correct pricing for Haiku."""
        manager = BudgetManager()

        record = manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="claude-haiku-3-20250307",
        )

        # Haiku pricing: $0.25/1M input, $1.25/1M output
        # Cost = (1M * $0.25/1M) + (0.1M * $1.25/1M) = $0.25 + $0.125 = $0.375
        assert record.cost_usd == pytest.approx(0.375, rel=0.01)

    def test_check_budget_within_ceiling(self):
        """check_budget MUST return status when within ceiling."""
        manager = BudgetManager(ceiling_usd=10.0)

        # Use $5 of budget
        manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="claude-sonnet-4-20250514",
        )

        status = manager.check_budget()

        assert isinstance(status, BudgetStatus)
        assert status.ceiling_usd == 10.0
        assert status.used_usd == pytest.approx(4.5, rel=0.01)
        assert status.remaining_usd == pytest.approx(5.5, rel=0.01)
        assert not status.is_exceeded

    def test_check_budget_exceeded_raises(self):
        """check_budget MUST raise BudgetExceededError when exceeded."""
        manager = BudgetManager(ceiling_usd=1.0)

        # Use more than $1 budget
        manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="claude-sonnet-4-20250514",
        )

        with pytest.raises(BudgetExceededError) as exc_info:
            manager.check_budget()

        assert exc_info.value.current_cost == pytest.approx(4.5, rel=0.01)
        assert exc_info.value.ceiling == 1.0

    def test_check_budget_exceeded_no_raise(self):
        """check_budget MUST return status without raising when configured."""
        manager = BudgetManager(ceiling_usd=1.0)

        manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="claude-sonnet-4-20250514",
        )

        status = manager.check_budget(raise_on_exceeded=False)

        assert status.is_exceeded
        assert status.used_usd == pytest.approx(4.5, rel=0.01)

    def test_warning_threshold(self):
        """check_budget MUST detect warning threshold."""
        manager = BudgetManager(ceiling_usd=10.0, warning_threshold=0.8)

        # Use $9 of $10 budget (90%)
        manager.track_usage(
            task_id="task-1",
            input_tokens=2_000_000,
            output_tokens=200_000,
            model="claude-sonnet-4-20250514",
        )

        status = manager.check_budget()

        assert status.warning_threshold_reached

    def test_can_afford_true(self):
        """can_afford MUST return True when within budget."""
        manager = BudgetManager(ceiling_usd=10.0)

        assert manager.can_afford(5.0)

    def test_can_afford_false(self):
        """can_afford MUST return False when would exceed budget."""
        manager = BudgetManager(ceiling_usd=10.0)

        # Use $8 of budget
        manager._total_cost = 8.0

        assert not manager.can_afford(5.0)

    def test_estimate_cost(self):
        """estimate_cost MUST calculate expected cost."""
        manager = BudgetManager()

        cost = manager.estimate_cost(
            input_tokens=500_000,
            output_tokens=50_000,
            model="claude-sonnet-4-20250514",
        )

        # (0.5M * $3/1M) + (0.05M * $15/1M) = $1.5 + $0.75 = $2.25
        assert cost == pytest.approx(2.25, rel=0.01)

    def test_get_usage_by_task(self):
        """get_usage_by_task MUST filter by task ID."""
        manager = BudgetManager()

        manager.track_usage("task-1", 1000, 100)
        manager.track_usage("task-2", 2000, 200)
        manager.track_usage("task-1", 3000, 300)

        records = manager.get_usage_by_task("task-1")

        assert len(records) == 2
        assert all(r.task_id == "task-1" for r in records)

    def test_get_task_cost(self):
        """get_task_cost MUST sum all costs for a task."""
        manager = BudgetManager()

        manager.track_usage("task-1", 1_000_000, 100_000)  # ~$4.5
        manager.track_usage("task-1", 1_000_000, 100_000)  # ~$4.5

        total = manager.get_task_cost("task-1")

        assert total == pytest.approx(9.0, rel=0.01)

    def test_set_ceiling(self):
        """set_ceiling MUST update the budget ceiling."""
        manager = BudgetManager(ceiling_usd=10.0)

        manager.set_ceiling(50.0)

        assert manager.ceiling_usd == 50.0

    def test_reset(self):
        """reset MUST clear all usage tracking."""
        manager = BudgetManager()

        manager.track_usage("task-1", 1000, 100)
        manager.track_usage("task-2", 2000, 200)

        manager.reset()

        assert manager._total_cost == 0.0
        assert manager._total_input_tokens == 0
        assert manager._total_output_tokens == 0
        assert len(manager._usage_records) == 0

    def test_custom_pricing(self):
        """BudgetManager MUST accept custom pricing."""
        custom_pricing = {
            "custom-model": {
                "input": 1.0,
                "output": 5.0,
            }
        }

        manager = BudgetManager(pricing=custom_pricing)

        record = manager.track_usage(
            task_id="task-1",
            input_tokens=1_000_000,
            output_tokens=100_000,
            model="custom-model",
        )

        # Custom pricing: $1/1M input, $5/1M output
        # Cost = (1M * $1/1M) + (0.1M * $5/1M) = $1 + $0.5 = $1.5
        assert record.cost_usd == pytest.approx(1.5, rel=0.01)

    def test_budget_status_fields(self):
        """BudgetStatus MUST contain all required fields."""
        manager = BudgetManager(ceiling_usd=100.0)

        manager.track_usage("task-1", 1000, 100)
        manager.track_usage("task-2", 2000, 200)

        status = manager.check_budget()

        assert hasattr(status, "ceiling_usd")
        assert hasattr(status, "used_usd")
        assert hasattr(status, "remaining_usd")
        assert hasattr(status, "usage_percent")
        assert hasattr(status, "total_input_tokens")
        assert hasattr(status, "total_output_tokens")
        assert hasattr(status, "task_count")
        assert hasattr(status, "is_exceeded")
        assert hasattr(status, "warning_threshold_reached")

        assert status.task_count == 2
        assert status.total_input_tokens == 3000
        assert status.total_output_tokens == 300


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_error_attributes(self):
        """BudgetExceededError MUST contain cost details."""
        error = BudgetExceededError(
            message="Budget exceeded",
            current_cost=15.0,
            ceiling=10.0,
            task_id="task-1",
        )

        assert error.current_cost == 15.0
        assert error.ceiling == 10.0
        assert error.task_id == "task-1"
        assert str(error) == "Budget exceeded"
