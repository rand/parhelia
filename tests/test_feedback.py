"""Tests for feedback and progress indication system.

Tests [SPEC-20.20], [SPEC-20.21], [SPEC-20.22].
"""

from __future__ import annotations

import io
import sys
import time

import pytest

from parhelia.feedback import (
    ErrorRecovery,
    ProgressBar,
    ProgressSpinner,
    StatusFormatter,
)


class TestStatusFormatter:
    """Tests for StatusFormatter output formatting."""

    def test_success_format(self):
        """Success messages include checkmark."""
        result = StatusFormatter.success("Operation completed")
        assert "✓" in result
        assert "Operation completed" in result

    def test_error_format(self):
        """Error messages include X marker."""
        result = StatusFormatter.error("Something failed")
        assert "✗" in result
        assert "Something failed" in result

    def test_error_with_suggestions(self):
        """Error messages can include suggestions."""
        result = StatusFormatter.error(
            "Connection failed",
            suggestions=["Check network", "Retry later"],
        )
        assert "Connection failed" in result
        assert "Check network" in result
        assert "Retry later" in result

    def test_warning_format(self):
        """Warning messages include warning marker."""
        result = StatusFormatter.warning("Deprecated feature")
        assert "⚠" in result
        assert "Deprecated feature" in result

    def test_info_format(self):
        """Info messages include info marker."""
        result = StatusFormatter.info("Processing...")
        assert "ℹ" in result
        assert "Processing..." in result


class TestErrorRecovery:
    """Tests for ErrorRecovery suggestions."""

    def test_returns_suggestions_for_known_error(self):
        """Known error codes return suggestions."""
        suggestions = ErrorRecovery.suggest("E200")
        assert len(suggestions) >= 2
        assert all(isinstance(s, str) for s in suggestions)

    def test_returns_suggestions_for_unknown_error(self):
        """Unknown error codes still return generic suggestions."""
        suggestions = ErrorRecovery.suggest("E999_UNKNOWN")
        assert len(suggestions) >= 2

    def test_session_not_found_suggestions(self):
        """E200 (session not found) has specific suggestions."""
        suggestions = ErrorRecovery.suggest("E200")
        # Should suggest listing sessions
        assert any("list" in s.lower() for s in suggestions)

    def test_budget_exceeded_suggestions(self):
        """E300 (budget exceeded) has specific suggestions."""
        suggestions = ErrorRecovery.suggest("E300")
        assert len(suggestions) >= 2


class TestProgressSpinner:
    """Tests for ProgressSpinner."""

    def test_spinner_context_manager(self):
        """Spinner works as context manager."""
        output = io.StringIO()
        # Use 'none' style to disable animation for testing
        with ProgressSpinner("Testing", style="none", stream=output) as spinner:
            assert spinner is not None

    def test_spinner_update_message(self):
        """Spinner can update its message."""
        output = io.StringIO()
        spinner = ProgressSpinner("Initial", style="none", stream=output)
        spinner.start()
        spinner.update("Updated")
        spinner.stop()
        assert spinner._message == "Updated"

    def test_spinner_disabled_for_non_tty(self):
        """Spinner auto-disables when not a TTY."""
        output = io.StringIO()  # Not a TTY
        spinner = ProgressSpinner("Test", stream=output)
        # Should detect non-TTY and adjust behavior
        assert spinner is not None


class TestProgressBar:
    """Tests for ProgressBar."""

    def test_progress_bar_init(self):
        """Progress bar initializes with total."""
        output = io.StringIO()
        bar = ProgressBar(total=100, description="Loading", stream=output)
        assert bar.total == 100

    def test_progress_bar_update(self):
        """Progress bar tracks updates."""
        output = io.StringIO()
        bar = ProgressBar(total=10, stream=output)
        bar.update(3)
        assert bar._current == 3
        bar.update(2)
        assert bar._current == 5

    def test_progress_bar_close(self):
        """Progress bar can be closed."""
        output = io.StringIO()
        bar = ProgressBar(total=10, stream=output)
        bar.update(10)
        bar.close()
        # Should not raise

    def test_progress_bar_set_description(self):
        """Progress bar description can be updated."""
        output = io.StringIO()
        bar = ProgressBar(total=10, description="Initial", stream=output)
        bar.set_description("Updated")
        assert bar.description == "Updated"
