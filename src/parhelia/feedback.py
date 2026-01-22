"""Feedback and progress indication for Parhelia CLI.

Implements:
- [SPEC-20.20] Feedback Excellence
- [SPEC-20.21] Progress Indication Standards
- [SPEC-20.22] Structured Error System

Provides visual feedback during long operations with spinners, progress bars,
and consistent status formatting.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TextIO

import click


# =============================================================================
# Constants
# =============================================================================

# Spinner animation frames per SPEC-20.21
SPINNER_STYLES: dict[str, list[str]] = {
    "dots": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "line": ["-", "\\", "|", "/"],
    "classic": ["|", "/", "-", "\\"],
    "none": [""],  # Disabled for CI/pipes
}

# Success/failure markers for StatusFormatter per SPEC-20.22
STATUS_MARKERS = {
    "success": ("✓", "green"),
    "error": ("✗", "red"),
    "warning": ("⚠", "yellow"),
    "info": ("ℹ", "blue"),
}

# Progress bar characters
PROGRESS_FILLED = "█"
PROGRESS_EMPTY = "░"


# =============================================================================
# Error Recovery System (SPEC-20.22)
# =============================================================================


class ErrorRecovery:
    """Generate recovery suggestions for errors.

    Implements [SPEC-20.22] - all errors include 2+ recovery commands.

    Error codes follow the SPEC-20.22 format:
    - 1xx: Validation errors
    - 2xx: Resource errors
    - 3xx: Budget errors
    - 4xx: Auth errors
    - 5xx: Infrastructure errors
    - 6xx: Operations errors
    - 9xx: Internal errors
    """

    # Recovery suggestions per error code
    _SUGGESTIONS: dict[str, list[str]] = {
        # Validation errors (1xx)
        "E100": [
            "Check the command syntax with --help",
            "Verify all required arguments are provided",
            "Try 'parhelia examples <command>' for usage examples",
        ],
        "E101": [
            "Verify the argument type and format",
            "Check valid values in 'parhelia <command> --help'",
        ],
        "E102": [
            "Provide the missing argument",
            "Check 'parhelia <command> --help' for required arguments",
        ],
        # Resource errors (2xx)
        "E200": [
            "List all sessions: parhelia session list",
            "Check if the session ID is correct",
            "The session may have completed or timed out",
        ],
        "E201": [
            "List tasks: parhelia task list",
            "Search by prefix: parhelia task list --query <prefix>",
            "Check recent tasks: parhelia task list --recent",
        ],
        "E202": [
            "List checkpoints: parhelia checkpoint list",
            "The checkpoint may have been deleted or expired",
        ],
        "E203": [
            "The worker may have terminated",
            "Check task status: parhelia task show <id>",
        ],
        # Budget errors (3xx)
        "E300": [
            "Check current usage: parhelia budget status",
            "Increase budget: parhelia budget set <amount>",
            "Use CPU instead of GPU for lower cost",
        ],
        "E301": [
            "Budget is running low - consider increasing",
            "Check usage breakdown: parhelia budget history",
        ],
        # Auth errors (4xx)
        "E400": [
            "Check Modal credentials: modal token show",
            "Re-authenticate: modal token set",
            "Verify environment variables are set correctly",
        ],
        "E401": [
            "Check permissions for the requested operation",
            "Verify your Modal workspace settings",
        ],
        "E402": [
            "Refresh your token: modal token set",
            "Check token expiration date",
        ],
        # Infrastructure errors (5xx)
        "E500": [
            "Try a different region in config",
            "Use CPU instead of GPU",
            "Wait a few minutes and retry",
        ],
        "E501": [
            "Increase timeout in parhelia.toml",
            "Check network connectivity",
            "Retry with smaller workload",
        ],
        "E502": [
            "Check your internet connection",
            "Verify Modal API status at status.modal.com",
            "Try again in a few moments",
        ],
        "E503": [
            "Check Modal connectivity",
            "Verify budget availability: parhelia budget status",
            "Try with --dry-run to diagnose",
        ],
        # Operations errors (6xx)
        "E600": [
            "Check disk space on volume",
            "Verify checkpoint directory permissions",
            "Try manual checkpoint: parhelia checkpoint create",
        ],
        "E601": [
            "Verify session is running: parhelia task show <id>",
            "Check network connectivity",
            "Wait for container startup",
        ],
        "E602": [
            "Check Modal connectivity",
            "Verify budget: parhelia budget status",
            "Try with --dry-run to test",
        ],
        "E603": [
            "Review pre-dispatch hook configuration",
            "Check hook rejection reason in logs",
            "Adjust task parameters to meet hook requirements",
        ],
        # Internal errors (9xx)
        "E900": [
            "Check logs for more details",
            "Retry the operation",
            "Report issue if persistent: https://github.com/parhelia/issues",
        ],
        "E901": [
            "This feature is not yet implemented",
            "Check documentation for alternatives",
        ],
    }

    @staticmethod
    def suggest(error_code: str, context: dict | None = None) -> list[str]:
        """Return 2+ recovery suggestions for an error.

        Args:
            error_code: The error code (e.g., "E200", "E300").
            context: Optional context dict with details like task_id, session_id.

        Returns:
            List of at least 2 recovery suggestions.
        """
        context = context or {}
        base_suggestions = ErrorRecovery._SUGGESTIONS.get(
            error_code,
            ["Check logs for more details", "Retry the operation"],
        )

        # Customize suggestions with context
        suggestions = []
        for suggestion in base_suggestions:
            if "{task_id}" in suggestion and "task_id" in context:
                suggestion = suggestion.replace("{task_id}", context["task_id"])
            if "{session_id}" in suggestion and "session_id" in context:
                suggestion = suggestion.replace("{session_id}", context["session_id"])
            suggestions.append(suggestion)

        # Ensure at least 2 suggestions per SPEC-20.22
        if len(suggestions) < 2:
            suggestions.append("Check 'parhelia --help' for usage information")

        return suggestions

    @staticmethod
    def get_all_codes() -> list[str]:
        """Return all known error codes."""
        return list(ErrorRecovery._SUGGESTIONS.keys())


# =============================================================================
# Status Formatter (SPEC-20.21)
# =============================================================================


class StatusFormatter:
    """Format status messages with consistent styling.

    Implements [SPEC-20.21] - consistent visual feedback.

    All methods return formatted strings ready for output. When stdout is not
    a TTY (piped output), styling is disabled for clean parsing.
    """

    def __init__(self, force_color: bool = False):
        """Initialize the formatter.

        Args:
            force_color: Force colored output even when not a TTY.
        """
        self._force_color = force_color

    def _is_tty(self) -> bool:
        """Check if stdout is a TTY."""
        return sys.stdout.isatty() or self._force_color

    @staticmethod
    def success(message: str) -> str:
        """Format a success message with green checkmark.

        Args:
            message: The success message.

        Returns:
            Formatted string with checkmark prefix.
        """
        marker, color = STATUS_MARKERS["success"]
        if sys.stdout.isatty():
            return click.style(f"{marker} {message}", fg=color)
        return f"{marker} {message}"

    @staticmethod
    def error(message: str, suggestions: list[str] | None = None) -> str:
        """Format an error message with red X and optional suggestions.

        Args:
            message: The error message.
            suggestions: Optional list of recovery suggestions.

        Returns:
            Formatted string with error marker and suggestions.
        """
        marker, color = STATUS_MARKERS["error"]
        if sys.stdout.isatty():
            lines = [click.style(f"{marker} {message}", fg=color)]
        else:
            lines = [f"{marker} {message}"]

        if suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)

    @staticmethod
    def warning(message: str) -> str:
        """Format a warning message with yellow warning sign.

        Args:
            message: The warning message.

        Returns:
            Formatted string with warning marker.
        """
        marker, color = STATUS_MARKERS["warning"]
        if sys.stdout.isatty():
            return click.style(f"{marker} {message}", fg=color)
        return f"{marker} {message}"

    @staticmethod
    def info(message: str) -> str:
        """Format an info message with blue info sign.

        Args:
            message: The info message.

        Returns:
            Formatted string with info marker.
        """
        marker, color = STATUS_MARKERS["info"]
        if sys.stdout.isatty():
            return click.style(f"{marker} {message}", fg=color)
        return f"{marker} {message}"


# =============================================================================
# Progress Spinner (SPEC-20.21)
# =============================================================================


class ProgressSpinner:
    """Animated spinner for long operations.

    Implements [SPEC-20.21] - visual feedback for 1-4s operations.

    Usage:
        with ProgressSpinner("Creating sandbox...") as spinner:
            # ... long operation ...
            spinner.update("Provisioning resources...")
            # ... more work ...

        # Or manual control:
        spinner = ProgressSpinner("Loading...")
        spinner.start()
        # ... work ...
        spinner.stop(success=True, final_message="Done!")
    """

    def __init__(
        self,
        message: str,
        style: str = "dots",
        stream: TextIO | None = None,
    ):
        """Initialize the spinner.

        Args:
            message: Initial status message to display.
            style: Spinner animation style (dots, line, classic, none).
            stream: Output stream (defaults to stderr for clean stdout).
        """
        self._message = message
        self._style = style
        self._stream = stream or sys.stderr
        self._frames = SPINNER_STYLES.get(style, SPINNER_STYLES["dots"])
        self._frame_index = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._disabled = not self._stream.isatty() or style == "none"

    @property
    def message(self) -> str:
        """Get the current message."""
        return self._message

    def start(self) -> None:
        """Start the spinner animation."""
        if self._disabled or self._running:
            # Still show the message even if animation disabled
            if self._disabled:
                self._stream.write(f"{self._message}\n")
                self._stream.flush()
            return

        self._running = True
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def _animate(self) -> None:
        """Animation loop running in background thread."""
        while self._running:
            with self._lock:
                frame = self._frames[self._frame_index]
                # Clear line and write current frame with message
                self._stream.write(f"\r{frame} {self._message}")
                self._stream.flush()
                self._frame_index = (self._frame_index + 1) % len(self._frames)
            time.sleep(0.1)  # 100ms between frames

    def update(self, message: str) -> None:
        """Update the spinner message.

        Args:
            message: New status message to display.
        """
        with self._lock:
            self._message = message
            if self._disabled and self._running:
                # For non-TTY, just print the new message
                self._stream.write(f"{message}\n")
                self._stream.flush()

    def stop(self, success: bool = True, final_message: str | None = None) -> None:
        """Stop the spinner and show final status.

        Args:
            success: Whether the operation succeeded.
            final_message: Optional final message (uses last message if None).
        """
        self._running = False

        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

        if self._disabled:
            # Just show final message for non-TTY
            msg = final_message or self._message
            if success:
                self._stream.write(f"{STATUS_MARKERS['success'][0]} {msg}\n")
            else:
                self._stream.write(f"{STATUS_MARKERS['error'][0]} {msg}\n")
            self._stream.flush()
            return

        # Clear the spinner line
        self._stream.write("\r" + " " * (len(self._message) + 5) + "\r")

        # Show final status
        msg = final_message or self._message
        marker, color = STATUS_MARKERS["success" if success else "error"]

        if self._stream.isatty():
            self._stream.write(click.style(f"{marker} {msg}", fg=color) + "\n")
        else:
            self._stream.write(f"{marker} {msg}\n")
        self._stream.flush()

    def __enter__(self) -> "ProgressSpinner":
        """Context manager entry - start the spinner."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the spinner."""
        success = exc_type is None
        self.stop(success=success)


# =============================================================================
# Progress Bar (SPEC-20.21)
# =============================================================================


@dataclass
class ProgressBar:
    """Progress bar for operations with known total.

    Implements [SPEC-20.21] - visual feedback for 4-10s+ operations.

    Usage:
        progress = ProgressBar(total=100, description="Processing files")
        for item in items:
            # process item
            progress.update(1)
        progress.close()
    """

    total: int
    description: str = ""
    width: int = 40
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    _current: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    _disabled: bool = field(default=False, init=False)

    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        self._disabled = not self.stream.isatty()
        self._render()

    def update(self, amount: int = 1) -> None:
        """Update progress by the given amount.

        Args:
            amount: Amount to increment progress by.
        """
        self._current = min(self._current + amount, self.total)
        self._render()

    def set_description(self, desc: str) -> None:
        """Update the progress description.

        Args:
            desc: New description text.
        """
        self.description = desc
        self._render()

    def _render(self) -> None:
        """Render the progress bar to the stream."""
        percent = (self._current / self.total * 100) if self.total > 0 else 0
        filled = int(self.width * self._current / self.total) if self.total > 0 else 0
        empty = self.width - filled

        bar = PROGRESS_FILLED * filled + PROGRESS_EMPTY * empty

        # Calculate ETA for operations > 10s
        elapsed = time.time() - self._start_time
        eta_str = ""
        if elapsed > 2 and self._current > 0 and self._current < self.total:
            rate = self._current / elapsed
            remaining = (self.total - self._current) / rate
            if remaining > 60:
                eta_str = f" ~{int(remaining/60)}m remaining"
            elif remaining > 10:
                eta_str = f" ~{int(remaining)}s remaining"

        desc = f"{self.description} " if self.description else ""

        if self._disabled:
            # For non-TTY, only print at milestones
            if self._current == self.total or self._current == 0 or percent % 25 == 0:
                self.stream.write(f"{desc}[{bar}] {percent:.0f}%{eta_str}\n")
                self.stream.flush()
        else:
            self.stream.write(f"\r{desc}[{bar}] {percent:.0f}%{eta_str}  ")
            self.stream.flush()

    def close(self) -> None:
        """Close the progress bar and move to new line."""
        if not self._disabled:
            self.stream.write("\n")
            self.stream.flush()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_spinner(message: str, style: str | None = None) -> ProgressSpinner:
    """Create a progress spinner with environment-aware defaults.

    Args:
        message: Status message to display.
        style: Optional spinner style override.

    Returns:
        Configured ProgressSpinner instance.
    """
    import os

    # Check PARHELIA_SPINNER environment variable
    env_style = os.environ.get("PARHELIA_SPINNER", "dots")
    final_style = style or env_style

    # Disable in CI environments
    if os.environ.get("CI") or os.environ.get("PARHELIA_NO_SPINNER"):
        final_style = "none"

    return ProgressSpinner(message, style=final_style)


def create_progress(
    total: int,
    description: str = "",
) -> ProgressBar:
    """Create a progress bar with environment-aware defaults.

    Args:
        total: Total number of items to process.
        description: Progress bar description.

    Returns:
        Configured ProgressBar instance.
    """
    return ProgressBar(total=total, description=description)
