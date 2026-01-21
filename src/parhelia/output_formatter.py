"""Output formatting for dual human/agent modes.

Implements [SPEC-11] Agent-Optimized Interfaces.

This module provides structured output formatting that works well for both
human operators (colored, formatted text) and AI agents (JSON with next_actions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standard error codes for structured responses."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    ATTACH_FAILED = "ATTACH_FAILED"
    DISPATCH_FAILED = "DISPATCH_FAILED"


# Error suggestions for each error code
ERROR_SUGGESTIONS: dict[ErrorCode, list[str]] = {
    ErrorCode.VALIDATION_ERROR: [
        "Check the command syntax with --help",
        "Verify all required arguments are provided",
    ],
    ErrorCode.SESSION_NOT_FOUND: [
        "List active sessions: parhelia list",
        "The session may have completed or timed out",
        "Check task status: parhelia status",
    ],
    ErrorCode.BUDGET_EXCEEDED: [
        "Reduce task scope or use CPU instead of GPU",
        "Increase budget: parhelia budget set <amount>",
        "Check current usage: parhelia budget show",
    ],
    ErrorCode.UNAUTHORIZED: [
        "Check Modal API credentials: modal token show",
        "Verify Anthropic API key is set",
        "Re-authenticate: modal token set",
    ],
    ErrorCode.RESOURCE_UNAVAILABLE: [
        "Try a different region in parhelia.toml",
        "Try CPU instead of GPU",
        "Wait a few minutes and retry",
    ],
    ErrorCode.TIMEOUT: [
        "Increase timeout in parhelia.toml",
        "Check network connectivity",
        "Try with --sync for immediate feedback",
    ],
    ErrorCode.INTERNAL_ERROR: [
        "Check logs: parhelia logs <session-id>",
        "Retry the operation",
        "Report issue if persistent",
    ],
    ErrorCode.CHECKPOINT_FAILED: [
        "Check disk space on volume",
        "Verify checkpoint directory permissions",
        "Try manual checkpoint: parhelia checkpoint create",
    ],
    ErrorCode.ATTACH_FAILED: [
        "Verify session is running: parhelia task show <id>",
        "Check network connectivity",
        "Wait for container startup if session is starting",
    ],
    ErrorCode.DISPATCH_FAILED: [
        "Check Modal connectivity",
        "Verify budget is available: parhelia budget show",
        "Try with --dry-run to test",
    ],
}


@dataclass
class NextAction:
    """A suggested next action for the user/agent."""

    action: str
    description: str
    command: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "description": self.description,
            "command": self.command,
        }


@dataclass
class OutputMetadata:
    """Metadata for output responses."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat() + "Z",
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
        }


@dataclass
class SuccessResponse:
    """Structured success response."""

    data: dict[str, Any]
    message: str | None = None
    metadata: OutputMetadata = field(default_factory=OutputMetadata)
    next_actions: list[NextAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "success": True,
            "data": self.data,
            "metadata": self.metadata.to_dict(),
            "next_actions": [a.to_dict() for a in self.next_actions],
        }


@dataclass
class ErrorResponse:
    """Structured error response."""

    code: ErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        suggestions = self.suggestions or ERROR_SUGGESTIONS.get(self.code, [])
        return {
            "success": False,
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            },
            "suggestions": suggestions,
        }


@dataclass
class ProgressEvent:
    """Progress event for streaming output."""

    phase: str
    percent: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "type": "progress",
            "phase": self.phase,
            "percent": self.percent,
            "message": self.message,
        }

    def to_json(self) -> str:
        """Convert to JSON string (one line)."""
        return json.dumps(self.to_dict())


class OutputFormatter:
    """Format output for human or agent consumption.

    Usage:
        formatter = OutputFormatter(json_mode=False)
        print(formatter.success(data, "Task completed"))

        formatter = OutputFormatter(json_mode=True)
        print(formatter.success(data))  # Returns JSON
    """

    def __init__(self, json_mode: bool = False):
        """Initialize formatter.

        Args:
            json_mode: If True, output JSON. If False, output human-readable text.
        """
        self.json_mode = json_mode

    def success(
        self,
        data: dict[str, Any],
        message: str | None = None,
        metadata: OutputMetadata | None = None,
        next_actions: list[NextAction] | None = None,
    ) -> str:
        """Format a success response.

        Args:
            data: The response data.
            message: Human-readable success message.
            metadata: Response metadata (timing, cost).
            next_actions: Suggested next actions.

        Returns:
            Formatted string (JSON or human-readable).
        """
        response = SuccessResponse(
            data=data,
            message=message,
            metadata=metadata or OutputMetadata(),
            next_actions=next_actions or [],
        )

        if self.json_mode:
            return json.dumps(response.to_dict(), indent=2)
        else:
            return self._format_success_human(response)

    def error(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
    ) -> str:
        """Format an error response.

        Args:
            code: Error code from ErrorCode enum.
            message: Human-readable error message.
            details: Additional error details.
            suggestions: Custom suggestions (uses defaults if not provided).

        Returns:
            Formatted string (JSON or human-readable).
        """
        response = ErrorResponse(
            code=code,
            message=message,
            details=details or {},
            suggestions=suggestions or [],
        )

        if self.json_mode:
            return json.dumps(response.to_dict(), indent=2)
        else:
            return self._format_error_human(response)

    def progress(self, phase: str, percent: int, message: str) -> str:
        """Format a progress event.

        Args:
            phase: Current phase name.
            percent: Completion percentage (0-100).
            message: Progress message.

        Returns:
            JSON line for progress event.
        """
        event = ProgressEvent(phase=phase, percent=percent, message=message)
        return event.to_json()

    def _format_success_human(self, response: SuccessResponse) -> str:
        """Format success response for human reading."""
        lines = []

        if response.message:
            lines.append(f"\u2713 {response.message}")
            lines.append("")

        # Format data as key-value pairs
        for key, value in response.data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}:")
                for item in value[:5]:  # Limit display
                    lines.append(f"  - {item}")
                if len(value) > 5:
                    lines.append(f"  ... and {len(value) - 5} more")
            else:
                lines.append(f"{key}: {value}")

        # Add next actions if present
        if response.next_actions:
            lines.append("")
            lines.append("Next steps:")
            for action in response.next_actions:
                lines.append(f"  {action.description}: {action.command}")

        return "\n".join(lines)

    def _format_error_human(self, response: ErrorResponse) -> str:
        """Format error response for human reading."""
        lines = []

        lines.append(f"\u2717 Error: {response.message}")

        if response.details:
            lines.append("")
            lines.append("Details:")
            for key, value in response.details.items():
                lines.append(f"  {key}: {value}")

        suggestions = response.suggestions or ERROR_SUGGESTIONS.get(response.code, [])
        if suggestions:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)


# Convenience functions for common responses


def format_task_submitted(
    task_id: str,
    worker_id: str | None = None,
    session_id: str | None = None,
    json_mode: bool = False,
) -> str:
    """Format task submission response."""
    formatter = OutputFormatter(json_mode=json_mode)

    data = {"task_id": task_id}
    if worker_id:
        data["worker_id"] = worker_id
    if session_id:
        data["session_id"] = session_id

    next_actions = [
        NextAction(
            action="status",
            description="Check task status",
            command=f"parhelia task show {task_id}",
        ),
    ]

    if session_id:
        next_actions.append(
            NextAction(
                action="attach",
                description="Attach to session",
                command=f"parhelia attach {task_id}",
            )
        )

    next_actions.append(
        NextAction(
            action="logs",
            description="View logs",
            command=f"parhelia logs {task_id}",
        )
    )

    return formatter.success(
        data=data,
        message="Task submitted successfully",
        next_actions=next_actions,
    )


def format_session_attached(
    session_id: str,
    container_info: dict[str, Any],
    json_mode: bool = False,
) -> str:
    """Format session attachment response."""
    formatter = OutputFormatter(json_mode=json_mode)

    data = {
        "session_id": session_id,
        "container": container_info,
        "status": "attached",
    }

    return formatter.success(
        data=data,
        message=f"Attached to session {session_id}",
        next_actions=[
            NextAction(
                action="detach",
                description="Detach from session",
                command="Press Ctrl+B, D",
            ),
        ],
    )


def format_budget_status(
    ceiling_usd: float,
    used_usd: float,
    remaining_usd: float,
    usage_percent: float,
    task_count: int,
    json_mode: bool = False,
) -> str:
    """Format budget status response."""
    formatter = OutputFormatter(json_mode=json_mode)

    data = {
        "ceiling_usd": ceiling_usd,
        "used_usd": used_usd,
        "remaining_usd": remaining_usd,
        "usage_percent": usage_percent,
        "task_count": task_count,
    }

    next_actions = []
    if usage_percent > 80:
        next_actions.append(
            NextAction(
                action="increase",
                description="Increase budget ceiling",
                command="parhelia budget set <amount>",
            )
        )

    return formatter.success(
        data=data,
        message="Budget status",
        next_actions=next_actions,
    )


def format_budget_exceeded(
    requested_usd: float,
    remaining_usd: float,
    ceiling_usd: float,
    json_mode: bool = False,
) -> str:
    """Format budget exceeded error."""
    formatter = OutputFormatter(json_mode=json_mode)

    return formatter.error(
        code=ErrorCode.BUDGET_EXCEEDED,
        message=f"Task would exceed budget ceiling (${ceiling_usd:.2f})",
        details={
            "requested_usd": requested_usd,
            "remaining_usd": remaining_usd,
            "ceiling_usd": ceiling_usd,
        },
    )


def format_session_not_found(
    session_id: str,
    json_mode: bool = False,
) -> str:
    """Format session not found error."""
    formatter = OutputFormatter(json_mode=json_mode)

    return formatter.error(
        code=ErrorCode.SESSION_NOT_FOUND,
        message=f"Session not found: {session_id}",
        details={"session_id": session_id},
    )
