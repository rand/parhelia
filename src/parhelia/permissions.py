"""Permission model for remote execution.

Implements:
- [SPEC-04.13] Permission Model for Remote Execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TrustLevel(Enum):
    """Trust level for task execution."""

    INTERACTIVE = "interactive"  # User is attached, can approve
    AUTOMATED = "automated"  # Headless, pre-approved permissions


@dataclass
class RemotePermissions:
    """Permissions granted to remote Claude Code instance.

    Implements [SPEC-04.13].
    """

    # Tool permissions
    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Bash",
            "Task",
            "TodoWrite",
        ]
    )
    denied_tools: list[str] = field(default_factory=lambda: ["WebFetch"])

    # Bash restrictions
    bash_allow_network: bool = True
    bash_allow_sudo: bool = False
    bash_blocked_commands: list[str] = field(
        default_factory=lambda: [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # Fork bomb
            "> /dev/sda",
            "chmod -R 777 /",
        ]
    )

    # File access
    allowed_paths: list[str] = field(
        default_factory=lambda: [
            "/vol/parhelia/workspaces",
            "/vol/parhelia/checkpoints",
            "/vol/parhelia/plugins",
            "/tmp",
        ]
    )
    denied_paths: list[str] = field(
        default_factory=lambda: [
            "/etc/shadow",
            "/etc/passwd",
            "/etc/sudoers",
            "/root/.ssh",
        ]
    )

    # Network egress
    allowed_domains: list[str] = field(
        default_factory=lambda: [
            "api.anthropic.com",
            "github.com",
            "*.githubusercontent.com",
            "pypi.org",
            "registry.npmjs.org",
            "crates.io",
        ]
    )


def build_claude_command(
    prompt: str,
    permissions: RemotePermissions,
    trust_level: TrustLevel = TrustLevel.INTERACTIVE,
    working_directory: str | None = None,
) -> list[str]:
    """Build Claude Code command with permission restrictions.

    Implements [SPEC-04.13].

    Args:
        prompt: The task prompt.
        permissions: Permission restrictions.
        trust_level: Whether task is interactive or automated.
        working_directory: Optional working directory.

    Returns:
        Command list for subprocess execution.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
    ]

    # Add allowed tools
    if permissions.allowed_tools:
        cmd.extend(["--allowedTools", ",".join(permissions.allowed_tools)])

    # In sandboxed Modal environment, skip permissions for automated tasks
    if trust_level == TrustLevel.AUTOMATED:
        cmd.append("--dangerously-skip-permissions")

    return cmd


def validate_command(command: str, permissions: RemotePermissions) -> bool:
    """Validate if a bash command is allowed.

    Implements [SPEC-04.13].

    Args:
        command: The command to validate.
        permissions: Permission restrictions.

    Returns:
        True if command is allowed, False otherwise.
    """
    command_lower = command.lower().strip()

    # Check for blocked commands
    for blocked in permissions.bash_blocked_commands:
        if blocked.lower() in command_lower:
            return False

    # Check for sudo when disabled
    if not permissions.bash_allow_sudo:
        if command_lower.startswith("sudo "):
            return False

    return True


def validate_path(path: str, permissions: RemotePermissions) -> bool:
    """Validate if a path is allowed for access.

    Implements [SPEC-04.13].

    Args:
        path: The path to validate.
        permissions: Permission restrictions.

    Returns:
        True if path is allowed, False otherwise.
    """
    # Check denied paths first (highest priority)
    for denied in permissions.denied_paths:
        if path.startswith(denied) or path == denied:
            return False

    # Check if path is under an allowed path
    for allowed in permissions.allowed_paths:
        if path.startswith(allowed) or path == allowed:
            return True

    # Default deny if not explicitly allowed
    return False
