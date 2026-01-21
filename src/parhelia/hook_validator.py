"""Claude Code hooks validation.

Validates that Claude Code hooks are properly configured and executable
in the remote sandbox environment.

Claude hooks are defined in ~/.claude/settings.json or settings.local.json:
{
  "hooks": {
    "PreToolUse": [{"matcher": "Bash", "hooks": ["~/.claude/hooks/pre-bash.sh"]}],
    "PostToolUse": [...],
    "SessionStart": [...]
  }
}
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HookIssue:
    """An issue found during hook validation."""

    hook_type: str  # PreToolUse, PostToolUse, etc.
    hook_path: str
    issue: str
    severity: str = "warning"  # warning or error


@dataclass
class HookValidationResult:
    """Result of hook validation."""

    hooks_found: int = 0
    hooks_valid: int = 0
    issues: list[HookIssue] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return self.hooks_found == self.hooks_valid

    def summary(self) -> str:
        if self.hooks_found == 0:
            return "No hooks configured"
        if self.all_valid:
            return f"All {self.hooks_found} hooks valid"
        return f"{self.hooks_valid}/{self.hooks_found} hooks valid, {len(self.issues)} issues"


class HookValidator:
    """Validates Claude Code hooks configuration."""

    SETTINGS_FILES = ["settings.json", "settings.local.json"]
    HOOK_TYPES = [
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "Notification",
    ]

    def __init__(self, claude_dir: str | Path | None = None):
        """Initialize hook validator.

        Args:
            claude_dir: Path to .claude directory. Defaults to ~/.claude
        """
        if claude_dir is None:
            claude_dir = Path.home() / ".claude"
        self.claude_dir = Path(claude_dir)

    def validate(self) -> HookValidationResult:
        """Validate all configured hooks.

        Returns:
            HookValidationResult with validation details
        """
        result = HookValidationResult()

        # Find and parse settings files
        hooks_config = self._load_hooks_config()
        if not hooks_config:
            return result

        # Validate each hook type
        for hook_type, hook_entries in hooks_config.items():
            if not isinstance(hook_entries, list):
                continue

            for entry in hook_entries:
                if not isinstance(entry, dict):
                    continue

                hook_paths = entry.get("hooks", [])
                if isinstance(hook_paths, str):
                    hook_paths = [hook_paths]

                for hook_path in hook_paths:
                    result.hooks_found += 1
                    issues = self._validate_hook(hook_type, hook_path)
                    if issues:
                        result.issues.extend(issues)
                    else:
                        result.hooks_valid += 1

        return result

    def _load_hooks_config(self) -> dict[str, Any]:
        """Load hooks configuration from settings files.

        Returns:
            Merged hooks configuration from all settings files
        """
        hooks: dict[str, Any] = {}

        for filename in self.SETTINGS_FILES:
            settings_path = self.claude_dir / filename
            if not settings_path.exists():
                continue

            try:
                with open(settings_path) as f:
                    data = json.load(f)
                    file_hooks = data.get("hooks", {})
                    # Merge hooks (later files override)
                    for hook_type, entries in file_hooks.items():
                        if hook_type not in hooks:
                            hooks[hook_type] = []
                        hooks[hook_type].extend(entries)
            except (json.JSONDecodeError, OSError):
                continue

        return hooks

    def _validate_hook(self, hook_type: str, hook_path: str) -> list[HookIssue]:
        """Validate a single hook.

        Args:
            hook_type: The hook type (PreToolUse, etc.)
            hook_path: Path to the hook script

        Returns:
            List of issues found (empty if valid)
        """
        issues = []

        # Expand path
        expanded = os.path.expanduser(hook_path)
        path = Path(expanded)

        # Check existence
        if not path.exists():
            issues.append(
                HookIssue(
                    hook_type=hook_type,
                    hook_path=hook_path,
                    issue=f"Hook script not found: {expanded}",
                    severity="error",
                )
            )
            return issues

        # Check if it's a file (not directory)
        if not path.is_file():
            issues.append(
                HookIssue(
                    hook_type=hook_type,
                    hook_path=hook_path,
                    issue=f"Hook path is not a file: {expanded}",
                    severity="error",
                )
            )
            return issues

        # Check executable permission
        mode = path.stat().st_mode
        if not (mode & stat.S_IXUSR or mode & stat.S_IXGRP or mode & stat.S_IXOTH):
            issues.append(
                HookIssue(
                    hook_type=hook_type,
                    hook_path=hook_path,
                    issue=f"Hook script not executable: {expanded}",
                    severity="warning",
                )
            )

        # Check shebang for shell scripts
        if path.suffix in (".sh", ".bash", ""):
            try:
                with open(path, "rb") as f:
                    first_bytes = f.read(2)
                    if first_bytes != b"#!":
                        issues.append(
                            HookIssue(
                                hook_type=hook_type,
                                hook_path=hook_path,
                                issue=f"Hook script missing shebang: {expanded}",
                                severity="warning",
                            )
                        )
            except OSError:
                pass

        return issues

    def fix_permissions(self) -> int:
        """Fix executable permissions on hook scripts.

        Returns:
            Number of hooks fixed
        """
        fixed = 0
        hooks_config = self._load_hooks_config()

        for hook_entries in hooks_config.values():
            if not isinstance(hook_entries, list):
                continue

            for entry in hook_entries:
                if not isinstance(entry, dict):
                    continue

                hook_paths = entry.get("hooks", [])
                if isinstance(hook_paths, str):
                    hook_paths = [hook_paths]

                for hook_path in hook_paths:
                    expanded = os.path.expanduser(hook_path)
                    path = Path(expanded)

                    if path.exists() and path.is_file():
                        mode = path.stat().st_mode
                        if not (mode & stat.S_IXUSR):
                            path.chmod(mode | stat.S_IXUSR)
                            fixed += 1

        return fixed


def validate_hooks(claude_dir: str | Path | None = None) -> HookValidationResult:
    """Convenience function to validate hooks.

    Args:
        claude_dir: Path to .claude directory

    Returns:
        HookValidationResult
    """
    validator = HookValidator(claude_dir)
    return validator.validate()


def ensure_hooks_executable(claude_dir: str | Path | None = None) -> int:
    """Ensure all hook scripts are executable.

    Args:
        claude_dir: Path to .claude directory

    Returns:
        Number of hooks fixed
    """
    validator = HookValidator(claude_dir)
    return validator.fix_permissions()
