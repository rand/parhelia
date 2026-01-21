"""Tests for Claude Code hooks validation."""

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from parhelia.hook_validator import (
    HookIssue,
    HookValidationResult,
    HookValidator,
    ensure_hooks_executable,
    validate_hooks,
)


@pytest.fixture
def temp_claude_dir():
    """Create a temporary .claude directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir = Path(tmpdir) / ".claude"
        claude_dir.mkdir()
        yield claude_dir


@pytest.fixture
def hook_script(temp_claude_dir):
    """Create a valid hook script."""
    hooks_dir = temp_claude_dir / "hooks"
    hooks_dir.mkdir()
    script = hooks_dir / "test-hook.sh"
    script.write_text("#!/bin/bash\necho 'hook ran'\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class TestHookValidationResult:
    def test_empty_result(self):
        result = HookValidationResult()
        assert result.hooks_found == 0
        assert result.hooks_valid == 0
        assert result.all_valid
        assert result.summary() == "No hooks configured"

    def test_all_valid(self):
        result = HookValidationResult(hooks_found=3, hooks_valid=3)
        assert result.all_valid
        assert result.summary() == "All 3 hooks valid"

    def test_some_invalid(self):
        result = HookValidationResult(
            hooks_found=3,
            hooks_valid=2,
            issues=[HookIssue("PreToolUse", "/path", "not found")],
        )
        assert not result.all_valid
        assert "2/3" in result.summary()
        assert "1 issues" in result.summary()


class TestHookValidator:
    def test_no_settings_file(self, temp_claude_dir):
        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 0
        assert result.all_valid

    def test_empty_hooks_config(self, temp_claude_dir):
        settings = temp_claude_dir / "settings.json"
        settings.write_text('{"hooks": {}}')

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 0
        assert result.all_valid

    def test_valid_hook(self, temp_claude_dir, hook_script):
        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [str(hook_script)]}
                        ]
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 1
        assert result.hooks_valid == 1
        assert result.all_valid

    def test_missing_hook_script(self, temp_claude_dir):
        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": ["/nonexistent/hook.sh"]}
                        ]
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 1
        assert result.hooks_valid == 0
        assert not result.all_valid
        assert len(result.issues) == 1
        assert "not found" in result.issues[0].issue

    def test_non_executable_hook(self, temp_claude_dir):
        hooks_dir = temp_claude_dir / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "test-hook.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")
        # Don't set executable bit

        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PostToolUse": [{"matcher": ".*", "hooks": [str(script)]}]
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 1
        assert result.hooks_valid == 0
        assert any("not executable" in i.issue for i in result.issues)

    def test_missing_shebang(self, temp_claude_dir):
        hooks_dir = temp_claude_dir / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "test-hook.sh"
        script.write_text("echo 'no shebang'\n")  # No #! line
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [str(script)]}]}}
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert any("shebang" in i.issue for i in result.issues)

    def test_multiple_hooks(self, temp_claude_dir, hook_script):
        # Create second hook
        hooks_dir = temp_claude_dir / "hooks"
        script2 = hooks_dir / "hook2.sh"
        script2.write_text("#!/bin/bash\necho 'hook2'\n")
        script2.chmod(script2.stat().st_mode | stat.S_IXUSR)

        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [str(hook_script)]}
                        ],
                        "PostToolUse": [{"matcher": ".*", "hooks": [str(script2)]}],
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 2
        assert result.hooks_valid == 2

    def test_settings_local_json(self, temp_claude_dir, hook_script):
        # Use settings.local.json instead
        settings = temp_claude_dir / "settings.local.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [str(hook_script)]}
                        ]
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert result.hooks_found == 1
        assert result.hooks_valid == 1

    def test_tilde_expansion(self, temp_claude_dir):
        # Test that ~ paths are expanded
        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": ".*", "hooks": ["~/.claude/hooks/test.sh"]}
                        ]
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        # Should fail because the file doesn't exist, but path should be expanded
        assert result.hooks_found == 1
        assert not result.all_valid
        # Check that the issue mentions the expanded path
        assert os.path.expanduser("~") in result.issues[0].issue

    def test_fix_permissions(self, temp_claude_dir):
        hooks_dir = temp_claude_dir / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "test-hook.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")
        # Don't set executable bit initially

        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [str(script)]}]}}
            )
        )

        validator = HookValidator(temp_claude_dir)

        # Initially not executable
        result = validator.validate()
        assert not result.all_valid

        # Fix permissions
        fixed = validator.fix_permissions()
        assert fixed == 1

        # Now should be valid
        result = validator.validate()
        assert result.all_valid

    def test_hook_is_directory(self, temp_claude_dir):
        hooks_dir = temp_claude_dir / "hooks"
        hooks_dir.mkdir()
        bad_hook = hooks_dir / "not-a-script"
        bad_hook.mkdir()  # It's a directory, not a file

        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [{"matcher": ".*", "hooks": [str(bad_hook)]}]
                    }
                }
            )
        )

        validator = HookValidator(temp_claude_dir)
        result = validator.validate()
        assert not result.all_valid
        assert any("not a file" in i.issue for i in result.issues)


class TestConvenienceFunctions:
    def test_validate_hooks(self, temp_claude_dir, hook_script):
        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [str(hook_script)]}
                        ]
                    }
                }
            )
        )

        result = validate_hooks(temp_claude_dir)
        assert result.all_valid

    def test_ensure_hooks_executable(self, temp_claude_dir):
        hooks_dir = temp_claude_dir / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "test-hook.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")

        settings = temp_claude_dir / "settings.json"
        settings.write_text(
            json.dumps(
                {"hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [str(script)]}]}}
            )
        )

        fixed = ensure_hooks_executable(temp_claude_dir)
        assert fixed == 1
        assert os.access(script, os.X_OK)
