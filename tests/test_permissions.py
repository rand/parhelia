"""Tests for permission model enforcement.

@trace SPEC-04.13 - Permission Model for Remote Execution
"""

import pytest


class TestRemotePermissions:
    """Tests for RemotePermissions dataclass - SPEC-04.13."""

    def test_permissions_default_allowed_tools(self):
        """@trace SPEC-04.13 - RemotePermissions MUST have default allowed tools."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert "Read" in perms.allowed_tools
        assert "Write" in perms.allowed_tools
        assert "Edit" in perms.allowed_tools
        assert "Bash" in perms.allowed_tools
        assert "Glob" in perms.allowed_tools
        assert "Grep" in perms.allowed_tools

    def test_permissions_default_denied_tools(self):
        """@trace SPEC-04.13 - RemotePermissions SHOULD have default denied tools."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert "WebFetch" in perms.denied_tools

    def test_permissions_bash_defaults(self):
        """@trace SPEC-04.13 - RemotePermissions MUST have safe bash defaults."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert perms.bash_allow_network is True
        assert perms.bash_allow_sudo is False

    def test_permissions_blocked_commands(self):
        """@trace SPEC-04.13 - RemotePermissions MUST block dangerous commands."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert len(perms.bash_blocked_commands) > 0
        assert any("rm -rf /" in cmd for cmd in perms.bash_blocked_commands)

    def test_permissions_allowed_paths(self):
        """@trace SPEC-04.13 - RemotePermissions MUST define allowed paths."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert "/vol/parhelia/workspaces" in perms.allowed_paths
        assert "/tmp" in perms.allowed_paths

    def test_permissions_denied_paths(self):
        """@trace SPEC-04.13 - RemotePermissions MUST deny sensitive paths."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert "/etc/shadow" in perms.denied_paths
        assert "/etc/passwd" in perms.denied_paths

    def test_permissions_allowed_domains(self):
        """@trace SPEC-04.13 - RemotePermissions MUST define allowed domains."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions()

        assert "api.anthropic.com" in perms.allowed_domains
        assert "github.com" in perms.allowed_domains

    def test_permissions_custom_allowed_tools(self):
        """@trace SPEC-04.13 - RemotePermissions SHOULD support custom tools."""
        from parhelia.permissions import RemotePermissions

        perms = RemotePermissions(
            allowed_tools=["Read", "Bash"],
            denied_tools=["Write", "Edit"],
        )

        assert perms.allowed_tools == ["Read", "Bash"]
        assert perms.denied_tools == ["Write", "Edit"]


class TestCommandBuilder:
    """Tests for build_claude_command - SPEC-04.13."""

    def test_build_command_basic(self):
        """@trace SPEC-04.13 - build_claude_command MUST include prompt."""
        from parhelia.permissions import RemotePermissions, build_claude_command

        perms = RemotePermissions()
        cmd = build_claude_command(
            prompt="Fix the bug",
            permissions=perms,
        )

        assert "claude" in cmd
        assert "-p" in cmd
        assert "Fix the bug" in cmd

    def test_build_command_stream_json(self):
        """@trace SPEC-04.13 - build_claude_command MUST use stream-json output."""
        from parhelia.permissions import RemotePermissions, build_claude_command

        perms = RemotePermissions()
        cmd = build_claude_command(prompt="test", permissions=perms)

        assert "--output-format" in cmd
        assert "stream-json" in cmd

    def test_build_command_allowed_tools(self):
        """@trace SPEC-04.13 - build_claude_command MUST include allowed tools."""
        from parhelia.permissions import RemotePermissions, build_claude_command

        perms = RemotePermissions(allowed_tools=["Read", "Bash"])
        cmd = build_claude_command(prompt="test", permissions=perms)

        assert "--allowedTools" in cmd
        # Tools should be comma-separated
        idx = cmd.index("--allowedTools")
        assert "Read" in cmd[idx + 1] or "Bash" in cmd[idx + 1]

    def test_build_command_skip_permissions_when_automated(self):
        """@trace SPEC-04.13 - build_claude_command SHOULD skip permissions for automated tasks."""
        from parhelia.permissions import (
            RemotePermissions,
            TrustLevel,
            build_claude_command,
        )

        perms = RemotePermissions()
        cmd = build_claude_command(
            prompt="test",
            permissions=perms,
            trust_level=TrustLevel.AUTOMATED,
        )

        assert "--dangerously-skip-permissions" in cmd

    def test_build_command_no_skip_for_interactive(self):
        """@trace SPEC-04.13 - build_claude_command MUST NOT skip permissions for interactive."""
        from parhelia.permissions import (
            RemotePermissions,
            TrustLevel,
            build_claude_command,
        )

        perms = RemotePermissions()
        cmd = build_claude_command(
            prompt="test",
            permissions=perms,
            trust_level=TrustLevel.INTERACTIVE,
        )

        assert "--dangerously-skip-permissions" not in cmd


class TestCommandValidator:
    """Tests for command validation - SPEC-04.13."""

    def test_validate_command_allows_safe(self):
        """@trace SPEC-04.13 - validate_command MUST allow safe commands."""
        from parhelia.permissions import RemotePermissions, validate_command

        perms = RemotePermissions()

        assert validate_command("ls -la", perms) is True
        assert validate_command("git status", perms) is True
        assert validate_command("npm install", perms) is True

    def test_validate_command_blocks_dangerous(self):
        """@trace SPEC-04.13 - validate_command MUST block dangerous commands."""
        from parhelia.permissions import RemotePermissions, validate_command

        perms = RemotePermissions()

        assert validate_command("rm -rf /", perms) is False
        assert validate_command("sudo rm -rf /", perms) is False

    def test_validate_command_blocks_sudo_when_disabled(self):
        """@trace SPEC-04.13 - validate_command MUST block sudo when disabled."""
        from parhelia.permissions import RemotePermissions, validate_command

        perms = RemotePermissions(bash_allow_sudo=False)

        assert validate_command("sudo apt install", perms) is False
        assert validate_command("apt install", perms) is True

    def test_validate_command_allows_sudo_when_enabled(self):
        """@trace SPEC-04.13 - validate_command SHOULD allow sudo when enabled."""
        from parhelia.permissions import RemotePermissions, validate_command

        perms = RemotePermissions(bash_allow_sudo=True)

        assert validate_command("sudo apt install git", perms) is True


class TestPathValidator:
    """Tests for path validation - SPEC-04.13."""

    def test_validate_path_allows_workspace(self):
        """@trace SPEC-04.13 - validate_path MUST allow workspace paths."""
        from parhelia.permissions import RemotePermissions, validate_path

        perms = RemotePermissions()

        assert validate_path("/vol/parhelia/workspaces/project", perms) is True
        assert validate_path("/tmp/build", perms) is True

    def test_validate_path_denies_sensitive(self):
        """@trace SPEC-04.13 - validate_path MUST deny sensitive paths."""
        from parhelia.permissions import RemotePermissions, validate_path

        perms = RemotePermissions()

        assert validate_path("/etc/shadow", perms) is False
        assert validate_path("/etc/passwd", perms) is False

    def test_validate_path_custom_allowed(self):
        """@trace SPEC-04.13 - validate_path SHOULD respect custom allowed paths."""
        from parhelia.permissions import RemotePermissions, validate_path

        perms = RemotePermissions(allowed_paths=["/custom/path"])

        assert validate_path("/custom/path/file.txt", perms) is True
        assert validate_path("/other/path", perms) is False
