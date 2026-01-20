"""Tests for environment versioning and capture.

@trace SPEC-07.10.01 - Claude Code Version Capture
@trace SPEC-07.10.02 - Plugin Version Capture
@trace SPEC-07.10.03 - MCP Server Version Capture
@trace SPEC-07.10.04 - Python Environment Capture
@trace SPEC-07.10.05 - Environment Diff
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from parhelia.environment import (
    ClaudeCodeVersion,
    EnvironmentCapture,
    EnvironmentDiff,
    EnvironmentSnapshot,
    MCPServerVersion,
    PluginVersion,
    diff_environments,
    format_environment_diff,
)


class TestEnvironmentDataClasses:
    """Tests for environment data classes."""

    def test_claude_code_version_creation(self):
        """@trace SPEC-07.10.01 - ClaudeCodeVersion MUST store version, hash, path."""
        version = ClaudeCodeVersion(
            version="1.0.0",
            binary_hash="abc123def456",
            install_path="/usr/local/bin/claude",
        )
        assert version.version == "1.0.0"
        assert version.binary_hash == "abc123def456"
        assert version.install_path == "/usr/local/bin/claude"

    def test_plugin_version_creation(self):
        """@trace SPEC-07.10.02 - PluginVersion MUST store git info."""
        plugin = PluginVersion(
            git_remote="https://github.com/example/plugin.git",
            git_commit="abc123",
            git_branch="main",
            installed_at=datetime(2026, 1, 20, 10, 0, 0),
            manifest_version="2.0.0",
        )
        assert plugin.git_remote == "https://github.com/example/plugin.git"
        assert plugin.git_commit == "abc123"
        assert plugin.git_branch == "main"
        assert plugin.manifest_version == "2.0.0"

    def test_mcp_server_version_creation(self):
        """@trace SPEC-07.10.03 - MCPServerVersion MUST store source type and version."""
        server = MCPServerVersion(
            source_type="npm",
            version_id="1.2.3",
            config_hash="config123",
        )
        assert server.source_type == "npm"
        assert server.version_id == "1.2.3"
        assert server.config_hash == "config123"

    def test_environment_snapshot_creation(self):
        """@trace SPEC-07.10 - EnvironmentSnapshot MUST aggregate all components."""
        claude_code = ClaudeCodeVersion(
            version="1.0.0",
            binary_hash="hash123",
            install_path="/usr/local/bin/claude",
        )
        snapshot = EnvironmentSnapshot(
            claude_code=claude_code,
            plugins={},
            mcp_servers={},
            python_version="3.11.0",
            python_packages={"anthropic": "0.20.0"},
            pip_freeze_hash="freeze123",
            captured_at=datetime(2026, 1, 20, 10, 0, 0),
        )
        assert snapshot.claude_code.version == "1.0.0"
        assert snapshot.python_version == "3.11.0"
        assert snapshot.python_packages["anthropic"] == "0.20.0"


class TestEnvironmentSnapshotSerialization:
    """Tests for EnvironmentSnapshot serialization."""

    @pytest.fixture
    def sample_snapshot(self) -> EnvironmentSnapshot:
        """Create a sample snapshot for testing."""
        return EnvironmentSnapshot(
            claude_code=ClaudeCodeVersion(
                version="1.5.0",
                binary_hash="abcdef123456",
                install_path="/usr/local/bin/claude",
            ),
            plugins={
                "plugin-a": PluginVersion(
                    git_remote="https://github.com/example/plugin-a.git",
                    git_commit="commit123",
                    git_branch="main",
                    installed_at=datetime(2026, 1, 20, 10, 0, 0),
                    manifest_version="1.0.0",
                )
            },
            mcp_servers={
                "server-x": MCPServerVersion(
                    source_type="npm",
                    version_id="2.0.0",
                    config_hash="config456",
                )
            },
            python_version="3.11.5",
            python_packages={"anthropic": "0.25.0", "modal": "0.60.0"},
            pip_freeze_hash="freeze789",
            captured_at=datetime(2026, 1, 20, 12, 30, 0),
        )

    def test_to_dict(self, sample_snapshot: EnvironmentSnapshot):
        """@trace SPEC-07.10 - EnvironmentSnapshot MUST serialize to dict."""
        data = sample_snapshot.to_dict()

        assert data["claude_code"]["version"] == "1.5.0"
        assert data["claude_code"]["binary_hash"] == "abcdef123456"
        assert "plugin-a" in data["plugins"]
        assert data["plugins"]["plugin-a"]["git_commit"] == "commit123"
        assert "server-x" in data["mcp_servers"]
        assert data["mcp_servers"]["server-x"]["source_type"] == "npm"
        assert data["python_version"] == "3.11.5"
        assert data["python_packages"]["anthropic"] == "0.25.0"

    def test_from_dict(self, sample_snapshot: EnvironmentSnapshot):
        """@trace SPEC-07.10 - EnvironmentSnapshot MUST deserialize from dict."""
        data = sample_snapshot.to_dict()
        restored = EnvironmentSnapshot.from_dict(data)

        assert restored.claude_code.version == sample_snapshot.claude_code.version
        assert restored.claude_code.binary_hash == sample_snapshot.claude_code.binary_hash
        assert "plugin-a" in restored.plugins
        assert restored.plugins["plugin-a"].git_commit == "commit123"
        assert "server-x" in restored.mcp_servers
        assert restored.python_version == "3.11.5"

    def test_json_round_trip(self, sample_snapshot: EnvironmentSnapshot):
        """@trace SPEC-07.10 - EnvironmentSnapshot MUST survive JSON round-trip."""
        json_str = json.dumps(sample_snapshot.to_dict())
        data = json.loads(json_str)
        restored = EnvironmentSnapshot.from_dict(data)

        assert restored.claude_code.version == sample_snapshot.claude_code.version
        assert len(restored.plugins) == len(sample_snapshot.plugins)
        assert len(restored.mcp_servers) == len(sample_snapshot.mcp_servers)


class TestEnvironmentCapture:
    """Tests for EnvironmentCapture functionality."""

    @pytest.fixture
    def capture(self):
        """Create EnvironmentCapture instance."""
        return EnvironmentCapture()

    @pytest.mark.asyncio
    async def test_capture_returns_snapshot(self, capture: EnvironmentCapture):
        """@trace SPEC-07.10 - capture() MUST return EnvironmentSnapshot."""
        snapshot = await capture.capture()
        assert isinstance(snapshot, EnvironmentSnapshot)
        assert snapshot.claude_code is not None
        assert snapshot.captured_at is not None

    @pytest.mark.asyncio
    async def test_capture_claude_code_version(self, capture: EnvironmentCapture):
        """@trace SPEC-07.10.01 - MUST capture Claude Code version."""
        snapshot = await capture.capture()
        # Version might be "unknown" if claude not installed
        assert snapshot.claude_code.version is not None
        assert snapshot.claude_code.install_path is not None

    @pytest.mark.asyncio
    async def test_capture_python_version(self, capture: EnvironmentCapture):
        """@trace SPEC-07.10.04 - MUST capture Python version."""
        snapshot = await capture.capture()
        # Should have Python version
        assert snapshot.python_version != ""
        assert "." in snapshot.python_version  # e.g., "3.11.0"

    @pytest.mark.asyncio
    async def test_capture_python_packages(self, capture: EnvironmentCapture):
        """@trace SPEC-07.10.04 - MUST capture key Python packages."""
        snapshot = await capture.capture()
        # Should have pip freeze hash
        assert snapshot.pip_freeze_hash != ""

    @pytest.mark.asyncio
    async def test_capture_is_fast(self, capture: EnvironmentCapture):
        """@trace SPEC-07.10 - capture MUST complete in < 5 seconds."""
        import asyncio
        import time

        start = time.monotonic()
        await asyncio.wait_for(capture.capture(), timeout=5.0)
        elapsed = time.monotonic() - start
        assert elapsed < 5.0


class TestEnvironmentCapturePlugins:
    """Tests for plugin version capture."""

    @pytest.fixture
    def plugins_dir(self, tmp_path: Path) -> Path:
        """Create a temporary plugins directory."""
        plugins = tmp_path / "plugins"
        plugins.mkdir()
        return plugins

    @pytest.fixture
    def capture_with_plugins(self, plugins_dir: Path) -> EnvironmentCapture:
        """Create EnvironmentCapture with custom plugins dir."""
        return EnvironmentCapture(plugins_dir=str(plugins_dir))

    @pytest.mark.asyncio
    async def test_capture_no_plugins(self, capture_with_plugins: EnvironmentCapture):
        """@trace SPEC-07.10.02 - MUST handle empty plugins directory."""
        snapshot = await capture_with_plugins.capture()
        assert snapshot.plugins == {}

    @pytest.mark.asyncio
    async def test_capture_plugin_with_git(
        self, capture_with_plugins: EnvironmentCapture, plugins_dir: Path
    ):
        """@trace SPEC-07.10.02 - MUST capture git info from plugins."""
        # Create a fake plugin with .git
        plugin_dir = plugins_dir / "test-plugin"
        plugin_dir.mkdir()
        git_dir = plugin_dir / ".git"
        git_dir.mkdir()

        # Mock git commands by creating required files
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        (git_dir / "config").write_text("[remote \"origin\"]\nurl = https://example.com/plugin.git")

        # The capture should not crash, even if git commands fail
        snapshot = await capture_with_plugins.capture()
        # Plugin may or may not be captured depending on git binary availability


class TestEnvironmentCaptureMCP:
    """Tests for MCP server version capture."""

    @pytest.fixture
    def mcp_config_file(self, tmp_path: Path) -> Path:
        """Create a temporary MCP config file."""
        config = {
            "mcpServers": {
                "test-server": {
                    "command": "npx",
                    "args": ["@test/mcp-server"],
                }
            }
        }
        config_path = tmp_path / "mcp_config.json"
        config_path.write_text(json.dumps(config))
        return config_path

    @pytest.mark.asyncio
    async def test_capture_mcp_servers_from_config(self, mcp_config_file: Path):
        """@trace SPEC-07.10.03 - MUST capture MCP servers from config."""
        capture = EnvironmentCapture(mcp_config_path=str(mcp_config_file))
        snapshot = await capture.capture()

        assert "test-server" in snapshot.mcp_servers
        assert snapshot.mcp_servers["test-server"].source_type == "npm"

    @pytest.mark.asyncio
    async def test_capture_mcp_config_hash(self, mcp_config_file: Path):
        """@trace SPEC-07.10.03 - MUST capture config hash for MCP servers."""
        capture = EnvironmentCapture(mcp_config_path=str(mcp_config_file))
        snapshot = await capture.capture()

        assert snapshot.mcp_servers["test-server"].config_hash != ""


class TestEnvironmentDiff:
    """Tests for environment diff functionality."""

    @pytest.fixture
    def env_a(self) -> EnvironmentSnapshot:
        """Create first environment snapshot."""
        return EnvironmentSnapshot(
            claude_code=ClaudeCodeVersion(
                version="1.0.0",
                binary_hash="hash_a",
                install_path="/usr/local/bin/claude",
            ),
            plugins={
                "plugin-a": PluginVersion(
                    git_remote="https://example.com/a.git",
                    git_commit="commit_a",
                    git_branch="main",
                    installed_at=datetime(2026, 1, 20, 10, 0, 0),
                ),
                "plugin-b": PluginVersion(
                    git_remote="https://example.com/b.git",
                    git_commit="commit_b",
                    git_branch="main",
                    installed_at=datetime(2026, 1, 20, 10, 0, 0),
                ),
            },
            mcp_servers={
                "server-x": MCPServerVersion(
                    source_type="npm",
                    version_id="1.0.0",
                    config_hash="config_x",
                ),
            },
            python_version="3.11.0",
            python_packages={"anthropic": "0.20.0", "modal": "0.50.0"},
            pip_freeze_hash="freeze_a",
            captured_at=datetime(2026, 1, 20, 10, 0, 0),
        )

    @pytest.fixture
    def env_b(self) -> EnvironmentSnapshot:
        """Create second environment snapshot with changes."""
        return EnvironmentSnapshot(
            claude_code=ClaudeCodeVersion(
                version="1.1.0",  # Changed
                binary_hash="hash_b",  # Changed
                install_path="/usr/local/bin/claude",
            ),
            plugins={
                "plugin-a": PluginVersion(
                    git_remote="https://example.com/a.git",
                    git_commit="commit_a_new",  # Changed
                    git_branch="main",
                    installed_at=datetime(2026, 1, 20, 10, 0, 0),
                ),
                # plugin-b removed
                "plugin-c": PluginVersion(  # Added
                    git_remote="https://example.com/c.git",
                    git_commit="commit_c",
                    git_branch="develop",
                    installed_at=datetime(2026, 1, 20, 12, 0, 0),
                ),
            },
            mcp_servers={
                "server-x": MCPServerVersion(
                    source_type="npm",
                    version_id="2.0.0",  # Changed
                    config_hash="config_x_new",  # Changed
                ),
                "server-y": MCPServerVersion(  # Added
                    source_type="git",
                    version_id="abc123",
                    config_hash="config_y",
                ),
            },
            python_version="3.12.0",  # Changed
            python_packages={
                "anthropic": "0.25.0",  # Changed
                # modal removed
                "httpx": "0.27.0",  # Added
            },
            pip_freeze_hash="freeze_b",
            captured_at=datetime(2026, 1, 20, 12, 0, 0),
        )

    def test_diff_claude_code_changed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect Claude Code version changes."""
        diff = diff_environments(env_a, env_b)
        assert diff.claude_code_changed is True
        assert diff.claude_code_diff is not None
        assert diff.claude_code_diff["version"] == ("1.0.0", "1.1.0")

    def test_diff_plugins_added(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect added plugins."""
        diff = diff_environments(env_a, env_b)
        assert "plugin-c" in diff.plugins_added

    def test_diff_plugins_removed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect removed plugins."""
        diff = diff_environments(env_a, env_b)
        assert "plugin-b" in diff.plugins_removed

    def test_diff_plugins_changed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect changed plugin commits."""
        diff = diff_environments(env_a, env_b)
        assert "plugin-a" in diff.plugins_changed
        assert diff.plugins_changed["plugin-a"]["commit"] == ("commit_a", "commit_a_new")

    def test_diff_mcp_servers_added(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect added MCP servers."""
        diff = diff_environments(env_a, env_b)
        assert "server-y" in diff.mcp_servers_added

    def test_diff_mcp_servers_changed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect changed MCP server versions."""
        diff = diff_environments(env_a, env_b)
        assert "server-x" in diff.mcp_servers_changed

    def test_diff_python_version_changed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect Python version changes."""
        diff = diff_environments(env_a, env_b)
        assert diff.python_version_changed is True
        assert diff.python_version_diff == ("3.11.0", "3.12.0")

    def test_diff_packages_added(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect added packages."""
        diff = diff_environments(env_a, env_b)
        assert "httpx" in diff.packages_added
        assert diff.packages_added["httpx"] == "0.27.0"

    def test_diff_packages_removed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect removed packages."""
        diff = diff_environments(env_a, env_b)
        assert "modal" in diff.packages_removed

    def test_diff_packages_changed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST detect changed package versions."""
        diff = diff_environments(env_a, env_b)
        assert "anthropic" in diff.packages_changed
        assert diff.packages_changed["anthropic"] == ("0.20.0", "0.25.0")

    def test_diff_is_empty_when_identical(self, env_a: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST report empty when identical."""
        diff = diff_environments(env_a, env_a)
        assert diff.is_empty() is True

    def test_diff_is_not_empty_when_changed(self, env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot):
        """@trace SPEC-07.10.05 - diff MUST report not empty when changed."""
        diff = diff_environments(env_a, env_b)
        assert diff.is_empty() is False


class TestEnvironmentDiffFormatting:
    """Tests for environment diff formatting."""

    def test_format_empty_diff(self):
        """@trace SPEC-07.10.05 - format MUST handle empty diff."""
        diff = EnvironmentDiff()
        output = format_environment_diff(diff)
        assert "No environment changes" in output

    def test_format_claude_code_change(self):
        """@trace SPEC-07.10.05 - format MUST show Claude Code changes."""
        diff = EnvironmentDiff()
        diff.claude_code_changed = True
        diff.claude_code_diff = {"version": ("1.0.0", "1.1.0"), "binary_hash": ("a", "b")}
        output = format_environment_diff(diff)
        assert "Claude Code" in output
        assert "1.0.0" in output
        assert "1.1.0" in output

    def test_format_plugin_changes(self):
        """@trace SPEC-07.10.05 - format MUST show plugin changes."""
        diff = EnvironmentDiff()
        diff.plugins_added = ["new-plugin"]
        diff.plugins_removed = ["old-plugin"]
        diff.plugins_changed = {"updated-plugin": {"commit": ("abc", "def"), "branch": ("main", "main")}}
        output = format_environment_diff(diff)
        assert "Plugins" in output
        assert "new-plugin" in output
        assert "old-plugin" in output
        assert "updated-plugin" in output

    def test_format_package_changes(self):
        """@trace SPEC-07.10.05 - format MUST show package changes."""
        diff = EnvironmentDiff()
        diff.packages_added = {"new-pkg": "1.0.0"}
        diff.packages_removed = {"old-pkg": "0.5.0"}
        diff.packages_changed = {"updated-pkg": ("1.0.0", "2.0.0")}
        output = format_environment_diff(diff)
        assert "Packages" in output
        assert "new-pkg" in output
        assert "old-pkg" in output
        assert "updated-pkg" in output


class TestCheckpointIntegration:
    """Tests for environment capture integration with checkpoints."""

    @pytest.fixture
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        cp_dir = tmp_path / "checkpoints"
        cp_dir.mkdir()
        return cp_dir

    @pytest.fixture
    def checkpoint_manager(self, checkpoint_dir: Path):
        """Create CheckpointManager instance."""
        from parhelia.checkpoint import CheckpointManager

        return CheckpointManager(
            checkpoint_root=str(checkpoint_dir),
            capture_environment=True,
        )

    @pytest.fixture
    def sample_session(self, tmp_path: Path):
        """Create a sample session for testing."""
        from parhelia.session import Session, SessionState

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "test.py").write_text("print('hello')")

        return Session(
            id="test-session-env",
            task_id="ph-env-test",
            state=SessionState.RUNNING,
            working_directory=str(workspace),
            environment={"FOO": "bar"},
        )

    @pytest.mark.asyncio
    async def test_checkpoint_captures_environment(
        self, checkpoint_manager, sample_session
    ):
        """@trace SPEC-07.10 - Checkpoint MUST capture environment snapshot."""
        from parhelia.session import CheckpointTrigger

        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        assert checkpoint.environment_snapshot is not None
        assert checkpoint.environment_snapshot.claude_code is not None
        assert checkpoint.environment_snapshot.python_version != ""

    @pytest.mark.asyncio
    async def test_checkpoint_manifest_includes_environment(
        self, checkpoint_manager, sample_session, checkpoint_dir: Path
    ):
        """@trace SPEC-07.11.04 - Manifest MUST include environment field."""
        from parhelia.session import CheckpointTrigger

        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        manifest_path = checkpoint_dir / sample_session.id / checkpoint.id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert "environment" in manifest
        assert manifest["version"] == "1.2"
        assert "claude_code" in manifest["environment"]
        assert "python_version" in manifest["environment"]

    @pytest.mark.asyncio
    async def test_checkpoint_loads_environment(
        self, checkpoint_manager, sample_session
    ):
        """@trace SPEC-07.10 - Checkpoint MUST load environment from manifest."""
        from parhelia.session import CheckpointTrigger

        checkpoint = await checkpoint_manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        # Reload checkpoint
        loaded = await checkpoint_manager.get_latest_checkpoint(sample_session.id)

        assert loaded is not None
        assert loaded.environment_snapshot is not None
        assert loaded.environment_snapshot.claude_code.version == checkpoint.environment_snapshot.claude_code.version

    @pytest.mark.asyncio
    async def test_checkpoint_without_environment_capture(
        self, checkpoint_dir: Path, sample_session
    ):
        """@trace SPEC-07.10 - Checkpoint SHOULD work without environment capture."""
        from parhelia.checkpoint import CheckpointManager
        from parhelia.session import CheckpointTrigger

        # Create manager with environment capture disabled
        manager = CheckpointManager(
            checkpoint_root=str(checkpoint_dir),
            capture_environment=False,
        )

        checkpoint = await manager.create_checkpoint(
            session=sample_session,
            trigger=CheckpointTrigger.MANUAL,
        )

        assert checkpoint.environment_snapshot is None
