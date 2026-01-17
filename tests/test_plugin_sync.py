"""Tests for plugin synchronization.

@trace SPEC-07.10 - Plugin Discovery
@trace SPEC-07.11 - Plugin Sync Strategy
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_plugin_info_creation(self):
        """@trace SPEC-07.10 - PluginInfo MUST capture plugin metadata."""
        from parhelia.plugin_sync import PluginInfo

        plugin = PluginInfo(
            name="test-plugin",
            path="/home/user/.claude/plugins/test-plugin",
            is_symlink=False,
            git_remote="https://github.com/user/test-plugin.git",
            git_branch="main",
            has_build_step=False,
            dependencies=[],
        )

        assert plugin.name == "test-plugin"
        assert plugin.git_remote is not None


class TestSyncStrategy:
    """Tests for sync strategy determination."""

    def test_git_clone_strategy_for_git_repo(self):
        """@trace SPEC-07.11 - Plugins with git remote SHOULD use GIT_CLONE strategy."""
        from parhelia.plugin_sync import PluginInfo, PluginSyncManager, SyncStrategy

        plugin = PluginInfo(
            name="test-plugin",
            path="/path/to/plugin",
            is_symlink=False,
            git_remote="https://github.com/user/plugin.git",
            git_branch="main",
            has_build_step=False,
            dependencies=[],
        )

        manager = PluginSyncManager(volume_path="/tmp/test-vol")
        strategy = manager.determine_strategy(plugin)

        assert strategy == SyncStrategy.GIT_CLONE

    def test_copy_strategy_for_local_plugin(self):
        """@trace SPEC-07.11 - Local plugins without git SHOULD use COPY strategy."""
        from parhelia.plugin_sync import PluginInfo, PluginSyncManager, SyncStrategy

        plugin = PluginInfo(
            name="local-plugin",
            path="/path/to/local/plugin",
            is_symlink=False,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )

        manager = PluginSyncManager(volume_path="/tmp/test-vol")
        strategy = manager.determine_strategy(plugin)

        assert strategy == SyncStrategy.COPY

    def test_skip_strategy_for_inaccessible_symlink(self, tmp_path: Path):
        """@trace SPEC-07.11 - Inaccessible symlinks SHOULD use SKIP strategy."""
        from parhelia.plugin_sync import PluginInfo, PluginSyncManager, SyncStrategy

        # Create a broken symlink
        broken_link = tmp_path / "broken-plugin"
        broken_link.symlink_to("/nonexistent/path")

        plugin = PluginInfo(
            name="broken-plugin",
            path=str(broken_link),
            is_symlink=True,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )

        manager = PluginSyncManager(volume_path="/tmp/test-vol")
        strategy = manager.determine_strategy(plugin)

        assert strategy == SyncStrategy.SKIP


class TestPluginDiscovery:
    """Tests for plugin discovery."""

    def test_discover_plugins_from_directory(self, tmp_path: Path):
        """@trace SPEC-07.10 - Discovery MUST find plugins in ~/.claude/plugins/."""
        from parhelia.plugin_sync import PluginDiscovery

        # Setup mock claude directory
        claude_dir = tmp_path / ".claude"
        plugins_dir = claude_dir / "plugins"
        plugins_dir.mkdir(parents=True)

        # Create a mock plugin
        plugin_dir = plugins_dir / "test-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "SKILL.md").write_text("# Test Plugin")

        discovery = PluginDiscovery(claude_dir=str(claude_dir))
        plugins = discovery.discover_all_sync()

        assert len(plugins) >= 1
        assert any(p.name == "test-plugin" for p in plugins)

    def test_discover_skills_from_directory(self, tmp_path: Path):
        """@trace SPEC-07.10 - Discovery MUST find skills in ~/.claude/skills/."""
        from parhelia.plugin_sync import PluginDiscovery

        # Setup mock claude directory
        claude_dir = tmp_path / ".claude"
        skills_dir = claude_dir / "skills"
        skills_dir.mkdir(parents=True)

        # Create a mock skill
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Test Skill")

        discovery = PluginDiscovery(claude_dir=str(claude_dir))
        plugins = discovery.discover_all_sync()

        assert len(plugins) >= 1
        assert any(p.name == "test-skill" for p in plugins)


class TestPluginSyncManager:
    """Tests for plugin sync manager."""

    def test_sync_manager_initialization(self, tmp_path: Path):
        """@trace SPEC-07.11 - SyncManager MUST initialize with volume path."""
        from parhelia.plugin_sync import PluginSyncManager

        manager = PluginSyncManager(volume_path=str(tmp_path))

        assert manager.volume_path == tmp_path
        assert manager.plugins_path == tmp_path / "plugins"

    @pytest.mark.asyncio
    async def test_sync_creates_target_directory(self, tmp_path: Path):
        """@trace SPEC-07.11 - Sync MUST create target directory if missing."""
        from parhelia.plugin_sync import PluginInfo, PluginSyncManager

        # Create source plugin
        source_dir = tmp_path / "source" / "my-plugin"
        source_dir.mkdir(parents=True)
        (source_dir / "SKILL.md").write_text("# My Plugin")

        plugin = PluginInfo(
            name="my-plugin",
            path=str(source_dir),
            is_symlink=False,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )

        volume_path = tmp_path / "volume"
        manager = PluginSyncManager(volume_path=str(volume_path))

        result = await manager.sync_plugin(plugin)

        assert result.success
        assert (volume_path / "plugins" / "my-plugin").exists()


class TestSyncResult:
    """Tests for sync result."""

    def test_sync_result_success(self):
        """@trace SPEC-07.11 - SyncResult MUST indicate success/failure."""
        from parhelia.plugin_sync import SyncResult

        result = SyncResult(
            plugin="test-plugin",
            success=True,
            strategy="copy",
        )

        assert result.success
        assert result.plugin == "test-plugin"

    def test_sync_result_failure_with_reason(self):
        """@trace SPEC-07.11 - Failed SyncResult MUST include reason."""
        from parhelia.plugin_sync import SyncResult

        result = SyncResult(
            plugin="test-plugin",
            success=False,
            reason="Git clone failed",
        )

        assert not result.success
        assert "Git clone failed" in result.reason
