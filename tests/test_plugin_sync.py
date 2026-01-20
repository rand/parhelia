"""Tests for plugin synchronization.

@trace SPEC-07.10 - Plugin Discovery
@trace SPEC-07.11 - Plugin Sync Strategy
@trace SPEC-08.16 - CAS Plugin Integration
"""

from pathlib import Path

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


# =============================================================================
# [SPEC-08.16] CAS Plugin Integration Tests
# =============================================================================


class TestPluginManifest:
    """Tests for PluginManifest dataclass."""

    def test_manifest_creation(self):
        """@trace SPEC-08.16 - Manifest MUST capture plugin metadata."""
        from parhelia.cas import Digest
        from parhelia.plugin_sync import PluginManifest

        manifest = PluginManifest(
            name="test-plugin",
            version="abc123",
            root_digest=Digest(hash="a" * 64, size_bytes=1000),
            git_remote="https://github.com/user/plugin.git",
            git_branch="main",
            has_build_step=True,
            dependencies=["npm"],
            file_count=50,
            total_size_bytes=102400,
        )

        assert manifest.name == "test-plugin"
        assert manifest.version == "abc123"
        assert manifest.file_count == 50

    def test_manifest_to_dict(self):
        """@trace SPEC-08.16 - Manifest MUST serialize to dict."""
        from parhelia.cas import Digest
        from parhelia.plugin_sync import PluginManifest

        manifest = PluginManifest(
            name="test-plugin",
            version="abc123",
            root_digest=Digest(hash="b" * 64, size_bytes=500),
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
            file_count=10,
            total_size_bytes=5000,
        )

        data = manifest.to_dict()

        assert data["name"] == "test-plugin"
        assert data["root_digest"]["hash"] == "b" * 64

    def test_manifest_from_dict(self):
        """@trace SPEC-08.16 - Manifest MUST deserialize from dict."""
        from parhelia.plugin_sync import PluginManifest

        data = {
            "name": "restored-plugin",
            "version": "xyz789",
            "root_digest": {"hash": "c" * 64, "size_bytes": 2000},
            "git_remote": "https://github.com/test/repo.git",
            "git_branch": "develop",
            "has_build_step": True,
            "dependencies": ["pip", "npm"],
            "file_count": 100,
            "total_size_bytes": 50000,
        }

        manifest = PluginManifest.from_dict(data)

        assert manifest.name == "restored-plugin"
        assert manifest.version == "xyz789"
        assert manifest.root_digest.hash == "c" * 64


class TestCASyncResult:
    """Tests for CASyncResult dataclass."""

    def test_casync_result_success(self):
        """@trace SPEC-08.16 - CASyncResult MUST indicate success."""
        from parhelia.cas import Digest
        from parhelia.plugin_sync import CASyncResult, PluginManifest

        manifest = PluginManifest(
            name="plugin",
            version="v1",
            root_digest=Digest(hash="d" * 64, size_bytes=100),
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
            file_count=5,
            total_size_bytes=1000,
        )

        result = CASyncResult(
            plugin="plugin",
            success=True,
            manifest=manifest,
        )

        assert result.success
        assert result.manifest is not None

    def test_casync_result_failure(self):
        """@trace SPEC-08.16 - CASyncResult MUST include failure reason."""
        from parhelia.plugin_sync import CASyncResult

        result = CASyncResult(
            plugin="plugin",
            success=False,
            reason="Source not found",
        )

        assert not result.success
        assert "Source not found" in result.reason


class TestPluginCASManager:
    """Tests for PluginCASManager."""

    @pytest.fixture
    def cas_setup(self, tmp_path: Path):
        """Set up CAS and manager for testing."""
        from parhelia.cas import ContentAddressableStorage
        from parhelia.plugin_sync import PluginCASManager

        cas_root = tmp_path / "cas"
        cas_root.mkdir()
        cas = ContentAddressableStorage(root_path=str(cas_root))

        manifest_path = tmp_path / "manifests"
        manager = PluginCASManager(cas=cas, manifest_path=str(manifest_path))

        return cas, manager, tmp_path

    @pytest.mark.asyncio
    async def test_sync_plugin_stores_in_cas(self, cas_setup):
        """@trace SPEC-08.16 - sync_plugin MUST store content in CAS."""
        cas, manager, tmp_path = cas_setup
        from parhelia.plugin_sync import PluginInfo

        # Create source plugin
        source_dir = tmp_path / "source" / "my-plugin"
        source_dir.mkdir(parents=True)
        (source_dir / "SKILL.md").write_text("# My Plugin\nDescription here.")
        (source_dir / "main.py").write_text("print('hello')")

        plugin = PluginInfo(
            name="my-plugin",
            path=str(source_dir),
            is_symlink=False,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )

        result = await manager.sync_plugin(plugin)

        assert result.success
        assert result.manifest is not None
        assert result.manifest.name == "my-plugin"
        assert result.manifest.file_count == 2
        assert await cas.contains(result.manifest.root_digest)

    @pytest.mark.asyncio
    async def test_materialize_plugin_restores_files(self, cas_setup):
        """@trace SPEC-08.16 - materialize_plugin MUST restore from CAS."""
        cas, manager, tmp_path = cas_setup
        from parhelia.plugin_sync import PluginInfo

        # Create and sync source plugin
        source_dir = tmp_path / "source" / "restore-test"
        source_dir.mkdir(parents=True)
        (source_dir / "config.json").write_text('{"key": "value"}')
        (source_dir / "lib").mkdir()
        (source_dir / "lib" / "utils.py").write_text("def helper(): pass")

        plugin = PluginInfo(
            name="restore-test",
            path=str(source_dir),
            is_symlink=False,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )

        await manager.sync_plugin(plugin)

        # Materialize to new location
        target_dir = tmp_path / "restored"
        success = await manager.materialize_plugin("restore-test", str(target_dir))

        assert success
        assert (target_dir / "restore-test" / "config.json").exists()
        assert (target_dir / "restore-test" / "lib" / "utils.py").exists()

        # Verify content
        restored_config = (target_dir / "restore-test" / "config.json").read_text()
        assert '{"key": "value"}' in restored_config

    @pytest.mark.asyncio
    async def test_get_manifest(self, cas_setup):
        """@trace SPEC-08.16 - get_manifest MUST return stored manifest."""
        cas, manager, tmp_path = cas_setup
        from parhelia.plugin_sync import PluginInfo

        # Create and sync plugin
        source_dir = tmp_path / "source" / "manifest-test"
        source_dir.mkdir(parents=True)
        (source_dir / "README.md").write_text("# Test")

        plugin = PluginInfo(
            name="manifest-test",
            path=str(source_dir),
            is_symlink=False,
            git_remote="https://github.com/test/repo.git",
            git_branch="main",
            has_build_step=False,
            dependencies=[],
        )

        await manager.sync_plugin(plugin)

        manifest = await manager.get_manifest("manifest-test")

        assert manifest is not None
        assert manifest.name == "manifest-test"
        assert manifest.git_remote == "https://github.com/test/repo.git"

    @pytest.mark.asyncio
    async def test_list_plugins(self, cas_setup):
        """@trace SPEC-08.16 - list_plugins MUST return all stored plugins."""
        cas, manager, tmp_path = cas_setup
        from parhelia.plugin_sync import PluginInfo

        # Create and sync multiple plugins
        for name in ["plugin-a", "plugin-b", "plugin-c"]:
            source_dir = tmp_path / "source" / name
            source_dir.mkdir(parents=True)
            (source_dir / "index.js").write_text(f"// {name}")

            plugin = PluginInfo(
                name=name,
                path=str(source_dir),
                is_symlink=False,
                git_remote=None,
                git_branch=None,
                has_build_step=False,
                dependencies=[],
            )
            await manager.sync_plugin(plugin)

        plugins = await manager.list_plugins()

        assert len(plugins) == 3
        names = [p.name for p in plugins]
        assert "plugin-a" in names
        assert "plugin-b" in names
        assert "plugin-c" in names

    @pytest.mark.asyncio
    async def test_plugin_exists(self, cas_setup):
        """@trace SPEC-08.16 - plugin_exists MUST check CAS."""
        cas, manager, tmp_path = cas_setup
        from parhelia.plugin_sync import PluginInfo

        # Initially doesn't exist
        assert not await manager.plugin_exists("nonexistent")

        # Create and sync
        source_dir = tmp_path / "source" / "exists-test"
        source_dir.mkdir(parents=True)
        (source_dir / "file.txt").write_text("content")

        plugin = PluginInfo(
            name="exists-test",
            path=str(source_dir),
            is_symlink=False,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )
        await manager.sync_plugin(plugin)

        # Now exists
        assert await manager.plugin_exists("exists-test")

    @pytest.mark.asyncio
    async def test_sync_nonexistent_path_fails(self, cas_setup):
        """@trace SPEC-08.16 - sync_plugin MUST fail for nonexistent path."""
        cas, manager, tmp_path = cas_setup
        from parhelia.plugin_sync import PluginInfo

        plugin = PluginInfo(
            name="missing-plugin",
            path="/nonexistent/path/to/plugin",
            is_symlink=False,
            git_remote=None,
            git_branch=None,
            has_build_step=False,
            dependencies=[],
        )

        result = await manager.sync_plugin(plugin)

        assert not result.success
        assert "not exist" in result.reason

    @pytest.mark.asyncio
    async def test_materialize_nonexistent_plugin_fails(self, cas_setup):
        """@trace SPEC-08.16 - materialize_plugin MUST fail for unknown plugin."""
        cas, manager, tmp_path = cas_setup

        target_dir = tmp_path / "target"
        success = await manager.materialize_plugin("unknown-plugin", str(target_dir))

        assert not success
