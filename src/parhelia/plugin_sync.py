"""Plugin synchronization for Modal volumes.

Implements:
- [SPEC-07.10] Plugin Discovery
- [SPEC-07.11] Plugin Sync Strategy
- [SPEC-08.16] CAS Plugin Integration
"""

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from parhelia.cas import ContentAddressableStorage, Digest, MerkleTreeBuilder


@dataclass
class PluginInfo:
    """Information about a discovered plugin.

    Implements [SPEC-07.10].
    """

    name: str
    path: str  # Resolved absolute path
    is_symlink: bool
    git_remote: str | None  # Git remote URL if available
    git_branch: str | None
    has_build_step: bool
    dependencies: list[str] = field(default_factory=list)


class SyncStrategy(Enum):
    """Plugin sync strategy.

    Implements [SPEC-07.11].
    """

    GIT_CLONE = "git_clone"  # Clone from git remote
    COPY = "copy"  # Direct copy (no git)
    SKIP = "skip"  # Don't sync (local-only)


@dataclass
class SyncResult:
    """Result of a plugin sync operation."""

    plugin: str
    success: bool
    strategy: str | None = None
    reason: str | None = None
    files_synced: int = 0


class PluginDiscovery:
    """Discover plugins from local Claude configuration.

    Implements [SPEC-07.10].
    """

    def __init__(self, claude_dir: str = "~/.claude"):
        self.claude_dir = Path(claude_dir).expanduser()

    async def discover_all(self) -> list[PluginInfo]:
        """Discover all plugins and skills asynchronously."""
        return self.discover_all_sync()

    def discover_all_sync(self) -> list[PluginInfo]:
        """Discover all plugins and skills synchronously."""
        plugins = []

        # Discover plugins
        plugins_dir = self.claude_dir / "plugins"
        if plugins_dir.exists():
            for item in plugins_dir.iterdir():
                if item.is_dir():
                    plugin = self._analyze_plugin(item)
                    if plugin:
                        plugins.append(plugin)

        # Discover skills
        skills_dir = self.claude_dir / "skills"
        if skills_dir.exists():
            for item in skills_dir.iterdir():
                if item.is_dir():
                    skill = self._analyze_plugin(item, is_skill=True)
                    if skill:
                        plugins.append(skill)

        return plugins

    def _analyze_plugin(
        self,
        path: Path,
        is_skill: bool = False,
    ) -> PluginInfo | None:
        """Analyze a plugin directory."""
        try:
            # Resolve symlinks
            resolved_path = path.resolve()
            is_symlink = path.is_symlink()

            # Check for git repo
            git_remote = None
            git_branch = None
            git_dir = resolved_path / ".git"
            if git_dir.exists():
                git_remote = self._get_git_remote(resolved_path)
                git_branch = self._get_git_branch(resolved_path)

            # Check for build steps
            has_build_step = (
                (resolved_path / "package.json").exists()
                or (resolved_path / "pyproject.toml").exists()
                or (resolved_path / "Makefile").exists()
            )

            # Get dependencies
            dependencies = self._get_dependencies(resolved_path)

            return PluginInfo(
                name=path.name,
                path=str(resolved_path),
                is_symlink=is_symlink,
                git_remote=git_remote,
                git_branch=git_branch,
                has_build_step=has_build_step,
                dependencies=dependencies,
            )
        except Exception:
            return None

    def _get_git_remote(self, path: Path) -> str | None:
        """Get git remote URL."""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_git_branch(self, path: Path) -> str | None:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_dependencies(self, path: Path) -> list[str]:
        """Get plugin dependencies (basic detection)."""
        deps = []

        # Check package.json for npm deps
        pkg_json = path / "package.json"
        if pkg_json.exists():
            deps.append("npm")

        # Check pyproject.toml for python deps
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            deps.append("pip")

        return deps


class PluginSyncManager:
    """Manage plugin synchronization to Modal Volume.

    Implements [SPEC-07.11].
    """

    def __init__(self, volume_path: str = "/vol/parhelia"):
        self.volume_path = Path(volume_path)
        self.plugins_path = self.volume_path / "plugins"
        self.skills_path = self.volume_path / "skills"

    def determine_strategy(self, plugin: PluginInfo) -> SyncStrategy:
        """Determine sync strategy for a plugin."""
        # If has git remote, prefer cloning
        if plugin.git_remote:
            return SyncStrategy.GIT_CLONE

        # If it's a symlink to a path that doesn't exist, skip
        if plugin.is_symlink and not Path(plugin.path).exists():
            return SyncStrategy.SKIP

        # Otherwise, copy
        return SyncStrategy.COPY

    async def sync_plugin(self, plugin: PluginInfo) -> SyncResult:
        """Sync a single plugin to Volume."""
        strategy = self.determine_strategy(plugin)
        target_dir = self.plugins_path / plugin.name

        # Ensure plugins directory exists
        self.plugins_path.mkdir(parents=True, exist_ok=True)

        match strategy:
            case SyncStrategy.GIT_CLONE:
                return await self._git_clone(plugin, target_dir)
            case SyncStrategy.COPY:
                return await self._copy_plugin(plugin, target_dir)
            case SyncStrategy.SKIP:
                return SyncResult(
                    plugin=plugin.name,
                    success=False,
                    strategy="skip",
                    reason="Plugin source not accessible for remote sync",
                )

    async def _git_clone(
        self,
        plugin: PluginInfo,
        target_dir: Path,
    ) -> SyncResult:
        """Clone plugin from git."""
        try:
            if target_dir.exists():
                # Pull updates
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "fetch", "origin"],
                    cwd=str(target_dir),
                    capture_output=True,
                    timeout=60,
                )

                if plugin.git_branch:
                    await asyncio.to_thread(
                        subprocess.run,
                        ["git", "reset", "--hard", f"origin/{plugin.git_branch}"],
                        cwd=str(target_dir),
                        capture_output=True,
                        timeout=30,
                    )
            else:
                # Clone fresh
                cmd = ["git", "clone", "--depth=1"]
                if plugin.git_branch:
                    cmd.extend(["--branch", plugin.git_branch])
                cmd.extend([plugin.git_remote, str(target_dir)])

                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    timeout=120,
                )

                if result.returncode != 0:
                    return SyncResult(
                        plugin=plugin.name,
                        success=False,
                        strategy="git_clone",
                        reason=f"Git clone failed: {result.stderr.decode()}",
                    )

            return SyncResult(
                plugin=plugin.name,
                success=True,
                strategy="git_clone",
            )

        except Exception as e:
            return SyncResult(
                plugin=plugin.name,
                success=False,
                strategy="git_clone",
                reason=str(e),
            )

    async def _copy_plugin(
        self,
        plugin: PluginInfo,
        target_dir: Path,
    ) -> SyncResult:
        """Copy plugin files directly."""
        try:
            source_path = Path(plugin.path)

            if not source_path.exists():
                return SyncResult(
                    plugin=plugin.name,
                    success=False,
                    strategy="copy",
                    reason=f"Source path does not exist: {plugin.path}",
                )

            # Remove existing target
            if target_dir.exists():
                shutil.rmtree(target_dir)

            # Copy files
            shutil.copytree(
                source_path,
                target_dir,
                symlinks=False,
                ignore=shutil.ignore_patterns(
                    "node_modules",
                    "__pycache__",
                    ".git",
                    "*.pyc",
                    ".venv",
                    "venv",
                ),
            )

            # Count files
            file_count = sum(1 for _ in target_dir.rglob("*") if _.is_file())

            return SyncResult(
                plugin=plugin.name,
                success=True,
                strategy="copy",
                files_synced=file_count,
            )

        except Exception as e:
            return SyncResult(
                plugin=plugin.name,
                success=False,
                strategy="copy",
                reason=str(e),
            )

    async def sync_all(
        self,
        plugins: list[PluginInfo],
    ) -> list[SyncResult]:
        """Sync all plugins to Volume."""
        results = []
        for plugin in plugins:
            result = await self.sync_plugin(plugin)
            results.append(result)
        return results


async def sync_plugins_to_volume(
    claude_dir: str = "~/.claude",
    volume_path: str = "/vol/parhelia",
) -> list[SyncResult]:
    """Convenience function to discover and sync all plugins.

    Args:
        claude_dir: Path to local Claude configuration
        volume_path: Path to Modal Volume mount

    Returns:
        List of sync results for each plugin
    """
    discovery = PluginDiscovery(claude_dir=claude_dir)
    plugins = await discovery.discover_all()

    manager = PluginSyncManager(volume_path=volume_path)
    return await manager.sync_all(plugins)


# =============================================================================
# [SPEC-08.16] CAS Plugin Integration
# =============================================================================


@dataclass
class PluginManifest:
    """Manifest for a plugin stored in CAS.

    Implements [SPEC-08.16].
    """

    name: str
    version: str | None  # Git commit hash or version string
    root_digest: Digest  # Merkle tree root
    git_remote: str | None
    git_branch: str | None
    has_build_step: bool
    dependencies: list[str]
    file_count: int
    total_size_bytes: int

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "root_digest": {
                "hash": self.root_digest.hash,
                "size_bytes": self.root_digest.size_bytes,
            },
            "git_remote": self.git_remote,
            "git_branch": self.git_branch,
            "has_build_step": self.has_build_step,
            "dependencies": self.dependencies,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PluginManifest":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            version=data.get("version"),
            root_digest=Digest(
                hash=data["root_digest"]["hash"],
                size_bytes=data["root_digest"]["size_bytes"],
            ),
            git_remote=data.get("git_remote"),
            git_branch=data.get("git_branch"),
            has_build_step=data.get("has_build_step", False),
            dependencies=data.get("dependencies", []),
            file_count=data.get("file_count", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
        )


@dataclass
class CASyncResult:
    """Result of a CAS plugin sync operation."""

    plugin: str
    success: bool
    manifest: PluginManifest | None = None
    reason: str | None = None
    blobs_created: int = 0
    blobs_reused: int = 0


class PluginCASManager:
    """Manage plugins using Content-Addressable Storage.

    Stores plugin content in CAS for deduplication across containers.
    Implements [SPEC-08.16].
    """

    def __init__(
        self,
        cas: ContentAddressableStorage,
        manifest_path: str = "/vol/parhelia/plugins/manifests",
    ):
        self.cas = cas
        self.tree_builder = MerkleTreeBuilder(cas)
        self.manifest_path = Path(manifest_path)

    async def sync_plugin(self, plugin: PluginInfo) -> CASyncResult:
        """Store plugin content in CAS.

        Args:
            plugin: Plugin info from discovery

        Returns:
            Sync result with manifest if successful
        """
        try:
            source_path = Path(plugin.path)

            if not source_path.exists():
                return CASyncResult(
                    plugin=plugin.name,
                    success=False,
                    reason=f"Source path does not exist: {plugin.path}",
                )

            # Get version from git if available
            version = self._get_git_version(source_path)

            # Count files and size before building tree
            file_count = 0
            total_size = 0
            for f in source_path.rglob("*"):
                if f.is_file() and not self._should_exclude(f):
                    file_count += 1
                    total_size += f.stat().st_size

            # Build Merkle tree (this stores all blobs in CAS)
            root_digest = await self.tree_builder.build_tree(str(source_path))

            # Create manifest
            manifest = PluginManifest(
                name=plugin.name,
                version=version,
                root_digest=root_digest,
                git_remote=plugin.git_remote,
                git_branch=plugin.git_branch,
                has_build_step=plugin.has_build_step,
                dependencies=plugin.dependencies,
                file_count=file_count,
                total_size_bytes=total_size,
            )

            # Store manifest
            await self._store_manifest(manifest)

            return CASyncResult(
                plugin=plugin.name,
                success=True,
                manifest=manifest,
            )

        except Exception as e:
            return CASyncResult(
                plugin=plugin.name,
                success=False,
                reason=str(e),
            )

    async def materialize_plugin(
        self,
        plugin_name: str,
        target_dir: str,
    ) -> bool:
        """Restore plugin from CAS to filesystem.

        Args:
            plugin_name: Name of the plugin to restore
            target_dir: Directory to restore plugin to

        Returns:
            True if successful
        """
        manifest = await self._load_manifest(plugin_name)
        if not manifest:
            return False

        target_path = Path(target_dir) / plugin_name
        target_path.mkdir(parents=True, exist_ok=True)

        # Restore from CAS
        await self._restore_tree(manifest.root_digest, target_path)
        return True

    async def get_manifest(self, plugin_name: str) -> PluginManifest | None:
        """Get manifest for a plugin."""
        return await self._load_manifest(plugin_name)

    async def list_plugins(self) -> list[PluginManifest]:
        """List all plugins stored in CAS."""
        manifests = []

        if not self.manifest_path.exists():
            return manifests

        for manifest_file in self.manifest_path.glob("*.json"):
            try:
                manifest = await self._load_manifest(manifest_file.stem)
                if manifest:
                    manifests.append(manifest)
            except Exception:
                continue

        return manifests

    async def plugin_exists(self, plugin_name: str, version: str | None = None) -> bool:
        """Check if plugin exists in CAS.

        Args:
            plugin_name: Name of the plugin
            version: Optional version to check for

        Returns:
            True if plugin (and optionally version) exists
        """
        manifest = await self._load_manifest(plugin_name)
        if not manifest:
            return False

        if version and manifest.version != version:
            return False

        # Verify root digest exists in CAS
        return await self.cas.contains(manifest.root_digest)

    async def _store_manifest(self, manifest: PluginManifest) -> None:
        """Store plugin manifest."""
        self.manifest_path.mkdir(parents=True, exist_ok=True)
        manifest_file = self.manifest_path / f"{manifest.name}.json"

        with open(manifest_file, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    async def _load_manifest(self, plugin_name: str) -> PluginManifest | None:
        """Load plugin manifest."""
        manifest_file = self.manifest_path / f"{plugin_name}.json"

        if not manifest_file.exists():
            return None

        with open(manifest_file, "r") as f:
            data = json.load(f)

        return PluginManifest.from_dict(data)

    async def _restore_tree(self, root_digest: Digest, target_path: Path) -> None:
        """Restore directory tree from CAS."""
        from parhelia.cas import Directory

        content = await self.cas.read_blob(root_digest)
        directory = Directory.deserialize(content)

        # Restore files
        for f in directory.files:
            file_path = target_path / f.name
            file_content = await self.cas.read_blob(f.digest)
            file_path.write_bytes(file_content)
            if f.is_executable:
                file_path.chmod(0o755)

        # Restore symlinks
        for s in directory.symlinks:
            link_path = target_path / s.name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(s.target)

        # Restore subdirectories
        for d in directory.directories:
            subdir_path = target_path / d.name
            subdir_path.mkdir(exist_ok=True)
            await self._restore_tree(d.digest, subdir_path)

    def _get_git_version(self, path: Path) -> str | None:
        """Get git commit hash as version."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]  # Short hash
        except Exception:
            pass
        return None

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from sync."""
        exclude_patterns = {
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            ".mypy_cache",
            ".pytest_cache",
        }
        return any(part in exclude_patterns for part in path.parts)


async def sync_plugins_to_cas(
    claude_dir: str = "~/.claude",
    cas_root: str = "/vol/parhelia/cas",
) -> list[CASyncResult]:
    """Convenience function to discover and sync all plugins to CAS.

    Args:
        claude_dir: Path to local Claude configuration
        cas_root: Path to CAS root directory

    Returns:
        List of sync results for each plugin
    """
    discovery = PluginDiscovery(claude_dir=claude_dir)
    plugins = await discovery.discover_all()

    cas = ContentAddressableStorage(root_path=cas_root)
    manager = PluginCASManager(cas=cas)

    results = []
    for plugin in plugins:
        result = await manager.sync_plugin(plugin)
        results.append(result)

    return results
