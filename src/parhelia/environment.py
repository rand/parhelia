"""Environment versioning and capture for checkpoint reproducibility.

Implements:
- [SPEC-07.10.01] Claude Code Version Capture
- [SPEC-07.10.02] Plugin Version Capture
- [SPEC-07.10.03] MCP Server Version Capture
- [SPEC-07.10.04] Python Environment Capture
- [SPEC-07.10.05] Environment Diff
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


@dataclass
class ClaudeCodeVersion:
    """Claude Code binary version information.

    Implements [SPEC-07.10.01].
    """

    version: str
    binary_hash: str  # SHA256 of the binary
    install_path: str


@dataclass
class PluginVersion:
    """Plugin version information from git.

    Implements [SPEC-07.10.02].
    """

    git_remote: str
    git_commit: str
    git_branch: str
    installed_at: datetime
    manifest_version: str | None = None


@dataclass
class MCPServerVersion:
    """MCP server version information.

    Implements [SPEC-07.10.03].
    """

    source_type: Literal["npm", "git", "local", "docker"]
    version_id: str  # npm version, git commit, path hash, or image digest
    config_hash: str  # Hash of the server's config section


@dataclass
class EnvironmentSnapshot:
    """Complete environment state at checkpoint time.

    Implements [SPEC-07.10].
    """

    claude_code: ClaudeCodeVersion
    plugins: dict[str, PluginVersion] = field(default_factory=dict)
    mcp_servers: dict[str, MCPServerVersion] = field(default_factory=dict)
    python_version: str = ""
    python_packages: dict[str, str] = field(default_factory=dict)  # name -> version
    pip_freeze_hash: str = ""  # SHA256 of full pip freeze output
    captured_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "claude_code": {
                "version": self.claude_code.version,
                "binary_hash": self.claude_code.binary_hash,
                "install_path": self.claude_code.install_path,
            },
            "plugins": {
                name: {
                    "git_remote": p.git_remote,
                    "git_commit": p.git_commit,
                    "git_branch": p.git_branch,
                    "installed_at": p.installed_at.isoformat(),
                    "manifest_version": p.manifest_version,
                }
                for name, p in self.plugins.items()
            },
            "mcp_servers": {
                name: {
                    "source_type": s.source_type,
                    "version_id": s.version_id,
                    "config_hash": s.config_hash,
                }
                for name, s in self.mcp_servers.items()
            },
            "python_version": self.python_version,
            "python_packages": self.python_packages,
            "pip_freeze_hash": self.pip_freeze_hash,
            "captured_at": self.captured_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> EnvironmentSnapshot:
        """Create from dictionary."""
        claude_code = ClaudeCodeVersion(
            version=data["claude_code"]["version"],
            binary_hash=data["claude_code"]["binary_hash"],
            install_path=data["claude_code"]["install_path"],
        )

        plugins = {}
        for name, p in data.get("plugins", {}).items():
            plugins[name] = PluginVersion(
                git_remote=p["git_remote"],
                git_commit=p["git_commit"],
                git_branch=p["git_branch"],
                installed_at=datetime.fromisoformat(p["installed_at"]),
                manifest_version=p.get("manifest_version"),
            )

        mcp_servers = {}
        for name, s in data.get("mcp_servers", {}).items():
            mcp_servers[name] = MCPServerVersion(
                source_type=s["source_type"],
                version_id=s["version_id"],
                config_hash=s["config_hash"],
            )

        return cls(
            claude_code=claude_code,
            plugins=plugins,
            mcp_servers=mcp_servers,
            python_version=data.get("python_version", ""),
            python_packages=data.get("python_packages", {}),
            pip_freeze_hash=data.get("pip_freeze_hash", ""),
            captured_at=datetime.fromisoformat(data["captured_at"]),
        )


class EnvironmentCapture:
    """Capture environment state for checkpoints.

    Implements [SPEC-07.10].
    """

    # Common Claude Code install locations
    CLAUDE_CODE_PATHS = [
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
        os.path.expanduser("~/.local/bin/claude"),
        os.path.expanduser("~/.npm-global/bin/claude"),
    ]

    # Claude Code plugins directory
    CLAUDE_PLUGINS_DIR = os.path.expanduser("~/.claude/plugins")

    def __init__(
        self,
        claude_code_path: str | None = None,
        plugins_dir: str | None = None,
        mcp_config_path: str | None = None,
    ):
        """Initialize environment capture.

        Args:
            claude_code_path: Path to Claude Code binary (auto-detected if None).
            plugins_dir: Path to plugins directory.
            mcp_config_path: Path to MCP config file.
        """
        self.claude_code_path = claude_code_path or self._find_claude_code()
        self.plugins_dir = Path(plugins_dir or self.CLAUDE_PLUGINS_DIR)
        self.mcp_config_path = mcp_config_path

    def _find_claude_code(self) -> str:
        """Find Claude Code binary path."""
        # Try which command first
        try:
            result = subprocess.run(
                ["which", "claude"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fall back to known paths
        for path in self.CLAUDE_CODE_PATHS:
            if os.path.isfile(path):
                return path

        return "/usr/local/bin/claude"  # Default

    async def capture(self) -> EnvironmentSnapshot:
        """Capture complete environment state.

        Implements [SPEC-07.10].

        Returns:
            EnvironmentSnapshot with all captured information.
        """
        # Capture all components concurrently
        claude_code, plugins, mcp_servers, (python_version, packages, freeze_hash) = (
            await asyncio.gather(
                self._capture_claude_code(),
                self._capture_plugins(),
                self._capture_mcp_servers(),
                self._capture_python_env(),
            )
        )

        return EnvironmentSnapshot(
            claude_code=claude_code,
            plugins=plugins,
            mcp_servers=mcp_servers,
            python_version=python_version,
            python_packages=packages,
            pip_freeze_hash=freeze_hash,
            captured_at=datetime.now(),
        )

    async def _capture_claude_code(self) -> ClaudeCodeVersion:
        """Capture Claude Code version and binary hash.

        Implements [SPEC-07.10.01].
        """
        version = "unknown"
        binary_hash = ""

        # Get version
        try:
            proc = await asyncio.create_subprocess_exec(
                self.claude_code_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            version_output = stdout.decode().strip()
            # Parse version from output (e.g., "claude 1.0.0" -> "1.0.0")
            parts = version_output.split()
            if len(parts) >= 2:
                version = parts[-1]
            elif parts:
                version = parts[0]
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            pass

        # Calculate binary hash
        if os.path.isfile(self.claude_code_path):
            binary_hash = await self._hash_file(self.claude_code_path)

        return ClaudeCodeVersion(
            version=version,
            binary_hash=binary_hash,
            install_path=self.claude_code_path,
        )

    async def _capture_plugins(self) -> dict[str, PluginVersion]:
        """Capture all plugin versions.

        Implements [SPEC-07.10.02].
        """
        plugins: dict[str, PluginVersion] = {}

        if not self.plugins_dir.exists():
            return plugins

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            plugin_name = plugin_dir.name
            git_dir = plugin_dir / ".git"

            if not git_dir.exists():
                continue

            try:
                # Get git information
                git_remote = await self._git_remote(plugin_dir)
                git_commit = await self._git_commit(plugin_dir)
                git_branch = await self._git_branch(plugin_dir)

                # Try to get manifest version
                manifest_version = await self._get_plugin_manifest_version(plugin_dir)

                # Get install time from .git directory mtime
                installed_at = datetime.fromtimestamp(git_dir.stat().st_mtime)

                plugins[plugin_name] = PluginVersion(
                    git_remote=git_remote,
                    git_commit=git_commit,
                    git_branch=git_branch,
                    installed_at=installed_at,
                    manifest_version=manifest_version,
                )
            except Exception:
                # Skip plugins that fail to capture
                continue

        return plugins

    async def _capture_mcp_servers(self) -> dict[str, MCPServerVersion]:
        """Capture MCP server versions.

        Implements [SPEC-07.10.03].
        """
        servers: dict[str, MCPServerVersion] = {}

        # Try to load MCP config
        config_paths = [
            self.mcp_config_path,
            os.path.expanduser("~/.claude/mcp_config.json"),
            os.path.expanduser("~/.config/claude/mcp.json"),
        ]

        mcp_config = None
        for path in config_paths:
            if path and os.path.isfile(path):
                try:
                    with open(path) as f:
                        mcp_config = json.load(f)
                    break
                except (json.JSONDecodeError, OSError):
                    continue

        if not mcp_config:
            return servers

        # Process each server in config
        for server_name, server_config in mcp_config.get("mcpServers", {}).items():
            try:
                source_type, version_id = await self._identify_mcp_server(
                    server_name, server_config
                )
                config_hash = hashlib.sha256(
                    json.dumps(server_config, sort_keys=True).encode()
                ).hexdigest()[:16]

                servers[server_name] = MCPServerVersion(
                    source_type=source_type,
                    version_id=version_id,
                    config_hash=config_hash,
                )
            except Exception:
                continue

        return servers

    async def _capture_python_env(self) -> tuple[str, dict[str, str], str]:
        """Capture Python environment.

        Implements [SPEC-07.10.04].

        Returns:
            Tuple of (python_version, packages_dict, pip_freeze_hash).
        """
        python_version = ""
        packages: dict[str, str] = {}
        freeze_hash = ""

        # Get Python version
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            version_output = stdout.decode().strip()
            # Parse "Python 3.11.0" -> "3.11.0"
            parts = version_output.split()
            if len(parts) >= 2:
                python_version = parts[-1]
        except (asyncio.TimeoutError, FileNotFoundError):
            pass

        # Get pip freeze output (try pip3 first, then pip)
        freeze_output = ""
        for pip_cmd in ["pip3", "pip"]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    pip_cmd,
                    "freeze",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0:
                    freeze_output = stdout.decode()
                    break
            except (asyncio.TimeoutError, FileNotFoundError):
                continue

        if freeze_output:
            # Hash the full output
            freeze_hash = hashlib.sha256(freeze_output.encode()).hexdigest()

            # Parse into dict (key packages only for manifest)
            key_packages = {
                "anthropic",
                "modal",
                "aiohttp",
                "click",
                "pydantic",
                "httpx",
                "pytest",
            }
            for line in freeze_output.strip().split("\n"):
                if "==" in line:
                    name, version = line.split("==", 1)
                    name_lower = name.lower()
                    # Include key packages in the dict
                    if name_lower in key_packages:
                        packages[name_lower] = version

        return python_version, packages, freeze_hash

    async def _hash_file(self, path: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except OSError:
            return ""

    async def _git_remote(self, repo_path: Path) -> str:
        """Get git remote URL."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(repo_path),
                "remote",
                "get-url",
                "origin",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return stdout.decode().strip()
        except (asyncio.TimeoutError, FileNotFoundError):
            return ""

    async def _git_commit(self, repo_path: Path) -> str:
        """Get current git commit hash."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(repo_path),
                "rev-parse",
                "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return stdout.decode().strip()
        except (asyncio.TimeoutError, FileNotFoundError):
            return ""

    async def _git_branch(self, repo_path: Path) -> str:
        """Get current git branch name."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(repo_path),
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            return stdout.decode().strip()
        except (asyncio.TimeoutError, FileNotFoundError):
            return ""

    async def _get_plugin_manifest_version(self, plugin_dir: Path) -> str | None:
        """Get version from plugin manifest if available."""
        manifest_paths = [
            plugin_dir / "package.json",
            plugin_dir / "manifest.json",
            plugin_dir / "plugin.json",
        ]

        for manifest_path in manifest_paths:
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    return manifest.get("version")
                except (json.JSONDecodeError, OSError):
                    continue

        return None

    async def _identify_mcp_server(
        self, name: str, config: dict
    ) -> tuple[Literal["npm", "git", "local", "docker"], str]:
        """Identify MCP server source type and version.

        Returns:
            Tuple of (source_type, version_id).
        """
        command = config.get("command", "")
        args = config.get("args", [])

        # Check for npx (npm package)
        if command == "npx" and args:
            package = args[0] if args else ""
            # Try to get version from npm
            version = await self._get_npm_version(package)
            return "npm", version or package

        # Check for docker
        if command == "docker" or "docker" in command:
            # Try to find image name in args
            for i, arg in enumerate(args):
                if arg == "run" and i + 1 < len(args):
                    image = args[i + 1]
                    return "docker", image
            return "docker", "unknown"

        # Check for local path
        if os.path.isfile(command):
            path_hash = hashlib.sha256(command.encode()).hexdigest()[:12]
            return "local", f"local:{path_hash}"

        # Check for git repo path
        if os.path.isdir(command):
            git_dir = Path(command) / ".git"
            if git_dir.exists():
                commit = await self._git_commit(Path(command))
                return "git", commit or "unknown"

        return "local", "unknown"

    async def _get_npm_version(self, package: str) -> str:
        """Get installed npm package version."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "npm",
                "list",
                package,
                "--depth=0",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            data = json.loads(stdout.decode())
            deps = data.get("dependencies", {})
            if package in deps:
                return deps[package].get("version", "")
        except (asyncio.TimeoutError, FileNotFoundError, json.JSONDecodeError):
            pass
        return ""


@dataclass
class EnvironmentDiff:
    """Differences between two environment snapshots.

    Implements [SPEC-07.10.05].
    """

    claude_code_changed: bool = False
    claude_code_diff: dict | None = None

    plugins_added: list[str] = field(default_factory=list)
    plugins_removed: list[str] = field(default_factory=list)
    plugins_changed: dict[str, dict] = field(default_factory=dict)

    mcp_servers_added: list[str] = field(default_factory=list)
    mcp_servers_removed: list[str] = field(default_factory=list)
    mcp_servers_changed: dict[str, dict] = field(default_factory=dict)

    python_version_changed: bool = False
    python_version_diff: tuple[str, str] | None = None

    packages_added: dict[str, str] = field(default_factory=dict)
    packages_removed: dict[str, str] = field(default_factory=dict)
    packages_changed: dict[str, tuple[str, str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if there are no differences."""
        return not (
            self.claude_code_changed
            or self.plugins_added
            or self.plugins_removed
            or self.plugins_changed
            or self.mcp_servers_added
            or self.mcp_servers_removed
            or self.mcp_servers_changed
            or self.python_version_changed
            or self.packages_added
            or self.packages_removed
            or self.packages_changed
        )


def diff_environments(
    env_a: EnvironmentSnapshot, env_b: EnvironmentSnapshot
) -> EnvironmentDiff:
    """Compare two environment snapshots.

    Implements [SPEC-07.10.05].

    Args:
        env_a: First (older) environment snapshot.
        env_b: Second (newer) environment snapshot.

    Returns:
        EnvironmentDiff describing the changes.
    """
    diff = EnvironmentDiff()

    # Compare Claude Code
    if env_a.claude_code.version != env_b.claude_code.version or env_a.claude_code.binary_hash != env_b.claude_code.binary_hash:
        diff.claude_code_changed = True
        diff.claude_code_diff = {
            "version": (env_a.claude_code.version, env_b.claude_code.version),
            "binary_hash": (env_a.claude_code.binary_hash, env_b.claude_code.binary_hash),
        }

    # Compare plugins
    plugins_a = set(env_a.plugins.keys())
    plugins_b = set(env_b.plugins.keys())

    diff.plugins_added = list(plugins_b - plugins_a)
    diff.plugins_removed = list(plugins_a - plugins_b)

    for name in plugins_a & plugins_b:
        p_a = env_a.plugins[name]
        p_b = env_b.plugins[name]
        if p_a.git_commit != p_b.git_commit:
            diff.plugins_changed[name] = {
                "commit": (p_a.git_commit, p_b.git_commit),
                "branch": (p_a.git_branch, p_b.git_branch),
            }

    # Compare MCP servers
    servers_a = set(env_a.mcp_servers.keys())
    servers_b = set(env_b.mcp_servers.keys())

    diff.mcp_servers_added = list(servers_b - servers_a)
    diff.mcp_servers_removed = list(servers_a - servers_b)

    for name in servers_a & servers_b:
        s_a = env_a.mcp_servers[name]
        s_b = env_b.mcp_servers[name]
        if s_a.version_id != s_b.version_id or s_a.config_hash != s_b.config_hash:
            diff.mcp_servers_changed[name] = {
                "version": (s_a.version_id, s_b.version_id),
                "config": (s_a.config_hash, s_b.config_hash),
            }

    # Compare Python version
    if env_a.python_version != env_b.python_version:
        diff.python_version_changed = True
        diff.python_version_diff = (env_a.python_version, env_b.python_version)

    # Compare packages
    packages_a = set(env_a.python_packages.keys())
    packages_b = set(env_b.python_packages.keys())

    for name in packages_b - packages_a:
        diff.packages_added[name] = env_b.python_packages[name]

    for name in packages_a - packages_b:
        diff.packages_removed[name] = env_a.python_packages[name]

    for name in packages_a & packages_b:
        v_a = env_a.python_packages[name]
        v_b = env_b.python_packages[name]
        if v_a != v_b:
            diff.packages_changed[name] = (v_a, v_b)

    return diff


def format_environment_diff(diff: EnvironmentDiff) -> str:
    """Format environment diff for display.

    Args:
        diff: The EnvironmentDiff to format.

    Returns:
        Human-readable string representation.
    """
    if diff.is_empty():
        return "No environment changes"

    lines = ["Environment Changes:", "=" * 40]

    # Claude Code
    if diff.claude_code_changed and diff.claude_code_diff:
        lines.append("\nClaude Code:")
        v_a, v_b = diff.claude_code_diff["version"]
        lines.append(f"  Version: {v_a} → {v_b}")

    # Plugins
    if diff.plugins_added or diff.plugins_removed or diff.plugins_changed:
        lines.append("\nPlugins:")
        for name in diff.plugins_added:
            lines.append(f"  + {name} (added)")
        for name in diff.plugins_removed:
            lines.append(f"  - {name} (removed)")
        for name, changes in diff.plugins_changed.items():
            c_a, c_b = changes["commit"]
            lines.append(f"  ~ {name}: {c_a[:8]} → {c_b[:8]}")

    # MCP Servers
    if diff.mcp_servers_added or diff.mcp_servers_removed or diff.mcp_servers_changed:
        lines.append("\nMCP Servers:")
        for name in diff.mcp_servers_added:
            lines.append(f"  + {name} (added)")
        for name in diff.mcp_servers_removed:
            lines.append(f"  - {name} (removed)")
        for name, changes in diff.mcp_servers_changed.items():
            v_a, v_b = changes["version"]
            lines.append(f"  ~ {name}: {v_a} → {v_b}")

    # Python
    if diff.python_version_changed and diff.python_version_diff:
        lines.append("\nPython:")
        v_a, v_b = diff.python_version_diff
        lines.append(f"  Version: {v_a} → {v_b}")

    # Packages
    if diff.packages_added or diff.packages_removed or diff.packages_changed:
        lines.append("\nPackages:")
        for name, version in diff.packages_added.items():
            lines.append(f"  + {name}=={version}")
        for name, version in diff.packages_removed.items():
            lines.append(f"  - {name}=={version}")
        for name, (v_a, v_b) in diff.packages_changed.items():
            lines.append(f"  ~ {name}: {v_a} → {v_b}")

    return "\n".join(lines)
