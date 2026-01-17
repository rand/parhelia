# SPEC-07: Plugin and MCP Synchronization

**Status**: Draft
**Issue**: ph-3jl
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines how Claude Code plugins and MCP servers are synchronized between local and remote environments, ensuring feature parity across execution contexts.

## Goals

- [SPEC-07.01] Clone and sync plugins from git to Modal Volume
- [SPEC-07.02] Mirror MCP server configuration to remote environments
- [SPEC-07.03] Handle plugin dependencies and build steps
- [SPEC-07.04] Support periodic sync for plugin updates
- [SPEC-07.05] Resolve symlinks and local paths for remote compatibility

## Non-Goals

- Plugin marketplace/registry (direct git clone only)
- Plugin version pinning UI (manual config for v1)
- Hot-reload of plugins in running sessions

---

## Architecture

### Sync Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL ENVIRONMENT                                  │
│                                                                              │
│  ~/.claude/                                                                  │
│  ├── plugins/                                                                │
│  │   ├── cc-polymath -> /Users/rand/src/cc-polymath (symlink)              │
│  │   └── other-plugin/                                                       │
│  ├── skills/                                                                 │
│  │   └── custom-skill/                                                       │
│  ├── mcp_config.json                                                         │
│  └── settings.json                                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Plugin Sync Manager                              │   │
│  │  - Resolves symlinks to actual paths                                 │   │
│  │  - Detects git repos for cloning                                     │   │
│  │  - Transforms MCP config for remote                                  │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │ sync
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL VOLUME                                       │
│                                                                              │
│  /vol/parhelia/                                                              │
│  ├── config/                                                                 │
│  │   └── claude/                                                             │
│  │       ├── mcp_config.json (transformed)                                   │
│  │       └── settings.json                                                   │
│  ├── plugins/                                                                │
│  │   ├── cc-polymath/  (git cloned)                                         │
│  │   └── other-plugin/ (copied)                                             │
│  └── skills/                                                                 │
│      └── custom-skill/ (copied)                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MCP Server Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL CONTAINER                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Claude Code                                   │   │
│  │                            │                                         │   │
│  │              ┌─────────────┼─────────────┐                          │   │
│  │              │             │             │                          │   │
│  │              ▼             ▼             ▼                          │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐             │   │
│  │  │ Beads MCP     │ │ Playwright    │ │ Custom MCP    │             │   │
│  │  │ Server        │ │ MCP Server    │ │ Servers       │             │   │
│  │  └───────────────┘ └───────────────┘ └───────────────┘             │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### [SPEC-07.10] Plugin Discovery

The sync manager MUST discover all plugins from local configuration:

```python
@dataclass
class PluginInfo:
    name: str
    path: str                    # Resolved absolute path
    is_symlink: bool
    git_remote: str | None       # Git remote URL if available
    git_branch: str | None
    has_build_step: bool
    dependencies: list[str]      # npm/pip dependencies

class PluginDiscovery:
    """Discover plugins from local Claude configuration."""

    def __init__(self, claude_dir: str = "~/.claude"):
        self.claude_dir = Path(claude_dir).expanduser()

    async def discover_all(self) -> list[PluginInfo]:
        """Discover all plugins and skills."""
        plugins = []

        # Discover plugins
        plugins_dir = self.claude_dir / "plugins"
        if plugins_dir.exists():
            for item in plugins_dir.iterdir():
                plugin = await self._analyze_plugin(item)
                if plugin:
                    plugins.append(plugin)

        # Discover skills
        skills_dir = self.claude_dir / "skills"
        if skills_dir.exists():
            for item in skills_dir.iterdir():
                skill = await self._analyze_plugin(item, is_skill=True)
                if skill:
                    plugins.append(skill)

        return plugins

    async def _analyze_plugin(
        self,
        path: Path,
        is_skill: bool = False,
    ) -> PluginInfo | None:
        """Analyze a plugin directory."""

        # Resolve symlinks
        resolved_path = path.resolve()
        is_symlink = path.is_symlink()

        # Check for git repo
        git_remote = None
        git_branch = None
        git_dir = resolved_path / ".git"
        if git_dir.exists():
            git_remote = await self._get_git_remote(resolved_path)
            git_branch = await self._get_git_branch(resolved_path)

        # Check for build step
        has_build = (
            (resolved_path / "package.json").exists() or
            (resolved_path / "pyproject.toml").exists() or
            (resolved_path / "Makefile").exists()
        )

        # Get dependencies
        dependencies = await self._get_dependencies(resolved_path)

        return PluginInfo(
            name=path.name,
            path=str(resolved_path),
            is_symlink=is_symlink,
            git_remote=git_remote,
            git_branch=git_branch,
            has_build_step=has_build,
            dependencies=dependencies,
        )

    async def _get_git_remote(self, path: Path) -> str | None:
        """Get git remote URL."""
        try:
            result = await run_command(
                ["git", "remote", "get-url", "origin"],
                cwd=str(path),
            )
            return result.stdout.strip()
        except Exception:
            return None
```

### [SPEC-07.11] Plugin Sync Strategy

Each plugin MUST be synced according to its characteristics:

```python
class SyncStrategy(Enum):
    GIT_CLONE = "git_clone"      # Clone from git remote
    COPY = "copy"                # Direct copy (no git)
    SKIP = "skip"                # Don't sync (local-only)

class PluginSyncManager:
    """Manage plugin synchronization to Modal Volume."""

    def __init__(self, volume_path: str = "/vol/parhelia"):
        self.volume_path = Path(volume_path)
        self.plugins_path = self.volume_path / "plugins"
        self.skills_path = self.volume_path / "skills"

    def determine_strategy(self, plugin: PluginInfo) -> SyncStrategy:
        """Determine sync strategy for a plugin."""

        # If has git remote, prefer cloning
        if plugin.git_remote:
            return SyncStrategy.GIT_CLONE

        # If it's a symlink to a git repo we can't access, skip
        if plugin.is_symlink and not Path(plugin.path).exists():
            return SyncStrategy.SKIP

        # Otherwise, copy
        return SyncStrategy.COPY

    async def sync_plugin(self, plugin: PluginInfo) -> SyncResult:
        """Sync a single plugin to Volume."""
        strategy = self.determine_strategy(plugin)
        target_dir = self.plugins_path / plugin.name

        match strategy:
            case SyncStrategy.GIT_CLONE:
                return await self._git_clone(plugin, target_dir)
            case SyncStrategy.COPY:
                return await self._copy_plugin(plugin, target_dir)
            case SyncStrategy.SKIP:
                return SyncResult(
                    plugin=plugin.name,
                    success=False,
                    reason="Plugin source not accessible for remote sync",
                )

    async def _git_clone(
        self,
        plugin: PluginInfo,
        target_dir: Path,
    ) -> SyncResult:
        """Clone plugin from git."""

        if target_dir.exists():
            # Pull updates
            await run_command(
                ["git", "fetch", "origin"],
                cwd=str(target_dir),
            )
            await run_command(
                ["git", "reset", "--hard", f"origin/{plugin.git_branch or 'main'}"],
                cwd=str(target_dir),
            )
        else:
            # Fresh clone
            await run_command([
                "git", "clone",
                "--depth", "1",
                "--branch", plugin.git_branch or "main",
                plugin.git_remote,
                str(target_dir),
            ])

        # Run build if needed
        if plugin.has_build_step:
            await self._run_build(target_dir)

        return SyncResult(plugin=plugin.name, success=True)

    async def _run_build(self, plugin_dir: Path):
        """Run plugin build step."""

        # npm/bun build
        if (plugin_dir / "package.json").exists():
            await run_command(["bun", "install"], cwd=str(plugin_dir))
            if await self._has_script(plugin_dir, "build"):
                await run_command(["bun", "run", "build"], cwd=str(plugin_dir))

        # Python build
        elif (plugin_dir / "pyproject.toml").exists():
            await run_command(
                ["pip", "install", "-e", "."],
                cwd=str(plugin_dir),
            )
```

### [SPEC-07.12] MCP Configuration Transform

MCP config MUST be transformed for remote execution:

```python
@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None

class MCPConfigTransformer:
    """Transform MCP config for remote environment."""

    def __init__(self, volume_path: str = "/vol/parhelia"):
        self.volume_path = volume_path

    async def transform(
        self,
        local_config: dict,
    ) -> dict:
        """Transform local MCP config for remote."""
        transformed = {"mcpServers": {}}

        for name, server in local_config.get("mcpServers", {}).items():
            transformed_server = await self._transform_server(name, server)
            if transformed_server:
                transformed["mcpServers"][name] = transformed_server

        return transformed

    async def _transform_server(
        self,
        name: str,
        server: dict,
    ) -> dict | None:
        """Transform a single MCP server config."""

        command = server.get("command", "")
        args = server.get("args", [])

        # Transform paths in command
        command = self._transform_path(command)

        # Transform paths in args
        args = [self._transform_path(arg) for arg in args]

        # Transform cwd
        cwd = server.get("cwd")
        if cwd:
            cwd = self._transform_path(cwd)

        # Transform env paths
        env = {}
        for key, value in server.get("env", {}).items():
            env[key] = self._transform_path(value)

        return {
            "command": command,
            "args": args,
            "env": env,
            "cwd": cwd,
        }

    def _transform_path(self, path: str) -> str:
        """Transform local path to remote path.

        Modal containers run as root by default, so ~ expands to /root.
        We transform common local paths to their remote equivalents.
        """

        # Home directory transformations
        # macOS: /Users/username -> /root
        # Linux: /home/username -> /root
        path = path.replace("~", "/root")
        path = re.sub(r"/Users/\w+", "/root", path)
        path = re.sub(r"/home/\w+", "/root", path)

        # Claude config directory -> Volume config
        if "/.claude/" in path:
            # Extract everything after .claude/
            match = re.search(r"/.claude/(.*)", path)
            if match:
                rest = match.group(1)

                # Plugins go to Volume plugins dir
                if rest.startswith("plugins/"):
                    plugin_rest = rest[8:]  # Remove "plugins/"
                    path = f"{self.volume_path}/plugins/{plugin_rest}"

                # Skills go to Volume skills dir
                elif rest.startswith("skills/"):
                    skill_rest = rest[7:]  # Remove "skills/"
                    path = f"{self.volume_path}/skills/{skill_rest}"

                # Other config files go to Volume config
                else:
                    path = f"{self.volume_path}/config/claude/{rest}"

        return path
```

### [SPEC-07.13] MCP Server Launcher

Remote containers MUST launch configured MCP servers:

```python
class MCPServerLauncher:
    """Launch MCP servers in remote container."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.processes: dict[str, subprocess.Popen] = {}

    async def launch_all(self) -> dict[str, bool]:
        """Launch all configured MCP servers."""
        with open(self.config_path) as f:
            config = json.load(f)

        results = {}
        for name, server_config in config.get("mcpServers", {}).items():
            try:
                await self.launch_server(name, server_config)
                results[name] = True
            except Exception as e:
                logger.error(f"Failed to launch MCP server {name}: {e}")
                results[name] = False

        return results

    async def launch_server(self, name: str, config: dict):
        """Launch a single MCP server."""
        command = config["command"]
        args = config.get("args", [])
        env = {**os.environ, **config.get("env", {})}
        cwd = config.get("cwd")

        # Start process
        process = subprocess.Popen(
            [command] + args,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.processes[name] = process

        # Wait for server to be ready
        await self._wait_for_ready(name, process)

    async def _wait_for_ready(
        self,
        name: str,
        process: subprocess.Popen,
        timeout: float = 10.0,
    ):
        """Wait for MCP server to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            if process.poll() is not None:
                # Process exited
                stderr = process.stderr.read().decode()
                raise Exception(f"MCP server {name} exited: {stderr}")

            # TODO: Check for ready signal (protocol-specific)
            await asyncio.sleep(0.1)

    async def shutdown_all(self):
        """Shutdown all MCP servers."""
        for name, process in self.processes.items():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
```

### [SPEC-07.14] Periodic Sync

Plugins MUST be synced periodically for updates:

```python
SYNC_INTERVAL_SECONDS = 3600  # 1 hour

class PeriodicSyncManager:
    """Manage periodic plugin synchronization."""

    def __init__(self, sync_manager: PluginSyncManager):
        self.sync_manager = sync_manager
        self.last_sync: datetime | None = None
        self.running = False

    async def start(self):
        """Start periodic sync background task."""
        self.running = True
        while self.running:
            await self.sync_all()
            await asyncio.sleep(SYNC_INTERVAL_SECONDS)

    async def stop(self):
        """Stop periodic sync."""
        self.running = False

    async def sync_all(self) -> SyncReport:
        """Sync all plugins."""
        discovery = PluginDiscovery()
        plugins = await discovery.discover_all()

        results = []
        for plugin in plugins:
            result = await self.sync_manager.sync_plugin(plugin)
            results.append(result)

        self.last_sync = datetime.now()

        return SyncReport(
            timestamp=self.last_sync,
            plugins_synced=len([r for r in results if r.success]),
            plugins_failed=len([r for r in results if not r.success]),
            results=results,
        )
```

### [SPEC-07.15] Sync Status Reporting

Users MUST be able to view sync status:

```python
@dataclass
class PluginSyncStatus:
    name: str
    strategy: SyncStrategy
    last_synced: datetime | None
    git_commit: str | None       # Current commit if git-based
    local_modified: bool         # Local changes not synced
    sync_error: str | None

async def get_sync_status() -> list[PluginSyncStatus]:
    """Get sync status for all plugins."""
    discovery = PluginDiscovery()
    plugins = await discovery.discover_all()

    statuses = []
    for plugin in plugins:
        status = await _get_plugin_status(plugin)
        statuses.append(status)

    return statuses
```

---

## CLI Commands

### `parhelia plugins list`

```bash
$ parhelia plugins list

NAME            TYPE      STRATEGY    LAST SYNCED      STATUS
cc-polymath     plugin    git_clone   5 minutes ago    synced (abc1234)
beads           plugin    git_clone   5 minutes ago    synced (def5678)
custom-skill    skill     copy        5 minutes ago    synced
local-only      plugin    skip        -                not synced (local only)
```

### `parhelia plugins sync`

```bash
$ parhelia plugins sync

Syncing plugins to Modal Volume...
  ├── cc-polymath: git pull... updated (abc1234 -> xyz9999)
  ├── beads: git pull... already up to date
  ├── custom-skill: copying... done
  └── local-only: skipped (no remote source)

Sync complete: 3 synced, 0 failed, 1 skipped
```

### `parhelia mcp status`

```bash
$ parhelia mcp status

SERVER              LOCAL    REMOTE   STATUS
beads               running  running  healthy
playwright          running  running  healthy
custom-server       running  -        local only
```

---

## Acceptance Criteria

- [ ] [SPEC-07.AC1] Plugins discovered from ~/.claude/plugins and skills
- [ ] [SPEC-07.AC2] Symlinks resolved to actual paths
- [ ] [SPEC-07.AC3] Git-based plugins cloned to Volume
- [ ] [SPEC-07.AC4] Non-git plugins copied to Volume
- [ ] [SPEC-07.AC5] MCP config transformed for remote paths
- [ ] [SPEC-07.AC6] MCP servers launch in remote container
- [ ] [SPEC-07.AC7] Periodic sync updates plugins hourly
- [ ] [SPEC-07.AC8] CLI shows sync status

---

## Resolved Questions

1. ~~**Private git repos**~~: **Resolved** - Use GitHub token from SPEC-04 secrets injection. Configure git credential helper to use `GITHUB_TOKEN` environment variable for HTTPS clones.

## Open Questions

1. **Large plugins**: What if a plugin exceeds reasonable size? Should we exclude node_modules, .git/objects during copy?
2. **MCP server conflicts**: What if same port used by multiple servers? (MCP uses stdio, not ports - may be non-issue)
3. **Plugin version pinning**: Should we support pinning to specific git tags/commits?

---

## References

- [Claude Code Plugins](https://docs.anthropic.com/en/docs/claude-code/plugins)
- [Claude Code Skills](https://docs.anthropic.com/en/docs/claude-code/skills)
- [MCP Protocol](https://modelcontextprotocol.io/)
- SPEC-01: Remote Environment Provisioning
