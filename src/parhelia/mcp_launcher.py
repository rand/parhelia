"""MCP server launcher with lazy loading support.

Implements [SPEC-01.15] Cold Start Optimization - Lazy MCP Loading.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Implements [SPEC-01.15].
    """

    name: str
    command: list[str]
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


class MCPLauncher:
    """MCP server launcher with lazy loading support.

    Implements [SPEC-01.15] - Lazy MCP loading for cold start optimization.

    When lazy=True, MCP servers are not started at initialization.
    Instead, they are started on-demand when first requested via ensure_running().
    """

    def __init__(self, lazy: bool = False, config_path: str | None = None):
        """Initialize the MCP launcher.

        Args:
            lazy: If True, servers start on-demand instead of at init.
            config_path: Path to MCP config file (mcp_config.json).
        """
        self.lazy = lazy
        self.config_path = config_path
        self._servers: dict[str, MCPServerConfig] = {}
        self._running: dict[str, subprocess.Popen] = {}

    @property
    def running_servers(self) -> dict[str, subprocess.Popen]:
        """Return dict of currently running server processes."""
        # Clean up any terminated processes
        terminated = []
        for name, proc in self._running.items():
            if proc.poll() is not None:
                terminated.append(name)

        for name in terminated:
            del self._running[name]

        return self._running

    def register_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration.

        Args:
            config: Server configuration to register.
        """
        self._servers[config.name] = config

    def ensure_running(self, server_name: str) -> subprocess.Popen:
        """Ensure an MCP server is running, starting it if needed.

        Implements lazy loading - server starts on first request.

        Args:
            server_name: Name of the server to ensure is running.

        Returns:
            The running process for the server.

        Raises:
            KeyError: If server_name is not registered.
            RuntimeError: If server fails to start.
        """
        # Check if already running
        if server_name in self._running:
            proc = self._running[server_name]
            if proc.poll() is None:
                # Still running
                return proc
            # Process terminated, will restart below
            del self._running[server_name]

        # Get config
        if server_name not in self._servers:
            raise KeyError(f"Server '{server_name}' not registered")

        config = self._servers[server_name]

        # Start the server
        proc = self._start_server(config)
        self._running[server_name] = proc

        return proc

    def _start_server(self, config: MCPServerConfig) -> subprocess.Popen:
        """Start an MCP server process.

        Args:
            config: Server configuration.

        Returns:
            The started process.
        """
        cmd = config.command + config.args

        env = os.environ.copy()
        env.update(config.env)

        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=config.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return proc

    def start_all(self) -> dict[str, subprocess.Popen]:
        """Start all registered servers.

        Returns:
            Dict mapping server names to their processes.
        """
        for name in self._servers:
            self.ensure_running(name)

        return self._running

    def stop_all(self) -> None:
        """Stop all running servers gracefully."""
        for name, proc in list(self._running.items()):
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            finally:
                del self._running[name]

    def stop_server(self, server_name: str) -> None:
        """Stop a specific server.

        Args:
            server_name: Name of the server to stop.
        """
        if server_name not in self._running:
            return

        proc = self._running[server_name]
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        finally:
            del self._running[server_name]

    def load_config(self, config_path: str | None = None) -> None:
        """Load MCP server configurations from file.

        Args:
            config_path: Path to config file. Defaults to self.config_path.
        """
        import json

        path = config_path or self.config_path
        if not path:
            return

        config_file = Path(path)
        if not config_file.exists():
            return

        with open(config_file) as f:
            data = json.load(f)

        # Parse mcpServers section
        servers = data.get("mcpServers", {})
        for name, server_data in servers.items():
            command = server_data.get("command", "")
            args = server_data.get("args", [])

            # Handle command as string or list
            if isinstance(command, str):
                command = [command]

            config = MCPServerConfig(
                name=name,
                command=command,
                args=args,
                env=server_data.get("env", {}),
                cwd=server_data.get("cwd"),
            )
            self.register_server(config)
