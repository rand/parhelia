"""MCP configuration mirroring for remote environments.

Implements:
- [SPEC-07.12] MCP Configuration Transform
"""

from __future__ import annotations

from dataclasses import dataclass, field

from parhelia.path_transformer import PathTransformer


@dataclass
class MCPMirror:
    """Transform MCP config for remote environment.

    Implements [SPEC-07.12].

    The mirror:
    - Transforms paths in command, args, cwd, and env
    - Filters out servers marked as local-only
    - Injects environment variables for remote context
    """

    volume_path: str = "/vol/parhelia"
    inject_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize path transformer."""
        self._transformer = PathTransformer(volume_path=self.volume_path)

    async def transform(self, local_config: dict) -> dict:
        """Transform local MCP config for remote.

        Args:
            local_config: Local MCP configuration dictionary.

        Returns:
            Transformed configuration for remote container.
        """
        transformed = {"mcpServers": {}}

        for name, server in local_config.get("mcpServers", {}).items():
            # Check if server is marked local-only
            if self._is_local_only(server):
                continue

            transformed_server = await self._transform_server(server)
            if transformed_server:
                transformed["mcpServers"][name] = transformed_server

        return transformed

    def _is_local_only(self, server: dict) -> bool:
        """Check if server is marked as local-only.

        Args:
            server: Server configuration.

        Returns:
            True if server should not be synced to remote.
        """
        parhelia_config = server.get("parhelia", {})
        return parhelia_config.get("localOnly", False)

    async def _transform_server(self, server: dict) -> dict:
        """Transform a single MCP server config.

        Args:
            server: Server configuration dictionary.

        Returns:
            Transformed server configuration.
        """
        # Transform command
        command = server.get("command", "")
        command = self._transformer.transform(command)

        # Transform args
        args = server.get("args", [])
        args = self._transformer.transform_list(args)

        # Transform cwd
        cwd = server.get("cwd")
        if cwd:
            cwd = self._transformer.transform(cwd)

        # Transform env paths and inject additional env vars
        env = {}
        for key, value in server.get("env", {}).items():
            env[key] = self._transformer.transform(value)

        # Inject environment variables
        for key, value in self.inject_env.items():
            env[key] = value

        result = {
            "command": command,
            "args": args,
            "env": env,
            "cwd": cwd,
        }

        return result
