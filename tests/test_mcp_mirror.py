"""Tests for MCP configuration mirroring for remote environments.

@trace SPEC-07.12 - MCP Configuration Transform
"""

import pytest


class TestMCPMirror:
    """Tests for MCPMirror class - SPEC-07.12."""

    @pytest.fixture
    def mirror(self):
        """Create MCPMirror instance."""
        from parhelia.mcp_mirror import MCPMirror

        return MCPMirror()

    def test_mirror_initialization(self, mirror):
        """@trace SPEC-07.12 - MCPMirror MUST initialize with volume path."""
        assert mirror is not None
        assert mirror.volume_path == "/vol/parhelia"

    def test_mirror_custom_volume_path(self):
        """@trace SPEC-07.12 - MCPMirror SHOULD support custom volume path."""
        from parhelia.mcp_mirror import MCPMirror

        mirror = MCPMirror(volume_path="/custom/vol")
        assert mirror.volume_path == "/custom/vol"

    @pytest.mark.asyncio
    async def test_transform_empty_config(self, mirror):
        """@trace SPEC-07.12 - MUST handle empty config."""
        config = {}
        result = await mirror.transform(config)

        assert result == {"mcpServers": {}}

    @pytest.mark.asyncio
    async def test_transform_empty_servers(self, mirror):
        """@trace SPEC-07.12 - MUST handle config with no servers."""
        config = {"mcpServers": {}}
        result = await mirror.transform(config)

        assert result == {"mcpServers": {}}

    @pytest.mark.asyncio
    async def test_transform_simple_server(self, mirror):
        """@trace SPEC-07.12 - MUST transform simple server config."""
        config = {
            "mcpServers": {
                "test-server": {
                    "command": "node",
                    "args": ["server.js"],
                }
            }
        }
        result = await mirror.transform(config)

        assert "test-server" in result["mcpServers"]
        assert result["mcpServers"]["test-server"]["command"] == "node"
        assert result["mcpServers"]["test-server"]["args"] == ["server.js"]

    @pytest.mark.asyncio
    async def test_transform_server_with_home_paths(self, mirror):
        """@trace SPEC-07.12 - MUST transform home paths in server config."""
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "/Users/alice/bin/server",
                    "args": ["--config", "/Users/alice/.config/server.yaml"],
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["my-server"]
        assert server["command"] == "/root/bin/server"
        assert server["args"] == ["--config", "/root/.config/server.yaml"]

    @pytest.mark.asyncio
    async def test_transform_server_with_cwd(self, mirror):
        """@trace SPEC-07.12 - MUST transform cwd path."""
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "python",
                    "args": ["-m", "server"],
                    "cwd": "/home/bob/projects/server",
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["my-server"]
        assert server["cwd"] == "/root/projects/server"

    @pytest.mark.asyncio
    async def test_transform_server_with_env(self, mirror):
        """@trace SPEC-07.12 - MUST transform env paths."""
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {
                        "HOME": "/Users/alice",
                        "CONFIG_PATH": "/Users/alice/.config/app",
                        "API_KEY": "secret-123",
                    },
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["my-server"]
        assert server["env"]["HOME"] == "/root"
        assert server["env"]["CONFIG_PATH"] == "/root/.config/app"
        assert server["env"]["API_KEY"] == "secret-123"

    @pytest.mark.asyncio
    async def test_transform_claude_plugin_paths(self, mirror):
        """@trace SPEC-07.12 - MUST transform .claude/plugins paths."""
        config = {
            "mcpServers": {
                "beads": {
                    "command": "node",
                    "args": [
                        "/Users/rand/.claude/plugins/beads/dist/server.js",
                    ],
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["beads"]
        assert server["args"] == [
            "/vol/parhelia/plugins/beads/dist/server.js",
        ]

    @pytest.mark.asyncio
    async def test_transform_multiple_servers(self, mirror):
        """@trace SPEC-07.12 - MUST transform all servers in config."""
        config = {
            "mcpServers": {
                "server1": {
                    "command": "/Users/alice/bin/s1",
                    "args": [],
                },
                "server2": {
                    "command": "/home/bob/bin/s2",
                    "args": [],
                },
            }
        }
        result = await mirror.transform(config)

        assert result["mcpServers"]["server1"]["command"] == "/root/bin/s1"
        assert result["mcpServers"]["server2"]["command"] == "/root/bin/s2"

    @pytest.mark.asyncio
    async def test_transform_preserves_system_paths(self, mirror):
        """@trace SPEC-07.12 - MUST preserve system paths."""
        config = {
            "mcpServers": {
                "node-server": {
                    "command": "/usr/local/bin/node",
                    "args": ["/app/server.js"],
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["node-server"]
        assert server["command"] == "/usr/local/bin/node"
        assert server["args"] == ["/app/server.js"]

    @pytest.mark.asyncio
    async def test_transform_preserves_null_cwd(self, mirror):
        """@trace SPEC-07.12 - MUST preserve null/missing cwd."""
        config = {
            "mcpServers": {
                "simple": {
                    "command": "node",
                    "args": ["server.js"],
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["simple"]
        assert server.get("cwd") is None


class TestMCPServerFilter:
    """Tests for MCP server filtering."""

    @pytest.fixture
    def mirror(self):
        """Create MCPMirror instance."""
        from parhelia.mcp_mirror import MCPMirror

        return MCPMirror()

    @pytest.mark.asyncio
    async def test_filter_local_only_servers(self, mirror):
        """@trace SPEC-07.12 - SHOULD filter servers marked local-only."""
        config = {
            "mcpServers": {
                "remote-ok": {
                    "command": "node",
                    "args": ["server.js"],
                },
                "local-only": {
                    "command": "node",
                    "args": ["local.js"],
                    "parhelia": {"localOnly": True},
                },
            }
        }
        result = await mirror.transform(config)

        assert "remote-ok" in result["mcpServers"]
        assert "local-only" not in result["mcpServers"]

    @pytest.mark.asyncio
    async def test_filter_preserves_non_filtered(self, mirror):
        """@trace SPEC-07.12 - MUST preserve servers not filtered."""
        config = {
            "mcpServers": {
                "server1": {"command": "cmd1", "args": []},
                "server2": {"command": "cmd2", "args": []},
            }
        }
        result = await mirror.transform(config)

        assert len(result["mcpServers"]) == 2


class TestMCPEnvInjection:
    """Tests for environment variable injection."""

    @pytest.fixture
    def mirror(self):
        """Create MCPMirror with env injection."""
        from parhelia.mcp_mirror import MCPMirror

        return MCPMirror(
            inject_env={
                "PARHELIA_REMOTE": "true",
                "PARHELIA_VOLUME": "/vol/parhelia",
            }
        )

    @pytest.mark.asyncio
    async def test_inject_env_to_server(self, mirror):
        """@trace SPEC-07.12 - SHOULD inject env vars to servers."""
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {"EXISTING": "value"},
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["my-server"]
        assert server["env"]["EXISTING"] == "value"
        assert server["env"]["PARHELIA_REMOTE"] == "true"
        assert server["env"]["PARHELIA_VOLUME"] == "/vol/parhelia"

    @pytest.mark.asyncio
    async def test_inject_env_to_server_without_env(self, mirror):
        """@trace SPEC-07.12 - SHOULD inject env even if none defined."""
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                }
            }
        }
        result = await mirror.transform(config)

        server = result["mcpServers"]["my-server"]
        assert server["env"]["PARHELIA_REMOTE"] == "true"
