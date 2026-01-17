"""Tests for path transformation for remote environments.

@trace SPEC-07.12 - MCP Configuration Transform (path transformation)
"""

import pytest


class TestPathTransformer:
    """Tests for PathTransformer class - SPEC-07.12."""

    @pytest.fixture
    def transformer(self):
        """Create PathTransformer instance."""
        from parhelia.path_transformer import PathTransformer

        return PathTransformer()

    def test_transformer_initialization(self, transformer):
        """@trace SPEC-07.12 - PathTransformer MUST initialize with volume path."""
        assert transformer is not None
        assert transformer.volume_path == "/vol/parhelia"

    def test_transformer_custom_volume_path(self):
        """@trace SPEC-07.12 - PathTransformer SHOULD support custom volume path."""
        from parhelia.path_transformer import PathTransformer

        transformer = PathTransformer(volume_path="/custom/vol")
        assert transformer.volume_path == "/custom/vol"

    def test_transform_tilde_to_root(self, transformer):
        """@trace SPEC-07.12 - MUST transform ~ to /root."""
        result = transformer.transform("~/projects/myapp")
        assert result == "/root/projects/myapp"

    def test_transform_macos_users_path(self, transformer):
        """@trace SPEC-07.12 - MUST transform /Users/username to /root."""
        result = transformer.transform("/Users/alice/projects/myapp")
        assert result == "/root/projects/myapp"

    def test_transform_linux_home_path(self, transformer):
        """@trace SPEC-07.12 - MUST transform /home/username to /root."""
        result = transformer.transform("/home/bob/projects/myapp")
        assert result == "/root/projects/myapp"

    def test_transform_claude_plugins_path(self, transformer):
        """@trace SPEC-07.12 - MUST transform .claude/plugins to volume plugins."""
        result = transformer.transform("/Users/alice/.claude/plugins/cc-polymath")
        assert result == "/vol/parhelia/plugins/cc-polymath"

    def test_transform_claude_skills_path(self, transformer):
        """@trace SPEC-07.12 - MUST transform .claude/skills to volume skills."""
        result = transformer.transform("/home/bob/.claude/skills/custom-skill")
        assert result == "/vol/parhelia/skills/custom-skill"

    def test_transform_claude_config_path(self, transformer):
        """@trace SPEC-07.12 - MUST transform .claude/* to volume config."""
        result = transformer.transform("/Users/alice/.claude/settings.json")
        assert result == "/vol/parhelia/config/claude/settings.json"

    def test_transform_claude_mcp_config(self, transformer):
        """@trace SPEC-07.12 - MUST transform mcp_config.json path."""
        result = transformer.transform("/home/user/.claude/mcp_config.json")
        assert result == "/vol/parhelia/config/claude/mcp_config.json"

    def test_transform_preserves_absolute_paths(self, transformer):
        """@trace SPEC-07.12 - MUST preserve system paths not matching patterns."""
        result = transformer.transform("/usr/local/bin/node")
        assert result == "/usr/local/bin/node"

    def test_transform_preserves_relative_paths(self, transformer):
        """@trace SPEC-07.12 - MUST preserve relative paths."""
        result = transformer.transform("./node_modules/.bin/tsx")
        assert result == "./node_modules/.bin/tsx"

    def test_transform_handles_nested_paths(self, transformer):
        """@trace SPEC-07.12 - MUST handle deeply nested plugin paths."""
        result = transformer.transform(
            "/Users/rand/.claude/plugins/my-plugin/dist/server.js"
        )
        assert result == "/vol/parhelia/plugins/my-plugin/dist/server.js"

    def test_transform_handles_usernames_with_underscores(self, transformer):
        """@trace SPEC-07.12 - MUST handle usernames with special chars."""
        result = transformer.transform("/Users/john_doe/projects")
        assert result == "/root/projects"

    def test_transform_handles_usernames_with_numbers(self, transformer):
        """@trace SPEC-07.12 - MUST handle usernames with numbers."""
        result = transformer.transform("/home/user123/workspace")
        assert result == "/root/workspace"


class TestPathTransformList:
    """Tests for batch path transformation."""

    @pytest.fixture
    def transformer(self):
        """Create PathTransformer instance."""
        from parhelia.path_transformer import PathTransformer

        return PathTransformer()

    def test_transform_list(self, transformer):
        """@trace SPEC-07.12 - MUST support transforming list of paths."""
        paths = [
            "/Users/alice/projects",
            "~/config",
            "/usr/bin/python",
        ]
        results = transformer.transform_list(paths)

        assert len(results) == 3
        assert results[0] == "/root/projects"
        assert results[1] == "/root/config"
        assert results[2] == "/usr/bin/python"

    def test_transform_list_empty(self, transformer):
        """@trace SPEC-07.12 - MUST handle empty list."""
        results = transformer.transform_list([])
        assert results == []


class TestPathTransformEnv:
    """Tests for environment variable path transformation."""

    @pytest.fixture
    def transformer(self):
        """Create PathTransformer instance."""
        from parhelia.path_transformer import PathTransformer

        return PathTransformer()

    def test_transform_env(self, transformer):
        """@trace SPEC-07.12 - MUST transform paths in env dict."""
        env = {
            "HOME": "/Users/alice",
            "PLUGIN_PATH": "/Users/alice/.claude/plugins/test",
            "API_KEY": "secret-key-123",  # Non-path values preserved
        }
        result = transformer.transform_env(env)

        assert result["HOME"] == "/root"
        assert result["PLUGIN_PATH"] == "/vol/parhelia/plugins/test"
        assert result["API_KEY"] == "secret-key-123"

    def test_transform_env_empty(self, transformer):
        """@trace SPEC-07.12 - MUST handle empty env dict."""
        result = transformer.transform_env({})
        assert result == {}
