"""Path transformation for remote environments.

Implements:
- [SPEC-07.12] MCP Configuration Transform (path transformation)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PathTransformer:
    """Transform local paths to remote container paths.

    Implements [SPEC-07.12].

    Modal containers run as root by default, so ~ expands to /root.
    This transformer converts:
    - ~ -> /root
    - /Users/<username> -> /root (macOS)
    - /home/<username> -> /root (Linux)
    - ~/.claude/plugins/... -> /vol/parhelia/plugins/...
    - ~/.claude/skills/... -> /vol/parhelia/skills/...
    - ~/.claude/... -> /vol/parhelia/config/claude/...
    """

    volume_path: str = "/vol/parhelia"

    def transform(self, path: str) -> str:
        """Transform a local path to its remote equivalent.

        Args:
            path: Local path to transform.

        Returns:
            Transformed path for remote container.
        """
        if not path:
            return path

        # First, handle .claude directory transformations (before home dir transform)
        # This needs to happen first to capture the full path structure

        # Check for .claude/ in the path (after any home dir prefix)
        claude_match = re.search(r"/.claude/(.*)", path)
        if claude_match:
            rest = claude_match.group(1)

            # Plugins go to Volume plugins dir
            if rest.startswith("plugins/"):
                plugin_rest = rest[8:]  # Remove "plugins/"
                return f"{self.volume_path}/plugins/{plugin_rest}"

            # Skills go to Volume skills dir
            if rest.startswith("skills/"):
                skill_rest = rest[7:]  # Remove "skills/"
                return f"{self.volume_path}/skills/{skill_rest}"

            # Other config files go to Volume config
            return f"{self.volume_path}/config/claude/{rest}"

        # Home directory transformations
        # Replace ~ with /root
        path = path.replace("~", "/root")

        # macOS: /Users/username -> /root
        # Username can contain letters, numbers, underscores, hyphens
        path = re.sub(r"/Users/[\w-]+", "/root", path)

        # Linux: /home/username -> /root
        path = re.sub(r"/home/[\w-]+", "/root", path)

        return path

    def transform_list(self, paths: list[str]) -> list[str]:
        """Transform a list of paths.

        Args:
            paths: List of paths to transform.

        Returns:
            List of transformed paths.
        """
        return [self.transform(p) for p in paths]

    def transform_env(self, env: dict[str, str]) -> dict[str, str]:
        """Transform path values in an environment dictionary.

        Non-path values (those not starting with / or ~) are preserved.

        Args:
            env: Environment variable dictionary.

        Returns:
            Dictionary with path values transformed.
        """
        result = {}
        for key, value in env.items():
            # Only transform values that look like paths
            if value.startswith("/") or value.startswith("~"):
                result[key] = self.transform(value)
            else:
                result[key] = value
        return result
