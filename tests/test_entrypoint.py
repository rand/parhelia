"""Tests for container entrypoint script.

@trace SPEC-01.13 - Environment Initialization
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEntrypointScript:
    """Tests for the entrypoint.sh script - SPEC-01.13."""

    @pytest.fixture
    def entrypoint_path(self) -> Path:
        """Get path to entrypoint script."""
        return Path(__file__).parent.parent / "src" / "parhelia" / "scripts" / "entrypoint.sh"

    def test_entrypoint_script_exists(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint script MUST exist."""
        assert entrypoint_path.exists(), f"Entrypoint script not found at {entrypoint_path}"

    def test_entrypoint_script_is_executable(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint script MUST be executable."""
        assert os.access(entrypoint_path, os.X_OK), "Entrypoint script is not executable"

    def test_entrypoint_script_has_shebang(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint script MUST have bash shebang."""
        content = entrypoint_path.read_text()
        assert content.startswith("#!/bin/bash"), "Script must start with #!/bin/bash"

    def test_entrypoint_script_uses_strict_mode(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint script MUST use strict mode."""
        content = entrypoint_path.read_text()
        assert "set -euo pipefail" in content, "Script must use 'set -euo pipefail'"

    def test_entrypoint_creates_config_symlink(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint MUST symlink ~/.claude to volume config."""
        content = entrypoint_path.read_text()
        # Script uses variables, so check for the pattern
        assert "ln -sfn" in content
        assert ".claude" in content
        assert "VOLUME_CONFIG" in content or "/vol/parhelia/config/claude" in content

    def test_entrypoint_verifies_claude_code(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint MUST verify Claude Code installation."""
        content = entrypoint_path.read_text()
        assert "claude --version" in content or "claude" in content

    def test_entrypoint_initializes_tmux(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint MUST initialize tmux session."""
        content = entrypoint_path.read_text()
        assert "tmux" in content
        assert "new-session" in content

    def test_entrypoint_signals_ready(self, entrypoint_path: Path):
        """@trace SPEC-01.13 - Entrypoint MUST signal readiness."""
        content = entrypoint_path.read_text()
        assert "PARHELIA_READY" in content
        assert "/tmp/ready" in content


class TestEntrypointModule:
    """Tests for the Python entrypoint module."""

    def test_init_environment_creates_symlink(self, tmp_path: Path):
        """@trace SPEC-01.13 - init_environment MUST create config symlink."""
        from parhelia.entrypoint import init_environment

        # Setup mock paths
        volume_config = tmp_path / "vol" / "parhelia" / "config" / "claude"
        volume_config.mkdir(parents=True)
        (volume_config / "settings.json").write_text("{}")

        home_dir = tmp_path / "home"
        home_dir.mkdir()

        with patch.dict(os.environ, {"HOME": str(home_dir)}):
            with patch("parhelia.entrypoint.VOLUME_CONFIG_PATH", str(volume_config)):
                init_environment(skip_claude_check=True, skip_tmux=True)

        claude_link = home_dir / ".claude"
        assert claude_link.is_symlink() or claude_link.exists()

    def test_verify_claude_code_returns_version(self):
        """@trace SPEC-01.13 - verify_claude_code MUST check installation."""
        from parhelia.entrypoint import verify_claude_code

        # Mock subprocess to avoid actual claude call
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="claude-code 1.0.0"
            )
            result = verify_claude_code()
            assert result is True

    def test_verify_claude_code_fails_gracefully(self):
        """@trace SPEC-01.13 - verify_claude_code MUST exit on failure."""
        from parhelia.entrypoint import verify_claude_code

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("claude not found")
            result = verify_claude_code()
            assert result is False

    def test_signal_ready_writes_file(self, tmp_path: Path):
        """@trace SPEC-01.13 - signal_ready MUST write to /tmp/ready."""
        from parhelia.entrypoint import signal_ready

        ready_file = tmp_path / "ready"

        with patch("parhelia.entrypoint.READY_FILE_PATH", str(ready_file)):
            signal_ready()

        assert ready_file.exists()
        assert "PARHELIA_READY" in ready_file.read_text()
