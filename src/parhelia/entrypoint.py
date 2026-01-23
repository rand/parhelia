"""Container entrypoint initialization.

Implements [SPEC-01.13] Environment Initialization.

This module provides Python functions for container initialization,
mirroring the functionality of scripts/entrypoint.sh for testing
and programmatic use.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration constants
VOLUME_ROOT = os.environ.get("PARHELIA_VOLUME_ROOT", "/vol/parhelia")
VOLUME_CONFIG_PATH = f"{VOLUME_ROOT}/config/claude"
WORKSPACE_DIR = f"{VOLUME_ROOT}/workspaces"
READY_FILE_PATH = "/tmp/ready"
# Claude Code binary path - default for non-root parhelia user
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/home/parhelia/.local/bin/claude")


def log(message: str) -> None:
    """Log message with prefix."""
    print(f"[parhelia-entrypoint] {message}", flush=True)


def error(message: str) -> None:
    """Log error message."""
    print(f"[parhelia-entrypoint] ERROR: {message}", file=sys.stderr, flush=True)


def init_environment(
    skip_claude_check: bool = False,
    skip_tmux: bool = False,
    skip_mcp: bool = False,
    skip_hooks: bool = False,
) -> bool:
    """Initialize the container environment.

    Args:
        skip_claude_check: Skip Claude Code verification (for testing)
        skip_tmux: Skip tmux initialization (for testing)
        skip_mcp: Skip MCP server startup
        skip_hooks: Skip hooks validation

    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        # 1. Link configuration
        link_config()

        # 2. Verify Claude Code
        if not skip_claude_check:
            if not verify_claude_code():
                return False

        # 3. Validate and fix hooks
        if not skip_hooks:
            verify_hooks()

        # 4. Start MCP servers
        if not skip_mcp:
            start_mcp_servers()

        # 5. Initialize tmux
        if not skip_tmux:
            init_tmux()

        # 6. Signal ready
        signal_ready()

        return True

    except Exception as e:
        error(f"Initialization failed: {e}")
        return False


def link_config() -> None:
    """Create symlink from ~/.claude to volume config.

    Implements step 1 of [SPEC-01.13].
    """
    home = Path.home()
    claude_dir = home / ".claude"
    volume_config = Path(VOLUME_CONFIG_PATH)

    if volume_config.exists():
        # Remove existing symlink or directory
        if claude_dir.is_symlink():
            claude_dir.unlink()
        elif claude_dir.exists():
            # Backup existing config
            backup = home / ".claude.backup"
            claude_dir.rename(backup)
            log(f"Backed up existing ~/.claude to {backup}")

        # Create symlink
        claude_dir.symlink_to(volume_config)
        log(f"Linked ~/.claude -> {volume_config}")
    else:
        log(f"Warning: Volume config not found at {volume_config}")
        claude_dir.mkdir(parents=True, exist_ok=True)


def verify_claude_code() -> bool:
    """Verify Claude Code installation.

    Implements step 2 of [SPEC-01.13].

    Returns:
        True if Claude Code is installed and working
    """
    claude_paths = [CLAUDE_BIN, "claude"]

    for claude_path in claude_paths:
        try:
            result = subprocess.run(
                [claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                log(f"Claude Code version: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    error(f"Claude Code not found at {CLAUDE_BIN} or in PATH")
    return False


def start_mcp_servers() -> int | None:
    """Start MCP servers if launcher is available.

    Implements step 3 of [SPEC-01.13].

    Returns:
        PID of MCP launcher process, or None if not started
    """
    try:
        # Check if launcher exists
        result = subprocess.run(
            ["which", "parhelia-mcp-launcher"],
            capture_output=True,
        )
        if result.returncode != 0:
            log("MCP launcher not found, skipping")
            return None

        # Start launcher in background
        process = subprocess.Popen(
            ["parhelia-mcp-launcher"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log(f"MCP launcher started (PID: {process.pid})")
        return process.pid

    except Exception as e:
        log(f"Failed to start MCP launcher: {e}")
        return None


def init_tmux() -> None:
    """Initialize tmux session.

    Implements step 4 of [SPEC-01.13].
    """
    workspace = Path(WORKSPACE_DIR)
    workspace.mkdir(parents=True, exist_ok=True)

    # Kill existing tmux server (clean slate)
    subprocess.run(["tmux", "kill-server"], capture_output=True)

    # Create new session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", "main", "-c", str(workspace)],
        check=True,
    )
    log(f"tmux session 'main' created in {workspace}")


def verify_hooks() -> bool:
    """Verify Claude Code hooks are properly configured.

    Validates that hook scripts exist and are executable.
    Automatically fixes executable permissions if needed.

    Returns:
        True if all hooks are valid (or no hooks configured)
    """
    try:
        from parhelia.hook_validator import HookValidator

        validator = HookValidator()
        result = validator.validate()

        if result.hooks_found == 0:
            log("No Claude hooks configured")
            return True

        # Log validation result
        log(f"Hooks validation: {result.summary()}")

        # Report issues
        for issue in result.issues:
            if issue.severity == "error":
                error(f"Hook {issue.hook_type}: {issue.issue}")
            else:
                log(f"Hook {issue.hook_type}: {issue.issue}")

        # Auto-fix permissions
        if not result.all_valid:
            fixed = validator.fix_permissions()
            if fixed > 0:
                log(f"Fixed permissions on {fixed} hook(s)")
                # Re-validate after fix
                result = validator.validate()
                log(f"After fix: {result.summary()}")

        return result.all_valid

    except Exception as e:
        log(f"Hook validation skipped: {e}")
        return True


def signal_ready() -> None:
    """Signal that the container is ready.

    Implements step 6 of [SPEC-01.13].
    """
    ready_file = Path(READY_FILE_PATH)
    ready_file.write_text("PARHELIA_READY\n")
    log(f"Container ready ({ready_file} written)")


def main() -> int:
    """Main entrypoint function."""
    success = init_environment()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
