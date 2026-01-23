"""Modal app definition for Parhelia.

Implements:
- [SPEC-01.10] Container Variants (CPU and GPU)
- [SPEC-01.11] Image Definition
- [SPEC-01.12] Volume Mounting

Key Design Decision: Use Sandboxes for interactive Claude Code sessions
(dynamic, long-lived), and Functions only for short batch operations.

Usage:
    # Deploy the app
    modal deploy src/parhelia/modal_app.py

    # Run health check
    modal run src/parhelia/modal_app.py::health_check

    # Initialize volume structure
    modal run src/parhelia/modal_app.py::init_volume_structure

    # Local development
    uv run python -m parhelia.modal_app
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

from parhelia.config import load_config

# Load configuration
config = load_config()

# Get the directory containing this file for local file references
_PACKAGE_DIR = Path(__file__).parent

# =============================================================================
# Modal App Definition - SPEC-01.11
# =============================================================================

app = modal.App("parhelia")

# =============================================================================
# Volume Definition - SPEC-01.12
# =============================================================================

volume = modal.Volume.from_name(config.modal.volume_name, create_if_missing=True)

# =============================================================================
# Container Variant Configuration - SPEC-01.10
# =============================================================================

CPU_CONFIG = {
    "cpu": config.modal.cpu_count,
    "memory": config.modal.memory_mb,
}

SUPPORTED_GPUS = ["A10G", "A100", "H100", "T4"]

# =============================================================================
# Image Definitions - SPEC-01.11
# =============================================================================

# Non-root user for Claude Code compatibility
# Claude Code blocks --dangerously-skip-permissions when running as root
CONTAINER_USER = "parhelia"
CONTAINER_HOME = f"/home/{CONTAINER_USER}"

# Base image for CPU workloads
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "tmux",
        "openssh-server",
        "git",
        "curl",
        "build-essential",
        "unzip",
        "procps",  # For ps, top commands
        "zstd",  # For checkpoint compression
        "sudo",  # For occasional privileged operations
        "pkg-config",  # For Rust crates that need system libraries
        "libssl-dev",  # OpenSSL development headers
        "libclang-dev",  # For bindgen (FFI generation)
    ])
    .pip_install([
        "anthropic>=0.40.0",
        "prometheus-client>=0.21.0",
        "aiofiles>=24.0.0",
        "psutil>=5.9.0",
        "toml>=0.10.0",
    ])
    # Create non-root user for Claude Code compatibility
    # This is required because Claude Code blocks --dangerously-skip-permissions as root
    .run_commands([
        # Create user with home directory
        f"useradd -m -s /bin/bash {CONTAINER_USER}",
        # Add to sudo group for occasional privileged ops (passwordless)
        f"usermod -aG sudo {CONTAINER_USER}",
        "echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers",
        # Create volume mount point with proper ownership
        "mkdir -p /vol/parhelia",
        f"chown -R {CONTAINER_USER}:{CONTAINER_USER} /vol/parhelia",
    ])
    # Copy entrypoint script into the image (copy=True needed for subsequent run_commands)
    .add_local_file(
        str(_PACKAGE_DIR / "scripts" / "entrypoint.sh"),
        "/entrypoint.sh",
        copy=True,
    )
    .run_commands([
        "chmod +x /entrypoint.sh",
    ])
    # Add parhelia package source for remote functions
    .add_local_dir(str(_PACKAGE_DIR), f"{CONTAINER_HOME}/parhelia", copy=True)
    # Fix ownership of entire home directory after all copies complete
    # This is critical - Modal's copy operations run as root
    .run_commands([
        f"chown -R {CONTAINER_USER}:{CONTAINER_USER} {CONTAINER_HOME}",
    ])
    # Install tools as non-root user (requires correct home dir ownership)
    .run_commands([
        # Install Bun for plugin tooling (as parhelia user)
        f"su - {CONTAINER_USER} -c 'curl -fsSL https://bun.sh/install | bash'",
        # Install Claude Code native binary (as parhelia user)
        f"su - {CONTAINER_USER} -c 'curl -fsSL https://claude.ai/install.sh | bash'",
        # Verify installation
        f"su - {CONTAINER_USER} -c '$HOME/.local/bin/claude --version' || echo 'Claude Code installation failed'",
        # Install Rust via rustup (as parhelia user)
        f"su - {CONTAINER_USER} -c 'curl --proto \"=https\" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'",
        # Verify Rust installation
        f"su - {CONTAINER_USER} -c '$HOME/.cargo/bin/rustc --version' || echo 'Rust installation failed'",
    ])
    # Configure environment for parhelia user
    .run_commands([
        f"echo 'export BUN_INSTALL=\"$HOME/.bun\"' >> {CONTAINER_HOME}/.bashrc",
        f"echo 'source \"$HOME/.cargo/env\"' >> {CONTAINER_HOME}/.bashrc",
        f"echo 'export PATH=\"$HOME/.local/bin:$BUN_INSTALL/bin:$HOME/.cargo/bin:$PATH\"' >> {CONTAINER_HOME}/.bashrc",
        f"echo 'export PYTHONPATH={CONTAINER_HOME}:$PYTHONPATH' >> {CONTAINER_HOME}/.bashrc",
    ])
    .env({
        "PYTHONPATH": CONTAINER_HOME,
        "HOME": CONTAINER_HOME,
        "USER": CONTAINER_USER,
    })
)

# GPU image extends CPU with CUDA support
gpu_image = cpu_image.run_commands([
    "pip install torch --index-url https://download.pytorch.org/whl/cu121",
])

# Secrets configuration - these must be created in Modal dashboard
# modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-...
# modal secret create github-token GITHUB_TOKEN=ghp_...
#
# For testing without secrets, set PARHELIA_SKIP_SECRETS=1
_SKIP_SECRETS = os.environ.get("PARHELIA_SKIP_SECRETS", "").lower() in ("1", "true", "yes")

# When skipping secrets, update images to include the env var for remote consistency
if _SKIP_SECRETS:
    cpu_image = cpu_image.env({"PARHELIA_SKIP_SECRETS": "1"})
    gpu_image = gpu_image.env({"PARHELIA_SKIP_SECRETS": "1"})
    PARHELIA_SECRETS: list = []
else:
    PARHELIA_SECRETS = [
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("github-token"),
    ]

# =============================================================================
# Sandbox Creation - For Interactive Claude Code Sessions
# =============================================================================


# Maximum memory without GPU: 336GB (344064 MiB) per Modal limits
MAX_MEMORY_MB = 344064


async def create_claude_sandbox(
    task_id: str,
    gpu: str | None = None,
    timeout_hours: int | None = None,
    memory_mb: int | None = None,
    cpu: float | None = None,
) -> modal.Sandbox:
    """Create a Sandbox for interactive Claude Code session.

    Sandboxes are preferred for:
    - Long-lived sessions (up to 24h)
    - Interactive stdin/stdout
    - Dynamic workloads
    - Sessions that may need SSH attachment

    Args:
        task_id: Unique identifier for the task/session
        gpu: GPU type (A10G, A100, etc.) or None for CPU-only
        timeout_hours: Session timeout in hours (default from config)
        memory_mb: Memory in MB (default from config, max 336GB without GPU)
        cpu: CPU cores (default from config)

    Returns:
        modal.Sandbox instance ready for Claude Code execution

    Raises:
        ValueError: If gpu is specified but not in SUPPORTED_GPUS
        ValueError: If memory_mb exceeds Modal's limit (336GB without GPU)
    """
    # Validate GPU type
    if gpu is not None and gpu not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU '{gpu}'. Must be one of: {SUPPORTED_GPUS}")

    # Use defaults from config if not specified
    if timeout_hours is None:
        timeout_hours = config.modal.default_timeout_hours
    if memory_mb is None:
        memory_mb = CPU_CONFIG["memory"]
    if cpu is None:
        cpu = CPU_CONFIG["cpu"]

    # Validate memory limit
    if memory_mb > MAX_MEMORY_MB:
        raise ValueError(f"Memory {memory_mb}MB exceeds Modal limit of {MAX_MEMORY_MB}MB (336GB)")

    image = gpu_image if gpu else cpu_image

    # Get app reference - required when running outside Modal container
    parhelia_app = modal.App.lookup("parhelia", create_if_missing=True)

    sandbox = await modal.Sandbox.create.aio(
        app=parhelia_app,
        image=image,
        secrets=PARHELIA_SECRETS,
        volumes={"/vol/parhelia": volume},
        gpu=gpu,
        timeout=timeout_hours * 3600,
        cpu=cpu,
        memory=memory_mb,
        # Pass task_id and user context as environment variables
        env={
            "PARHELIA_TASK_ID": task_id,
            "HOME": CONTAINER_HOME,
            "USER": CONTAINER_USER,
            "PYTHONPATH": CONTAINER_HOME,
        },
    )

    return sandbox


async def run_in_sandbox(
    sandbox: modal.Sandbox,
    command: list[str],
    timeout_seconds: int = 300,
    as_root: bool = False,
) -> str:
    """Execute command in sandbox and return output.

    Commands run as the non-root parhelia user by default. This is required
    for Claude Code compatibility (it blocks --dangerously-skip-permissions as root).

    Args:
        sandbox: The Modal Sandbox instance
        command: Command and arguments to execute
        timeout_seconds: Maximum time to wait for command (default 5 minutes)
        as_root: Run as root instead of parhelia user (default False)

    Returns:
        stdout from the command
    """
    # Build command string with proper escaping
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in command)

    # For simple commands (ls, echo, cat), use direct execution
    simple_commands = {"ls", "cat", "echo", "pwd", "env", "whoami", "which", "head", "tail"}
    is_simple = command and command[0].split("/")[-1] in simple_commands

    if is_simple and as_root:
        # Simple command as root - direct execution
        process = await sandbox.exec.aio(*command)
        stdout_lines = []
        for line in process.stdout:
            stdout_lines.append(line)
        await process.wait.aio()
        return "".join(stdout_lines)

    if is_simple:
        # Simple command as parhelia user using sudo -E to preserve environment
        wrapper = f"sudo -E -u {CONTAINER_USER} bash -c '{cmd_str}'"
        process = await sandbox.exec.aio("bash", "-c", wrapper)
        stdout_lines = []
        for line in process.stdout:
            stdout_lines.append(line)
        await process.wait.aio()
        return "".join(stdout_lines)

    # For complex commands (like Claude Code), close stdin to ensure clean exit.
    # Claude Code in -p mode waits for stdin by default; closing it with < /dev/null
    # allows the process to exit cleanly after producing output.
    if as_root:
        wrapper = f"timeout {timeout_seconds} {cmd_str} < /dev/null 2>&1"
    else:
        # Run as parhelia user with sudo -E to preserve environment (esp. ANTHROPIC_API_KEY)
        # Use login shell (-i) to get PATH from .bashrc
        escaped_cmd = cmd_str.replace("'", "'\\''")
        wrapper = f"sudo -E -u {CONTAINER_USER} bash -l -c 'timeout {timeout_seconds} {escaped_cmd} < /dev/null 2>&1'"

    process = await sandbox.exec.aio("bash", "-c", wrapper)
    stdout_lines = []
    for line in process.stdout:
        stdout_lines.append(line)
    await process.wait.aio()
    return "".join(stdout_lines)


# =============================================================================
# Functions - For Short Batch Operations Only
# =============================================================================


@app.function(
    image=cpu_image,
    volumes={"/vol/parhelia": volume},
    secrets=PARHELIA_SECRETS,
    cpu=CPU_CONFIG["cpu"],
    memory=CPU_CONFIG["memory"],
    timeout=300,  # 5 min max for batch ops
)
def health_check() -> dict:
    """Quick health check for the Parhelia environment.

    Returns:
        dict with status information
    """
    import os
    import subprocess

    result = {
        "status": "ok",
        "volume_mounted": os.path.exists("/vol/parhelia"),
        "claude_installed": False,
        "anthropic_key_set": "ANTHROPIC_API_KEY" in os.environ,
    }

    # Check Claude Code installation - installed to ~/.local/bin/claude
    claude_bin = os.path.expanduser("~/.local/bin/claude")
    try:
        claude_check = subprocess.run(
            [claude_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["claude_installed"] = claude_check.returncode == 0
        result["claude_version"] = claude_check.stdout.strip()
        result["claude_path"] = claude_bin
    except Exception as e:
        result["claude_error"] = str(e)
        result["claude_path"] = claude_bin

    return result


@app.function(
    image=cpu_image,
    volumes={"/vol/parhelia": volume},
    timeout=60,
)
def init_volume_structure() -> dict:
    """Initialize the volume directory structure.

    Creates the required directories if they don't exist:
    - /vol/parhelia/config/
    - /vol/parhelia/plugins/
    - /vol/parhelia/checkpoints/
    - /vol/parhelia/workspaces/

    Returns:
        dict with created directories
    """
    import os

    base = "/vol/parhelia"
    directories = [
        f"{base}/config/claude",
        f"{base}/config/env",
        f"{base}/plugins",
        f"{base}/checkpoints",
        f"{base}/workspaces",
    ]

    created = []
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            created.append(directory)

    # Commit volume changes
    volume.commit()

    return {"created": created, "all_directories": directories}


# =============================================================================
# Sandbox Manager - Track and manage active sandboxes
# =============================================================================


@dataclass
class SandboxInfo:
    """Information about an active sandbox.

    Implements [SPEC-01].
    """

    sandbox_id: str
    task_id: str
    gpu: str | None
    created_at: datetime = field(default_factory=datetime.now)
    sandbox: Any = None  # modal.Sandbox type at runtime


class SandboxManager:
    """Manage active sandboxes and their lifecycle.

    Implements [SPEC-01].
    """

    DEFAULT_MAX_SANDBOXES = 10

    def __init__(self, max_sandboxes: int | None = None):
        """Initialize the sandbox manager.

        Args:
            max_sandboxes: Maximum number of concurrent sandboxes.
        """
        self.max_sandboxes = max_sandboxes or self.DEFAULT_MAX_SANDBOXES
        self.active_sandboxes: dict[str, SandboxInfo] = {}

    def can_accept_sandbox(self) -> bool:
        """Check if we can accept a new sandbox.

        Returns:
            True if capacity available.
        """
        return len(self.active_sandboxes) < self.max_sandboxes

    def register_sandbox(
        self,
        sandbox_id: str,
        task_id: str,
        sandbox: Any,
        gpu: str | None = None,
    ) -> SandboxInfo:
        """Register a new active sandbox.

        Args:
            sandbox_id: Unique sandbox identifier.
            task_id: Associated task ID.
            sandbox: The Modal Sandbox instance.
            gpu: GPU type if GPU sandbox.

        Returns:
            SandboxInfo for the registered sandbox.
        """
        info = SandboxInfo(
            sandbox_id=sandbox_id,
            task_id=task_id,
            gpu=gpu,
            sandbox=sandbox,
        )
        self.active_sandboxes[sandbox_id] = info
        return info

    def unregister_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Remove a sandbox from tracking.

        Args:
            sandbox_id: The sandbox to unregister.

        Returns:
            The removed SandboxInfo, or None if not found.
        """
        return self.active_sandboxes.pop(sandbox_id, None)

    def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get sandbox info by ID.

        Args:
            sandbox_id: The sandbox ID.

        Returns:
            SandboxInfo or None if not found.
        """
        return self.active_sandboxes.get(sandbox_id)

    def list_sandboxes(self) -> list[SandboxInfo]:
        """List all active sandboxes.

        Returns:
            List of SandboxInfo for all active sandboxes.
        """
        return list(self.active_sandboxes.values())


# =============================================================================
# CLI Entrypoints - For local development and testing
# =============================================================================


@app.local_entrypoint()
def main(
    command: str = "status",
    task_id: str = "cli-test",
    gpu: str | None = None,
):
    """Parhelia CLI entrypoint.

    Usage:
        modal run src/parhelia/modal_app.py  # Show status
        modal run src/parhelia/modal_app.py --command health  # Health check
        modal run src/parhelia/modal_app.py --command init  # Init volume
        modal run src/parhelia/modal_app.py --command sandbox --task-id my-task  # Create sandbox

    Args:
        command: Command to run (status, health, init, sandbox)
        task_id: Task ID for sandbox creation
        gpu: GPU type for sandbox (A10G, A100, etc.)
    """
    import json

    if command == "status":
        print("Parhelia Modal App Status")
        print("=" * 40)
        print(f"App Name: {app.name}")
        print(f"Volume: {config.modal.volume_name}")
        print(f"CPU Config: {CPU_CONFIG}")
        print(f"Supported GPUs: {SUPPORTED_GPUS}")
        print("\nTo deploy: modal deploy src/parhelia/modal_app.py")
        print("To run health check: modal run src/parhelia/modal_app.py --command health")

    elif command == "health":
        print("Running health check...")
        result = health_check.remote()
        print(json.dumps(result, indent=2))

    elif command == "init":
        print("Initializing volume structure...")
        result = init_volume_structure.remote()
        print(json.dumps(result, indent=2))

    elif command == "sandbox":
        import asyncio

        async def create_and_test():
            print(f"Creating sandbox for task: {task_id}")
            if gpu:
                print(f"GPU: {gpu}")

            sandbox = await create_claude_sandbox(task_id, gpu=gpu)
            print(f"Sandbox created: {sandbox}")

            # Run a simple test
            print("\nRunning test command...")
            output = await run_in_sandbox(sandbox, ["echo", "Hello from Parhelia!"])
            print(f"Output: {output}")

            # Check Claude (runs as parhelia user by default)
            print("\nChecking Claude Code...")
            try:
                claude_output = await run_in_sandbox(
                    sandbox,
                    [f"{CONTAINER_HOME}/.local/bin/claude", "--version"],
                )
                print(f"Claude version: {claude_output}")
            except Exception as e:
                print(f"Claude check failed: {e}")

            return sandbox

        asyncio.run(create_and_test())

    else:
        print(f"Unknown command: {command}")
        print("Available commands: status, health, init, sandbox")


# Allow running as module
if __name__ == "__main__":
    print("Use 'modal run src/parhelia/modal_app.py' to run this app")
    print("Or 'modal deploy src/parhelia/modal_app.py' to deploy")
