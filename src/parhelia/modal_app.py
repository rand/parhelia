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
    ])
    .pip_install([
        "anthropic>=0.40.0",
        "prometheus-client>=0.21.0",
        "aiofiles>=24.0.0",
        "psutil>=5.9.0",
    ])
    .run_commands([
        # Install Bun for plugin tooling
        "curl -fsSL https://bun.sh/install | bash",
        # Add bun to PATH for subsequent commands
        "echo 'export BUN_INSTALL=\"$HOME/.bun\"' >> ~/.bashrc",
        "echo 'export PATH=\"$BUN_INSTALL/bin:$PATH\"' >> ~/.bashrc",
        # Install Claude Code native binary
        "curl -fsSL https://claude.ai/install.sh | sh || echo 'Claude install may need manual setup'",
    ])
    # Copy entrypoint script into the image
    .add_local_file(
        str(_PACKAGE_DIR / "scripts" / "entrypoint.sh"),
        "/entrypoint.sh",
    )
    .run_commands([
        "chmod +x /entrypoint.sh",
    ])
)

# GPU image extends CPU with CUDA support
gpu_image = cpu_image.run_commands([
    "pip install torch --index-url https://download.pytorch.org/whl/cu121",
])

# Secrets configuration - these must be created in Modal dashboard
# modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-...
# modal secret create github-token GITHUB_TOKEN=ghp_...
PARHELIA_SECRETS = [
    modal.Secret.from_name("anthropic-api-key"),
    modal.Secret.from_name("github-token"),
]

# =============================================================================
# Sandbox Creation - For Interactive Claude Code Sessions
# =============================================================================


async def create_claude_sandbox(
    task_id: str,
    gpu: str | None = None,
    timeout_hours: int | None = None,
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

    Returns:
        modal.Sandbox instance ready for Claude Code execution

    Raises:
        ValueError: If gpu is specified but not in SUPPORTED_GPUS
    """
    # Validate GPU type
    if gpu is not None and gpu not in SUPPORTED_GPUS:
        raise ValueError(f"Unsupported GPU '{gpu}'. Must be one of: {SUPPORTED_GPUS}")

    if timeout_hours is None:
        timeout_hours = config.modal.default_timeout_hours

    image = gpu_image if gpu else cpu_image

    sandbox = await modal.Sandbox.create.aio(
        image=image,
        secrets=PARHELIA_SECRETS,
        volumes={"/vol/parhelia": volume},
        gpu=gpu,
        timeout=timeout_hours * 3600,
        cpu=CPU_CONFIG["cpu"],
        memory=CPU_CONFIG["memory"],
        # Pass task_id as environment variable for tracking/logging
        environment={"PARHELIA_TASK_ID": task_id},
    )

    return sandbox


async def run_in_sandbox(sandbox: modal.Sandbox, command: list[str]) -> str:
    """Execute command in sandbox and return output.

    Args:
        sandbox: The Modal Sandbox instance
        command: Command and arguments to execute

    Returns:
        stdout from the command
    """
    process = await sandbox.exec.aio(*command)
    stdout = await process.stdout.read()
    return stdout


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

    # Check Claude Code installation
    try:
        claude_check = subprocess.run(
            ["/root/.claude/local/claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["claude_installed"] = claude_check.returncode == 0
        result["claude_version"] = claude_check.stdout.strip()
    except Exception as e:
        result["claude_error"] = str(e)

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

            # Check Claude
            print("\nChecking Claude Code...")
            try:
                claude_output = await run_in_sandbox(
                    sandbox,
                    ["/root/.claude/local/claude", "--version"],
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
