"""Modal app definition for Parhelia.

Implements:
- [SPEC-01.10] Container Variants (CPU and GPU)
- [SPEC-01.11] Image Definition
- [SPEC-01.12] Volume Mounting

Key Design Decision: Use Sandboxes for interactive Claude Code sessions
(dynamic, long-lived), and Functions only for short batch operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import modal

from parhelia.config import load_config

# Load configuration
config = load_config()

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
    ])
    .pip_install([
        "anthropic>=0.40.0",
        "prometheus-client>=0.21.0",
    ])
    .run_commands([
        # Install Bun for plugin tooling
        "curl -fsSL https://bun.sh/install | bash",
        # Install Claude Code native binary
        "curl -fsSL https://claude.ai/install.sh | sh",
    ])
)

# GPU image extends CPU with CUDA support
gpu_image = cpu_image.run_commands([
    "pip install torch --index-url https://download.pytorch.org/whl/cu121",
])

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
        secrets=[
            modal.Secret.from_name("anthropic-api-key"),
            modal.Secret.from_name("github-token"),
        ],
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
    secrets=[modal.Secret.from_name("anthropic-api-key")],
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
