# SPEC-01: Remote Environment Provisioning

**Status**: Draft
**Issue**: ph-3h1
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines how Parhelia provisions and configures Modal.com containers for running Claude Code with full configuration (plugins, skills, CLAUDE.md, MCP servers).

## Goals

- [SPEC-01.01] Provision Modal containers with Claude Code and all dependencies
- [SPEC-01.02] Support both CPU-only and GPU-enabled container variants
- [SPEC-01.03] Mount persistent storage for config, plugins, checkpoints, and workspaces
- [SPEC-01.04] Initialize environment with user's Claude configuration
- [SPEC-01.05] Support cold start optimization via image layering

## Non-Goals

- Container orchestration (handled by Modal)
- Network configuration beyond Modal defaults
- Multi-cloud support (Modal-only for v1)

---

## Architecture

### Container Image Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Runtime Config (mounted at startup)                   │
│  - ~/.claude/CLAUDE.md, settings.json, mcp_config.json         │
│  - Synced from Volume on container start                        │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Plugins & Skills (Volume mount, read-mostly)          │
│  - ~/.claude/plugins/*, ~/.claude/skills/*                      │
│  - Git-cloned to Volume, synced periodically                    │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Claude Code + Dependencies (baked into image)         │
│  - Claude Code native binary (no Node.js/npm required)          │
│  - tmux, git, common dev tools                                  │
│  - Python 3.11+ for MCP servers                                 │
│  - Bun (for any JS tooling that plugins may need)               │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Base OS (Modal default or custom)                     │
│  - Debian 12 (bookworm) base                                    │
│  - CUDA drivers (GPU variant only)                              │
└─────────────────────────────────────────────────────────────────┘
```

### Volume Structure

```
/vol/parhelia/                          # Modal Volume mount point
├── config/                             # User configuration
│   ├── claude/                         # ~/.claude/ contents
│   │   ├── CLAUDE.md
│   │   ├── settings.json
│   │   ├── mcp_config.json
│   │   └── agents/
│   └── env/                            # Environment-specific overrides
│       └── remote.env                  # Remote-only env vars
├── plugins/                            # Cloned plugin repos
│   ├── cc-polymath/
│   ├── beads/
│   └── ...
├── checkpoints/                        # Session state persistence
│   ├── {session-id}/
│   │   ├── conversation.json
│   │   ├── state.json
│   │   └── workspace.tar.gz
│   └── ...
└── workspaces/                         # Cloned project repos
    ├── {project-hash}/
    └── ...
```

---

## Requirements

### [SPEC-01.10] Container Variants

The system MUST support two container variants:

| Variant | GPU | Use Case | Modal Config |
|---------|-----|----------|--------------|
| `parhelia-cpu` | None | General tasks, tests, builds | `cpu=4, memory=16384` |
| `parhelia-gpu` | A10G/A100 | ML inference, training, CUDA | `gpu="A10G"` or `gpu="A100"` |

**Rationale**: GPU instances are 10-50x more expensive; most tasks don't need GPU.

### [SPEC-01.11] Image Definition

The container image MUST be defined as a Modal `Image` with:

```python
# Pseudo-code structure
parhelia_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "tmux", "openssh-server", "git", "curl", "build-essential", "unzip",
    ])
    .pip_install([
        "anthropic",      # For API access
        # MCP server dependencies
    ])
    .run_commands([
        # Install Bun (for plugin tooling)
        "curl -fsSL https://bun.sh/install | bash",
        # Install Claude Code native binary
        "curl -fsSL https://claude.ai/install.sh | sh",
        "claude --version",  # Verify installation
    ])
)
```

**Note**: Claude Code's native binary eliminates Node.js/npm dependency. Bun is included for any JS tooling that plugins may require.

### [SPEC-01.11a] Image Update Strategy

Images MUST be rebuilt weekly via CI to capture Claude Code updates:

```yaml
# .github/workflows/image-rebuild.yml
name: Rebuild Parhelia Images
on:
  schedule:
    - cron: '0 6 * * 0'  # Weekly Sunday 6am UTC
  workflow_dispatch: {}  # Manual trigger

jobs:
  rebuild:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: modal-labs/modal-setup@v1
      - run: modal deploy --tag weekly-$(date +%Y%m%d)
```

### [SPEC-01.11b] Region Configuration

Users MUST be able to configure Modal region:

```python
# parhelia.toml
[modal]
region = "us-east"  # Options: us-east, us-west, eu-west

# In Python
@app.function(region=config.modal.region, ...)
def run_claude(...): ...
```

### [SPEC-01.12] Volume Mounting

Containers MUST mount the Parhelia volume at `/vol/parhelia` with:

- **Read-write** for `checkpoints/` and `workspaces/`
- **Read-mostly** for `config/` and `plugins/` (writes only during sync)

```python
volume = modal.Volume.from_name("parhelia-vol", create_if_missing=True)

@app.function(volumes={"/vol/parhelia": volume})
def run_claude(...):
    ...
```

### [SPEC-01.13] Environment Initialization

On container start, the init script MUST:

1. **Symlink config**: `~/.claude -> /vol/parhelia/config/claude`
2. **Verify Claude Code**: Run `claude --version`
3. **Start MCP servers**: Launch configured MCP servers from `mcp_config.json`
4. **Initialize tmux**: Create tmux server with standard session
5. **Report ready**: Signal readiness to orchestrator

```bash
#!/bin/bash
# /entrypoint.sh

set -euo pipefail

# 1. Link configuration
ln -sfn /vol/parhelia/config/claude ~/.claude

# 2. Verify Claude Code
claude --version || exit 1

# 3. Start MCP servers (background)
parhelia-mcp-launcher &

# 4. Initialize tmux
tmux new-session -d -s main -c /vol/parhelia/workspaces

# 5. Signal ready
echo "PARHELIA_READY" > /tmp/ready
```

### [SPEC-01.14] Secrets Injection

Secrets MUST be injected via Modal's Secrets API, NOT baked into images:

| Secret | Environment Variable | Purpose |
|--------|---------------------|---------|
| Anthropic API Key | `ANTHROPIC_API_KEY` | Claude API access |
| GitHub Token | `GITHUB_TOKEN` | Plugin cloning, repo access |
| Custom secrets | User-defined | Project-specific needs |

```python
@app.function(
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("github-token"),
    ]
)
def run_claude(...):
    ...
```

### [SPEC-01.15] Cold Start Optimization

To minimize cold start latency:

1. **Image layer caching**: Stable dependencies in lower layers
2. **Volume pre-warming**: Config/plugins already on Volume
3. **Memory snapshots** (GPU): Use Modal's snapshot feature for GPU containers
4. **Lazy MCP loading**: Start MCP servers on-demand, not at init

Target cold start times:
- CPU variant: < 5 seconds
- GPU variant: < 10 seconds (with memory snapshot)

---

## Container Lifecycle

### Startup Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Modal provisions container from image                        │
│    - Pulls image layers (cached after first run)                │
│    - Mounts Volume at /vol/parhelia                             │
│    - Injects secrets as environment variables                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Entrypoint script runs                                       │
│    - Links ~/.claude to Volume config                           │
│    - Verifies Claude Code installation                          │
│    - Starts tmux server                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Ready signal                                                 │
│    - Container reports ready to orchestrator                    │
│    - Begins accepting work or interactive connections           │
└─────────────────────────────────────────────────────────────────┘
```

### Shutdown Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Graceful shutdown signal (SIGTERM)                           │
│    - Checkpoint current session state                           │
│    - Flush any pending Volume writes                            │
│    - Terminate MCP servers                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Volume commit                                                │
│    - Modal commits Volume changes                               │
│    - Changes visible to future containers                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Container termination                                        │
│    - Resources released                                         │
│    - Scale to zero if no other containers active                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Schema

### Remote Environment Config (`/vol/parhelia/config/env/remote.env`)

```bash
# Parhelia remote environment configuration

# Override local paths for remote context
PARHELIA_VOLUME_ROOT=/vol/parhelia
PARHELIA_CHECKPOINT_DIR=/vol/parhelia/checkpoints
PARHELIA_WORKSPACE_DIR=/vol/parhelia/workspaces

# Claude Code behavior
CLAUDE_CODE_HEADLESS=1
CLAUDE_CODE_NO_BROWSER=1

# Telemetry (optional)
PARHELIA_METRICS_ENABLED=true
PARHELIA_METRICS_PORT=9090
```

### Modal App Definition (`parhelia/modal_app.py`)

**Key Design Decision**: Use **Sandboxes** for interactive Claude Code sessions (dynamic, long-lived), and **Functions** only for short batch operations.

```python
import modal
from parhelia.config import load_config

config = load_config()
app = modal.App("parhelia")

# Volume for persistent storage (using v2 for better performance)
volume = modal.Volume.from_name("parhelia-vol", create_if_missing=True)

# Base image for CPU workloads
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "tmux", "openssh-server", "git", "curl",
        "build-essential", "unzip", "psutil",
    ])
    .run_commands([
        # Install Bun for plugin tooling
        "curl -fsSL https://bun.sh/install | bash",
        # Install Claude Code native binary
        "curl -fsSL https://claude.ai/install.sh | sh",
        # Verify installation
        "/root/.claude/local/claude --version",
    ])
)

# GPU image extends CPU with CUDA
gpu_image = cpu_image.run_commands([
    "pip install torch --index-url https://download.pytorch.org/whl/cu121",
])

# ============================================================
# SANDBOXES: For interactive Claude Code sessions
# ============================================================

async def create_claude_sandbox(
    task_id: str,
    gpu: str | None = None,
    timeout_hours: int = 24,
) -> modal.Sandbox:
    """Create a Sandbox for interactive Claude Code session.

    Sandboxes are preferred for:
    - Long-lived sessions (up to 24h)
    - Interactive stdin/stdout
    - Dynamic workloads
    - Sessions that may need SSH attachment
    """
    image = gpu_image if gpu else cpu_image

    sandbox = modal.Sandbox.create(
        image=image,
        secrets=[modal.Secret.from_name("anthropic-api-key")],
        volumes={"/vol/parhelia": volume},
        gpu=gpu,
        timeout=timeout_hours * 3600,
        # Enable filesystem snapshots for checkpoint/resume
        enable_snapshot=True,
    )

    return sandbox

async def run_in_sandbox(sandbox: modal.Sandbox, command: list[str]) -> str:
    """Execute command in sandbox and return output."""
    process = await sandbox.exec(*command)
    return await process.stdout.read()

# ============================================================
# FUNCTIONS: For short batch operations only
# ============================================================

@app.function(
    image=cpu_image,
    volumes={"/vol/parhelia": volume},
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    cpu=4,
    memory=16384,
    timeout=300,  # 5 min max for batch ops
)
def run_batch_task(task: dict) -> dict:
    """Run short batch task (sync, health check, etc.)."""
    ...
```

### Sandbox vs Function Decision Matrix

| Use Case | Abstraction | Rationale |
|----------|-------------|-----------|
| Interactive Claude session | **Sandbox** | Long-lived, stdin/stdout, SSH attachment |
| Headless multi-turn task | **Sandbox** | May run for hours, needs state |
| Quick health check | Function | Short, stateless |
| Plugin sync | Function | Short batch operation |
| Metrics collection | Function | Periodic, stateless |

---

## Acceptance Criteria

- [ ] [SPEC-01.AC1] CPU container starts and runs `claude --version` successfully
- [ ] [SPEC-01.AC2] GPU container starts with CUDA available (`nvidia-smi` works)
- [ ] [SPEC-01.AC3] Volume mounts correctly and persists data across container restarts
- [ ] [SPEC-01.AC4] Secrets are available as environment variables, not logged
- [ ] [SPEC-01.AC5] Cold start time < 5s for CPU, < 10s for GPU (with snapshot)
- [ ] [SPEC-01.AC6] ~/.claude symlink points to Volume config correctly
- [ ] [SPEC-01.AC7] tmux session created and accessible on startup

---

## Resolved Questions

1. ~~**Node.js version management**~~: **Resolved** - Use Claude Code native binary (no Node.js needed). Bun installed for plugin tooling.
2. ~~**Image update strategy**~~: **Resolved** - Weekly CI rebuild via GitHub Actions.
3. ~~**Multi-region**~~: **Resolved** - User-configurable via `parhelia.toml`.

## Open Questions

1. **Volume size limits**: What's the expected growth rate of checkpoints/workspaces? Need retention policy.
2. **Auto-update in container**: Should running containers auto-update Claude Code, or only get updates via image rebuild?

---

## References

- [Modal Image Documentation](https://modal.com/docs/guide/images)
- [Modal Volumes Documentation](https://modal.com/docs/guide/volumes)
- [Modal Secrets Documentation](https://modal.com/docs/guide/secrets)
- [Claude Code CLI Reference](https://docs.anthropic.com/en/docs/claude-code)
