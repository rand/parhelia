# Parhelia

**Remote Claude Code execution with checkpoint/resume, GPU support, and full local configuration sync.**

[![Tests](https://img.shields.io/badge/tests-1479%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## Why Parhelia?

Local Claude Code sessions hit limits during intensive work—full test suites exhaust memory, large builds max out CPU, GPU workloads require specialized hardware. Network disconnections lose hours of progress.

Parhelia solves this by running Claude Code in Modal.com's cloud infrastructure while preserving everything that makes your local setup productive:

- **Your full configuration**: plugins, skills, CLAUDE.md, MCP servers—all synced
- **Checkpoint/resume**: Sessions survive container restarts and network failures
- **GPU access**: A10G, A100, H100 for ML workloads
- **Parallel execution**: Dispatch multiple tasks across workers
- **Budget controls**: Set spending limits, track costs in real-time
- **Interactive sessions**: SSH/tmux attachment for debugging

---

## Quick Start

```bash
# Install Parhelia
uv pip install -e .

# Configure Modal (one-time setup)
modal token set
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# Deploy to Modal
modal deploy src/parhelia/modal_app.py

# Create your first task
parhelia task create "List the files in /tmp and count them" --sync
```

---

## CLI Commands

Parhelia uses a structured command hierarchy with intuitive aliases:

### Task Management

```bash
# Create and run a task
parhelia task create "Run the test suite" --sync          # Wait for result
parhelia task create "Refactor auth module"               # Async dispatch
parhelia task create "Train model" --gpu A10G             # With GPU

# Monitor tasks
parhelia task list                                        # List all tasks
parhelia task show <task-id>                              # Task details
parhelia task watch <task-id>                             # Real-time updates

# Control execution
parhelia task dispatch                                    # Dispatch pending tasks
```

### Session Management

```bash
# Interactive sessions
parhelia session list                                     # List active sessions
parhelia session attach <session-id>                      # SSH/tmux attach
parhelia session kill <session-id>                        # Terminate session

# Recovery
parhelia session recover <session-id>                     # Guided recovery flow
```

### Checkpoint & Resume

```bash
# Checkpoints
parhelia checkpoint create <session-id>                   # Manual checkpoint
parhelia checkpoint list                                  # List all checkpoints
parhelia checkpoint list <session-id>                     # For specific session

# Recovery
parhelia checkpoint rollback <checkpoint-id>              # Restore workspace
parhelia checkpoint diff <cp-a> <cp-b>                    # Compare checkpoints
parhelia resume <session-id>                              # Resume from latest
```

### Container Introspection

```bash
# View containers
parhelia container list                                   # All containers
parhelia container list --state running                   # Filter by state
parhelia container show <container-id>                    # Full details
parhelia container events <container-id>                  # Event history

# Health monitoring
parhelia container health                                 # Health summary
parhelia container watch                                  # Real-time updates
```

### Event Streaming

```bash
# Query events
parhelia events list                                      # Recent events
parhelia events list --type container_started             # Filter by type
parhelia events list --level error                        # Errors only
parhelia events watch                                     # Real-time stream

# Export
parhelia events export events.jsonl                       # Export to file
parhelia events replay <container-id>                     # Replay history
```

### Budget Management

```bash
# Monitor spending
parhelia budget show                                      # Current status
parhelia budget history                                   # Usage over time

# Set limits
parhelia budget set 50.00                                 # Set ceiling ($50)
```

### System Operations

```bash
# Health and status
parhelia status                                           # System health
parhelia reconciler status                                # Reconciler state
parhelia reconciler run                                   # Force reconciliation

# Cleanup
parhelia cleanup                                          # Find orphaned containers
parhelia cleanup --terminate                              # Auto-terminate orphans
```

### Help & Examples

```bash
# Contextual help
parhelia help task                                        # Topic help
parhelia help E200                                        # Error code help
parhelia help checkpoint                                  # Checkpoint guide

# Example workflows
parhelia examples gpu                                     # GPU usage examples
parhelia examples checkpoint                              # Checkpoint examples
parhelia examples budget                                  # Budget management
```

### Aliases

| Alias | Expands To |
|-------|------------|
| `t` | `task` |
| `s` | `session` |
| `c` | `checkpoint` |
| `b` | `budget` |
| `e` | `events` |

```bash
parhelia t list                                           # Same as: task list
parhelia c list                                           # Same as: checkpoint list
```

---

## Architecture

```
Local Machine                         Modal.com
┌─────────────────────┐              ┌────────────────────────────────────┐
│                     │              │  Modal Sandbox                     │
│  parhelia CLI       │              │  ┌──────────────────────────────┐  │
│  ┌───────────────┐  │              │  │ Claude Code                  │  │
│  │ Orchestrator  │──┼──────────────┼─▶│ + Your plugins & skills      │  │
│  │ State Store   │  │   dispatch   │  │ + CLAUDE.md config           │  │
│  │ Reconciler    │  │              │  │ + MCP servers                │  │
│  └───────────────┘  │              │  └──────────────────────────────┘  │
│         │           │              │              │                     │
│         ▼           │              │              ▼                     │
│  ┌───────────────┐  │              │  ┌──────────────────────────────┐  │
│  │ SQLite DB     │  │              │  │ Modal Volume                 │  │
│  │ - containers  │  │              │  │ /vol/parhelia/               │  │
│  │ - events      │  │              │  │ ├── config/claude/           │  │
│  │ - heartbeats  │  │              │  │ ├── plugins/                 │  │
│  └───────────────┘  │              │  │ ├── checkpoints/             │  │
│                     │              │  │ └── workspaces/              │  │
│  ┌───────────────┐  │              │  └──────────────────────────────┘  │
│  │ MCP Server    │  │              │                                    │
│  │ (24 tools)    │  │              │  Container Variants:               │
│  └───────────────┘  │              │  • CPU: 4 cores, 16GB RAM          │
│                     │              │  • A10G: + 24GB VRAM               │
└─────────────────────┘              │  • A100: + 40/80GB VRAM            │
                                     │  • H100: + 80GB VRAM               │
                                     └────────────────────────────────────┘
```

### Control Plane

Parhelia maintains a local control plane that tracks:

- **Containers**: State, health, lifecycle events
- **Events**: Full audit trail of all operations
- **Heartbeats**: Liveness monitoring for running sessions
- **Reconciliation**: Automatic drift detection between local state and Modal

### MCP Integration

Parhelia exposes 24 MCP tools for programmatic access:

```bash
# Start MCP server
parhelia mcp-server
```

Tools include:
- `parhelia_task_create`, `parhelia_task_list`, `parhelia_task_show`
- `parhelia_containers`, `parhelia_container_show`, `parhelia_container_events`
- `parhelia_events_list`, `parhelia_events_subscribe`, `parhelia_events_stream`
- `parhelia_checkpoint_create`, `parhelia_checkpoint_list`, `parhelia_checkpoint_restore`
- `parhelia_budget_status`, `parhelia_budget_estimate`
- `parhelia_health`, `parhelia_reconciler_status`

---

## Configuration

### Local Configuration

Create `.parhelia/config.toml`:

```toml
[modal]
volume_name = "parhelia-data"
cpu_count = 4
memory_mb = 16384
default_timeout_hours = 4

[budget]
default_ceiling_usd = 50.0
warning_threshold_percent = 80

[orchestrator]
db_path = ".parhelia/orchestrator.db"
max_concurrent_workers = 10

[reconciler]
poll_interval_seconds = 60
stale_threshold_seconds = 300
auto_terminate_orphans = false
```

### Modal Secrets

```bash
# Required
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# Optional (for repo access)
modal secret create github-token GITHUB_TOKEN=ghp_...
```

---

## Container Variants

| Variant | Resources | Use Case | Cost |
|---------|-----------|----------|------|
| **CPU** | 4 CPU, 16GB RAM | Tests, builds, general coding | ~$0.0001/sec |
| **A10G** | + 24GB VRAM | ML inference, medium training | ~$0.001/sec |
| **A100** | + 40/80GB VRAM | Large model training | ~$0.003/sec |
| **H100** | + 80GB VRAM | Maximum performance | ~$0.005/sec |

```bash
# Request specific GPU
parhelia task create "Train model" --gpu A10G
parhelia task create "Large inference" --gpu A100
```

---

## Checkpoint & Resume

Parhelia automatically checkpoints session state:

- **Workspace**: All files in working directory
- **Conversation**: Full Claude conversation history
- **Environment**: Claude Code version, plugins, configuration
- **Git state**: Branch, uncommitted changes, stash

### Automatic Checkpoints

Checkpoints are created automatically on:
- Periodic intervals (configurable)
- Before risky operations
- On detach from interactive session
- On error conditions

### Manual Checkpoints

```bash
# Create checkpoint with message
parhelia checkpoint create <session-id> -m "Before major refactor"

# List checkpoints for session
parhelia checkpoint list <session-id>

# Rollback to checkpoint
parhelia checkpoint rollback <checkpoint-id>

# Compare two checkpoints
parhelia checkpoint diff cp-abc123 cp-def456
```

### Resume from Failure

If a container dies or network disconnects:

```bash
# Check session state
parhelia session list

# Resume from latest checkpoint
parhelia resume <session-id>

# Or guided recovery
parhelia session recover <session-id>
```

---

## Budget Management

### Set Spending Limits

```bash
# Set budget ceiling
parhelia budget set 50.00

# Check current status
parhelia budget show
```

Output:
```
Budget Status
─────────────────────────────────
Ceiling:    $50.00
Used:       $12.45 (24.9%)
Remaining:  $37.55

Tasks:      47
Tokens:     1.2M input, 450K output
```

### Cost Estimation

```bash
# Estimate before running
parhelia task create "Large task" --estimate-only

# Output: Estimated cost: $0.50-2.00 (1-4 hours CPU)
```

### Monthly Cost Estimates

| Usage Pattern | Est. Monthly Cost |
|---------------|-------------------|
| Light (10 tasks/day, CPU) | $5-15 |
| Medium (50 tasks/day, CPU) | $25-75 |
| Heavy (200 tasks/day, mixed GPU) | $100-300 |

---

## Interactive Sessions

For complex work requiring human oversight:

```bash
# Start interactive session
parhelia task create "Debug failing tests" --interactive

# Attach to running session
parhelia session attach <session-id>

# Inside tmux session:
# - Full Claude Code interface
# - Your plugins and skills
# - Ctrl+B, D to detach

# Session continues after detach
parhelia session list  # Shows session still running
```

### SSH Configuration

Parhelia configures SSH with:
- Keep-alive intervals for stability
- Automatic reconnection
- tmux session persistence

---

## Help System

### Topic Help

```bash
parhelia help task        # Task management guide
parhelia help session     # Session lifecycle
parhelia help checkpoint  # Checkpoint/resume guide
parhelia help budget      # Budget management
parhelia help container   # Container introspection
```

### Error Code Help

```bash
parhelia help E200        # SESSION_NOT_FOUND
parhelia help E300        # BUDGET_EXCEEDED
parhelia help E500        # MODAL_ERROR
```

Each error includes:
- What went wrong
- Why it happened
- Recovery steps

### Example Workflows

```bash
parhelia examples gpu         # GPU task examples
parhelia examples checkpoint  # Checkpoint workflows
parhelia examples budget      # Budget management
parhelia examples parallel    # Parallel execution
```

---

## Development

### Install

```bash
# Clone repository
git clone https://github.com/rand/parhelia.git
cd parhelia

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests (1479 tests)
uv run pytest

# Specific test files
uv run pytest tests/test_cli.py
uv run pytest tests/test_mcp.py

# With coverage
uv run pytest --cov=parhelia --cov-report=html
```

### Property-Based Tests

Parhelia includes Hypothesis property tests for system invariants:

```bash
uv run pytest tests/test_properties.py -v
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [Quick Start](#quick-start) | Get running in 5 minutes |
| [CLI Commands](#cli-commands) | Full command reference |
| [Production Guide](docs/PRODUCTION.md) | Deployment, security, operations |
| [Validation Plan](docs/VALIDATION-PLAN.md) | Test coverage and results |
| [Specifications](docs/spec/) | Detailed technical specs |
| [ADRs](docs/adr/) | Architecture decisions |

---

## Requirements

- Python 3.11+
- Modal account with billing enabled
- Anthropic API key

---

## License

MIT

---

## Support

- **Issues**: [github.com/rand/parhelia/issues](https://github.com/rand/parhelia/issues)
- **Modal Docs**: [modal.com/docs](https://modal.com/docs)
- **Claude Code**: [claude.ai/claude-code](https://claude.ai/claude-code)
