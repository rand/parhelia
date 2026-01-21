# Parhelia

Run Claude Code in remote Modal.com containers with full local configuration.

## Why Parhelia?

Local Claude Code sessions exhaust machine resources during intensive work—full test suites, large builds, parallel operations. GPU access requires specialized hardware. Parhelia offloads execution to Modal's cloud infrastructure while preserving your complete setup: plugins, skills, CLAUDE.md, MCP servers.

**Key capabilities:**
- CPU and GPU containers (A10G, A100, H100)
- Checkpoint/resume across container restarts
- Your full Claude configuration synced to remote
- Parallel task dispatch with budget controls

## Quick Start

```bash
# Install
uv pip install -e .

# Configure Modal (one-time)
modal token set
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# Deploy
modal deploy src/parhelia/modal_app.py

# Submit a task
parhelia submit "Run the test suite and fix any failures" --sync
```

## Usage

### Submit Tasks

```bash
# Synchronous (wait for result)
parhelia submit "What is 2+2?" --sync

# Asynchronous (dispatch and return)
parhelia submit "Refactor the auth module"

# With GPU
parhelia submit "Train the model on dataset X" --gpu A10G

# Dry run (no Modal execution)
parhelia submit "Test prompt" --dry-run
```

### Check Status

```bash
# Health check
modal run src/parhelia/modal_app.py::health_check

# List tasks
parhelia tasks list

# View task output
parhelia tasks show <task-id>
```

### Interactive Sessions

For complex work requiring human oversight:

```bash
# Attach to running session (coming soon)
parhelia attach <session-id>
```

## Architecture

```
Local Machine                    Modal.com
┌─────────────────┐             ┌─────────────────────────────────┐
│  parhelia CLI   │────────────▶│  Modal Sandbox                  │
│                 │             │  ┌─────────────────────────────┐│
│  - submit tasks │             │  │ Claude Code + your config   ││
│  - view results │             │  │ tmux session                ││
│  - attach       │             │  │ checkpoint manager          ││
└─────────────────┘             │  └─────────────────────────────┘│
                                │              │                   │
                                │              ▼                   │
                                │  ┌─────────────────────────────┐│
                                │  │ Modal Volume                ││
                                │  │ - ~/.claude/ config         ││
                                │  │ - plugins/                  ││
                                │  │ - checkpoints/              ││
                                │  │ - workspaces/               ││
                                │  └─────────────────────────────┘│
                                └─────────────────────────────────┘
```

## Configuration

Create `.parhelia/config.toml`:

```toml
[modal]
volume_name = "parhelia-data"
cpu_count = 4
memory_mb = 16384
default_timeout_hours = 1

[orchestrator]
db_path = ".parhelia/orchestrator.db"
max_concurrent_workers = 10
```

## Container Variants

| Variant | Resources | Use Case |
|---------|-----------|----------|
| CPU | 4 CPU, 16GB RAM | Tests, builds, general coding |
| A10G | + 24GB VRAM | ML inference, medium training |
| A100 | + 40/80GB VRAM | Large model training |
| H100 | + 80GB VRAM | Maximum performance |

## Checkpoint/Resume

Parhelia automatically checkpoints session state:
- Conversation history
- Working directory changes
- Environment state

If a container dies, the next dispatch resumes from the last checkpoint.

```bash
# List checkpoints
parhelia session checkpoints <session-id>

# Resume from specific checkpoint
parhelia session resume <checkpoint-id>
```

## Cost Estimates

| Usage | Monthly Cost |
|-------|--------------|
| Light (10 tasks/day, CPU) | $5-15 |
| Medium (50 tasks/day, CPU) | $25-75 |
| Heavy (200 tasks/day, mixed) | $100-300 |

Set spending limits in Modal dashboard to prevent surprises.

## Documentation

| Document | Purpose |
|----------|---------|
| [Production Guide](docs/PRODUCTION.md) | Deployment, security, operations |
| [Specs](docs/spec/) | Detailed technical specifications |
| [ADRs](docs/adr/) | Architecture decision records |

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run E2E tests (requires Modal)
uv run python scripts/e2e_modal_tests.py --level core
```

## Requirements

- Python 3.11+
- Modal account with billing enabled
- Anthropic API key

## License

MIT
