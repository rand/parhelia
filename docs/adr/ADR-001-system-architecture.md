# ADR-001: Remote Claude Code Execution System Architecture

**Status**: Proposed
**Date**: 2026-01-16
**Deciders**: rand

## Context

Claude Code sessions frequently exhaust local machine resources (memory, CPU) during intensive operations like running full test suites. Additionally, GPU access is required for ML inference, model training, and CUDA compilation workloads. A system is needed to offload Claude Code execution to remote environments while maintaining the full local configuration (plugins, skills, CLAUDE.md).

### Requirements

1. Run Claude Code with complete configuration in remote Modal.com environments
2. Support both automated orchestration and interactive debugging sessions
3. Survive network failures with full session state recovery
4. Broadcast resource capacity for optimal work dispatch
5. Scale dynamically based on task complexity and budget constraints
6. Mirror all MCP servers and plugins to remote environment

## Decision

We will build **Parhelia** (named for the atmospheric phenomenon of multiple sun images), a hybrid local-remote Claude Code execution system with the following architecture:

### Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOCAL ENVIRONMENT                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Claude Code    │───▶│  Parhelia       │───▶│  Resource       │         │
│  │  (Interactive)  │    │  Orchestrator   │    │  Monitor        │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                      ▲                   │
│                                  │ dispatch             │ capacity          │
└──────────────────────────────────┼──────────────────────┼───────────────────┘
                                   │                      │
                          ┌────────▼──────────────────────┴───────┐
                          │         Modal.com API                  │
                          └────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────────┐
│                           MODAL ENVIRONMENT                                  │
│                                  │                                           │
│  ┌───────────────────────────────▼───────────────────────────────────────┐  │
│  │                     Parhelia Modal Function                            │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │   tmux      │  │  Claude     │  │  Checkpoint │  │  Metrics    │   │  │
│  │  │   Server    │  │  Code       │  │  Manager    │  │  Exporter   │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                  │                                           │
│  ┌───────────────────────────────▼───────────────────────────────────────┐  │
│  │                        Modal Volume                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
│  │  │  ~/.claude/ │  │  plugins/   │  │  checkpoints│  │  workspaces/│   │  │
│  │  │  config     │  │  (cloned)   │  │  (sessions) │  │  (projects) │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

#### 1. Hybrid Interaction Model

**Decision**: Support both headless orchestration and interactive attachment.

- **Headless mode**: Local orchestrator dispatches tasks to Modal functions running `claude -p` with `--output-format stream-json`
- **Interactive mode**: User can attach to running tmux session via SSH tunnel for debugging or complex multi-step work
- **Rationale**: Automated dispatch enables parallelism and optimal resource usage; interactive attach enables human-in-the-loop when needed

#### 2. Checkpoint and Resume

**Decision**: Implement full conversation state persistence with automatic checkpoint/resume.

- Serialize Claude session state to Modal Volume at configurable intervals
- On container death: detect via heartbeat timeout, restore from last checkpoint, resume task
- State includes: conversation history, tool outputs, working directory, uncommitted changes
- **Rationale**: User chose robustness over simplicity; 24-hour Modal timeout means long tasks need checkpointing anyway

#### 3. Scale to Zero with Cold Start Tolerance

**Decision**: Containers scale to zero when idle; accept 2-4 second cold starts.

- No warm pool by default (cost optimization)
- Use Modal memory snapshots to reduce cold start for GPU workloads
- Optional warm pool for time-sensitive work (configurable)
- **Rationale**: Cost efficiency is priority; cold starts are acceptable for batch/background work

#### 4. Dynamic Concurrency

**Decision**: System determines parallelism based on task decomposition and budget.

- Orchestrator analyzes task complexity (file count, test count, etc.)
- Spawns 1-N workers up to configured budget ceiling
- Workers coordinate via shared checkpoint storage
- **Rationale**: Maximizes throughput while preventing runaway costs

#### 5. Full MCP Server Mirroring

**Decision**: All local MCP servers run in remote environment.

- MCP server definitions synced to Volume
- Remote Claude Code starts all configured MCP servers
- Some servers may need remote-specific config (e.g., Playwright headless)
- **Rationale**: Full feature parity between local and remote execution

#### 6. Plugin Cloning to Volume

**Decision**: Plugins cloned via git to Modal Volume, synced periodically.

- On first run: clone all plugin repos to Volume
- Periodic sync (configurable, default 1 hour) pulls updates
- Symlinks resolved to actual paths
- **Rationale**: Plugins evolve; git clone enables version control and easy updates

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Parhelia Orchestrator** | Task dispatch, worker lifecycle, result aggregation |
| **Resource Monitor** | Collect capacity metrics from all environments, inform dispatch |
| **Modal Function** | Run Claude Code, manage tmux, checkpoint state, export metrics |
| **Checkpoint Manager** | Serialize/deserialize session state, manage retention |
| **Metrics Exporter** | Expose Prometheus metrics for capacity and utilization |
| **Volume Manager** | Sync config, plugins, checkpoints between local and Modal |

### Communication Protocols

| Path | Protocol | Purpose |
|------|----------|---------|
| Local → Modal dispatch | Modal Python SDK | Task invocation, parameter passing |
| Modal → Local results | Modal return values / streaming | Task output, incremental results |
| Interactive attach | SSH tunnel via `modal.forward()` | tmux attachment |
| Metrics broadcast | HTTP pull (Modal web endpoint) | Resource capacity polling |
| Checkpoint sync | Modal Volume API | State persistence |

## Consequences

### Positive

- **Resource scalability**: Access to Modal's GPU fleet (A10G, A100, H100)
- **Resilience**: Full state recovery from any failure point
- **Cost efficiency**: Pay only for compute used, scale to zero
- **Feature parity**: Full Claude Code capability in remote environment
- **Parallelism**: Dynamic worker spawning for large tasks

### Negative

- **Complexity**: Checkpoint/resume and state sync add significant complexity
- **Latency**: Cold starts add 2-4 seconds; network round-trips for all operations
- **Cost unpredictability**: Dynamic scaling could surprise with high bills
- **Debugging difficulty**: Distributed system failures harder to diagnose

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Checkpoint corruption | Versioned checkpoints, keep N most recent |
| Volume storage limits (500K inodes) | Aggressive checkpoint pruning, archive old sessions |
| OAuth auth in headless | Use `ANTHROPIC_API_KEY` directly; monitor Claude Code headless auth developments |
| Plugin sync conflicts | Git-based sync with conflict detection and alerting |
| Cost runaway | Hard budget ceiling, automatic scale-down when approaching limit |

## Alternatives Considered

### 1. Direct SSH Only (No Orchestrator)

Simple SSH into Modal container, run Claude Code interactively.

**Rejected because**: No parallelism, no automatic dispatch, no resilience to connection loss.

### 2. Kubernetes Instead of Modal

Deploy to self-managed or cloud Kubernetes cluster.

**Rejected because**: Higher operational overhead; Modal provides serverless GPU access out of box.

### 3. Ephemeral Sessions (No Checkpointing)

Let sessions die on failure, accept lost work.

**Rejected because**: User explicitly chose robustness; long-running tasks need persistence.

### 4. Local Docker with Remote Mount

Run Claude Code locally in Docker, mount remote filesystem.

**Rejected because**: Doesn't solve resource constraints; network latency for file ops would be prohibitive.

## References

- [Modal.com Documentation](https://modal.com/docs)
- [Claude Code Headless Mode](https://docs.anthropic.com/en/docs/claude-code/headless)
- [tmux Session Management](https://github.com/tmux/tmux/wiki)
