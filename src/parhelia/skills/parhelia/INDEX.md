---
name: parhelia-index
description: Index of all Parhelia skills for remote Claude Code execution
category: parhelia
---

# Parhelia Skills Index

**Category**: Remote Execution
**Skills**: 7
**Total Lines**: ~2,100

## Skills in This Category

### modal-deployment.md
**Description**: Setting up Modal accounts, API keys, volumes, and secrets
**Lines**: ~300
**Use When**:
- Setting up Parhelia for the first time
- Configuring Modal volumes and secrets
- Managing API keys and credentials
- Troubleshooting Modal connectivity

### task-dispatch.md
**Description**: Submitting tasks and understanding dispatch modes
**Lines**: ~300
**Use When**:
- Submitting tasks for remote execution
- Choosing between sync/async modes
- Understanding task lifecycle
- Debugging dispatch failures

### checkpoint-resume.md
**Description**: Session persistence, checkpoints, and recovery
**Lines**: ~350
**Use When**:
- Understanding checkpoint triggers
- Resuming from failed sessions
- Managing checkpoint storage
- Configuring auto-checkpoint intervals

### gpu-configuration.md
**Description**: GPU selection, optimization, and cost trade-offs
**Lines**: ~280
**Use When**:
- Choosing the right GPU for your workload
- Optimizing GPU utilization
- Understanding cost implications
- Debugging GPU-related issues

### interactive-attach.md
**Description**: SSH tunnels, tmux sessions, and detach workflows
**Lines**: ~300
**Use When**:
- Attaching to running sessions
- Understanding tmux integration
- Troubleshooting SSH connections
- Managing detach and checkpoint flows

### budget-management.md
**Description**: Cost tracking, spending limits, and alerts
**Lines**: ~250
**Use When**:
- Setting up budget controls
- Understanding cost breakdowns
- Configuring spending alerts
- Troubleshooting budget blocks

### troubleshooting.md
**Description**: Common issues, diagnostics, and solutions
**Lines**: ~320
**Use When**:
- Sessions fail to start
- SSH tunnels won't connect
- Checkpoints not restoring
- Budget unexpectedly exceeded

## Related Categories

- `remote-execution/` - Cross-cutting remote execution patterns
- `discover-parhelia/` - Gateway skill for auto-activation

## How to Load Skills

Ask for a specific topic:
- "Explain how checkpoints work in Parhelia"
- "Help me set up Modal for Parhelia"
- "What GPU should I use for training?"

Or load directly:
- "Load the parhelia/task-dispatch skill"
