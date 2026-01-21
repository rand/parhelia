---
name: discover-parhelia
description: Gateway skill for Parhelia remote execution on Modal.com
category: remote-execution
keywords: [parhelia, modal, remote, gpu, cloud, checkpoint, attach, dispatch, sandbox]
---

# Parhelia Remote Execution

**Scope**: Remote Claude Code execution on Modal.com infrastructure
**Lines**: ~200 (gateway)
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Gateway)

## When This Skill Activates

This gateway auto-activates when you mention:
- "remote", "modal", "parhelia", "cloud execution"
- "GPU", "A10G", "A100", "H100", "T4"
- "checkpoint", "resume session", "session recovery"
- "attach", "detach", "interactive session"
- "dispatch", "offload", "run remotely"

## What Parhelia Does

Parhelia runs Claude Code sessions on Modal.com infrastructure:

1. **Remote Execution** - Dispatch tasks to cloud containers
2. **GPU Support** - Access A10G, A100, H100 GPUs for ML workloads
3. **Checkpoint/Resume** - Persist and restore session state
4. **Interactive Sessions** - SSH tunnel attachment to running sessions
5. **Budget Control** - Track and limit spending

## Quick Start

**Submit a task**:
```bash
parhelia submit "Run the full test suite" --sync
```

**Submit with GPU**:
```bash
parhelia submit "Train the model" --gpu A10G
```

**Check status**:
```bash
parhelia status
```

**Attach to session**:
```bash
parhelia attach task-abc12345
```

## Available Skills

For deeper guidance, load the specific skill:

| Skill | Use When |
|-------|----------|
| `parhelia/modal-deployment` | Setting up Modal account, volumes, secrets |
| `parhelia/task-dispatch` | Submitting tasks, understanding dispatch modes |
| `parhelia/checkpoint-resume` | Session persistence, recovery workflows |
| `parhelia/gpu-configuration` | GPU selection, optimization, cost trade-offs |
| `parhelia/interactive-attach` | SSH tunnels, tmux sessions, detach flows |
| `parhelia/budget-management` | Cost tracking, limits, alerts |
| `parhelia/troubleshooting` | Common issues and solutions |

## Key Commands

| Command | Purpose |
|---------|---------|
| `/parhelia-submit` | Submit task for remote execution |
| `/parhelia-status` | System and session status |
| `/parhelia-attach` | Attach to running session |
| `/parhelia-session` | Session management |
| `/parhelia-checkpoint` | Checkpoint operations |
| `/parhelia-budget` | Budget management |

## Architecture Overview

```
Local Machine                    Modal.com
┌─────────────────┐             ┌─────────────────────────┐
│  Claude Code    │             │  Modal Sandbox          │
│  + Parhelia CLI │────────────▶│  + Claude Code          │
│                 │   dispatch  │  + tmux session         │
│                 │◀────────────│  + workspace volume     │
└─────────────────┘   results   └─────────────────────────┘
        │                                   │
        │           SSH Tunnel              │
        └───────────────────────────────────┘
                   (attach)
```

## Cost Estimates

| Resource | Cost/Hour |
|----------|-----------|
| CPU only | ~$0.05 |
| A10G GPU | ~$1.10 |
| A100 GPU | ~$2.50 |
| H100 GPU | ~$4.50 |

API tokens are additional (~$3/$15 per 1M input/output tokens).

## Next Steps

1. Run `parhelia status` to check system health
2. Submit a simple task with `parhelia submit "hello world" --dry-run`
3. Review budget with `parhelia budget show`

For detailed guidance, ask for a specific skill like "explain checkpoint/resume".
