---
name: parhelia-checkpoint-resume
description: Session persistence, checkpoints, and recovery workflows
category: parhelia
keywords: [checkpoint, resume, recovery, persistence, state, rollback]
---

# Checkpoint and Resume

**Scope**: Session state persistence, checkpoint management, and recovery
**Lines**: ~350
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Atomic)

## When to Use This Skill

- Understanding checkpoint triggers and timing
- Resuming from failed or interrupted sessions
- Managing checkpoint storage and retention
- Configuring auto-checkpoint intervals
- Rolling back to previous states

## Core Concepts

### What's in a Checkpoint

Each checkpoint captures:

| Component | Contents |
|-----------|----------|
| **Conversation** | Messages, context, Claude state |
| **Workspace** | Modified files since last checkpoint |
| **Environment** | Variables, working directory |
| **tmux State** | Window layout, scroll position |

### Checkpoint Triggers

| Trigger | When |
|---------|------|
| `PERIODIC` | Every N minutes (configurable) |
| `DETACH` | User detaches from session (Ctrl+B, D) |
| `MANUAL` | Explicit `parhelia checkpoint create` |
| `ERROR` | Before risky operations |
| `COMPLETE` | Session completes successfully |
| `SHUTDOWN` | Container shutting down |

### Checkpoint Storage

```
/vol/parhelia/checkpoints/
└── {session-id}/
    └── cp-{timestamp}/
        ├── manifest.json      # Metadata
        ├── conversation.json  # Claude state
        ├── workspace.tar.gz   # Changed files
        ├── tmux-layout.txt    # Terminal state
        └── env.json           # Environment
```

## Patterns

### Pattern 1: Manual Checkpoint

**When**: Before risky changes, at logical breakpoints

```bash
parhelia checkpoint create task-abc12345 -m "Before refactoring auth"
```

### Pattern 2: List Checkpoints

**When**: Finding a checkpoint to resume from

```bash
parhelia checkpoint list --session task-abc12345
```

Output:
```
ID               Session              Trigger    Created
cp-20260121-001  task-abc12345        PERIODIC   2026-01-21 14:30
cp-20260121-002  task-abc12345        MANUAL     2026-01-21 14:45
cp-20260121-003  task-abc12345        DETACH     2026-01-21 15:00
```

### Pattern 3: Resume from Checkpoint

**When**: Session failed, want to continue

```bash
parhelia resume task-abc12345
```

Resumes from latest checkpoint. For specific checkpoint:

```bash
parhelia resume task-abc12345 --checkpoint-id cp-20260121-002
```

### Pattern 4: Rollback Workspace

**When**: Made mistakes, want to revert

```bash
parhelia checkpoint rollback cp-20260121-001
```

Safety guarantees:
- Creates safety checkpoint before rollback
- Stashes uncommitted git changes
- Can recover if rollback fails

### Pattern 5: Compare Checkpoints

**When**: Understanding what changed

```bash
parhelia checkpoint diff cp-20260121-001 cp-20260121-003
```

Shows:
- Files added/modified/deleted
- Conversation turn count
- Token usage difference

## Recovery Wizard

For failed sessions, use the interactive wizard:

```bash
parhelia session recover task-abc12345
```

Output:
```
Session Recovery: task-abc12345
==============================

Status: FAILED
  Reason: Container timeout after 24 hours
  Last checkpoint: cp-xyz789 (5 minutes before timeout)

Available Actions:
  [1] Resume from checkpoint (recommended)
  [2] Start fresh session with same prompt
  [3] View checkpoint contents
  [4] Cancel

Select action [1]:
```

## Auto-Checkpoint Configuration

In `parhelia.toml`:

```toml
[checkpoint]
# Periodic checkpoints
auto_interval_minutes = 5

# Maximum checkpoints per session
max_per_session = 20

# Retention policy
retention_days = 7

# Checkpoint on these triggers
triggers = ["periodic", "detach", "error", "complete"]
```

## Anti-Patterns

### Anti-Pattern 1: Ignoring Checkpoints

**Bad**: Running long task without checkpoints
```bash
parhelia submit "Train for 24 hours" --no-checkpoint
```

**Good**: Let checkpoints run
```bash
parhelia submit "Train for 24 hours"
# Periodic checkpoints every 5 min by default
```

### Anti-Pattern 2: Manual Checkpoint Spam

**Bad**: Checkpointing every minute
```bash
while true; do parhelia checkpoint create ...; sleep 60; done
```

**Good**: Trust auto-checkpoints, manual only at key points
```bash
parhelia checkpoint create task-abc -m "Before major change"
```

### Anti-Pattern 3: Skipping Recovery Wizard

**Bad**: Starting fresh after failure
```bash
parhelia submit "Same task again"  # Loses progress
```

**Good**: Use recovery wizard
```bash
parhelia session recover task-abc12345  # Resumes from checkpoint
```

## Checkpoint Diff Example

```bash
$ parhelia checkpoint diff cp-001 cp-003

Checkpoint Comparison
=====================
From: cp-001 (2026-01-21 14:30)
To:   cp-003 (2026-01-21 15:00)

Files Changed:
  + src/auth/login.py (new)
  M src/auth/session.py (+45, -12)
  M tests/test_auth.py (+30, -0)

Conversation:
  Turns: 15 → 47 (+32)
  Tokens: 12,500 → 45,000 (+32,500)

Cost:
  $0.12 → $0.45 (+$0.33)
```

## Storage Management

Check checkpoint disk usage:

```bash
du -sh /vol/parhelia/checkpoints/
```

Clean old checkpoints:

```bash
# List checkpoints older than 7 days
parhelia checkpoint list --older-than 7d

# Delete specific checkpoint
parhelia checkpoint delete cp-old-001
```

## Related Skills

- `parhelia/interactive-attach` - Detach triggers checkpoint
- `parhelia/troubleshooting` - Checkpoint recovery issues
- `parhelia/task-dispatch` - Task lifecycle
