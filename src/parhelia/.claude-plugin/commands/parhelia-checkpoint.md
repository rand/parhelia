---
name: parhelia-checkpoint
description: Checkpoint operations - create, list, rollback, diff
argument-hint: <subcommand> [args]
---

# Checkpoint Management

Manage checkpoints for session state persistence and recovery.

## Subcommands

### create - Create Checkpoint

```bash
parhelia checkpoint create <session-id> [-m MESSAGE]
```

Creates a manual checkpoint of the current session state.

Options:
- `-m, --message`: Checkpoint description

### list - List Checkpoints

```bash
parhelia checkpoint list [--session ID] [--limit N]
```

Lists checkpoints, optionally filtered by session.

### rollback - Rollback to Checkpoint

```bash
parhelia checkpoint rollback <checkpoint-id> [--session ID] [-y]
```

Rolls back workspace to a previous checkpoint state.

Safety guarantees:
- Creates safety checkpoint before rollback
- Stashes uncommitted changes
- Provides recovery on failure

Options:
- `--session`: Session ID if checkpoint is ambiguous
- `-y, --yes`: Skip confirmation prompt

### diff - Compare Checkpoints

```bash
parhelia checkpoint diff <checkpoint-a> <checkpoint-b> [options]
```

Compare two checkpoints.

Options:
- `--session`: Session ID if ambiguous
- `--file PATH`: Show diff for specific file
- `--conversation`: Show conversation diff
- `--json`: Output as JSON

## Checkpoint Contents

Each checkpoint includes:
- **Conversation state** - Messages and context
- **Workspace files** - Changed files since last checkpoint
- **Environment** - Variables and working directory
- **tmux state** - Window layout and scroll position

## Examples

**Create checkpoint with message**:
```bash
parhelia checkpoint create task-abc12345 -m "Before refactoring auth"
```

**List all checkpoints**:
```bash
parhelia checkpoint list
```

**Rollback with confirmation**:
```bash
parhelia checkpoint rollback cp-xyz789
```

**Compare two checkpoints**:
```bash
parhelia checkpoint diff cp-abc123 cp-def456
```

## Automatic Checkpoints

Checkpoints are created automatically on:
- Session detach (Ctrl+B, D)
- Periodic intervals (configurable)
- Before risky operations
- On session completion

## Related Commands

- `/parhelia-session` - Session management
- `parhelia session recover` - Recovery wizard
- `parhelia env diff` - Environment comparison

## Arguments

$ARGUMENTS
