---
name: parhelia-session
description: Session management - list, show, kill, logs, recover
argument-hint: <subcommand> [args]
---

# Session Management

Manage Claude Code sessions running on Modal infrastructure.

## Subcommands

### list - List Sessions

```bash
parhelia list [--status STATUS] [--limit N]
```

Options:
- `--status`: Filter by all, pending, running, completed, failed
- `--limit`: Maximum sessions to show (default: 20)

### show - Show Session Details

```bash
parhelia task show <task-id> [--json]
```

Shows:
- Task status and type
- Requirements (GPU, memory)
- Worker information
- Result summary

### logs - View Session Logs

```bash
parhelia logs <session-id> [--follow] [--lines N]
```

Options:
- `-f, --follow`: Stream logs in real-time
- `-n, --lines`: Number of lines to show (default: 50)

### kill - Terminate Session

```bash
parhelia detach <session-id>
```

Gracefully terminates a running session.

### recover - Recovery Wizard

```bash
parhelia session recover <session-id> [--from CHECKPOINT] [--action ACTION]
```

Interactive recovery for failed or stopped sessions.

Actions:
- `resume`: Resume from checkpoint (default)
- `new`: Start fresh session with same prompt
- `wait`: Manual intervention required

## Examples

**List running sessions**:
```bash
parhelia list --status running
```

**Show task details as JSON**:
```bash
parhelia task show task-abc12345 --json
```

**Follow logs**:
```bash
parhelia logs task-abc12345 -f
```

**Recover failed session**:
```bash
parhelia session recover task-abc12345 --action resume
```

## Related Commands

- `/parhelia-attach` - Attach to session
- `/parhelia-status` - System status
- `/parhelia-checkpoint` - Checkpoint management

## Arguments

$ARGUMENTS
