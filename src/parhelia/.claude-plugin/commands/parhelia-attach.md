---
name: parhelia-attach
description: Attach to a running session via SSH tunnel and tmux
argument-hint: <session-id>
---

# Attach to Session

Attach to a running Claude Code session on Modal via SSH tunnel and tmux.

## Usage

```bash
parhelia attach <session-id>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `session-id` | Task ID or session ID to attach to |

## Attach Flow

1. **Validate** - Verify session exists and is running
2. **Tunnel** - Retrieve SSH tunnel endpoint from Modal
3. **Connect** - Establish SSH tunnel to sandbox
4. **Attach** - Connect to tmux session

## Interactive Session

Once attached, you have a live terminal in the Modal sandbox:
- Claude Code is running in tmux
- Full terminal interaction available
- File system access to workspace

## Detach

Press `Ctrl+B, D` to detach from the tmux session.

On detach:
- Automatic checkpoint is created
- Session continues running in background
- Re-attach anytime with same command

## Examples

**Attach to task**:
```bash
parhelia attach task-abc12345
```

**Expected output**:
```
Connecting to session ph-task-abc12345-20260121T143022...
  Container: us-east-1a (CPU, 4 cores, 16GB)
  Running for: 12m 34s
  Last activity: 2s ago

Establishing SSH tunnel...
Attaching to tmux session...
[Press Ctrl+B, D to detach]
```

## Session States

| State | Behavior |
|-------|----------|
| RUNNING | Attach immediately |
| SUSPENDED | Resume first, then attach |
| STARTING | Wait for ready, then attach |
| COMPLETED | Error - session ended |
| FAILED | Offer recovery wizard |

## Related Commands

- `/parhelia-status` - Check session status
- `/parhelia-session` - Session management
- `parhelia detach <id>` - Explicit detach
- `parhelia session recover <id>` - Recovery wizard

## Arguments

$ARGUMENTS
