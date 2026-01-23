---
name: parhelia-interactive-attach
description: SSH tunnels, tmux sessions, and detach workflows
category: parhelia
keywords: [attach, detach, ssh, tunnel, tmux, interactive, session]
---

# Interactive Attach

**Scope**: Attaching to running sessions via SSH tunnel and tmux
**Lines**: ~300
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Atomic)

## When to Use This Skill

- Attaching to running Claude Code sessions
- Understanding tmux integration
- Troubleshooting SSH connection issues
- Managing detach and checkpoint flows
- Navigating the remote terminal

## Core Concepts

### Attach Architecture

```
Local Machine                    Modal Sandbox
┌─────────────────┐             ┌─────────────────────────┐
│  Terminal       │             │  tmux server            │
│    │            │             │    │                    │
│    └──SSH───────┼─────────────┼────┴──▶ tmux session   │
│       tunnel    │             │           │             │
│                 │             │           ▼             │
│                 │             │       Claude Code       │
└─────────────────┘             └─────────────────────────┘
```

### Session States

| State | Attach Behavior |
|-------|-----------------|
| RUNNING | Attach immediately |
| STARTING | Wait for ready signal |
| SUSPENDED | Resume container first |
| COMPLETED | Error (session ended) |
| FAILED | Offer recovery wizard |

### tmux Basics

Parhelia uses tmux for session persistence:

| Key | Action |
|-----|--------|
| `Ctrl+B, D` | Detach (keeps session running) |
| `Ctrl+B, [` | Scroll mode (use arrows, `q` to exit) |
| `Ctrl+B, c` | New window |
| `Ctrl+B, n/p` | Next/previous window |
| `Ctrl+B, %` | Split vertically |
| `Ctrl+B, "` | Split horizontally |

## Patterns

### Pattern 1: Basic Attach

**When**: Connect to running session

```bash
parhelia session attach task-abc12345
```

Output:
```
Connecting to session ph-task-abc12345-20260121T143022...
  Container: us-east-1a (CPU, 4 cores, 16GB)
  Running for: 12m 34s

Establishing SSH tunnel...
Attaching to tmux session...
[Press Ctrl+B, D to detach]
```

### Pattern 2: Detach with Checkpoint

**When**: Step away but keep session running

Press `Ctrl+B, D` in the attached session.

Output:
```
Detaching from session ph-task-abc12345...

Creating checkpoint... done
  Checkpoint ID: cp-xyz789
  Workspace: 45 files, 2.3 MB

Session continues running in background.
```

### Pattern 3: View Without Attaching

**When**: Check status without full attachment

```bash
parhelia task show task-abc12345
parhelia logs task-abc12345 --tail 50
```

### Pattern 4: Multiple Windows

**When**: Need multiple terminals in session

Once attached:
```
Ctrl+B, c       # Create new window
Ctrl+B, n       # Switch to next window
Ctrl+B, 0-9     # Switch to window by number
```

### Pattern 5: Session Recovery After Disconnect

**When**: Network dropped, need to reconnect

```bash
# Just attach again - tmux preserves state
parhelia session attach task-abc12345
```

## Attach Flow Detail

```
1. parhelia session attach <task-id>
   └─ Validate task exists

2. Get session info from orchestrator
   └─ Container ID, region, state

3. Retrieve tunnel endpoint from Modal
   └─ modal.Sandbox.tunnel() → (host, port)

4. Establish SSH tunnel
   └─ ssh -L 2222:localhost:22 user@host

5. Attach to tmux session
   └─ tmux attach-session -t ph-{task-id}

6. User interacts with Claude Code
   └─ Full terminal access
   └─ File system access
   └─ All tools available

7. On detach (Ctrl+B, D)
   └─ Checkpoint created
   └─ SSH tunnel closed
   └─ Session keeps running
```

## Anti-Patterns

### Anti-Pattern 1: Killing Instead of Detaching

**Bad**: Closing terminal window (kills tunnel)
```
[X] Close window  # Abrupt disconnect
```

**Good**: Proper detach
```
Ctrl+B, D  # Clean detach, checkpoint created
```

### Anti-Pattern 2: Forgetting Sessions

**Bad**: Leaving sessions running unmonitored
```bash
parhelia session attach task-123
# Detach and forget
# Session runs for 24h, burns budget
```

**Good**: Monitor and manage
```bash
parhelia status           # Check active sessions
parhelia session kill task-123  # When done
```

### Anti-Pattern 3: Too Many Attachments

**Bad**: Opening many parallel attachments
```bash
# Terminal 1: parhelia session attach task-123
# Terminal 2: parhelia session attach task-123
# Terminal 3: parhelia session attach task-123
# All fighting for same tmux session
```

**Good**: One attachment at a time
```bash
parhelia session attach task-123
# Use tmux windows for multiple terminals
Ctrl+B, c  # New window within same session
```

## Troubleshooting

### SSH Connection Refused

```
Connection refused: port 2222
```

**Solutions**:
1. Wait for container startup (STARTING state)
2. Check if session is still running: `parhelia task show <id>`
3. Modal may be scaling up - retry in 30 seconds

### tmux Session Not Found

```
can't find session: ph-task-abc12345
```

**Solutions**:
1. Session may have completed - check status
2. Container may have restarted - recovery needed
3. Use recovery wizard: `parhelia session recover <id>`

### Tunnel Drops Frequently

```
Connection reset by peer
```

**Solutions**:
1. Check network stability
2. Container may be hibernating - increase activity
3. Add SSH keepalive: configured automatically by Parhelia

### Slow/Laggy Terminal

**Solutions**:
1. Check network latency to Modal region
2. Try different region in config
3. Reduce terminal output (less verbose logging)

## Advanced: Direct SSH

For debugging, you can SSH directly:

```bash
# Get tunnel info
parhelia session attach task-123 --info-only

# Manual SSH
ssh -p 2222 -o StrictHostKeyChecking=no root@localhost

# Manual tmux attach
tmux attach-session -t ph-task-123
```

## Related Skills

- `parhelia/checkpoint-resume` - Checkpoint on detach
- `parhelia/troubleshooting` - Connection issues
- `parhelia/task-dispatch` - Starting sessions
