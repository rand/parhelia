# SPEC-12: Interactive Sessions

**Status**: Draft
**Author**: Claude + rand
**Date**: 2026-01-21

## Overview

This specification defines world-class interactive session UX for Parhelia, enabling seamless attach/detach workflows with automatic checkpoint management and session recovery.

## Goals

- [SPEC-12.01] Seamless SSH tunnel attachment to running sessions
- [SPEC-12.02] Automatic checkpoint on detach
- [SPEC-12.03] Session recovery wizard for failed sessions
- [SPEC-12.04] Real-time status updates via event streaming

## Non-Goals

- Web-based terminal (v1 is SSH/tmux only)
- VS Code Remote integration (future work)

---

## Attach Flow

### [SPEC-12.10] Attach Command

```bash
parhelia attach <session-id>
```

**Sequence**:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Validate session exists and is in RUNNING state              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Retrieve tunnel endpoint from Modal sandbox                  │
│    - Call sandbox.tunnel() to get (host, port)                  │
│    - Verify endpoint is reachable                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Display connection info and establish SSH tunnel             │
│    - Show container details (region, resources, uptime)         │
│    - Spawn SSH process with port forwarding                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Attach to tmux session                                       │
│    - Execute: tmux attach -t <session-name>                     │
│    - User now has interactive terminal                          │
└─────────────────────────────────────────────────────────────────┘
```

### [SPEC-12.11] Attach Output

```
$ parhelia attach task-abc123

Connecting to session ph-task-abc123-20260121T143022...
  Container: us-east-1a (CPU, 4 cores, 16GB RAM)
  Running for: 12m 34s
  Last activity: 2s ago

Establishing SSH tunnel...
  Tunnel: localhost:2222 -> modal-sandbox:22

Attaching to tmux session...
[Press Ctrl+B, D to detach]

# [tmux session appears with Claude Code running]
```

### [SPEC-12.12] Attach States

| State | Action |
|-------|--------|
| RUNNING | Proceed with attach |
| SUSPENDED | Resume sandbox first, then attach |
| STARTING | Wait for ready signal, then attach |
| COMPLETED | Error: session has ended |
| FAILED | Offer recovery wizard |

---

## Detach Flow

### [SPEC-12.20] Detach Trigger

Detach occurs when user presses `Ctrl+B, D` (tmux detach).

**Automatic actions on detach**:
1. Create checkpoint with current state
2. Keep sandbox running in background
3. Display re-attach instructions

### [SPEC-12.21] Detach Output

```
[Ctrl+B, D pressed]

Detaching from session ph-task-abc123-20260121T143022...

Creating checkpoint... done
  Checkpoint ID: cp-xyz789
  Workspace: 45 files, 2.3 MB
  Conversation: 127 turns

Session continues running in background.
  Time remaining: 11h 28m
  Estimated cost if left running: $0.38

Commands:
  Re-attach:    parhelia attach task-abc123
  View logs:    parhelia session logs task-abc123
  Kill session: parhelia session kill task-abc123
```

### [SPEC-12.22] Checkpoint Content

Each checkpoint MUST include:
- Conversation state (messages, context)
- Workspace files (changed since last checkpoint)
- Environment state (env vars, working directory)
- tmux state (window layout, scroll position)

```
/vol/parhelia/checkpoints/{session-id}/cp-{timestamp}/
├── manifest.json          # Checkpoint metadata
├── conversation.json      # Claude conversation state
├── workspace.tar.gz       # Changed workspace files
├── tmux-layout.txt        # tmux window configuration
└── env.json               # Environment snapshot
```

---

## Session Recovery

### [SPEC-12.30] Recovery Wizard

When a session fails or times out, offer interactive recovery:

```bash
$ parhelia attach task-abc123

Session Recovery: task-abc123
==============================

Status: FAILED
  Reason: Container timeout after 24 hours
  Last checkpoint: cp-xyz789 (5 minutes before timeout)
  Checkpoint age: 24h 5m ago

Recoverable state:
  ✓ Conversation: 127 turns preserved
  ✓ Workspace: 45 files, 2.3 MB
  ✓ Environment: All variables captured

Available Actions:
  [1] Resume from checkpoint (recommended)
  [2] Start fresh session with same prompt
  [3] View checkpoint contents
  [4] Download checkpoint locally
  [5] Cancel

Select action [1]:
```

### [SPEC-12.31] Resume from Checkpoint

```
Select action [1]: 1

Resuming from checkpoint cp-xyz789...

Creating new sandbox... done
  Container: us-east-1b (CPU, 4 cores, 16GB)
  New session ID: ph-task-abc123-20260122T103000

Restoring state...
  Workspace files: 45/45 restored
  Conversation: 127 turns loaded
  Environment: 12 variables set

Session resumed successfully.

Attaching...
[Press Ctrl+B, D to detach]
```

### [SPEC-12.32] Recovery States

| Failure Type | Recovery Option |
|--------------|-----------------|
| Timeout | Resume from last checkpoint |
| OOM Kill | Resume with more memory (`--memory 32`) |
| Network Error | Resume, auto-retry pending operations |
| User Kill | Offer fresh start or resume |
| Internal Error | Download checkpoint, file bug report |

---

## Event Streaming

### [SPEC-12.40] Event Stream Endpoint

Modal sandboxes expose SSE endpoint for real-time status:

```bash
# CLI polling mode
parhelia status --watch task-abc123

# Or direct SSE connection
curl -N https://parhelia-sandbox-abc123.modal.run/events
```

### [SPEC-12.41] Event Types

```json
{"type": "heartbeat", "timestamp": "2026-01-21T14:35:00Z"}
{"type": "activity", "tool": "Edit", "file": "src/main.py", "lines_changed": 15}
{"type": "checkpoint", "id": "cp-auto-001", "trigger": "periodic"}
{"type": "warning", "message": "Session will timeout in 1 hour"}
{"type": "completed", "result": "success", "summary": "Fixed 3 bugs"}
```

### [SPEC-12.42] Status Watch Output

```bash
$ parhelia status --watch task-abc123

Session: ph-task-abc123-20260121T143022
Status: RUNNING

[14:35:02] Claude used Edit on src/main.py (+15 lines)
[14:35:15] Claude used Bash: pytest tests/
[14:35:28] Tests: 12 passed, 0 failed
[14:36:00] Auto-checkpoint created: cp-auto-001
[14:36:45] Claude used Edit on src/utils.py (+8 lines)
...

Press Ctrl+C to stop watching
```

---

## Session Management Commands

### [SPEC-12.50] List Sessions

```bash
$ parhelia session list

Active Sessions
===============
ID                              Status    Age       Container       Cost
ph-task-abc123-20260121T143022  RUNNING   45m       us-east, CPU    $0.12
ph-task-def456-20260121T120000  RUNNING   3h 15m    us-west, A10G   $3.58

Suspended Sessions
==================
ID                              Status     Suspended   Checkpoint
ph-task-ghi789-20260120T090000  SUSPENDED  18h ago     cp-xyz123

Total running cost: $0.04/hour
```

### [SPEC-12.51] Show Session Details

```bash
$ parhelia session show task-abc123

Session: ph-task-abc123-20260121T143022
=======================================

Status:      RUNNING
Started:     2026-01-21 14:30:22 UTC (45 minutes ago)
Container:   us-east-1a
Resources:   4 CPU cores, 16 GB RAM
Cost so far: $0.12

Task:
  Prompt: "Fix the authentication bug in login.py"
  Type: CODE_FIX

Activity:
  Total tool calls: 47
  Last activity: 30 seconds ago
  Files modified: 5

Checkpoints:
  cp-auto-001  14:36:00  Periodic (5 min)
  cp-auto-002  14:41:00  Periodic (5 min)
  cp-detach-001 (pending)

Commands:
  Attach:     parhelia attach task-abc123
  Logs:       parhelia session logs task-abc123
  Kill:       parhelia session kill task-abc123
```

### [SPEC-12.52] Session Logs

```bash
$ parhelia session logs task-abc123 --tail 50

[14:30:22] Session started
[14:30:25] Claude initialized with 127 context tokens
[14:30:30] [Claude] Reading src/auth/login.py...
[14:30:32] [Claude] Found potential issue on line 45
[14:30:35] [Claude] Editing src/auth/login.py...
[14:30:38] [Claude] Running pytest tests/test_login.py...
[14:30:45] [Output] 3 passed, 1 failed
...
```

---

## Acceptance Criteria

- [ ] [SPEC-12.AC1] `parhelia attach` establishes SSH tunnel in < 5 seconds
- [ ] [SPEC-12.AC2] Checkpoint created automatically on every detach
- [ ] [SPEC-12.AC3] Recovery wizard offers resume from checkpoint
- [ ] [SPEC-12.AC4] `parhelia status --watch` streams events in real-time
- [ ] [SPEC-12.AC5] Session list shows all sessions with cost information
- [ ] [SPEC-12.AC6] Session survives client disconnect (keeps running)

---

## References

- [Modal Sandbox Tunnels](https://modal.com/docs/guide/sandbox-tunnels)
- [SPEC-03: Checkpoint Resume](./SPEC-03-checkpoint-resume.md)
- [SPEC-02: Session Management](./SPEC-02-session-management.md)
