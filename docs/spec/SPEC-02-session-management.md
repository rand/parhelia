# SPEC-02: Session Management and tmux Integration

**Status**: Draft
**Issue**: ph-by1
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines how Parhelia manages Claude Code sessions in remote Modal containers, including tmux integration for persistence and interactive attachment.

## Goals

- [SPEC-02.01] Provide persistent sessions that survive network disconnection
- [SPEC-02.02] Support both headless (automated) and interactive (human-attached) modes
- [SPEC-02.03] Enable seamless attachment to running sessions via SSH tunnel
- [SPEC-02.04] Support multiple concurrent sessions per container
- [SPEC-02.05] Integrate session lifecycle with checkpoint system

## Non-Goals

- Terminal multiplexer alternatives (screen, zellij) - tmux only for v1
- Web-based terminal UI (future consideration)
- Session recording/playback (future consideration)

---

## Architecture

### Session Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL CONTAINER                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         tmux Server                                   │   │
│  │                                                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │   │
│  │  │ Session: main   │  │ Session: task-1 │  │ Session: task-2 │       │   │
│  │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │       │   │
│  │  │ │ Window 0    │ │  │ │ Window 0    │ │  │ │ Window 0    │ │       │   │
│  │  │ │ Claude Code │ │  │ │ Claude Code │ │  │ │ Claude Code │ │       │   │
│  │  │ │ (headless)  │ │  │ │ (headless)  │ │  │ │ (attached)  │ │       │   │
│  │  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │       │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              │ SSH Tunnel (port 2222)                        │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Local Terminal    │
                    │   ssh + tmux attach │
                    └─────────────────────┘
```

### Session States

```
                    ┌─────────────┐
                    │   CREATED   │
                    └──────┬──────┘
                           │ start
                           ▼
         ┌─────────────────────────────────────┐
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│    RUNNING      │◀───────────────────────────┤
│   (headless)    │                            │
└────────┬────────┘                            │
         │ attach                              │
         ▼                                     │
┌─────────────────┐                            │
│    ATTACHED     │─────── detach ─────────────┤
│  (interactive)  │                            │
└────────┬────────┘                            │
         │                                     │
         │ complete/error/timeout              │
         ▼                                     │
┌─────────────────┐                            │
│   CHECKPOINTED  │─────── resume ─────────────┘
└────────┬────────┘
         │ cleanup
         ▼
┌─────────────────┐
│    TERMINATED   │
└─────────────────┘
```

---

## Requirements

### [SPEC-02.10] Session Identification

Each session MUST have a unique identifier:

```
{prefix}-{task-id}-{timestamp}
```

Example: `ph-fix-auth-20260116T143022`

| Component | Format | Purpose |
|-----------|--------|---------|
| `prefix` | `ph` | Parhelia namespace |
| `task-id` | User-provided or auto | Links to beads issue or task description |
| `timestamp` | ISO8601 compact | Uniqueness, ordering |

### [SPEC-02.11] tmux Server Configuration

The tmux server MUST be configured for Parhelia use:

```bash
# /vol/parhelia/config/tmux.conf

# Server options
set-option -g default-shell /bin/bash
set-option -g history-limit 50000
set-option -g escape-time 0

# Session options
set-option -g base-index 1
set-option -g renumber-windows on

# Enable mouse for interactive sessions
set-option -g mouse on

# Status bar shows session ID and resource usage
set-option -g status-right '#{session_name} | CPU: #{cpu_percentage} | MEM: #{mem_percentage}'

# Parhelia-specific: hook for checkpoint on detach
set-hook -g client-detached 'run-shell "parhelia-checkpoint #{session_name}"'
```

### [SPEC-02.12] Session Creation

Sessions MUST be created via the Parhelia session manager:

```python
class SessionManager:
    async def create_session(
        self,
        task_id: str,
        working_dir: str = "/vol/parhelia/workspaces",
        env: dict[str, str] | None = None,
    ) -> Session:
        """Create a new tmux session for a Claude Code task."""
        session_id = f"ph-{task_id}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"

        # Create tmux session
        await run_command([
            "tmux", "new-session",
            "-d",                    # Detached
            "-s", session_id,        # Session name
            "-c", working_dir,       # Working directory
        ])

        # Set environment variables in session
        for key, value in (env or {}).items():
            await run_command([
                "tmux", "set-environment", "-t", session_id, key, value
            ])

        return Session(id=session_id, state=SessionState.CREATED)
```

### [SPEC-02.13] Headless Execution

For automated dispatch, sessions run Claude Code headlessly:

```python
async def run_headless(
    self,
    session: Session,
    prompt: str,
    allowed_tools: list[str] | None = None,
    max_turns: int = 50,
) -> TaskResult:
    """Run Claude Code headlessly in the session."""

    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "stream-json",
        "--max-turns", str(max_turns),
    ]

    if allowed_tools:
        cmd.extend(["--allowedTools", ",".join(allowed_tools)])

    # Send command to tmux session
    await run_command([
        "tmux", "send-keys", "-t", session.id,
        " ".join(shlex.quote(c) for c in cmd),
        "Enter"
    ])

    session.state = SessionState.RUNNING
    return await self._monitor_completion(session)
```

### [SPEC-02.14] Interactive Attachment

Users MUST be able to attach to running sessions:

#### SSH Tunnel Setup

```python
# In Modal function
@app.function(...)
def run_with_ssh_tunnel():
    """Start SSH server for interactive attachment."""

    # Generate ephemeral host key
    subprocess.run(["ssh-keygen", "-A"], check=True)

    # Start SSH daemon on port 2222
    subprocess.Popen(["/usr/sbin/sshd", "-D", "-p", "2222"])

    # Forward port via Modal tunnel
    with modal.forward(2222) as tunnel:
        print(f"SSH available at: {tunnel.url}")
        # ... run tasks
```

#### SSH Tunnel Characteristics

**Important Modal constraints** (verified January 2026):
- `modal.forward()` is an **experimental API** - may change
- Tunnel ports are **randomly assigned** (not predictable)
- **TCP only** - Modal does not support UDP (no mosh)
- Tunnel terminates when context manager exits

```python
# In Modal Sandbox
with modal.forward(2222, unencrypted=True) as tunnel:
    # tunnel.tcp_socket returns (hostname, port) tuple
    host, port = tunnel.tcp_socket
    print(f"SSH available at: ssh -p {port} user@{host}")
    # Port is random, e.g., r3.modal.host:23447
```

#### SSH Keepalive Configuration

Since mosh is not available, configure SSH for resilience:

```bash
# ~/.ssh/config (local)
Host parhelia-*
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes
    ConnectionAttempts 3
```

#### Local Attachment Command

```bash
# SSH attachment
parhelia attach <session-id>
# Resolves tunnel endpoint and connects:
# → ssh -t -o ServerAliveInterval=30 -p $TUNNEL_PORT $TUNNEL_HOST \
#     "tmux attach-session -t $SESSION_ID"
```

### [SPEC-02.15] Session Lifecycle Hooks

Sessions MUST trigger hooks at lifecycle events:

| Event | Hook | Purpose |
|-------|------|---------|
| `session.created` | `on_session_created` | Initialize session state file |
| `session.started` | `on_session_started` | Begin heartbeat, log start |
| `session.attached` | `on_session_attached` | Pause auto-checkpoint |
| `session.detached` | `on_session_detached` | Trigger checkpoint |
| `session.completed` | `on_session_completed` | Final checkpoint, report results |
| `session.error` | `on_session_error` | Checkpoint, log error |
| `session.timeout` | `on_session_timeout` | Checkpoint, notify orchestrator |

```python
class SessionHooks:
    async def on_session_detached(self, session: Session):
        """Handle client detachment - checkpoint immediately."""
        await checkpoint_manager.create_checkpoint(session)
        logger.info(f"Session {session.id} checkpointed on detach")

    async def on_session_completed(self, session: Session, result: TaskResult):
        """Handle task completion."""
        await checkpoint_manager.create_final_checkpoint(session, result)
        await metrics.record_completion(session, result)
        session.state = SessionState.CHECKPOINTED
```

### [SPEC-02.16] Multi-Session Support

A single container MAY run multiple concurrent sessions:

```python
MAX_SESSIONS_PER_CONTAINER = 4  # Configurable

class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    async def can_accept_session(self) -> bool:
        """Check if container can accept another session."""
        active = sum(1 for s in self.sessions.values()
                     if s.state in (SessionState.RUNNING, SessionState.ATTACHED))
        return active < MAX_SESSIONS_PER_CONTAINER

    async def get_resource_availability(self) -> ResourceStatus:
        """Report available capacity for orchestrator."""
        return ResourceStatus(
            sessions_available=MAX_SESSIONS_PER_CONTAINER - len(self.sessions),
            cpu_available=get_available_cpu(),
            memory_available=get_available_memory(),
        )
```

### [SPEC-02.17] Human Intervention Signaling

The system MUST detect when human intervention is needed via two mechanisms:

#### Claude-Requested Intervention

Claude Code can signal it needs human help via structured output:

```json
// In stream-json output
{
  "type": "needs_human",
  "reason": "Permission denied for destructive operation",
  "context": "Attempting to drop database table, need explicit approval",
  "suggested_action": "Review and approve or reject the operation"
}
```

The session manager watches for this signal:

```python
async def monitor_claude_output(self, session: Session):
    """Watch Claude output for intervention signals."""
    async for line in session.output_stream():
        if '"type": "needs_human"' in line:
            event = json.loads(line)
            await self.request_human_intervention(
                session,
                reason=event["reason"],
                context=event["context"],
            )
```

#### Timeout-Based Intervention

If no progress for configurable duration, trigger notification:

```python
PROGRESS_TIMEOUT_MINUTES = 10  # Configurable

async def monitor_progress(self, session: Session):
    """Watch for stalled sessions."""
    while session.state == SessionState.RUNNING:
        await asyncio.sleep(60)

        minutes_since_activity = (
            datetime.now() - session.last_activity
        ).total_seconds() / 60

        if minutes_since_activity > PROGRESS_TIMEOUT_MINUTES:
            await self.request_human_intervention(
                session,
                reason="timeout",
                context=f"No progress for {minutes_since_activity:.0f} minutes",
            )
```

#### Intervention Request Flow

```python
async def request_human_intervention(
    self,
    session: Session,
    reason: str,
    context: str,
):
    """Request human intervention for a session."""
    # 1. Notify orchestrator
    await orchestrator.notify_needs_human(session.id, reason, context)

    # 2. Create checkpoint
    await checkpoint_manager.create_checkpoint(session)

    # 3. Send notification (webhook, email, CLI alert)
    await notifications.send(
        title=f"Session {session.id} needs attention",
        body=f"Reason: {reason}\n\nContext: {context}",
        actions=[
            {"label": "Attach", "url": f"parhelia://attach/{session.id}"},
            {"label": "Dismiss", "url": f"parhelia://dismiss/{session.id}"},
        ]
    )

    # 4. Update session state
    session.needs_human = True
    session.intervention_reason = reason
```

### [SPEC-02.18] Session Cleanup

Terminated sessions MUST be cleaned up:

```python
async def cleanup_session(self, session: Session):
    """Clean up terminated session resources."""

    # Kill tmux session
    await run_command(["tmux", "kill-session", "-t", session.id])

    # Archive logs
    await archive_session_logs(session)

    # Remove from active sessions
    del self.sessions[session.id]

    # Notify orchestrator
    await orchestrator.session_terminated(session.id)
```

---

## Interaction Patterns

### Pattern 1: Fully Automated (Headless)

```
Orchestrator                    Modal Container
    │                                │
    │─── dispatch(task) ────────────▶│
    │                                │── create_session()
    │                                │── run_headless(prompt)
    │                                │── [Claude Code runs]
    │                                │── checkpoint()
    │◀── result ─────────────────────│
    │                                │── cleanup_session()
```

### Pattern 2: Interactive Debug

```
Orchestrator                    Modal Container                  User Terminal
    │                                │                                │
    │─── dispatch(task) ────────────▶│                                │
    │                                │── create_session()             │
    │                                │── run_headless(prompt)         │
    │                                │                                │
    │                                │   [task hits issue]            │
    │                                │                                │
    │◀── needs_human ────────────────│                                │
    │                                │                                │
    │─── request_tunnel ────────────▶│                                │
    │◀── tunnel_url ─────────────────│                                │
    │                                │                                │
    │─────────────────────────────────────── tunnel_url ─────────────▶│
    │                                │                                │
    │                                │◀─────── ssh + attach ──────────│
    │                                │                                │
    │                                │   [user interacts]             │
    │                                │                                │
    │                                │◀─────── detach ────────────────│
    │                                │── checkpoint()                 │
    │                                │── resume_headless()            │
```

### Pattern 3: Direct Interactive

```
User                            Parhelia CLI                     Modal Container
  │                                  │                                │
  │─── parhelia shell ──────────────▶│                                │
  │                                  │─── spawn_interactive ─────────▶│
  │                                  │                                │── create_session()
  │                                  │◀── tunnel_url ─────────────────│
  │                                  │                                │
  │◀── connecting... ────────────────│                                │
  │                                  │                                │
  │────────────────────────── ssh tunnel ─────────────────────────────▶│
  │                                  │                                │── attach_session()
  │                                  │                                │
  │   [interactive Claude Code session]                               │
  │                                  │                                │
  │─── Ctrl+B, D (detach) ──────────────────────────────────────────▶│
  │                                  │                                │── checkpoint()
  │◀── session checkpointed ─────────│                                │
```

---

## Data Structures

### Session State File

```json
// /vol/parhelia/sessions/{session-id}/state.json
{
  "id": "ph-fix-auth-20260116T143022",
  "task_id": "fix-auth",
  "created_at": "2026-01-16T14:30:22Z",
  "state": "running",
  "working_dir": "/vol/parhelia/workspaces/myproject",
  "env": {
    "ANTHROPIC_API_KEY": "[REDACTED]",
    "PROJECT_NAME": "myproject"
  },
  "claude_pid": 12345,
  "last_heartbeat": "2026-01-16T14:35:00Z",
  "last_checkpoint": "2026-01-16T14:34:00Z",
  "attached_clients": [],
  "metrics": {
    "turns_completed": 12,
    "tokens_used": 45000,
    "tools_invoked": ["Read", "Edit", "Bash"]
  }
}
```

### Session List Response

```python
@dataclass
class SessionInfo:
    id: str
    task_id: str
    state: SessionState
    created_at: datetime
    last_activity: datetime
    attached: bool

@dataclass
class ListSessionsResponse:
    sessions: list[SessionInfo]
    container_id: str
    capacity: ResourceStatus
```

---

## CLI Commands

### `parhelia session list`

```
$ parhelia session list

CONTAINER    SESSION                         STATE      CREATED         ATTACHED
us-east-1a   ph-fix-auth-20260116T143022    running    5 minutes ago   no
us-east-1a   ph-add-tests-20260116T140000   attached   35 minutes ago  yes
us-east-1b   ph-refactor-20260116T120000    checkpointed 2 hours ago   no
```

### `parhelia session attach`

```
$ parhelia session attach ph-fix-auth-20260116T143022

Connecting to us-east-1a...
Attaching to session ph-fix-auth-20260116T143022...

[tmux session appears]
```

### `parhelia session kill`

```
$ parhelia session kill ph-fix-auth-20260116T143022

Checkpointing session... done
Terminating session... done
Session ph-fix-auth-20260116T143022 terminated.
```

---

## Acceptance Criteria

- [ ] [SPEC-02.AC1] Sessions persist across network disconnection (SSH drop)
- [ ] [SPEC-02.AC2] Headless execution completes without human intervention
- [ ] [SPEC-02.AC3] Interactive attachment works via SSH tunnel
- [ ] [SPEC-02.AC4] Detach triggers automatic checkpoint
- [ ] [SPEC-02.AC5] Multiple sessions can run concurrently in one container
- [ ] [SPEC-02.AC6] Session cleanup removes all resources
- [ ] [SPEC-02.AC7] CLI commands work as documented

---

## Resolved Questions

1. ~~**mosh support**~~: **Resolved (NOT SUPPORTED)** - Modal does not support UDP port forwarding. SSH only with keepalive configuration for resilience.
2. ~~**Human intervention signaling**~~: **Resolved** - Both Claude-requested (`needs_human` output) AND timeout-based (10 min default).
3. ~~**UDP forwarding**~~: **Resolved** - Modal tunnels are TCP-only. No UDP support.

## Open Questions

1. **Web terminal**: Future consideration for browser-based attachment? (Could use Modal's Sandbox Connect Tokens)
2. **Session transfer**: Can a session be migrated between containers?

---

## References

- [tmux Manual](https://man7.org/linux/man-pages/man1/tmux.1.html)
- [Modal SSH Tunnels](https://modal.com/docs/guide/tunnels)
- [Claude Code Headless Mode](https://docs.anthropic.com/en/docs/claude-code/headless)
- ADR-001: System Architecture
- SPEC-01: Remote Environment Provisioning
