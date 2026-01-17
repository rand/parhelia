# SPEC-03: Checkpoint and Resume System

**Status**: Draft
**Issue**: ph-532
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines how Parhelia persists Claude Code session state and recovers from failures, ensuring no work is lost when containers die, networks fail, or sessions timeout.

## Goals

- [SPEC-03.01] Persist full conversation state at configurable intervals
- [SPEC-03.02] Detect container/session failure via heartbeat monitoring
- [SPEC-03.03] Automatically resume sessions from last checkpoint
- [SPEC-03.04] Preserve uncommitted code changes across failures
- [SPEC-03.05] Maintain checkpoint history for rollback capability

## Non-Goals

- Real-time replication (eventual consistency is acceptable)
- Cross-region checkpoint sync (single region for v1)
- Incremental/differential checkpoints (full snapshots for v1)

---

## Architecture

### Checkpoint Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODAL CONTAINER                                    │
│                                                                              │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐          │
│  │  Claude Code   │────▶│  Checkpoint    │────▶│  Modal Volume  │          │
│  │  Session       │     │  Manager       │     │  /checkpoints/ │          │
│  └────────────────┘     └───────┬────────┘     └────────────────┘          │
│                                 │                                           │
│                                 │ heartbeat                                 │
│                                 ▼                                           │
│                         ┌────────────────┐                                  │
│                         │   Heartbeat    │                                  │
│                         │   Emitter      │                                  │
│                         └───────┬────────┘                                  │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                       │
│                                                                              │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐          │
│  │   Heartbeat    │────▶│   Failure      │────▶│   Resume       │          │
│  │   Monitor      │     │   Detector     │     │   Scheduler    │          │
│  └────────────────┘     └────────────────┘     └────────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Checkpoint Contents

```
/vol/parhelia/checkpoints/{session-id}/{checkpoint-id}/
├── manifest.json           # Checkpoint metadata
├── conversation.json       # Claude conversation history
├── session_state.json      # Session manager state
├── environment.json        # Environment variables (non-secret)
├── workspace.tar.zst       # Working directory snapshot (zstd compressed)
├── tmux_state.txt          # tmux layout and history
└── mcp_state/              # MCP server states
    ├── beads.json
    └── ...
```

---

## Requirements

### [SPEC-03.10] Checkpoint State Schema

Each checkpoint MUST capture the following state:

```python
@dataclass
class Checkpoint:
    # Identity
    id: str                          # Unique checkpoint ID (ULID)
    session_id: str                  # Parent session
    created_at: datetime             # Creation timestamp
    trigger: CheckpointTrigger       # What caused this checkpoint

    # Conversation state
    conversation: ConversationState  # Full Claude conversation
    turn_count: int                  # Number of turns completed
    last_message_id: str             # For continuation

    # Session state
    working_directory: str           # Current working dir
    environment: dict[str, str]      # Non-secret env vars
    tmux_layout: str                 # tmux window/pane layout

    # Code state
    workspace_snapshot: str          # Path to workspace.tar.zst
    uncommitted_changes: list[str]   # List of modified files
    git_state: GitState              # Branch, HEAD, stash

    # MCP state
    mcp_server_states: dict[str, Any]  # Per-server state

    # Metrics
    tokens_used: int
    cost_estimate: float
    tools_invoked: list[str]

class CheckpointTrigger(Enum):
    PERIODIC = "periodic"            # Scheduled interval
    DETACH = "detach"                # User detached
    ERROR = "error"                  # Error occurred
    COMPLETE = "complete"            # Task completed
    SHUTDOWN = "shutdown"            # Container shutting down
    MANUAL = "manual"                # User requested
```

### [SPEC-03.11] Conversation State Capture

The conversation state MUST include all information needed to resume:

```python
@dataclass
class ConversationState:
    # Message history
    messages: list[Message]          # All messages in conversation
    system_prompt: str               # Current system prompt

    # Claude Code state
    session_id: str                  # Claude Code session ID
    resume_token: str | None         # For --resume flag

    # Tool state
    pending_tool_calls: list[ToolCall]  # In-flight tool calls
    tool_results: dict[str, Any]        # Cached tool results

    # Context
    loaded_files: list[str]          # Files in context window
    todos: list[TodoItem]            # TodoWrite state
```

**Capture mechanism**: Parse Claude Code's `--output-format stream-json` output to reconstruct state.

### [SPEC-03.12] Checkpoint Triggers

Checkpoints MUST be created on these events:

| Trigger | Interval/Condition | Rationale |
|---------|-------------------|-----------|
| `PERIODIC` | Every 5 minutes (configurable) | Regular safety net |
| `DETACH` | Immediately on tmux detach | Preserve interactive state |
| `ERROR` | On any error | Capture state for debugging |
| `COMPLETE` | On task completion | Final state for audit |
| `SHUTDOWN` | On SIGTERM | Last chance before container dies |
| `MANUAL` | User request via CLI | On-demand backup |

```python
CHECKPOINT_INTERVAL_SECONDS = 300  # 5 minutes, configurable

class CheckpointManager:
    async def start_periodic_checkpointing(self, session: Session):
        """Background task for periodic checkpoints."""
        while session.state in (SessionState.RUNNING, SessionState.ATTACHED):
            await asyncio.sleep(CHECKPOINT_INTERVAL_SECONDS)
            await self.create_checkpoint(session, trigger=CheckpointTrigger.PERIODIC)
```

### [SPEC-03.13] Workspace Snapshot

The working directory MUST be captured efficiently:

```python
async def snapshot_workspace(self, session: Session) -> str:
    """Create compressed snapshot of workspace."""
    workspace_dir = session.working_directory
    snapshot_path = f"/vol/parhelia/checkpoints/{session.id}/workspace.tar.zst"

    # Use zstd for fast compression
    await run_command([
        "tar",
        "--zstd",
        "-cf", snapshot_path,
        "-C", workspace_dir,
        ".",
        # Exclude large/generated directories
        "--exclude", "node_modules",
        "--exclude", ".git/objects",
        "--exclude", "__pycache__",
        "--exclude", "target",
        "--exclude", ".venv",
    ])

    return snapshot_path
```

**Size handling**:
- < 100MB: Store directly on Volume
- 100MB - 1GB: Warn user, store on Volume
- > 1GB: Stream to object storage (S3/R2)

```python
VOLUME_SIZE_LIMIT_MB = 1024  # 1GB

async def snapshot_workspace(self, session: Session) -> str:
    """Create workspace snapshot, using object storage for large workspaces."""
    workspace_dir = session.working_directory
    temp_path = f"/tmp/workspace-{session.id}.tar.zst"

    # Create compressed archive
    await run_command([
        "tar", "--zstd", "-cf", temp_path,
        "-C", workspace_dir, ".",
        "--exclude", "node_modules",
        "--exclude", ".git/objects",
        # ... other excludes
    ])

    size_mb = os.path.getsize(temp_path) / (1024 * 1024)

    if size_mb > VOLUME_SIZE_LIMIT_MB:
        # Stream to object storage
        return await self.upload_to_object_storage(temp_path, session.id)
    else:
        # Store on Volume
        volume_path = f"/vol/parhelia/checkpoints/{session.id}/workspace.tar.zst"
        shutil.move(temp_path, volume_path)
        return volume_path

async def upload_to_object_storage(self, local_path: str, session_id: str) -> str:
    """Upload large snapshot to S3/R2."""
    import boto3  # or cloudflare R2 client

    key = f"checkpoints/{session_id}/workspace.tar.zst"
    s3.upload_file(local_path, CHECKPOINT_BUCKET, key)

    return f"s3://{CHECKPOINT_BUCKET}/{key}"
```

### [SPEC-03.14] Heartbeat Monitoring

Sessions MUST emit heartbeats for failure detection:

```python
HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_TIMEOUT_SECONDS = 90  # 3 missed heartbeats

class HeartbeatEmitter:
    async def run(self, session: Session):
        """Emit heartbeats to orchestrator."""
        while session.state != SessionState.TERMINATED:
            await orchestrator.heartbeat(
                session_id=session.id,
                container_id=get_container_id(),
                state=session.state,
                last_activity=session.last_activity,
                metrics=session.get_metrics(),
            )
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

class HeartbeatMonitor:
    async def monitor(self, session_id: str):
        """Detect session failure via missed heartbeats."""
        last_heartbeat = datetime.now()

        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

            if session_id not in self.heartbeats:
                continue

            time_since_heartbeat = (
                datetime.now() - self.heartbeats[session_id]
            ).total_seconds()

            if time_since_heartbeat > HEARTBEAT_TIMEOUT_SECONDS:
                await self.handle_session_failure(session_id)
                break

    async def handle_session_failure(self, session_id: str):
        """Handle detected session failure."""
        logger.warning(f"Session {session_id} failed (heartbeat timeout)")

        # Get latest checkpoint
        checkpoint = await self.get_latest_checkpoint(session_id)

        if checkpoint:
            # Schedule resume
            await resume_scheduler.schedule_resume(session_id, checkpoint)
        else:
            # No checkpoint - notify user of total loss
            await notifications.send(
                title=f"Session {session_id} lost",
                body="Session failed with no checkpoint available",
                severity="error",
            )
```

### [SPEC-03.15] Resume Workflow

Sessions MUST be resumable from checkpoint:

```python
class ResumeScheduler:
    async def schedule_resume(
        self,
        session_id: str,
        checkpoint: Checkpoint,
        delay_seconds: int = 5,
    ):
        """Schedule session resume from checkpoint."""
        await asyncio.sleep(delay_seconds)  # Brief delay for cleanup

        # Spawn new container
        new_session = await container_manager.spawn_session(
            task_id=checkpoint.session_id,
            resume_from=checkpoint,
        )

        logger.info(f"Resumed {session_id} as {new_session.id}")

    async def restore_from_checkpoint(
        self,
        checkpoint: Checkpoint,
    ) -> Session:
        """Restore session state from checkpoint."""

        # 1. Restore workspace
        workspace_dir = f"/vol/parhelia/workspaces/{checkpoint.session_id}"
        await run_command([
            "tar", "--zstd", "-xf", checkpoint.workspace_snapshot,
            "-C", workspace_dir,
        ])

        # 2. Restore environment
        env = checkpoint.environment.copy()
        env["PARHELIA_RESUMED_FROM"] = checkpoint.id

        # 3. Create new tmux session
        session = await session_manager.create_session(
            task_id=checkpoint.session_id,
            working_dir=workspace_dir,
            env=env,
        )

        # 4. Restore tmux layout
        await restore_tmux_layout(session, checkpoint.tmux_layout)

        # 5. Resume Claude Code with conversation
        await self.resume_claude_code(session, checkpoint)

        return session

    async def resume_claude_code(
        self,
        session: Session,
        checkpoint: Checkpoint,
    ):
        """Resume Claude Code from checkpoint."""

        # Claude Code's --resume continues the most recent conversation
        # in the current working directory. Since we restored the workspace
        # to the same path, this should find the previous session.
        cmd = [
            "claude",
            "--resume",  # Continues most recent conversation in this directory
            "--output-format", "stream-json",
        ]

        # If --resume fails (no session found), fall back to prompt reconstruction
        try:
            await session_manager.send_command(session, cmd)
        except SessionNotFoundError:
            # Reconstruct conversation via prompt
            resume_prompt = self.build_resume_prompt(checkpoint)
            cmd = [
                "claude",
                "-p", resume_prompt,
                "--output-format", "stream-json",
            ]
            await session_manager.send_command(session, cmd)

    def build_resume_prompt(self, checkpoint: Checkpoint) -> str:
        """Build prompt to resume conversation."""
        return f"""You are resuming a previous session that was interrupted.

Previous conversation summary:
{self.summarize_conversation(checkpoint.conversation)}

Last task in progress:
{checkpoint.conversation.messages[-1].content if checkpoint.conversation.messages else 'None'}

Uncommitted changes in workspace:
{', '.join(checkpoint.uncommitted_changes) or 'None'}

Please continue from where you left off."""
```

### [SPEC-03.16] Checkpoint Retention

Checkpoints MUST be retained according to policy:

```python
@dataclass
class RetentionPolicy:
    max_checkpoints_per_session: int = 10      # Keep last N
    max_age_hours: int = 168                   # 7 days
    keep_final_checkpoint: bool = True         # Always keep completion checkpoint
    keep_error_checkpoints: bool = True        # Keep for debugging

class CheckpointCleaner:
    async def cleanup_old_checkpoints(self, session_id: str):
        """Remove checkpoints according to retention policy."""
        checkpoints = await self.list_checkpoints(session_id)
        policy = self.get_retention_policy()

        to_delete = []
        kept = 0

        for cp in sorted(checkpoints, key=lambda c: c.created_at, reverse=True):
            # Always keep final and error checkpoints
            if policy.keep_final_checkpoint and cp.trigger == CheckpointTrigger.COMPLETE:
                continue
            if policy.keep_error_checkpoints and cp.trigger == CheckpointTrigger.ERROR:
                continue

            # Check age
            age_hours = (datetime.now() - cp.created_at).total_seconds() / 3600
            if age_hours > policy.max_age_hours:
                to_delete.append(cp)
                continue

            # Check count
            kept += 1
            if kept > policy.max_checkpoints_per_session:
                to_delete.append(cp)

        for cp in to_delete:
            await self.delete_checkpoint(cp)
```

### [SPEC-03.17] MCP Server State Checkpointing

All MCP servers MUST have their state captured:

```python
class MCPStateManager:
    async def capture_all_mcp_state(self, session: Session) -> dict[str, Any]:
        """Capture state from all configured MCP servers."""
        mcp_states = {}

        for server_name, server_config in session.mcp_servers.items():
            try:
                state = await self.capture_server_state(server_name, server_config)
                mcp_states[server_name] = state
            except Exception as e:
                logger.warning(f"Failed to capture MCP state for {server_name}: {e}")
                mcp_states[server_name] = {"error": str(e)}

        return mcp_states

    async def capture_server_state(
        self,
        server_name: str,
        config: MCPServerConfig,
    ) -> dict[str, Any]:
        """Capture state from a single MCP server."""

        # Standard MCP state capture protocol
        # Servers that support checkpointing expose a checkpoint_state resource
        if config.supports_checkpoint:
            state = await self.mcp_client.read_resource(
                server_name,
                "checkpoint://state"
            )
            return state

        # Fallback: capture any files the server uses
        if config.state_files:
            return {
                "files": await self.capture_files(config.state_files)
            }

        return {"type": "stateless"}

    async def restore_mcp_state(
        self,
        session: Session,
        mcp_states: dict[str, Any],
    ):
        """Restore MCP server states after resume."""

        for server_name, state in mcp_states.items():
            if state.get("error"):
                logger.warning(f"Skipping restore for {server_name}: had capture error")
                continue

            server_config = session.mcp_servers.get(server_name)
            if not server_config:
                continue

            if server_config.supports_checkpoint:
                await self.mcp_client.call_tool(
                    server_name,
                    "restore_checkpoint",
                    {"state": state}
                )
            elif state.get("files"):
                await self.restore_files(state["files"])
```

**MCP checkpoint protocol**: Servers that support checkpointing expose:
- Resource: `checkpoint://state` - Returns serializable state
- Tool: `restore_checkpoint` - Accepts state and restores

**Beads-specific**: Beads MCP server syncs to JSONL in git, so checkpoint captures the JSONL file state.

### [SPEC-03.18] Checkpoint Verification

Checkpoints MUST be verified for integrity:

```python
async def verify_checkpoint(self, checkpoint: Checkpoint) -> bool:
    """Verify checkpoint is complete and restorable."""
    required_files = [
        "manifest.json",
        "conversation.json",
        "session_state.json",
        "workspace.tar.zst",
    ]

    checkpoint_dir = f"/vol/parhelia/checkpoints/{checkpoint.session_id}/{checkpoint.id}"

    for file in required_files:
        path = f"{checkpoint_dir}/{file}"
        if not os.path.exists(path):
            logger.error(f"Checkpoint {checkpoint.id} missing {file}")
            return False

    # Verify workspace archive
    try:
        await run_command(["tar", "--zstd", "-tf", f"{checkpoint_dir}/workspace.tar.zst"])
    except Exception as e:
        logger.error(f"Checkpoint {checkpoint.id} workspace corrupt: {e}")
        return False

    return True
```

---

## Checkpoint Manifest Schema

```json
// manifest.json
{
  "version": "1.0",
  "id": "01HQXYZ123456789ABCDEF",
  "session_id": "ph-fix-auth-20260116T143022",
  "created_at": "2026-01-16T14:35:00Z",
  "trigger": "periodic",

  "conversation": {
    "turn_count": 15,
    "last_message_id": "msg_abc123",
    "tokens_used": 45000,
    "has_resume_token": true
  },

  "workspace": {
    "path": "workspace.tar.zst",
    "size_bytes": 15728640,
    "file_count": 234,
    "uncommitted_files": ["src/auth.py", "tests/test_auth.py"]
  },

  "git": {
    "branch": "feature/fix-auth",
    "head": "a1b2c3d4e5f6",
    "has_stash": true,
    "dirty": true
  },

  "metrics": {
    "cost_estimate_usd": 0.45,
    "tools_invoked": ["Read", "Edit", "Bash", "Grep"],
    "files_modified": 3
  },

  "checksum": "sha256:abc123..."
}
```

---

## Failure Scenarios

### Scenario 1: Container Crash

```
Container                      Orchestrator                    New Container
    │                               │                               │
    │── heartbeat ─────────────────▶│                               │
    │                               │                               │
    ✕ [crash]                       │                               │
                                    │                               │
                                    │◀── [timeout: 90s] ───────────│
                                    │                               │
                                    │── detect failure              │
                                    │── get latest checkpoint       │
                                    │                               │
                                    │── spawn new container ───────▶│
                                    │                               │── restore checkpoint
                                    │                               │── resume session
                                    │◀── heartbeat ─────────────────│
```

### Scenario 2: Network Partition

```
Container                      Orchestrator                    Container
    │                               │                               │
    │── heartbeat ──────────X       │                               │
    │                               │                               │
    │   [network partition]         │                               │
    │                               │                               │
    │── checkpoint (local) ─────────│                               │
    │                               │                               │
    │                               │◀── [timeout: 90s] ───────────│
    │                               │                               │
    │                               │── detect failure              │
    │                               │── schedule resume             │
    │                               │                               │
    │   [network restored]          │                               │
    │                               │                               │
    │── heartbeat ─────────────────▶│                               │
    │                               │── cancel resume (still alive)│
```

### Scenario 3: Modal Timeout (24h)

```
Container                      Orchestrator
    │                               │
    │── [approaching 24h timeout]   │
    │                               │
    │── checkpoint(SHUTDOWN) ───────│
    │── notify orchestrator ───────▶│
    │                               │── schedule resume
    │                               │
    ✕ [Modal terminates]            │
                                    │── spawn new container
                                    │── restore & resume
```

---

## CLI Commands

### `parhelia checkpoint list`

```
$ parhelia checkpoint list ph-fix-auth-20260116T143022

ID                          TRIGGER    CREATED         SIZE    VERIFIED
01HQXYZ123456789ABCDEF     periodic   2 minutes ago   15MB    ✓
01HQXYZ123456789ABCDEE     periodic   7 minutes ago   14MB    ✓
01HQXYZ123456789ABCDED     detach     15 minutes ago  14MB    ✓
```

### `parhelia checkpoint restore`

```
$ parhelia checkpoint restore 01HQXYZ123456789ABCDEF

Restoring checkpoint 01HQXYZ123456789ABCDEF...
  ├── Extracting workspace (15MB)... done
  ├── Restoring environment... done
  ├── Recreating tmux session... done
  └── Resuming Claude Code... done

Session restored as ph-fix-auth-20260116T143522
```

### `parhelia checkpoint create`

```
$ parhelia checkpoint create ph-fix-auth-20260116T143022

Creating manual checkpoint...
  ├── Capturing conversation state... done
  ├── Snapshotting workspace... done (15MB)
  ├── Saving tmux layout... done
  └── Verifying checkpoint... done

Checkpoint 01HQXYZ123456789ABCDFG created.
```

---

## Acceptance Criteria

- [ ] [SPEC-03.AC1] Periodic checkpoints created every 5 minutes
- [ ] [SPEC-03.AC2] Checkpoint on detach captures full state
- [ ] [SPEC-03.AC3] Container crash triggers automatic resume
- [ ] [SPEC-03.AC4] Resume restores conversation context
- [ ] [SPEC-03.AC5] Resume restores uncommitted code changes
- [ ] [SPEC-03.AC6] Checkpoint retention policy enforced
- [ ] [SPEC-03.AC7] Checkpoint verification catches corruption
- [ ] [SPEC-03.AC8] CLI commands work as documented

---

## Resolved Questions

1. ~~**Large workspace handling**~~: **Resolved** - Stream to S3/R2 for workspaces > 1GB.
2. ~~**MCP server state**~~: **Resolved** - Checkpoint all MCP servers via standard protocol (checkpoint://state resource).

## Resolved Questions

3. ~~**Resume mechanism**~~: **Resolved** - Claude Code supports `claude --resume` which continues the most recent conversation in the current directory, and `claude --continue` which continues the most recent conversation regardless of directory. We should use `--resume` when workspace is restored to same path.

## Open Questions

1. **Session ID persistence**: Claude Code uses session IDs internally - can we capture and reuse these across container restarts?
2. **Object storage choice**: S3 vs R2 vs Modal's native storage? (R2 has no egress fees)

---

## References

- [Modal Volume Performance](https://modal.com/docs/guide/volumes)
- [Claude Code Session Resume](https://docs.anthropic.com/en/docs/claude-code/headless)
- [zstd Compression](https://facebook.github.io/zstd/)
- SPEC-02: Session Management
