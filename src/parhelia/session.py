"""Session state management.

Implements:
- [SPEC-03.10] Checkpoint State Schema
- [SPEC-03.11] Conversation State Capture
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parhelia.cas import Digest
    from parhelia.environment import EnvironmentSnapshot


class SessionState(Enum):
    """Session lifecycle states.

    Implements [SPEC-03.10].
    """

    STARTING = "starting"  # Container spawning, initialization
    RUNNING = "running"  # Actively executing
    SUSPENDED = "suspended"  # Paused/detached but preservable
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"  # Task failed or container died


class CheckpointTrigger(Enum):
    """What caused a checkpoint to be created.

    Implements [SPEC-03.10].
    """

    PERIODIC = "periodic"  # Scheduled interval
    DETACH = "detach"  # User detached
    ERROR = "error"  # Error occurred
    COMPLETE = "complete"  # Task completed
    SHUTDOWN = "shutdown"  # Container shutting down
    MANUAL = "manual"  # User requested


@dataclass
class Message:
    """A message in the conversation.

    Implements [SPEC-03.11].
    """

    role: str  # "user", "assistant", or "system"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ConversationState:
    """Full conversation state for checkpointing.

    Implements [SPEC-03.11].
    """

    messages: list[Message]
    system_prompt: str
    session_id: str  # Claude Code session ID
    resume_token: str | None = None
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    loaded_files: list[str] = field(default_factory=list)
    todos: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GitState:
    """Git repository state.

    Implements [SPEC-03.10].
    """

    branch: str
    head_commit: str
    is_dirty: bool
    has_stash: bool
    uncommitted_files: list[str] = field(default_factory=list)


@dataclass
class Session:
    """A Claude Code execution session.

    Implements [SPEC-03.10].
    """

    id: str  # Unique session identifier
    task_id: str  # Parent task from orchestrator
    state: SessionState
    working_directory: str

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # Environment
    environment: dict[str, str] = field(default_factory=dict)

    # Container info
    container_id: str | None = None
    gpu: str | None = None

    # MCP servers
    mcp_servers: dict[str, Any] = field(default_factory=dict)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def transition_to(self, new_state: SessionState) -> None:
        """Transition to a new state.

        Args:
            new_state: The state to transition to.
        """
        self.state = new_state
        self.update_activity()


@dataclass
class Checkpoint:
    """A checkpoint of session state.

    Implements [SPEC-03.10], [SPEC-08.14].

    Supports two storage modes:
    - Legacy: workspace_snapshot (path to tar.gz archive)
    - CAS: workspace_root (Merkle tree digest in Content-Addressable Storage)
    """

    id: str  # Unique checkpoint ID (ULID recommended)
    session_id: str  # Parent session
    trigger: CheckpointTrigger
    working_directory: str

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    # Conversation state
    conversation: ConversationState | None = None
    turn_count: int = 0
    last_message_id: str | None = None

    # Environment
    environment: dict[str, str] = field(default_factory=dict)
    tmux_layout: str | None = None

    # Code state - Legacy (tar.gz archive)
    workspace_snapshot: str | None = None  # Path to workspace.tar.zst

    # Code state - CAS (Merkle tree) [SPEC-08.14]
    workspace_root: Digest | None = None  # Merkle tree root digest

    uncommitted_changes: list[str] = field(default_factory=list)
    git_state: GitState | None = None

    # MCP state
    mcp_server_states: dict[str, Any] = field(default_factory=dict)

    # Metrics
    tokens_used: int = 0
    cost_estimate: float = 0.0
    tools_invoked: list[str] = field(default_factory=list)

    # Environment versioning [SPEC-07.10]
    environment_snapshot: EnvironmentSnapshot | None = None

    # Verification
    verified: bool = False
    checksum: str | None = None
