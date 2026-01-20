"""Session state management.

Implements:
- [SPEC-03.10] Checkpoint State Schema
- [SPEC-03.11] Conversation State Capture
- [SPEC-07.11] Checkpoint Metadata Schema v1.2
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


class ApprovalStatus(Enum):
    """Checkpoint approval state.

    Implements [SPEC-07.11.03].
    """

    PENDING = "pending"  # Awaiting human review
    APPROVED = "approved"  # Human approved
    REJECTED = "rejected"  # Human rejected
    AUTO_APPROVED = "auto_approved"  # Policy auto-approved


@dataclass
class CheckpointApproval:
    """Approval decision for a checkpoint.

    Implements [SPEC-07.11.03].
    """

    status: ApprovalStatus
    user: str | None = None  # Username who approved/rejected
    timestamp: datetime | None = None  # When decision was made
    reason: str | None = None  # Free-text explanation
    policy: str | None = None  # Which escalation policy was applied

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "user": self.user,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "reason": self.reason,
            "policy": self.policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointApproval":
        """Create from dictionary."""
        return cls(
            status=ApprovalStatus(data["status"]),
            user=data.get("user"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            reason=data.get("reason"),
            policy=data.get("policy"),
        )


@dataclass
class CheckpointAnnotation:
    """A timestamped annotation on a checkpoint.

    Implements [SPEC-07.11.05].
    """

    timestamp: datetime
    user: str
    text: str

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointAnnotation":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user=data["user"],
            text=data["text"],
        )


@dataclass
class LinkedIssue:
    """Reference to an external issue tracker.

    Implements [SPEC-07.11.05].
    """

    tracker: str  # "github", "linear", "beads"
    id: str  # Issue ID
    url: str | None = None  # Full URL if available

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "tracker": self.tracker,
            "id": self.id,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LinkedIssue":
        """Create from dictionary."""
        return cls(
            tracker=data["tracker"],
            id=data["id"],
            url=data.get("url"),
        )


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

    # Provenance [SPEC-07.11.02]
    parent_checkpoint_id: str | None = None  # Previous checkpoint in chain
    checkpoint_chain_depth: int = 1  # Number of checkpoints in session

    # Approval [SPEC-07.11.03]
    approval: CheckpointApproval | None = None

    # Annotations [SPEC-07.11.05]
    tags: list[str] = field(default_factory=list)  # Hierarchical tags
    annotations: list[CheckpointAnnotation] = field(default_factory=list)
    linked_issues: list[LinkedIssue] = field(default_factory=list)

    # Verification
    verified: bool = False
    checksum: str | None = None
