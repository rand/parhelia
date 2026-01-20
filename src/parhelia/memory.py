"""Session memory and summarization.

Implements:
- [SPEC-07.30.01] Memory Hierarchy
- [SPEC-07.30.02] Session Summary Generation
- [SPEC-07.30.03] Summary Storage
- [SPEC-07.30.04] Context Window Management
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import aiofiles
import aiofiles.os


@dataclass
class FileChange:
    """Record of a file change in a session."""

    path: str
    change_type: Literal["added", "modified", "deleted"]
    additions: int = 0
    deletions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "change_type": self.change_type,
            "additions": self.additions,
            "deletions": self.deletions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileChange":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            change_type=data["change_type"],
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
        )


@dataclass
class Decision:
    """A decision made during a session."""

    description: str
    rationale: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        """Create from dictionary."""
        return cls(
            description=data["description"],
            rationale=data.get("rationale"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
        )


@dataclass
class FailedApproach:
    """A failed approach attempted during a session."""

    description: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedApproach":
        """Create from dictionary."""
        return cls(
            description=data["description"],
            reason=data["reason"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
        )


@dataclass
class SessionSummary:
    """Summary of a session's progress.

    Implements [SPEC-07.30.02].
    """

    session_id: str
    session_name: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    # Progress tracking
    progress_summary: str = ""
    files_changed: list[FileChange] = field(default_factory=list)

    # Decisions and approaches
    decisions: list[Decision] = field(default_factory=list)
    failed_approaches: list[FailedApproach] = field(default_factory=list)

    # Blockers and context
    blockers: list[str] = field(default_factory=list)
    resume_context: str = ""

    # Metrics
    conversation_turns: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "created_at": self.created_at.isoformat(),
            "progress_summary": self.progress_summary,
            "files_changed": [f.to_dict() for f in self.files_changed],
            "decisions": [d.to_dict() for d in self.decisions],
            "failed_approaches": [f.to_dict() for f in self.failed_approaches],
            "blockers": self.blockers,
            "resume_context": self.resume_context,
            "conversation_turns": self.conversation_turns,
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSummary":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            session_name=data.get("session_name"),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            progress_summary=data.get("progress_summary", ""),
            files_changed=[
                FileChange.from_dict(f) for f in data.get("files_changed", [])
            ],
            decisions=[Decision.from_dict(d) for d in data.get("decisions", [])],
            failed_approaches=[
                FailedApproach.from_dict(f) for f in data.get("failed_approaches", [])
            ],
            blockers=data.get("blockers", []),
            resume_context=data.get("resume_context", ""),
            conversation_turns=data.get("conversation_turns", 0),
            tokens_used=data.get("tokens_used", 0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
        )

    def to_markdown(self) -> str:
        """Generate markdown representation of the summary.

        Implements [SPEC-07.30.02].
        """
        lines = [
            f"## Session Summary: {self.session_name or self.session_id}",
            "",
        ]

        # Progress
        lines.append("### Progress")
        if self.progress_summary:
            lines.append(self.progress_summary)
        else:
            lines.append("- No progress recorded")
        lines.append("")

        # Files changed
        if self.files_changed:
            lines.append("### Files Changed")
            for fc in self.files_changed:
                change_str = f"({fc.change_type})"
                if fc.additions or fc.deletions:
                    change_str = f"+{fc.additions}, -{fc.deletions}"
                lines.append(f"- {fc.path} {change_str}")
            lines.append("")

        # Decisions
        if self.decisions:
            lines.append("### Decisions")
            for d in self.decisions:
                lines.append(f"- {d.description}")
                if d.rationale:
                    lines.append(f"  - Rationale: {d.rationale}")
            lines.append("")

        # Failed approaches
        if self.failed_approaches:
            lines.append("### Failed Approaches")
            for fa in self.failed_approaches:
                lines.append(f"- {fa.description}")
                lines.append(f"  - Reason: {fa.reason}")
            lines.append("")

        # Blockers
        if self.blockers:
            lines.append("### Blockers")
            for b in self.blockers:
                lines.append(f"- {b}")
            lines.append("")

        # Context for resume
        if self.resume_context:
            lines.append("### Context for Resume")
            lines.append(self.resume_context)
            lines.append("")

        # Metrics
        lines.append("### Metrics")
        lines.append(f"- Conversation turns: {self.conversation_turns}")
        lines.append(f"- Tokens used: {self.tokens_used:,}")
        lines.append(f"- Estimated cost: ${self.estimated_cost_usd:.2f}")

        return "\n".join(lines)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    tool_calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "tool_calls": self.tool_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            token_count=data.get("token_count", 0),
            tool_calls=data.get("tool_calls", []),
        )


@dataclass
class ContextBudget:
    """Context window budget allocation.

    Implements [SPEC-07.32.01].
    """

    total_tokens: int = 200000  # Default Claude context
    system_prompt_pct: float = 0.10
    project_memory_pct: float = 0.10
    session_summary_pct: float = 0.15
    recent_conversation_pct: float = 0.50
    tool_results_buffer_pct: float = 0.15

    @property
    def system_prompt_budget(self) -> int:
        """Token budget for system prompt."""
        return int(self.total_tokens * self.system_prompt_pct)

    @property
    def project_memory_budget(self) -> int:
        """Token budget for project memory."""
        return int(self.total_tokens * self.project_memory_pct)

    @property
    def session_summary_budget(self) -> int:
        """Token budget for session summary."""
        return int(self.total_tokens * self.session_summary_pct)

    @property
    def recent_conversation_budget(self) -> int:
        """Token budget for recent conversation."""
        return int(self.total_tokens * self.recent_conversation_pct)

    @property
    def tool_results_buffer_budget(self) -> int:
        """Token budget for tool results."""
        return int(self.total_tokens * self.tool_results_buffer_pct)


@dataclass
class CompressionResult:
    """Result of context compression."""

    original_tokens: int
    compressed_tokens: int
    turns_removed: int
    summary_generated: bool

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original tokens."""
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


class MemoryManager:
    """Manage session memory and context.

    Implements [SPEC-07.30].
    """

    # Compression threshold (80% of context limit)
    COMPRESSION_THRESHOLD = 0.80

    def __init__(
        self,
        session_id: str,
        sessions_root: str = "/vol/parhelia/sessions",
        context_budget: ContextBudget | None = None,
    ):
        """Initialize the memory manager.

        Args:
            session_id: The session ID.
            sessions_root: Root directory for session storage.
            context_budget: Optional context budget configuration.
        """
        self.session_id = session_id
        self.sessions_root = Path(sessions_root)
        self.context_budget = context_budget or ContextBudget()

        # In-memory state
        self._summary: SessionSummary | None = None
        self._conversation: list[ConversationTurn] = []
        self._current_tokens = 0

    @property
    def summary(self) -> SessionSummary | None:
        """Get current session summary."""
        return self._summary

    @property
    def conversation(self) -> list[ConversationTurn]:
        """Get conversation history."""
        return self._conversation

    @property
    def current_tokens(self) -> int:
        """Get current token count."""
        return self._current_tokens

    @property
    def context_usage_pct(self) -> float:
        """Get context usage as percentage of budget."""
        return self._current_tokens / self.context_budget.total_tokens

    def should_compress(self) -> bool:
        """Check if context compression is needed.

        Implements [SPEC-07.30.04].
        """
        return self.context_usage_pct >= self.COMPRESSION_THRESHOLD

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn.

        Args:
            turn: The conversation turn to add.
        """
        self._conversation.append(turn)
        self._current_tokens += turn.token_count

    def add_decision(self, description: str, rationale: str | None = None) -> None:
        """Record a decision made during the session.

        Args:
            description: What was decided.
            rationale: Why this decision was made.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.decisions.append(
            Decision(description=description, rationale=rationale)
        )

    def add_failed_approach(self, description: str, reason: str) -> None:
        """Record a failed approach.

        Args:
            description: What was tried.
            reason: Why it failed.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.failed_approaches.append(
            FailedApproach(description=description, reason=reason)
        )

    def add_file_change(
        self,
        path: str,
        change_type: Literal["added", "modified", "deleted"],
        additions: int = 0,
        deletions: int = 0,
    ) -> None:
        """Record a file change.

        Args:
            path: File path.
            change_type: Type of change.
            additions: Lines added.
            deletions: Lines deleted.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.files_changed.append(
            FileChange(
                path=path,
                change_type=change_type,
                additions=additions,
                deletions=deletions,
            )
        )

    def add_blocker(self, blocker: str) -> None:
        """Record a blocker.

        Args:
            blocker: Description of the blocker.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.blockers.append(blocker)

    def set_progress(self, progress: str) -> None:
        """Set the progress summary.

        Args:
            progress: Progress summary text.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.progress_summary = progress

    def set_resume_context(self, context: str) -> None:
        """Set context for session resumption.

        Args:
            context: Context text for resuming.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.resume_context = context

    def update_metrics(
        self,
        conversation_turns: int | None = None,
        tokens_used: int | None = None,
        estimated_cost_usd: float | None = None,
    ) -> None:
        """Update session metrics.

        Args:
            conversation_turns: Total conversation turns.
            tokens_used: Total tokens used.
            estimated_cost_usd: Estimated cost in USD.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        if conversation_turns is not None:
            self._summary.conversation_turns = conversation_turns
        if tokens_used is not None:
            self._summary.tokens_used = tokens_used
        if estimated_cost_usd is not None:
            self._summary.estimated_cost_usd = estimated_cost_usd

    def generate_summary(
        self,
        session_name: str | None = None,
    ) -> SessionSummary:
        """Generate a session summary.

        Implements [SPEC-07.30.02].

        Args:
            session_name: Optional session name.

        Returns:
            Generated SessionSummary.
        """
        if self._summary is None:
            self._summary = SessionSummary(session_id=self.session_id)

        self._summary.session_name = session_name
        self._summary.created_at = datetime.now()

        # Update metrics from conversation
        if self._conversation:
            self._summary.conversation_turns = len(self._conversation)
            self._summary.tokens_used = sum(t.token_count for t in self._conversation)

        return self._summary

    def compress_context(self, target_pct: float = 0.60) -> CompressionResult:
        """Compress conversation context to fit within budget.

        Implements [SPEC-07.30.04].

        Args:
            target_pct: Target context usage percentage after compression.

        Returns:
            CompressionResult with compression details.
        """
        original_tokens = self._current_tokens
        target_tokens = int(self.context_budget.total_tokens * target_pct)

        if self._current_tokens <= target_tokens:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=self._current_tokens,
                turns_removed=0,
                summary_generated=False,
            )

        # Strategy: Remove oldest turns until we're under budget
        # but preserve at least the last 5 turns
        turns_removed = 0
        min_preserved_turns = 5

        while (
            self._current_tokens > target_tokens
            and len(self._conversation) > min_preserved_turns
        ):
            removed = self._conversation.pop(0)
            self._current_tokens -= removed.token_count
            turns_removed += 1

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=self._current_tokens,
            turns_removed=turns_removed,
            summary_generated=turns_removed > 0,
        )

    async def save_summary(self, summary: SessionSummary | None = None) -> Path:
        """Save summary to markdown file.

        Implements [SPEC-07.30.03].

        Args:
            summary: Summary to save. Uses current summary if None.

        Returns:
            Path to saved summary file.
        """
        summary = summary or self._summary
        if summary is None:
            summary = self.generate_summary()

        session_dir = self.sessions_root / self.session_id
        await aiofiles.os.makedirs(str(session_dir), exist_ok=True)

        summary_path = session_dir / "summary.md"
        async with aiofiles.open(str(summary_path), "w") as f:
            await f.write(summary.to_markdown())

        return summary_path

    async def load_summary(self) -> SessionSummary | None:
        """Load summary from storage.

        Returns:
            Loaded SessionSummary or None if not found.
        """
        session_dir = self.sessions_root / self.session_id

        # Try JSON first
        json_path = session_dir / "summary.json"
        if json_path.exists():
            async with aiofiles.open(str(json_path)) as f:
                data = json.loads(await f.read())
                self._summary = SessionSummary.from_dict(data)
                return self._summary

        return None

    async def save_summary_json(self, summary: SessionSummary | None = None) -> Path:
        """Save summary to JSON file.

        Args:
            summary: Summary to save. Uses current summary if None.

        Returns:
            Path to saved summary file.
        """
        summary = summary or self._summary
        if summary is None:
            summary = self.generate_summary()

        session_dir = self.sessions_root / self.session_id
        await aiofiles.os.makedirs(str(session_dir), exist_ok=True)

        json_path = session_dir / "summary.json"
        async with aiofiles.open(str(json_path), "w") as f:
            await f.write(json.dumps(summary.to_dict(), indent=2))

        return json_path

    def get_context_for_model(self) -> dict[str, Any]:
        """Get context formatted for model input.

        Returns:
            Dict with session_summary and recent_turns.
        """
        return {
            "session_summary": self._summary.to_dict() if self._summary else None,
            "recent_turns": [t.to_dict() for t in self._conversation[-10:]],
            "context_usage_pct": self.context_usage_pct,
        }


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic of ~4 characters per token.

    Args:
        text: Text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


def summarize_tool_output(output: str, max_length: int = 500) -> str:
    """Summarize verbose tool output.

    Implements [SPEC-07.32.02].

    Args:
        output: Original tool output.
        max_length: Maximum length of summary.

    Returns:
        Summarized output.
    """
    if len(output) <= max_length:
        return output

    # For file listings, keep first and last items
    lines = output.strip().split("\n")
    if len(lines) > 10:
        kept = lines[:5] + ["...", f"({len(lines) - 10} lines omitted)", "..."] + lines[-5:]
        summary = "\n".join(kept)
        if len(summary) <= max_length:
            return summary

    # Default truncation
    return output[: max_length - 20] + "\n... (truncated)"
