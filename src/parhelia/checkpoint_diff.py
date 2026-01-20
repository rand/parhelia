"""Checkpoint comparison and diff.

Implements:
- [SPEC-07.41.01] Diff Command
- [SPEC-07.41.02] File Diff
- [SPEC-07.41.03] Conversation Diff
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from parhelia.checkpoint import Checkpoint


@dataclass
class FileChange:
    """Change to a file between checkpoints."""

    path: str
    change_type: str  # 'added', 'modified', 'deleted'
    additions: int = 0
    deletions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "change_type": self.change_type,
            "additions": self.additions,
            "deletions": self.deletions,
        }


@dataclass
class ConversationStats:
    """Statistics about conversation at a checkpoint."""

    turns: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "turns": self.turns,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class ConversationDiff:
    """Difference in conversation between checkpoints."""

    turns_added: int = 0
    tokens_added: int = 0
    cost_added: float = 0.0
    from_stats: ConversationStats = field(default_factory=ConversationStats)
    to_stats: ConversationStats = field(default_factory=ConversationStats)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "turns_added": self.turns_added,
            "tokens_added": self.tokens_added,
            "cost_added": self.cost_added,
            "from_stats": self.from_stats.to_dict(),
            "to_stats": self.to_stats.to_dict(),
        }


@dataclass
class EnvironmentDiff:
    """Difference in environment between checkpoints."""

    has_changes: bool = False
    claude_code_changed: bool = False
    plugins_changed: list[str] = field(default_factory=list)
    mcp_servers_changed: list[str] = field(default_factory=list)
    packages_changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "has_changes": self.has_changes,
            "claude_code_changed": self.claude_code_changed,
            "plugins_changed": self.plugins_changed,
            "mcp_servers_changed": self.mcp_servers_changed,
            "packages_changed": self.packages_changed,
        }


@dataclass
class ApprovalDiff:
    """Difference in approval status between checkpoints."""

    from_status: str | None = None
    from_user: str | None = None
    to_status: str | None = None
    to_user: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "from_status": self.from_status,
            "from_user": self.from_user,
            "to_status": self.to_status,
            "to_user": self.to_user,
        }


@dataclass
class CheckpointComparison:
    """Full comparison between two checkpoints.

    Implements [SPEC-07.41.01].
    """

    from_checkpoint_id: str
    to_checkpoint_id: str
    from_time: datetime
    to_time: datetime
    time_delta: timedelta
    conversation: ConversationDiff
    files_changed: list[FileChange]
    environment: EnvironmentDiff
    approval: ApprovalDiff

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "from_checkpoint_id": self.from_checkpoint_id,
            "to_checkpoint_id": self.to_checkpoint_id,
            "from_time": self.from_time.isoformat(),
            "to_time": self.to_time.isoformat(),
            "time_delta_seconds": self.time_delta.total_seconds(),
            "conversation": self.conversation.to_dict(),
            "files_changed": [f.to_dict() for f in self.files_changed],
            "environment": self.environment.to_dict(),
            "approval": self.approval.to_dict(),
        }


class CheckpointDiffer:
    """Compare checkpoints and generate diffs.

    Implements [SPEC-07.41].
    """

    def compare(
        self,
        from_cp: Checkpoint,
        to_cp: Checkpoint,
    ) -> CheckpointComparison:
        """Compare two checkpoints.

        Implements [SPEC-07.41.01].

        Args:
            from_cp: Earlier checkpoint.
            to_cp: Later checkpoint.

        Returns:
            CheckpointComparison with all differences.
        """
        # Time comparison
        time_delta = to_cp.created_at - from_cp.created_at

        # Conversation comparison
        conversation = self._compare_conversation(from_cp, to_cp)

        # Files comparison
        files_changed = self._compare_files(from_cp, to_cp)

        # Environment comparison
        environment = self._compare_environment(from_cp, to_cp)

        # Approval comparison
        approval = self._compare_approval(from_cp, to_cp)

        return CheckpointComparison(
            from_checkpoint_id=from_cp.id,
            to_checkpoint_id=to_cp.id,
            from_time=from_cp.created_at,
            to_time=to_cp.created_at,
            time_delta=time_delta,
            conversation=conversation,
            files_changed=files_changed,
            environment=environment,
            approval=approval,
        )

    def _compare_conversation(
        self,
        from_cp: Checkpoint,
        to_cp: Checkpoint,
    ) -> ConversationDiff:
        """Compare conversation stats between checkpoints."""
        from_stats = ConversationStats(
            turns=0,  # Would need conversation data in checkpoint
            tokens=from_cp.tokens_used or 0,
            cost_usd=from_cp.cost_estimate or 0.0,
        )

        to_stats = ConversationStats(
            turns=0,
            tokens=to_cp.tokens_used or 0,
            cost_usd=to_cp.cost_estimate or 0.0,
        )

        return ConversationDiff(
            turns_added=to_stats.turns - from_stats.turns,
            tokens_added=to_stats.tokens - from_stats.tokens,
            cost_added=to_stats.cost_usd - from_stats.cost_usd,
            from_stats=from_stats,
            to_stats=to_stats,
        )

    def _compare_files(
        self,
        from_cp: Checkpoint,
        to_cp: Checkpoint,
    ) -> list[FileChange]:
        """Compare files between checkpoints."""
        changes: list[FileChange] = []

        from_files = set(from_cp.uncommitted_changes or [])
        to_files = set(to_cp.uncommitted_changes or [])

        # Added files (in to but not in from)
        for path in to_files - from_files:
            changes.append(
                FileChange(
                    path=path,
                    change_type="added",
                    additions=0,  # Would need actual file content
                    deletions=0,
                )
            )

        # Deleted files (in from but not in to)
        for path in from_files - to_files:
            changes.append(
                FileChange(
                    path=path,
                    change_type="deleted",
                    additions=0,
                    deletions=0,
                )
            )

        # Modified files (in both)
        for path in from_files & to_files:
            changes.append(
                FileChange(
                    path=path,
                    change_type="modified",
                    additions=0,
                    deletions=0,
                )
            )

        return sorted(changes, key=lambda c: c.path)

    def _compare_environment(
        self,
        from_cp: Checkpoint,
        to_cp: Checkpoint,
    ) -> EnvironmentDiff:
        """Compare environment versions between checkpoints."""
        diff = EnvironmentDiff()

        from_env = from_cp.environment_snapshot
        to_env = to_cp.environment_snapshot

        if not from_env or not to_env:
            return diff

        # Compare Claude Code version
        if from_env.claude_code_version != to_env.claude_code_version:
            diff.has_changes = True
            diff.claude_code_changed = True

        # Compare plugins
        from_plugins = {p.name: p.version for p in from_env.plugins}
        to_plugins = {p.name: p.version for p in to_env.plugins}
        for name in set(from_plugins.keys()) | set(to_plugins.keys()):
            if from_plugins.get(name) != to_plugins.get(name):
                diff.has_changes = True
                diff.plugins_changed.append(name)

        # Compare MCP servers
        from_mcp = {s.name: s.version for s in from_env.mcp_servers}
        to_mcp = {s.name: s.version for s in to_env.mcp_servers}
        for name in set(from_mcp.keys()) | set(to_mcp.keys()):
            if from_mcp.get(name) != to_mcp.get(name):
                diff.has_changes = True
                diff.mcp_servers_changed.append(name)

        # Compare packages
        from_pkgs = {p.name: p.version for p in from_env.packages}
        to_pkgs = {p.name: p.version for p in to_env.packages}
        for name in set(from_pkgs.keys()) | set(to_pkgs.keys()):
            if from_pkgs.get(name) != to_pkgs.get(name):
                diff.has_changes = True
                diff.packages_changed.append(name)

        return diff

    def _compare_approval(
        self,
        from_cp: Checkpoint,
        to_cp: Checkpoint,
    ) -> ApprovalDiff:
        """Compare approval status between checkpoints."""
        diff = ApprovalDiff()

        if from_cp.approval:
            diff.from_status = from_cp.approval.status.value
            diff.from_user = from_cp.approval.user

        if to_cp.approval:
            diff.to_status = to_cp.approval.status.value
            diff.to_user = to_cp.approval.user

        return diff

    # =========================================================================
    # File Diff [SPEC-07.41.02]
    # =========================================================================

    def diff_file(
        self,
        from_content: str,
        to_content: str,
        filename: str = "file",
        context_lines: int = 3,
    ) -> str:
        """Generate unified diff for a file.

        Implements [SPEC-07.41.02].

        Args:
            from_content: Content in earlier checkpoint.
            to_content: Content in later checkpoint.
            filename: File path for header.
            context_lines: Lines of context around changes.

        Returns:
            Unified diff string.
        """
        from_lines = from_content.splitlines(keepends=True)
        to_lines = to_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            from_lines,
            to_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=context_lines,
        )

        return "".join(diff)

    def diff_file_stats(
        self,
        from_content: str,
        to_content: str,
    ) -> tuple[int, int]:
        """Calculate additions and deletions for a file diff.

        Args:
            from_content: Content in earlier checkpoint.
            to_content: Content in later checkpoint.

        Returns:
            Tuple of (additions, deletions).
        """
        from_lines = set(from_content.splitlines())
        to_lines = set(to_content.splitlines())

        additions = len(to_lines - from_lines)
        deletions = len(from_lines - to_lines)

        return additions, deletions

    # =========================================================================
    # Conversation Diff [SPEC-07.41.03]
    # =========================================================================

    def diff_conversation(
        self,
        from_turns: list[dict[str, Any]],
        to_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Get conversation turns added between checkpoints.

        Implements [SPEC-07.41.03].

        Args:
            from_turns: Conversation turns at earlier checkpoint.
            to_turns: Conversation turns at later checkpoint.

        Returns:
            List of added turns.
        """
        # Simple approach: return turns after from_turns length
        if len(to_turns) > len(from_turns):
            return to_turns[len(from_turns) :]
        return []

    # =========================================================================
    # Formatting [SPEC-07.41.01]
    # =========================================================================

    def format_comparison(self, comparison: CheckpointComparison) -> str:
        """Format comparison for CLI output.

        Implements [SPEC-07.41.01].

        Args:
            comparison: The checkpoint comparison.

        Returns:
            Formatted string for display.
        """
        lines: list[str] = []

        lines.append(
            f"Checkpoint Comparison: {comparison.from_checkpoint_id} → {comparison.to_checkpoint_id}"
        )
        lines.append("")

        # Time
        time_delta = comparison.time_delta
        minutes = int(time_delta.total_seconds() / 60)
        lines.append(
            f"Time: {comparison.from_time.isoformat()} → {comparison.to_time.isoformat()} ({minutes} minutes)"
        )
        lines.append("")

        # Conversation
        lines.append("Conversation:")
        conv = comparison.conversation
        lines.append(
            f"  Turns: {conv.from_stats.turns} → {conv.to_stats.turns} ({conv.turns_added:+d})"
        )
        lines.append(
            f"  Tokens: {conv.from_stats.tokens:,} → {conv.to_stats.tokens:,} ({conv.tokens_added:+,})"
        )
        lines.append(
            f"  Cost: ${conv.from_stats.cost_usd:.2f} → ${conv.to_stats.cost_usd:.2f} ({conv.cost_added:+.2f})"
        )
        lines.append("")

        # Files
        lines.append("Files Changed:")
        if comparison.files_changed:
            for fc in comparison.files_changed:
                type_char = {"added": "A", "modified": "M", "deleted": "D"}.get(
                    fc.change_type, "?"
                )
                stats = ""
                if fc.additions or fc.deletions:
                    stats = f"  +{fc.additions}  -{fc.deletions}"
                lines.append(f"  {type_char} {fc.path}{stats}")
        else:
            lines.append("  No file changes")
        lines.append("")

        # Environment
        lines.append("Environment:")
        env = comparison.environment
        if env.has_changes:
            if env.claude_code_changed:
                lines.append("  Claude Code version changed")
            if env.plugins_changed:
                lines.append(f"  Plugins: {', '.join(env.plugins_changed)}")
            if env.mcp_servers_changed:
                lines.append(f"  MCP Servers: {', '.join(env.mcp_servers_changed)}")
            if env.packages_changed:
                lines.append(f"  Packages: {', '.join(env.packages_changed[:5])}")
                if len(env.packages_changed) > 5:
                    lines.append(f"    ... and {len(env.packages_changed) - 5} more")
        else:
            lines.append("  No changes")
        lines.append("")

        # Approval
        lines.append("Approval:")
        appr = comparison.approval
        from_status = appr.from_status or "none"
        to_status = appr.to_status or "none"
        from_user = f" (by {appr.from_user})" if appr.from_user else ""
        to_user = f" (by {appr.to_user})" if appr.to_user else ""
        lines.append(f"  {comparison.from_checkpoint_id}: {from_status}{from_user}")
        lines.append(f"  {comparison.to_checkpoint_id}: {to_status}{to_user}")

        return "\n".join(lines)

    def format_file_diff(
        self,
        diff_text: str,
        from_cp_id: str,
        to_cp_id: str,
        filename: str,
    ) -> str:
        """Format file diff for CLI output.

        Args:
            diff_text: The unified diff text.
            from_cp_id: Source checkpoint ID.
            to_cp_id: Target checkpoint ID.
            filename: File being diffed.

        Returns:
            Formatted string for display.
        """
        lines: list[str] = [
            f"File Diff: {filename}",
            f"Checkpoints: {from_cp_id} → {to_cp_id}",
            "",
            diff_text if diff_text else "(no changes)",
        ]
        return "\n".join(lines)

    def format_conversation_diff(
        self,
        added_turns: list[dict[str, Any]],
        from_cp_id: str,
        to_cp_id: str,
    ) -> str:
        """Format conversation diff for CLI output.

        Args:
            added_turns: Turns added between checkpoints.
            from_cp_id: Source checkpoint ID.
            to_cp_id: Target checkpoint ID.

        Returns:
            Formatted string for display.
        """
        lines: list[str] = [
            f"Conversation Diff: {from_cp_id} → {to_cp_id}",
            f"Added {len(added_turns)} turns:",
            "",
        ]

        for i, turn in enumerate(added_turns):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            # Truncate long content
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"[{i + 1}] {role}: {content}")
            lines.append("")

        return "\n".join(lines)
