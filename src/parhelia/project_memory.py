"""Project-level persistent memory.

Implements:
- [SPEC-07.31.01] Project Memory Store
- [SPEC-07.31.02] Memory Update Triggers
- [SPEC-07.31.03] Memory Retrieval
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class ArchitectureKnowledge:
    """Knowledge about project architecture."""

    summary: str
    key_files: list[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "summary": self.summary,
            "key_files": self.key_files,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchitectureKnowledge:
        """Deserialize from dictionary."""
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()

        return cls(
            summary=data.get("summary", ""),
            key_files=data.get("key_files", []),
            updated_at=updated_at,
        )


@dataclass
class ConventionKnowledge:
    """Knowledge about project conventions."""

    conventions: dict[str, str] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = dict(self.conventions)
        result["updated_at"] = self.updated_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConventionKnowledge:
        """Deserialize from dictionary."""
        conventions = {k: v for k, v in data.items() if k != "updated_at"}
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()

        return cls(conventions=conventions, updated_at=updated_at)


@dataclass
class Gotcha:
    """A lesson learned or gotcha discovered during work."""

    description: str
    discovered_in: str  # Session ID where discovered
    added_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "description": self.description,
            "discovered_in": self.discovered_in,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Gotcha:
        """Deserialize from dictionary."""
        added_at = data.get("added_at")
        if isinstance(added_at, str):
            added_at = datetime.fromisoformat(added_at)
        else:
            added_at = datetime.now()

        return cls(
            description=data.get("description", ""),
            discovered_in=data.get("discovered_in", ""),
            added_at=added_at,
        )


@dataclass
class SessionHistoryEntry:
    """Entry in session history."""

    session_id: str
    summary: str
    outcome: Literal["approved", "rejected", "auto_approved", "pending"]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionHistoryEntry:
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        else:
            timestamp = datetime.now()

        return cls(
            session_id=data.get("session_id", ""),
            summary=data.get("summary", ""),
            outcome=data.get("outcome", "pending"),
            timestamp=timestamp,
        )


@dataclass
class ProjectKnowledge:
    """Collection of project knowledge."""

    architecture: ArchitectureKnowledge | None = None
    conventions: ConventionKnowledge | None = None
    gotchas: list[Gotcha] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {}
        if self.architecture:
            result["architecture"] = self.architecture.to_dict()
        if self.conventions:
            result["conventions"] = self.conventions.to_dict()
        if self.gotchas:
            result["gotchas"] = [g.to_dict() for g in self.gotchas]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectKnowledge:
        """Deserialize from dictionary."""
        architecture = None
        if "architecture" in data:
            architecture = ArchitectureKnowledge.from_dict(data["architecture"])

        conventions = None
        if "conventions" in data:
            conventions = ConventionKnowledge.from_dict(data["conventions"])

        gotchas = []
        if "gotchas" in data:
            gotchas = [Gotcha.from_dict(g) for g in data["gotchas"]]

        return cls(
            architecture=architecture,
            conventions=conventions,
            gotchas=gotchas,
        )


@dataclass
class ProjectMemory:
    """Project-level persistent memory.

    Implements [SPEC-07.31.01].
    """

    project_id: str
    last_updated: datetime = field(default_factory=datetime.now)
    knowledge: ProjectKnowledge = field(default_factory=ProjectKnowledge)
    session_history: list[SessionHistoryEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "project_id": self.project_id,
            "last_updated": self.last_updated.isoformat(),
            "knowledge": self.knowledge.to_dict(),
            "session_history": [s.to_dict() for s in self.session_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectMemory:
        """Deserialize from dictionary."""
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        else:
            last_updated = datetime.now()

        knowledge = ProjectKnowledge()
        if "knowledge" in data:
            knowledge = ProjectKnowledge.from_dict(data["knowledge"])

        session_history = []
        if "session_history" in data:
            session_history = [
                SessionHistoryEntry.from_dict(s) for s in data["session_history"]
            ]

        return cls(
            project_id=data.get("project_id", ""),
            last_updated=last_updated,
            knowledge=knowledge,
            session_history=session_history,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ProjectMemory:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ProjectMemoryManager:
    """Manage project-level memory persistence.

    Implements [SPEC-07.31].
    """

    DEFAULT_PATH = "/vol/parhelia/memory/project.json"
    MAX_SESSION_HISTORY = 100  # Keep last N sessions
    MAX_GOTCHAS = 50  # Keep most recent gotchas

    def __init__(
        self,
        project_id: str,
        memory_path: str | Path | None = None,
    ):
        """Initialize the memory manager.

        Args:
            project_id: Unique project identifier.
            memory_path: Path to memory file. Defaults to DEFAULT_PATH.
        """
        self.project_id = project_id
        self.memory_path = Path(memory_path) if memory_path else Path(self.DEFAULT_PATH)
        self._memory: ProjectMemory | None = None

    @property
    def memory(self) -> ProjectMemory:
        """Get current memory, loading if needed."""
        if self._memory is None:
            self._memory = self.load()
        return self._memory

    # =========================================================================
    # Persistence [SPEC-07.31.01]
    # =========================================================================

    def load(self) -> ProjectMemory:
        """Load project memory from disk.

        Returns:
            ProjectMemory, or new empty memory if file doesn't exist.
        """
        if self.memory_path.exists():
            try:
                content = self.memory_path.read_text()
                return ProjectMemory.from_json(content)
            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted file, start fresh but log warning
                import logging

                logging.warning(f"Failed to load project memory: {e}")

        # Create new memory
        return ProjectMemory(project_id=self.project_id)

    async def save(self) -> Path:
        """Save project memory to disk.

        Returns:
            Path to saved file.
        """
        self.memory.last_updated = datetime.now()

        # Ensure directory exists
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically
        temp_path = self.memory_path.with_suffix(".tmp")
        temp_path.write_text(self.memory.to_json())
        temp_path.rename(self.memory_path)

        return self.memory_path

    # =========================================================================
    # Knowledge Operations [SPEC-07.31.02]
    # =========================================================================

    def set_architecture(
        self,
        summary: str,
        key_files: list[str] | None = None,
    ) -> None:
        """Set architecture knowledge.

        Args:
            summary: Architecture summary.
            key_files: List of key file paths.
        """
        self.memory.knowledge.architecture = ArchitectureKnowledge(
            summary=summary,
            key_files=key_files or [],
            updated_at=datetime.now(),
        )

    def set_convention(self, key: str, value: str) -> None:
        """Set a project convention.

        Args:
            key: Convention key (e.g., "testing", "specs").
            value: Convention description.
        """
        if self.memory.knowledge.conventions is None:
            self.memory.knowledge.conventions = ConventionKnowledge()

        self.memory.knowledge.conventions.conventions[key] = value
        self.memory.knowledge.conventions.updated_at = datetime.now()

    def add_gotcha(self, description: str, session_id: str) -> Gotcha:
        """Add a gotcha/lesson learned.

        Args:
            description: Description of the gotcha.
            session_id: Session where it was discovered.

        Returns:
            The created Gotcha.
        """
        gotcha = Gotcha(
            description=description,
            discovered_in=session_id,
            added_at=datetime.now(),
        )
        self.memory.knowledge.gotchas.append(gotcha)

        # Trim to max size (keep most recent)
        if len(self.memory.knowledge.gotchas) > self.MAX_GOTCHAS:
            self.memory.knowledge.gotchas = self.memory.knowledge.gotchas[
                -self.MAX_GOTCHAS :
            ]

        return gotcha

    def remove_gotcha(self, description: str) -> bool:
        """Remove a gotcha by description.

        Args:
            description: Description to match.

        Returns:
            True if removed, False if not found.
        """
        for i, gotcha in enumerate(self.memory.knowledge.gotchas):
            if gotcha.description == description:
                self.memory.knowledge.gotchas.pop(i)
                return True
        return False

    # =========================================================================
    # Session History [SPEC-07.31.02]
    # =========================================================================

    def add_session(
        self,
        session_id: str,
        summary: str,
        outcome: Literal["approved", "rejected", "auto_approved", "pending"],
    ) -> SessionHistoryEntry:
        """Add a session to history.

        Implements [SPEC-07.31.02] - update on session approval.

        Args:
            session_id: Session identifier.
            summary: Brief session summary.
            outcome: Session outcome.

        Returns:
            The created SessionHistoryEntry.
        """
        entry = SessionHistoryEntry(
            session_id=session_id,
            summary=summary,
            outcome=outcome,
            timestamp=datetime.now(),
        )
        self.memory.session_history.append(entry)

        # Trim to max size (keep most recent)
        if len(self.memory.session_history) > self.MAX_SESSION_HISTORY:
            self.memory.session_history = self.memory.session_history[
                -self.MAX_SESSION_HISTORY :
            ]

        return entry

    def get_recent_sessions(
        self,
        limit: int = 10,
        outcome: str | None = None,
    ) -> list[SessionHistoryEntry]:
        """Get recent sessions from history.

        Args:
            limit: Maximum number of sessions to return.
            outcome: Filter by outcome if specified.

        Returns:
            List of recent sessions (most recent first).
        """
        sessions = self.memory.session_history
        if outcome:
            sessions = [s for s in sessions if s.outcome == outcome]

        return list(reversed(sessions[-limit:]))

    # =========================================================================
    # Memory Retrieval [SPEC-07.31.03]
    # =========================================================================

    def recall(self, query: str) -> dict[str, Any]:
        """Recall relevant memory based on query.

        Implements [SPEC-07.31.03] - mid-session retrieval.

        Args:
            query: Search query (case-insensitive substring match).

        Returns:
            Dictionary with matching memory sections.
        """
        query_lower = query.lower()
        results: dict[str, Any] = {}

        # Search architecture
        arch = self.memory.knowledge.architecture
        if arch and query_lower in arch.summary.lower():
            results["architecture"] = arch.to_dict()

        # Search conventions
        conv = self.memory.knowledge.conventions
        if conv:
            matching_conventions = {}
            for key, value in conv.conventions.items():
                if query_lower in key.lower() or query_lower in value.lower():
                    matching_conventions[key] = value
            if matching_conventions:
                results["conventions"] = matching_conventions

        # Search gotchas
        matching_gotchas = []
        for gotcha in self.memory.knowledge.gotchas:
            if query_lower in gotcha.description.lower():
                matching_gotchas.append(gotcha.to_dict())
        if matching_gotchas:
            results["gotchas"] = matching_gotchas

        # Search session history
        matching_sessions = []
        for session in self.memory.session_history:
            if (
                query_lower in session.summary.lower()
                or query_lower in session.session_id.lower()
            ):
                matching_sessions.append(session.to_dict())
        if matching_sessions:
            results["session_history"] = matching_sessions[-10:]  # Last 10 matches

        return results

    def get_system_prompt_context(self, max_tokens: int = 2000) -> str:
        """Get memory context for system prompt.

        Implements [SPEC-07.31.03] - include in system prompt.

        Args:
            max_tokens: Approximate max tokens (~4 chars/token).

        Returns:
            Formatted memory context string.
        """
        max_chars = max_tokens * 4
        sections: list[str] = []

        # Architecture summary (prioritized)
        arch = self.memory.knowledge.architecture
        if arch and arch.summary:
            arch_section = f"## Architecture\n{arch.summary}"
            if arch.key_files:
                arch_section += f"\nKey files: {', '.join(arch.key_files[:5])}"
            sections.append(arch_section)

        # Conventions
        conv = self.memory.knowledge.conventions
        if conv and conv.conventions:
            conv_lines = [f"- {k}: {v}" for k, v in list(conv.conventions.items())[:10]]
            sections.append("## Conventions\n" + "\n".join(conv_lines))

        # Gotchas (most recent first)
        gotchas = self.memory.knowledge.gotchas
        if gotchas:
            gotcha_lines = [f"- {g.description}" for g in gotchas[-5:]]
            sections.append("## Gotchas\n" + "\n".join(gotcha_lines))

        # Recent approved sessions
        recent = self.get_recent_sessions(limit=5, outcome="approved")
        if recent:
            session_lines = [f"- {s.summary}" for s in recent]
            sections.append("## Recent Sessions\n" + "\n".join(session_lines))

        # Combine sections up to max length
        result = ""
        for section in sections:
            if len(result) + len(section) + 2 > max_chars:
                break
            if result:
                result += "\n\n"
            result += section

        return result

    # =========================================================================
    # Save/Recall Commands [SPEC-07.31.02/03]
    # =========================================================================

    def save_knowledge(
        self,
        key: str,
        value: str,
        session_id: str | None = None,
    ) -> None:
        """Save knowledge to memory.

        Implements `parhelia memory save "<key>" "<value>"`.

        Args:
            key: Knowledge key (e.g., "architecture", "convention:testing", "gotcha").
            value: Knowledge value.
            session_id: Current session ID (for gotchas).
        """
        if key == "architecture":
            self.set_architecture(value)
        elif key.startswith("convention:"):
            convention_key = key[11:]  # Remove "convention:" prefix
            self.set_convention(convention_key, value)
        elif key == "gotcha":
            self.add_gotcha(value, session_id or "unknown")
        else:
            # Treat as convention
            self.set_convention(key, value)

    def recall_knowledge(self, query: str) -> str:
        """Recall knowledge from memory.

        Implements `parhelia memory recall <query>`.

        Args:
            query: Search query.

        Returns:
            Formatted recall results.
        """
        results = self.recall(query)
        if not results:
            return f"No memory found matching '{query}'"

        lines: list[str] = [f"Memory recall for '{query}':", ""]

        if "architecture" in results:
            arch = results["architecture"]
            lines.append("## Architecture")
            lines.append(arch.get("summary", ""))
            if arch.get("key_files"):
                lines.append(f"Key files: {', '.join(arch['key_files'])}")
            lines.append("")

        if "conventions" in results:
            lines.append("## Conventions")
            for k, v in results["conventions"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if "gotchas" in results:
            lines.append("## Gotchas")
            for g in results["gotchas"]:
                lines.append(f"- {g['description']}")
            lines.append("")

        if "session_history" in results:
            lines.append("## Related Sessions")
            for s in results["session_history"]:
                lines.append(f"- [{s['outcome']}] {s['summary']}")
            lines.append("")

        return "\n".join(lines)
