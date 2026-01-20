"""Tests for project-level persistent memory.

@trace SPEC-07.31.01 - Project Memory Store
@trace SPEC-07.31.02 - Memory Update Triggers
@trace SPEC-07.31.03 - Memory Retrieval
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from parhelia.project_memory import (
    ArchitectureKnowledge,
    ConventionKnowledge,
    Gotcha,
    ProjectKnowledge,
    ProjectMemory,
    ProjectMemoryManager,
    SessionHistoryEntry,
)


class TestArchitectureKnowledge:
    """Tests for ArchitectureKnowledge dataclass."""

    def test_creation(self):
        """ArchitectureKnowledge MUST capture summary and key files."""
        arch = ArchitectureKnowledge(
            summary="Modal-based remote execution",
            key_files=["src/modal_app.py", "src/checkpoint.py"],
        )

        assert arch.summary == "Modal-based remote execution"
        assert len(arch.key_files) == 2
        assert arch.updated_at is not None

    def test_serialization(self):
        """ArchitectureKnowledge MUST serialize to/from dict."""
        arch = ArchitectureKnowledge(
            summary="Test summary",
            key_files=["file1.py"],
        )

        data = arch.to_dict()
        restored = ArchitectureKnowledge.from_dict(data)

        assert restored.summary == arch.summary
        assert restored.key_files == arch.key_files


class TestConventionKnowledge:
    """Tests for ConventionKnowledge dataclass."""

    def test_creation(self):
        """ConventionKnowledge MUST capture conventions dict."""
        conv = ConventionKnowledge(
            conventions={
                "testing": "pytest with 80% coverage",
                "specs": "SPEC-XX.YY format",
            }
        )

        assert conv.conventions["testing"] == "pytest with 80% coverage"
        assert conv.updated_at is not None

    def test_serialization(self):
        """ConventionKnowledge MUST serialize to/from dict."""
        conv = ConventionKnowledge(
            conventions={"key": "value"},
        )

        data = conv.to_dict()
        restored = ConventionKnowledge.from_dict(data)

        assert restored.conventions == conv.conventions


class TestGotcha:
    """Tests for Gotcha dataclass."""

    def test_creation(self):
        """Gotcha MUST capture description and session."""
        gotcha = Gotcha(
            description="Modal sandbox requires network flag",
            discovered_in="ph-fix-123",
        )

        assert gotcha.description == "Modal sandbox requires network flag"
        assert gotcha.discovered_in == "ph-fix-123"
        assert gotcha.added_at is not None

    def test_serialization(self):
        """Gotcha MUST serialize to/from dict."""
        gotcha = Gotcha(
            description="Test gotcha",
            discovered_in="session-1",
        )

        data = gotcha.to_dict()
        restored = Gotcha.from_dict(data)

        assert restored.description == gotcha.description
        assert restored.discovered_in == gotcha.discovered_in


class TestSessionHistoryEntry:
    """Tests for SessionHistoryEntry dataclass."""

    def test_creation(self):
        """SessionHistoryEntry MUST capture session info and outcome."""
        entry = SessionHistoryEntry(
            session_id="ph-auth-123",
            summary="Implemented JWT auth",
            outcome="approved",
        )

        assert entry.session_id == "ph-auth-123"
        assert entry.summary == "Implemented JWT auth"
        assert entry.outcome == "approved"
        assert entry.timestamp is not None

    def test_serialization(self):
        """SessionHistoryEntry MUST serialize to/from dict."""
        entry = SessionHistoryEntry(
            session_id="test-session",
            summary="Test summary",
            outcome="rejected",
        )

        data = entry.to_dict()
        restored = SessionHistoryEntry.from_dict(data)

        assert restored.session_id == entry.session_id
        assert restored.outcome == entry.outcome


class TestProjectKnowledge:
    """Tests for ProjectKnowledge dataclass."""

    def test_creation_empty(self):
        """ProjectKnowledge MUST allow empty creation."""
        knowledge = ProjectKnowledge()

        assert knowledge.architecture is None
        assert knowledge.conventions is None
        assert knowledge.gotchas == []

    def test_creation_full(self):
        """ProjectKnowledge MUST support all fields."""
        knowledge = ProjectKnowledge(
            architecture=ArchitectureKnowledge("Summary", ["file.py"]),
            conventions=ConventionKnowledge({"key": "value"}),
            gotchas=[Gotcha("Gotcha 1", "session-1")],
        )

        assert knowledge.architecture is not None
        assert knowledge.conventions is not None
        assert len(knowledge.gotchas) == 1

    def test_serialization(self):
        """ProjectKnowledge MUST serialize to/from dict."""
        knowledge = ProjectKnowledge(
            architecture=ArchitectureKnowledge("Summary", ["file.py"]),
            gotchas=[Gotcha("Gotcha 1", "session-1")],
        )

        data = knowledge.to_dict()
        restored = ProjectKnowledge.from_dict(data)

        assert restored.architecture is not None
        assert restored.architecture.summary == "Summary"
        assert len(restored.gotchas) == 1


class TestProjectMemory:
    """Tests for ProjectMemory dataclass."""

    def test_creation(self):
        """@trace SPEC-07.31.01 - ProjectMemory MUST have required fields."""
        memory = ProjectMemory(project_id="parhelia")

        assert memory.project_id == "parhelia"
        assert memory.last_updated is not None
        assert memory.knowledge is not None
        assert memory.session_history == []

    def test_serialization_dict(self):
        """@trace SPEC-07.31.01 - ProjectMemory MUST serialize to/from dict."""
        memory = ProjectMemory(
            project_id="test-project",
            knowledge=ProjectKnowledge(
                architecture=ArchitectureKnowledge("Architecture", ["main.py"]),
            ),
            session_history=[
                SessionHistoryEntry("session-1", "Did work", "approved"),
            ],
        )

        data = memory.to_dict()
        restored = ProjectMemory.from_dict(data)

        assert restored.project_id == memory.project_id
        assert restored.knowledge.architecture is not None
        assert len(restored.session_history) == 1

    def test_serialization_json(self):
        """@trace SPEC-07.31.01 - ProjectMemory MUST serialize to/from JSON."""
        memory = ProjectMemory(project_id="test")

        json_str = memory.to_json()
        restored = ProjectMemory.from_json(json_str)

        assert restored.project_id == "test"


class TestProjectMemoryManager:
    """Tests for ProjectMemoryManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_dir) -> ProjectMemoryManager:
        """Create ProjectMemoryManager with temp storage."""
        memory_path = Path(temp_dir) / "memory" / "project.json"
        return ProjectMemoryManager(
            project_id="test-project",
            memory_path=memory_path,
        )

    def test_manager_creation(self, manager):
        """ProjectMemoryManager MUST initialize with project ID."""
        assert manager.project_id == "test-project"

    def test_load_nonexistent(self, manager):
        """@trace SPEC-07.31.01 - Load MUST create empty memory if file missing."""
        memory = manager.load()

        assert memory.project_id == "test-project"
        assert memory.session_history == []

    @pytest.mark.asyncio
    async def test_save_and_load(self, manager):
        """@trace SPEC-07.31.01 - Manager MUST persist memory to disk."""
        manager.set_architecture("Test architecture", ["file.py"])
        await manager.save()

        # Create new manager and load
        new_manager = ProjectMemoryManager(
            project_id="test-project",
            memory_path=manager.memory_path,
        )
        loaded = new_manager.load()

        assert loaded.knowledge.architecture is not None
        assert loaded.knowledge.architecture.summary == "Test architecture"

    def test_set_architecture(self, manager):
        """@trace SPEC-07.31.02 - Manager MUST set architecture knowledge."""
        manager.set_architecture(
            summary="Modal-based execution",
            key_files=["modal_app.py", "checkpoint.py"],
        )

        arch = manager.memory.knowledge.architecture
        assert arch is not None
        assert arch.summary == "Modal-based execution"
        assert len(arch.key_files) == 2

    def test_set_convention(self, manager):
        """@trace SPEC-07.31.02 - Manager MUST set conventions."""
        manager.set_convention("testing", "pytest with async")
        manager.set_convention("specs", "SPEC-XX.YY format")

        conv = manager.memory.knowledge.conventions
        assert conv is not None
        assert conv.conventions["testing"] == "pytest with async"
        assert conv.conventions["specs"] == "SPEC-XX.YY format"

    def test_add_gotcha(self, manager):
        """@trace SPEC-07.31.02 - Manager MUST add gotchas."""
        gotcha = manager.add_gotcha(
            description="Modal requires network flag",
            session_id="ph-fix-123",
        )

        assert gotcha.description == "Modal requires network flag"
        assert len(manager.memory.knowledge.gotchas) == 1

    def test_add_gotcha_limit(self, manager):
        """@trace SPEC-07.31.02 - Gotchas MUST be limited to prevent bloat."""
        # Add more than max gotchas
        for i in range(60):
            manager.add_gotcha(f"Gotcha {i}", f"session-{i}")

        assert len(manager.memory.knowledge.gotchas) == manager.MAX_GOTCHAS

    def test_remove_gotcha(self, manager):
        """@trace SPEC-07.31.02 - Manager MUST support gotcha removal."""
        manager.add_gotcha("Keep this", "session-1")
        manager.add_gotcha("Remove this", "session-2")

        result = manager.remove_gotcha("Remove this")
        assert result is True
        assert len(manager.memory.knowledge.gotchas) == 1
        assert manager.memory.knowledge.gotchas[0].description == "Keep this"

    def test_remove_gotcha_not_found(self, manager):
        """@trace SPEC-07.31.02 - Remove MUST return False if not found."""
        result = manager.remove_gotcha("Nonexistent")
        assert result is False

    def test_add_session(self, manager):
        """@trace SPEC-07.31.02 - Manager MUST add sessions on approval."""
        entry = manager.add_session(
            session_id="ph-auth-123",
            summary="Implemented JWT authentication",
            outcome="approved",
        )

        assert entry.session_id == "ph-auth-123"
        assert len(manager.memory.session_history) == 1

    def test_add_session_limit(self, manager):
        """@trace SPEC-07.31.02 - Session history MUST be limited."""
        # Add more than max sessions
        for i in range(110):
            manager.add_session(f"session-{i}", f"Summary {i}", "approved")

        assert len(manager.memory.session_history) == manager.MAX_SESSION_HISTORY

    def test_get_recent_sessions(self, manager):
        """@trace SPEC-07.31.03 - Manager MUST retrieve recent sessions."""
        manager.add_session("session-1", "First", "approved")
        manager.add_session("session-2", "Second", "rejected")
        manager.add_session("session-3", "Third", "approved")

        recent = manager.get_recent_sessions(limit=5)
        assert len(recent) == 3
        assert recent[0].session_id == "session-3"  # Most recent first

    def test_get_recent_sessions_filtered(self, manager):
        """@trace SPEC-07.31.03 - Get recent MUST filter by outcome."""
        manager.add_session("session-1", "First", "approved")
        manager.add_session("session-2", "Second", "rejected")
        manager.add_session("session-3", "Third", "approved")

        approved = manager.get_recent_sessions(limit=5, outcome="approved")
        assert len(approved) == 2
        assert all(s.outcome == "approved" for s in approved)

    def test_recall_architecture(self, manager):
        """@trace SPEC-07.31.03 - Recall MUST search architecture."""
        manager.set_architecture("Modal-based remote execution")

        results = manager.recall("modal")
        assert "architecture" in results
        assert "Modal-based" in results["architecture"]["summary"]

    def test_recall_conventions(self, manager):
        """@trace SPEC-07.31.03 - Recall MUST search conventions."""
        manager.set_convention("testing", "pytest with async support")

        results = manager.recall("pytest")
        assert "conventions" in results
        assert "testing" in results["conventions"]

    def test_recall_gotchas(self, manager):
        """@trace SPEC-07.31.03 - Recall MUST search gotchas."""
        manager.add_gotcha("Network flag required for sandbox", "session-1")

        results = manager.recall("sandbox")
        assert "gotchas" in results
        assert len(results["gotchas"]) == 1

    def test_recall_sessions(self, manager):
        """@trace SPEC-07.31.03 - Recall MUST search session history."""
        manager.add_session("ph-auth", "Implemented JWT auth", "approved")

        results = manager.recall("JWT")
        assert "session_history" in results

    def test_recall_no_match(self, manager):
        """@trace SPEC-07.31.03 - Recall MUST return empty dict if no match."""
        results = manager.recall("nonexistent")
        assert results == {}

    def test_get_system_prompt_context(self, manager):
        """@trace SPEC-07.31.03 - Manager MUST generate system prompt context."""
        manager.set_architecture("Modal-based execution", ["modal_app.py"])
        manager.set_convention("testing", "pytest")
        manager.add_gotcha("Network flag needed", "session-1")
        manager.add_session("session-1", "Added auth", "approved")

        context = manager.get_system_prompt_context()

        assert "## Architecture" in context
        assert "Modal-based" in context
        assert "## Conventions" in context
        assert "## Gotchas" in context
        assert "## Recent Sessions" in context

    def test_get_system_prompt_context_max_tokens(self, manager):
        """@trace SPEC-07.31.03 - Context MUST respect token limit."""
        # Add lots of content
        manager.set_architecture("A" * 5000, ["file.py"])
        for i in range(20):
            manager.set_convention(f"conv_{i}", "B" * 200)

        context = manager.get_system_prompt_context(max_tokens=500)
        # ~500 tokens * 4 chars/token = 2000 chars max
        assert len(context) <= 2500  # Some buffer for formatting

    def test_save_knowledge_architecture(self, manager):
        """@trace SPEC-07.31.02 - save_knowledge MUST handle architecture key."""
        manager.save_knowledge("architecture", "New architecture summary")

        arch = manager.memory.knowledge.architecture
        assert arch is not None
        assert arch.summary == "New architecture summary"

    def test_save_knowledge_convention(self, manager):
        """@trace SPEC-07.31.02 - save_knowledge MUST handle convention: prefix."""
        manager.save_knowledge("convention:testing", "Use pytest")

        conv = manager.memory.knowledge.conventions
        assert conv is not None
        assert conv.conventions["testing"] == "Use pytest"

    def test_save_knowledge_gotcha(self, manager):
        """@trace SPEC-07.31.02 - save_knowledge MUST handle gotcha key."""
        manager.save_knowledge("gotcha", "Remember to flush cache", "session-1")

        assert len(manager.memory.knowledge.gotchas) == 1
        assert "flush cache" in manager.memory.knowledge.gotchas[0].description

    def test_save_knowledge_generic(self, manager):
        """@trace SPEC-07.31.02 - save_knowledge MUST treat unknown keys as conventions."""
        manager.save_knowledge("custom_key", "Custom value")

        conv = manager.memory.knowledge.conventions
        assert conv is not None
        assert conv.conventions["custom_key"] == "Custom value"

    def test_recall_knowledge_found(self, manager):
        """@trace SPEC-07.31.03 - recall_knowledge MUST return formatted results."""
        manager.set_architecture("Modal execution system")

        result = manager.recall_knowledge("modal")

        assert "Memory recall for 'modal'" in result
        assert "## Architecture" in result
        assert "Modal execution" in result

    def test_recall_knowledge_not_found(self, manager):
        """@trace SPEC-07.31.03 - recall_knowledge MUST indicate no match."""
        result = manager.recall_knowledge("nonexistent")

        assert "No memory found" in result

    @pytest.mark.asyncio
    async def test_save_creates_directory(self, temp_dir):
        """@trace SPEC-07.31.01 - Save MUST create parent directories."""
        memory_path = Path(temp_dir) / "deep" / "nested" / "project.json"
        manager = ProjectMemoryManager(
            project_id="test",
            memory_path=memory_path,
        )

        manager.set_architecture("Test")
        path = await manager.save()

        assert path.exists()
        assert path.parent.exists()

    def test_load_corrupted_file(self, temp_dir):
        """@trace SPEC-07.31.01 - Load MUST handle corrupted files gracefully."""
        memory_path = Path(temp_dir) / "project.json"
        memory_path.write_text("invalid json {{{")

        manager = ProjectMemoryManager(
            project_id="test",
            memory_path=memory_path,
        )
        memory = manager.load()

        # Should return fresh memory instead of crashing
        assert memory.project_id == "test"
        assert memory.session_history == []
