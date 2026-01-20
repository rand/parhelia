"""Tests for session memory and summarization.

@trace SPEC-07.30.01 - Memory Hierarchy
@trace SPEC-07.30.02 - Session Summary Generation
@trace SPEC-07.30.03 - Summary Storage
@trace SPEC-07.30.04 - Context Window Management
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from parhelia.memory import (
    CompressionResult,
    ContextBudget,
    ConversationTurn,
    Decision,
    FailedApproach,
    FileChange,
    MemoryManager,
    SessionSummary,
    estimate_tokens,
    summarize_tool_output,
)


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_file_change_creation(self):
        """FileChange MUST capture path, type, and line counts."""
        fc = FileChange(
            path="src/auth.py",
            change_type="modified",
            additions=45,
            deletions=12,
        )

        assert fc.path == "src/auth.py"
        assert fc.change_type == "modified"
        assert fc.additions == 45
        assert fc.deletions == 12

    def test_file_change_serialization(self):
        """FileChange MUST serialize to/from dict."""
        fc = FileChange(
            path="test.py",
            change_type="added",
            additions=100,
            deletions=0,
        )

        data = fc.to_dict()
        restored = FileChange.from_dict(data)

        assert restored.path == fc.path
        assert restored.change_type == fc.change_type
        assert restored.additions == fc.additions


class TestDecision:
    """Tests for Decision dataclass."""

    def test_decision_creation(self):
        """Decision MUST capture description and rationale."""
        d = Decision(
            description="Use JWT for authentication",
            rationale="Stateless, works well with API",
        )

        assert d.description == "Use JWT for authentication"
        assert d.rationale == "Stateless, works well with API"
        assert d.timestamp is not None

    def test_decision_serialization(self):
        """Decision MUST serialize to/from dict."""
        d = Decision(
            description="Test decision",
            rationale="Test rationale",
        )

        data = d.to_dict()
        restored = Decision.from_dict(data)

        assert restored.description == d.description
        assert restored.rationale == d.rationale


class TestFailedApproach:
    """Tests for FailedApproach dataclass."""

    def test_failed_approach_creation(self):
        """FailedApproach MUST capture description and reason."""
        fa = FailedApproach(
            description="Tried using authlib",
            reason="Incompatible with async",
        )

        assert fa.description == "Tried using authlib"
        assert fa.reason == "Incompatible with async"

    def test_failed_approach_serialization(self):
        """FailedApproach MUST serialize to/from dict."""
        fa = FailedApproach(
            description="Test approach",
            reason="Did not work",
        )

        data = fa.to_dict()
        restored = FailedApproach.from_dict(data)

        assert restored.description == fa.description
        assert restored.reason == fa.reason


class TestSessionSummary:
    """Tests for SessionSummary."""

    def test_summary_creation(self):
        """@trace SPEC-07.30.02 - Summary MUST include required fields."""
        summary = SessionSummary(
            session_id="test-session-123",
            session_name="Fix Auth Bug",
            progress_summary="Implemented JWT authentication",
        )

        assert summary.session_id == "test-session-123"
        assert summary.session_name == "Fix Auth Bug"
        assert summary.progress_summary == "Implemented JWT authentication"

    def test_summary_with_all_fields(self):
        """@trace SPEC-07.30.02 - Summary MUST support all sections."""
        summary = SessionSummary(
            session_id="test-session",
            session_name="Full Test",
            progress_summary="Made good progress",
            files_changed=[
                FileChange("src/auth.py", "modified", 45, 12),
                FileChange("tests/test_auth.py", "added", 120, 0),
            ],
            decisions=[
                Decision("Use JWT", "Stateless auth"),
                Decision("PyJWT library", "Well maintained"),
            ],
            failed_approaches=[
                FailedApproach("authlib", "Async incompatible"),
            ],
            blockers=["Need rate limiting"],
            resume_context="Tests passing, ready for review",
            conversation_turns=15,
            tokens_used=45000,
            estimated_cost_usd=0.45,
        )

        assert len(summary.files_changed) == 2
        assert len(summary.decisions) == 2
        assert len(summary.failed_approaches) == 1
        assert len(summary.blockers) == 1

    def test_summary_serialization(self):
        """@trace SPEC-07.30.03 - Summary MUST serialize to/from dict."""
        summary = SessionSummary(
            session_id="test-session",
            progress_summary="Test progress",
            decisions=[Decision("Test decision", "Test rationale")],
            conversation_turns=10,
            tokens_used=5000,
        )

        data = summary.to_dict()
        restored = SessionSummary.from_dict(data)

        assert restored.session_id == summary.session_id
        assert restored.progress_summary == summary.progress_summary
        assert len(restored.decisions) == 1
        assert restored.conversation_turns == 10

    def test_summary_to_markdown(self):
        """@trace SPEC-07.30.02 - Summary MUST generate markdown."""
        summary = SessionSummary(
            session_id="test-session",
            session_name="Test Session",
            progress_summary="- Implemented feature X\n- Added tests",
            files_changed=[
                FileChange("src/feature.py", "modified", 50, 10),
            ],
            decisions=[
                Decision("Used approach A", "Better performance"),
            ],
            failed_approaches=[
                FailedApproach("Approach B", "Too complex"),
            ],
            blockers=["Need API key"],
            resume_context="Ready to continue with step 2",
            conversation_turns=20,
            tokens_used=10000,
            estimated_cost_usd=0.10,
        )

        md = summary.to_markdown()

        assert "## Session Summary: Test Session" in md
        assert "### Progress" in md
        assert "Implemented feature X" in md
        assert "### Files Changed" in md
        assert "src/feature.py" in md
        assert "### Decisions" in md
        assert "Used approach A" in md
        assert "### Failed Approaches" in md
        assert "Approach B" in md
        assert "### Blockers" in md
        assert "Need API key" in md
        assert "### Context for Resume" in md
        assert "step 2" in md
        assert "### Metrics" in md
        assert "20" in md  # turns


class TestContextBudget:
    """Tests for ContextBudget."""

    def test_default_budget(self):
        """@trace SPEC-07.32.01 - Budget MUST have default allocations."""
        budget = ContextBudget()

        assert budget.total_tokens == 200000
        assert budget.system_prompt_pct == 0.10
        assert budget.project_memory_pct == 0.10
        assert budget.session_summary_pct == 0.15
        assert budget.recent_conversation_pct == 0.50
        assert budget.tool_results_buffer_pct == 0.15

    def test_budget_calculations(self):
        """@trace SPEC-07.32.01 - Budget MUST calculate token allocations."""
        budget = ContextBudget(total_tokens=100000)

        assert budget.system_prompt_budget == 10000
        assert budget.project_memory_budget == 10000
        assert budget.session_summary_budget == 15000
        assert budget.recent_conversation_budget == 50000
        assert budget.tool_results_buffer_budget == 15000


class TestConversationTurn:
    """Tests for ConversationTurn."""

    def test_turn_creation(self):
        """ConversationTurn MUST capture role and content."""
        turn = ConversationTurn(
            role="assistant",
            content="I'll help you with that.",
            token_count=10,
        )

        assert turn.role == "assistant"
        assert turn.content == "I'll help you with that."
        assert turn.token_count == 10

    def test_turn_serialization(self):
        """ConversationTurn MUST serialize to/from dict."""
        turn = ConversationTurn(
            role="user",
            content="Fix the bug",
            token_count=5,
            tool_calls=["read", "edit"],
        )

        data = turn.to_dict()
        restored = ConversationTurn.from_dict(data)

        assert restored.role == turn.role
        assert restored.content == turn.content
        assert restored.tool_calls == turn.tool_calls


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_dir) -> MemoryManager:
        """Create MemoryManager with temp storage."""
        return MemoryManager(
            session_id="test-session-123",
            sessions_root=temp_dir,
        )

    def test_manager_creation(self, manager):
        """MemoryManager MUST initialize with session ID."""
        assert manager.session_id == "test-session-123"
        assert manager.summary is None
        assert manager.conversation == []

    def test_add_turn(self, manager):
        """@trace SPEC-07.30.01 - Manager MUST track conversation turns."""
        turn = ConversationTurn(role="user", content="Hello", token_count=5)
        manager.add_turn(turn)

        assert len(manager.conversation) == 1
        assert manager.current_tokens == 5

    def test_add_decision(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST track decisions."""
        manager.add_decision("Use JWT", "Stateless auth")

        assert manager.summary is not None
        assert len(manager.summary.decisions) == 1
        assert manager.summary.decisions[0].description == "Use JWT"

    def test_add_failed_approach(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST track failed approaches."""
        manager.add_failed_approach("authlib", "Async incompatible")

        assert manager.summary is not None
        assert len(manager.summary.failed_approaches) == 1

    def test_add_file_change(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST track file changes."""
        manager.add_file_change("src/auth.py", "modified", 45, 12)

        assert manager.summary is not None
        assert len(manager.summary.files_changed) == 1

    def test_add_blocker(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST track blockers."""
        manager.add_blocker("Need API key")

        assert manager.summary is not None
        assert "Need API key" in manager.summary.blockers

    def test_set_progress(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST set progress summary."""
        manager.set_progress("Implemented JWT authentication")

        assert manager.summary is not None
        assert manager.summary.progress_summary == "Implemented JWT authentication"

    def test_set_resume_context(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST set resume context."""
        manager.set_resume_context("Ready for code review")

        assert manager.summary is not None
        assert manager.summary.resume_context == "Ready for code review"

    def test_update_metrics(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST update metrics."""
        manager.update_metrics(
            conversation_turns=15,
            tokens_used=45000,
            estimated_cost_usd=0.45,
        )

        assert manager.summary is not None
        assert manager.summary.conversation_turns == 15
        assert manager.summary.tokens_used == 45000
        assert manager.summary.estimated_cost_usd == 0.45

    def test_generate_summary(self, manager):
        """@trace SPEC-07.30.02 - Manager MUST generate summary."""
        manager.set_progress("Good progress")
        manager.add_decision("Use JWT", "Stateless")

        summary = manager.generate_summary(session_name="Test Session")

        assert summary.session_id == "test-session-123"
        assert summary.session_name == "Test Session"
        assert summary.progress_summary == "Good progress"
        assert len(summary.decisions) == 1

    def test_should_compress(self, manager):
        """@trace SPEC-07.30.04 - Manager MUST detect when compression needed."""
        # Initially not at threshold
        assert manager.should_compress() is False

        # Add enough tokens to exceed threshold
        for i in range(100):
            turn = ConversationTurn(role="user", content="x" * 1000, token_count=2000)
            manager.add_turn(turn)

        # Now should compress (200000 * 0.8 = 160000, we added 200000)
        assert manager.should_compress() is True

    def test_compress_context(self, manager):
        """@trace SPEC-07.30.04 - Manager MUST compress context."""
        # Add many turns
        for i in range(50):
            turn = ConversationTurn(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Turn {i}",
                token_count=100,
            )
            manager.add_turn(turn)

        # Should have 5000 tokens
        assert manager.current_tokens == 5000

        # Compress to very low target
        result = manager.compress_context(target_pct=0.01)

        # Should have removed turns
        assert result.turns_removed > 0
        assert result.original_tokens == 5000
        assert result.compressed_tokens < 5000
        assert len(manager.conversation) >= 5  # Preserves minimum

    def test_context_usage_pct(self, manager):
        """@trace SPEC-07.30.04 - Manager MUST calculate context usage."""
        turn = ConversationTurn(role="user", content="Hello", token_count=20000)
        manager.add_turn(turn)

        # 20000 / 200000 = 0.1 = 10%
        assert manager.context_usage_pct == 0.1

    @pytest.mark.asyncio
    async def test_save_summary_markdown(self, manager):
        """@trace SPEC-07.30.03 - Manager MUST save summary as markdown."""
        manager.set_progress("Test progress")
        manager.generate_summary(session_name="Test")

        path = await manager.save_summary()

        assert path.exists()
        assert path.suffix == ".md"

        content = path.read_text()
        assert "Test progress" in content

    @pytest.mark.asyncio
    async def test_save_summary_json(self, manager):
        """@trace SPEC-07.30.03 - Manager MUST save summary as JSON."""
        manager.set_progress("Test progress")
        manager.generate_summary()

        path = await manager.save_summary_json()

        assert path.exists()
        assert path.suffix == ".json"

    def test_get_context_for_model(self, manager):
        """@trace SPEC-07.30.01 - Manager MUST provide context for model."""
        manager.set_progress("Progress")
        manager.add_turn(ConversationTurn(role="user", content="Hello", token_count=5))
        manager.generate_summary()

        context = manager.get_context_for_model()

        assert "session_summary" in context
        assert "recent_turns" in context
        assert "context_usage_pct" in context
        assert len(context["recent_turns"]) == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_estimate_tokens(self):
        """estimate_tokens MUST provide reasonable estimates."""
        # ~4 chars per token
        text = "a" * 100
        estimate = estimate_tokens(text)

        assert estimate == 25  # 100 / 4

    def test_summarize_tool_output_short(self):
        """@trace SPEC-07.32.02 - Short output MUST not be truncated."""
        short_output = "Hello world"
        result = summarize_tool_output(short_output, max_length=500)

        assert result == short_output

    def test_summarize_tool_output_long(self):
        """@trace SPEC-07.32.02 - Long output MUST be truncated."""
        long_output = "x" * 1000
        result = summarize_tool_output(long_output, max_length=100)

        assert len(result) <= 100
        assert "truncated" in result

    def test_summarize_tool_output_multiline(self):
        """@trace SPEC-07.32.02 - Multiline output MUST preserve first/last when possible."""
        # Create enough lines to trigger line-based summarization
        lines = [f"Line {i}" for i in range(50)]
        output = "\n".join(lines)
        # Use small max_length to trigger summarization, but large enough to fit the summary
        result = summarize_tool_output(output, max_length=200)

        # Should keep first 5 and last 5 lines
        assert "Line 0" in result
        assert "Line 49" in result
        assert "omitted" in result


class TestCompressionResult:
    """Tests for CompressionResult."""

    def test_compression_ratio(self):
        """CompressionResult MUST calculate compression ratio."""
        result = CompressionResult(
            original_tokens=1000,
            compressed_tokens=500,
            turns_removed=10,
            summary_generated=True,
        )

        assert result.compression_ratio == 0.5

    def test_compression_ratio_zero_original(self):
        """CompressionResult MUST handle zero original tokens."""
        result = CompressionResult(
            original_tokens=0,
            compressed_tokens=0,
            turns_removed=0,
            summary_generated=False,
        )

        assert result.compression_ratio == 1.0
