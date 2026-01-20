"""Tests for context window optimization.

@trace SPEC-07.32.01 - Context Budget Allocation
@trace SPEC-07.32.02 - Compression Strategies
@trace SPEC-07.32.03 - Context Metrics
"""

import pytest

from parhelia.context_optimization import (
    ContextMetrics,
    ContextOptimizer,
    ExtractedDecision,
    MergedTurn,
    calculate_unique_information,
    extract_decisions_from_text,
    extract_error_resolutions,
    extract_key_decisions,
    identify_similar_turns,
    merge_similar_turns,
)
from parhelia.memory import ContextBudget, ConversationTurn


class TestContextMetrics:
    """Tests for ContextMetrics dataclass."""

    def test_creation(self):
        """@trace SPEC-07.32.03 - Metrics MUST track required fields."""
        metrics = ContextMetrics(
            context_tokens_used=50000,
            context_budget_total=200000,
            compression_applied=False,
            unique_information_tokens=40000,
        )

        assert metrics.context_tokens_used == 50000
        assert metrics.context_budget_total == 200000
        assert metrics.compression_applied is False

    def test_context_budget_percent(self):
        """@trace SPEC-07.32.03 - Metrics MUST calculate budget percentage."""
        metrics = ContextMetrics(
            context_tokens_used=50000,
            context_budget_total=200000,
        )

        assert metrics.context_budget_percent == 0.25

    def test_context_budget_percent_zero(self):
        """@trace SPEC-07.32.03 - Budget percent MUST handle zero total."""
        metrics = ContextMetrics(
            context_tokens_used=100,
            context_budget_total=0,
        )

        assert metrics.context_budget_percent == 0.0

    def test_information_density(self):
        """@trace SPEC-07.32.03 - Metrics MUST calculate information density."""
        metrics = ContextMetrics(
            context_tokens_used=100,
            unique_information_tokens=80,
        )

        assert metrics.information_density == 0.8

    def test_information_density_zero(self):
        """@trace SPEC-07.32.03 - Info density MUST handle zero tokens."""
        metrics = ContextMetrics(
            context_tokens_used=0,
            unique_information_tokens=0,
        )

        assert metrics.information_density == 1.0

    def test_to_dict(self):
        """@trace SPEC-07.32.03 - Metrics MUST serialize to dict."""
        metrics = ContextMetrics(
            context_tokens_used=50000,
            context_budget_total=200000,
            compression_applied=True,
            unique_information_tokens=40000,
        )

        data = metrics.to_dict()

        assert data["context_tokens_used"] == 50000
        assert data["context_budget_percent"] == 0.25
        assert data["compression_applied"] is True
        assert data["information_density"] == 0.8


class TestExtractDecisions:
    """Tests for decision extraction functions."""

    def test_extract_ill_use(self):
        """@trace SPEC-07.32.02 - Extract MUST find 'I'll use' patterns."""
        text = "I'll use JWT for authentication because it's stateless."
        decisions = extract_decisions_from_text(text)

        assert len(decisions) >= 1
        assert any("JWT" in d for d in decisions)

    def test_extract_lets_go_with(self):
        """@trace SPEC-07.32.02 - Extract MUST find 'Let's go with' patterns."""
        text = "Let's go with the Redis approach for caching."
        decisions = extract_decisions_from_text(text)

        assert len(decisions) >= 1
        assert any("Redis" in d for d in decisions)

    def test_extract_decided_to(self):
        """@trace SPEC-07.32.02 - Extract MUST find 'Decided to' patterns."""
        text = "Decided to implement pagination using cursor-based approach."
        decisions = extract_decisions_from_text(text)

        assert len(decisions) >= 1

    def test_extract_approach_will_be(self):
        """@trace SPEC-07.32.02 - Extract MUST find 'approach will be' patterns."""
        text = "The approach will be to use async/await for all I/O operations."
        decisions = extract_decisions_from_text(text)

        assert len(decisions) >= 1

    def test_extract_no_decisions(self):
        """@trace SPEC-07.32.02 - Extract MUST return empty for no decisions."""
        text = "This is just a regular statement without decisions."
        decisions = extract_decisions_from_text(text)

        assert len(decisions) == 0


class TestExtractErrorResolutions:
    """Tests for error resolution extraction."""

    def test_extract_error_and_fix(self):
        """@trace SPEC-07.32.02 - Extract MUST find error-resolution pairs."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="Error: Module 'foo' not found",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="Fixed: Added foo to requirements.txt",
                token_count=10,
            ),
        ]

        resolutions = extract_error_resolutions(turns)
        assert len(resolutions) >= 1

    def test_extract_exception_resolved(self):
        """@trace SPEC-07.32.02 - Extract MUST find exception resolutions."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="TypeError: Cannot read property 'x' of undefined",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="Tests passing now after null check added.",
                token_count=10,
            ),
        ]

        resolutions = extract_error_resolutions(turns)
        assert len(resolutions) >= 1

    def test_extract_no_resolution(self):
        """@trace SPEC-07.32.02 - Extract MUST handle unresolved errors."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="Error: Connection refused",
                token_count=10,
            ),
            ConversationTurn(
                role="user",
                content="Please continue",
                token_count=5,
            ),
        ]

        resolutions = extract_error_resolutions(turns)
        # Error without resolution should not be included
        assert len(resolutions) == 0


class TestIdentifySimilarTurns:
    """Tests for similar turn identification."""

    def test_identify_similar_content(self):
        """@trace SPEC-07.32.02 - Identify MUST find similar turns."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="The quick brown fox jumps over the lazy dog.",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="A quick brown fox jumps over lazy dogs.",
                token_count=10,
            ),
            ConversationTurn(
                role="user",
                content="Something completely different.",
                token_count=10,
            ),
        ]

        groups = identify_similar_turns(turns, similarity_threshold=0.5)
        # First two should be grouped (same role, similar content)
        assert len(groups) >= 1
        assert 0 in groups[0] and 1 in groups[0]

    def test_identify_different_roles(self):
        """@trace SPEC-07.32.02 - Identify MUST not group different roles."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="The quick brown fox",
                token_count=10,
            ),
            ConversationTurn(
                role="user",
                content="The quick brown fox",
                token_count=10,
            ),
        ]

        groups = identify_similar_turns(turns, similarity_threshold=0.9)
        # Different roles should not be grouped
        assert len(groups) == 0

    def test_identify_no_similar(self):
        """@trace SPEC-07.32.02 - Identify MUST return empty for no similar."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="Apples oranges bananas",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="Cars trucks motorcycles",
                token_count=10,
            ),
        ]

        groups = identify_similar_turns(turns, similarity_threshold=0.7)
        assert len(groups) == 0


class TestMergeSimilarTurns:
    """Tests for turn merging."""

    def test_merge_creates_merged_turn(self):
        """@trace SPEC-07.32.02 - Merge MUST combine turn content."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="First point about the topic.",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="Second point about the topic.",
                token_count=10,
            ),
        ]

        merged = merge_similar_turns(turns, [0, 1])

        assert merged.original_count == 2
        assert "First point" in merged.merged_content
        assert "Second point" in merged.merged_content
        assert merged.role == "assistant"

    def test_merge_deduplicates(self):
        """@trace SPEC-07.32.02 - Merge MUST deduplicate similar sentences."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="The approach is good. The approach works well.",
                token_count=20,
            ),
            ConversationTurn(
                role="assistant",
                content="The approach is good. It handles edge cases.",
                token_count=20,
            ),
        ]

        merged = merge_similar_turns(turns, [0, 1])

        # Should not have duplicate "The approach is good"
        count = merged.merged_content.lower().count("the approach is good")
        assert count == 1

    def test_merge_calculates_savings(self):
        """@trace SPEC-07.32.02 - Merge MUST calculate token savings."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="A" * 100,
                token_count=25,
            ),
            ConversationTurn(
                role="assistant",
                content="A" * 100,
                token_count=25,
            ),
        ]

        merged = merge_similar_turns(turns, [0, 1])

        # Should save tokens from deduplication
        assert merged.token_savings >= 0


class TestExtractKeyDecisions:
    """Tests for key decision extraction from turns."""

    def test_extract_from_assistant_turns(self):
        """@trace SPEC-07.32.02 - Extract MUST find decisions in assistant turns."""
        turns = [
            ConversationTurn(
                role="user",
                content="How should we handle auth?",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="I'll use JWT for authentication because it's stateless.",
                token_count=20,
            ),
        ]

        decisions = extract_key_decisions(turns)
        assert len(decisions) >= 1
        assert decisions[0].source_turn_index == 1

    def test_extract_with_rationale(self):
        """@trace SPEC-07.32.02 - Extract MUST capture rationale."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="Let's go with Redis because it has better performance.",
                token_count=20,
            ),
        ]

        decisions = extract_key_decisions(turns)
        assert len(decisions) >= 1
        # Should capture rationale
        assert any(d.rationale is not None for d in decisions)

    def test_extract_ignores_user_turns(self):
        """@trace SPEC-07.32.02 - Extract MUST ignore user turns."""
        turns = [
            ConversationTurn(
                role="user",
                content="I'll use the simple approach.",
                token_count=10,
            ),
        ]

        decisions = extract_key_decisions(turns)
        assert len(decisions) == 0


class TestCalculateUniqueInformation:
    """Tests for unique information calculation."""

    def test_unique_with_duplicates(self):
        """@trace SPEC-07.32.03 - Calculate MUST deduplicate content."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="This is a unique statement about the project architecture.",
                token_count=20,
            ),
            ConversationTurn(
                role="assistant",
                content="This is a unique statement about the project architecture.",
                token_count=20,
            ),
        ]

        unique = calculate_unique_information(turns)
        total = sum(t.token_count for t in turns)

        # Unique should be less than total due to deduplication
        assert unique < total

    def test_unique_all_different(self):
        """@trace SPEC-07.32.03 - Calculate MUST count distinct content."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="First completely unique statement here.",
                token_count=10,
            ),
            ConversationTurn(
                role="assistant",
                content="Second completely different statement here.",
                token_count=10,
            ),
        ]

        unique = calculate_unique_information(turns)
        # Should have significant unique content
        assert unique > 0


class TestContextOptimizer:
    """Tests for ContextOptimizer."""

    @pytest.fixture
    def optimizer(self) -> ContextOptimizer:
        """Create optimizer with test budget."""
        budget = ContextBudget(total_tokens=10000)
        return ContextOptimizer(budget=budget, compression_threshold=0.8)

    @pytest.fixture
    def sample_turns(self) -> list[ConversationTurn]:
        """Create sample conversation turns."""
        return [
            ConversationTurn(role="user", content="Help me with auth", token_count=10),
            ConversationTurn(
                role="assistant",
                content="I'll use JWT for authentication.",
                token_count=20,
            ),
            ConversationTurn(role="user", content="Sounds good", token_count=5),
            ConversationTurn(
                role="assistant",
                content="Let me implement it now.",
                token_count=15,
            ),
        ]

    def test_optimizer_creation(self, optimizer):
        """ContextOptimizer MUST initialize with budget."""
        assert optimizer.budget.total_tokens == 10000
        assert optimizer.compression_threshold == 0.8

    def test_get_metrics(self, optimizer, sample_turns):
        """@trace SPEC-07.32.03 - Optimizer MUST calculate metrics."""
        metrics = optimizer.get_metrics(sample_turns)

        assert metrics.context_tokens_used == 50  # Sum of token_counts
        assert metrics.context_budget_total == 10000
        assert metrics.context_budget_percent == 0.005

    def test_should_compress_below_threshold(self, optimizer, sample_turns):
        """@trace SPEC-07.32.02 - Should not compress below threshold."""
        assert optimizer.should_compress(sample_turns) is False

    def test_should_compress_above_threshold(self, optimizer):
        """@trace SPEC-07.32.02 - Should compress above threshold."""
        # Create turns that exceed 80% of 10000 tokens
        large_turns = [
            ConversationTurn(
                role="assistant",
                content="x" * 4000,
                token_count=4000,
            )
            for _ in range(3)
        ]

        assert optimizer.should_compress(large_turns) is True

    def test_compress_empty(self, optimizer):
        """@trace SPEC-07.32.02 - Compress MUST handle empty turns."""
        result_turns, result = optimizer.compress([])

        assert result_turns == []
        assert result.original_tokens == 0
        assert result.compressed_tokens == 0

    def test_compress_applies_strategies(self, optimizer):
        """@trace SPEC-07.32.02 - Compress MUST apply compression strategies."""
        # Create turns with tool calls (will be summarized)
        turns = []
        for i in range(20):
            turns.append(
                ConversationTurn(
                    role="assistant",
                    content=f"Tool output {i}: " + "x" * 500,
                    token_count=150,
                    tool_calls=["read"] if i < 10 else None,
                )
            )

        compressed, result = optimizer.compress(turns, target_pct=0.3)

        # Should have compressed
        assert result.compressed_tokens <= result.original_tokens
        assert len(compressed) <= len(turns)

    def test_compress_preserves_recent(self, optimizer):
        """@trace SPEC-07.32.02 - Compress MUST preserve recent turns."""
        turns = [
            ConversationTurn(
                role="assistant",
                content=f"Turn {i}",
                token_count=100,
            )
            for i in range(20)
        ]

        compressed, _ = optimizer.compress(turns, target_pct=0.5)

        # Recent turns should be preserved
        recent_content = [t.content for t in compressed[-5:]]
        assert any("Turn 19" in c for c in recent_content)

    def test_metrics_history(self, optimizer):
        """@trace SPEC-07.32.03 - Optimizer MUST track metrics history."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="x" * 1000,
                token_count=1000,
            )
            for _ in range(15)
        ]

        optimizer.compress(turns, target_pct=0.3)

        history = optimizer.get_metrics_history()
        assert len(history) >= 1
        assert history[-1].compression_applied is True

    def test_reset_metrics_history(self, optimizer):
        """@trace SPEC-07.32.03 - Optimizer MUST support history reset."""
        turns = [
            ConversationTurn(role="assistant", content="x" * 500, token_count=500)
            for _ in range(15)
        ]

        optimizer.compress(turns, target_pct=0.3)
        optimizer.reset_metrics_history()

        assert len(optimizer.get_metrics_history()) == 0

    def test_compress_preserves_error_resolutions(self, optimizer):
        """@trace SPEC-07.32.02 - Compress MUST preserve error resolutions."""
        turns = [
            ConversationTurn(
                role="assistant",
                content="Starting work...",
                token_count=100,
            )
            for _ in range(10)
        ]
        # Add error and resolution in middle
        turns.insert(
            5,
            ConversationTurn(
                role="assistant",
                content="Error: Module not found",
                token_count=100,
            ),
        )
        turns.insert(
            6,
            ConversationTurn(
                role="assistant",
                content="Fixed: Added missing import. Tests passing.",
                token_count=100,
            ),
        )

        compressed, _ = optimizer.compress(turns, target_pct=0.5)

        # Error resolution should be preserved
        all_content = " ".join(t.content for t in compressed)
        assert "Error:" in all_content or "Fixed:" in all_content
