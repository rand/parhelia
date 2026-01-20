"""Context window optimization.

Implements:
- [SPEC-07.32.01] Context Budget Allocation
- [SPEC-07.32.02] Compression Strategies
- [SPEC-07.32.03] Context Metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from parhelia.memory import (
    CompressionResult,
    ContextBudget,
    ConversationTurn,
    estimate_tokens,
    summarize_tool_output,
)


@dataclass
class ContextMetrics:
    """Context usage metrics.

    Implements [SPEC-07.32.03].
    """

    context_tokens_used: int = 0
    context_budget_total: int = 200000
    compression_applied: bool = False
    unique_information_tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def context_budget_percent(self) -> float:
        """Usage as percentage of limit."""
        if self.context_budget_total == 0:
            return 0.0
        return self.context_tokens_used / self.context_budget_total

    @property
    def information_density(self) -> float:
        """Ratio of unique information to tokens.

        Higher is better - means less redundancy.
        """
        if self.context_tokens_used == 0:
            return 1.0
        return self.unique_information_tokens / self.context_tokens_used

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "context_tokens_used": self.context_tokens_used,
            "context_budget_total": self.context_budget_total,
            "context_budget_percent": round(self.context_budget_percent, 4),
            "compression_applied": self.compression_applied,
            "unique_information_tokens": self.unique_information_tokens,
            "information_density": round(self.information_density, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MergedTurn:
    """Result of merging multiple conversation turns."""

    original_count: int
    merged_content: str
    token_savings: int
    role: str


@dataclass
class ExtractedDecision:
    """A decision extracted from conversation."""

    description: str
    rationale: str | None
    source_turn_index: int


def extract_decisions_from_text(text: str) -> list[str]:
    """Extract decision statements from text.

    Looks for patterns like:
    - "I'll use X because Y"
    - "Let's go with X"
    - "The approach will be X"
    - "Decided to X"

    Args:
        text: Text to extract decisions from.

    Returns:
        List of decision statements.
    """
    decisions: list[str] = []

    # Patterns that indicate decisions
    patterns = [
        r"(?:I'll|I will|Let's|We'll|We will)\s+(?:use|go with|implement|create|add)\s+([^.!?\n]+)",
        r"(?:Decided|Choosing|Going)\s+(?:to|with)\s+([^.!?\n]+)",
        r"(?:The|Our)\s+(?:approach|solution|plan|strategy)\s+(?:is|will be)\s+([^.!?\n]+)",
        r"(?:Using|Implementing|Creating)\s+([^.!?\n]+)\s+(?:because|since|as)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 10:  # Filter out very short matches
                decisions.append(match.strip())

    return decisions


def extract_error_resolutions(
    turns: list[ConversationTurn],
) -> list[tuple[str, str]]:
    """Extract error messages and their resolutions.

    Implements [SPEC-07.32.02] - preserve error messages and resolutions.

    Args:
        turns: Conversation turns to analyze.

    Returns:
        List of (error, resolution) tuples.
    """
    resolutions: list[tuple[str, str]] = []
    current_error: str | None = None

    for turn in turns:
        content = turn.content

        # Look for error indicators
        error_patterns = [
            r"(?:Error|Exception|Failed|Failure):\s*([^\n]+)",
            r"(?:error|failed|exception)\[.*?\]:\s*([^\n]+)",
            r"AssertionError:\s*([^\n]+)",
            r"TypeError:\s*([^\n]+)",
            r"ValueError:\s*([^\n]+)",
        ]

        for pattern in error_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                current_error = match.group(0)
                break

        # Look for resolution indicators after an error
        if current_error:
            resolution_patterns = [
                r"(?:Fixed|Resolved|Solution|The fix):\s*([^\n]+)",
                r"(?:fixed|resolved|working now)",
                r"(?:tests? (?:pass|passing|passed))",
            ]

            for pattern in resolution_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Found resolution
                    resolution = content[:200]  # First 200 chars as context
                    resolutions.append((current_error, resolution))
                    current_error = None
                    break

    return resolutions


def identify_similar_turns(
    turns: list[ConversationTurn],
    similarity_threshold: float = 0.7,
) -> list[list[int]]:
    """Identify groups of similar turns that can be merged.

    Args:
        turns: Conversation turns to analyze.
        similarity_threshold: Minimum similarity ratio (0-1).

    Returns:
        List of groups, where each group is a list of turn indices.
    """
    groups: list[list[int]] = []
    used: set[int] = set()

    for i, turn_a in enumerate(turns):
        if i in used:
            continue

        # Start a new group
        group = [i]
        used.add(i)

        for j, turn_b in enumerate(turns[i + 1 :], start=i + 1):
            if j in used:
                continue

            # Same role is required for merging
            if turn_a.role != turn_b.role:
                continue

            # Check content similarity
            similarity = _compute_similarity(turn_a.content, turn_b.content)
            if similarity >= similarity_threshold:
                group.append(j)
                used.add(j)

        # Only keep groups with multiple turns
        if len(group) > 1:
            groups.append(group)

    return groups


def _compute_similarity(text_a: str, text_b: str) -> float:
    """Compute simple word-based Jaccard similarity.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Similarity ratio (0-1).
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union)


def merge_similar_turns(
    turns: list[ConversationTurn],
    indices: list[int],
) -> MergedTurn:
    """Merge multiple similar turns into one.

    Implements [SPEC-07.32.02] - merge similar conversation turns.

    Args:
        turns: All conversation turns.
        indices: Indices of turns to merge.

    Returns:
        MergedTurn with combined content.
    """
    to_merge = [turns[i] for i in indices]
    role = to_merge[0].role

    # Deduplicate content while preserving order
    seen_content: set[str] = set()
    unique_parts: list[str] = []

    for turn in to_merge:
        # Split by sentences and deduplicate
        sentences = re.split(r"[.!?]\s+", turn.content)
        for sentence in sentences:
            normalized = sentence.strip().lower()
            if normalized and normalized not in seen_content:
                seen_content.add(normalized)
                unique_parts.append(sentence.strip())

    merged_content = ". ".join(unique_parts)
    if merged_content and not merged_content.endswith((".", "!", "?")):
        merged_content += "."

    original_tokens = sum(turn.token_count for turn in to_merge)
    merged_tokens = estimate_tokens(merged_content)
    token_savings = original_tokens - merged_tokens

    return MergedTurn(
        original_count=len(to_merge),
        merged_content=merged_content,
        token_savings=max(0, token_savings),
        role=role,
    )


def extract_key_decisions(
    turns: list[ConversationTurn],
) -> list[ExtractedDecision]:
    """Extract key decisions from conversation, discarding deliberation.

    Implements [SPEC-07.32.02] - extract key decisions, discard deliberation.

    Args:
        turns: Conversation turns to analyze.

    Returns:
        List of extracted decisions.
    """
    decisions: list[ExtractedDecision] = []

    for i, turn in enumerate(turns):
        if turn.role != "assistant":
            continue

        extracted = extract_decisions_from_text(turn.content)
        for decision_text in extracted:
            # Look for rationale in same turn
            rationale = None
            rationale_match = re.search(
                r"(?:because|since|as|due to)\s+([^.!?\n]+)",
                turn.content,
                re.IGNORECASE,
            )
            if rationale_match:
                rationale = rationale_match.group(1).strip()

            decisions.append(
                ExtractedDecision(
                    description=decision_text,
                    rationale=rationale,
                    source_turn_index=i,
                )
            )

    return decisions


def calculate_unique_information(turns: list[ConversationTurn]) -> int:
    """Calculate unique information tokens in conversation.

    Uses content deduplication to measure information density.

    Args:
        turns: Conversation turns.

    Returns:
        Estimated unique information tokens.
    """
    all_content: list[str] = []
    seen_phrases: set[str] = set()

    for turn in turns:
        # Split into phrases
        phrases = re.split(r"[.!?,;:\n]+", turn.content)
        for phrase in phrases:
            normalized = phrase.strip().lower()
            if len(normalized) > 20 and normalized not in seen_phrases:
                seen_phrases.add(normalized)
                all_content.append(phrase.strip())

    unique_text = " ".join(all_content)
    return estimate_tokens(unique_text)


class ContextOptimizer:
    """Advanced context window optimization.

    Implements [SPEC-07.32].
    """

    def __init__(
        self,
        budget: ContextBudget | None = None,
        compression_threshold: float = 0.80,
    ):
        """Initialize the optimizer.

        Args:
            budget: Context budget allocation. Defaults to standard allocation.
            compression_threshold: Trigger compression at this usage percentage.
        """
        self.budget = budget or ContextBudget()
        self.compression_threshold = compression_threshold
        self._metrics_history: list[ContextMetrics] = []

    def get_metrics(self, turns: list[ConversationTurn]) -> ContextMetrics:
        """Calculate current context metrics.

        Implements [SPEC-07.32.03].

        Args:
            turns: Current conversation turns.

        Returns:
            ContextMetrics with current state.
        """
        total_tokens = sum(turn.token_count for turn in turns)
        unique_tokens = calculate_unique_information(turns)

        metrics = ContextMetrics(
            context_tokens_used=total_tokens,
            context_budget_total=self.budget.total_tokens,
            compression_applied=len(self._metrics_history) > 0
            and any(m.compression_applied for m in self._metrics_history[-5:]),
            unique_information_tokens=unique_tokens,
        )

        return metrics

    def should_compress(self, turns: list[ConversationTurn]) -> bool:
        """Check if compression is needed.

        Args:
            turns: Current conversation turns.

        Returns:
            True if compression should be triggered.
        """
        total_tokens = sum(turn.token_count for turn in turns)
        usage = total_tokens / self.budget.total_tokens
        return usage >= self.compression_threshold

    def compress(
        self,
        turns: list[ConversationTurn],
        target_pct: float = 0.60,
    ) -> tuple[list[ConversationTurn], CompressionResult]:
        """Apply compression strategies to conversation.

        Implements [SPEC-07.32.02].

        Strategies applied in order:
        1. Summarize old tool outputs
        2. Merge similar conversation turns
        3. Extract decisions, discard deliberation for old turns
        4. Preserve error resolutions

        Args:
            turns: Conversation turns to compress.
            target_pct: Target context usage percentage.

        Returns:
            Tuple of (compressed turns, compression result).
        """
        if not turns:
            return turns, CompressionResult(
                original_tokens=0,
                compressed_tokens=0,
                turns_removed=0,
                summary_generated=False,
            )

        original_tokens = sum(turn.token_count for turn in turns)
        target_tokens = int(self.budget.total_tokens * target_pct)

        # Start with a copy
        compressed: list[ConversationTurn] = list(turns)
        turns_removed = 0

        # Strategy 1: Summarize old tool outputs
        compressed = self._summarize_tool_outputs(compressed)

        # Check if target reached
        current_tokens = sum(turn.token_count for turn in compressed)
        if current_tokens <= target_tokens:
            return compressed, self._make_result(
                original_tokens, current_tokens, turns_removed
            )

        # Strategy 2: Merge similar turns
        compressed, merge_removed = self._merge_similar(compressed)
        turns_removed += merge_removed

        current_tokens = sum(turn.token_count for turn in compressed)
        if current_tokens <= target_tokens:
            return compressed, self._make_result(
                original_tokens, current_tokens, turns_removed
            )

        # Strategy 3: Extract decisions from old turns
        compressed, decision_removed = self._extract_and_compact(compressed)
        turns_removed += decision_removed

        current_tokens = sum(turn.token_count for turn in compressed)
        if current_tokens <= target_tokens:
            return compressed, self._make_result(
                original_tokens, current_tokens, turns_removed
            )

        # Strategy 4: Remove oldest turns (preserve recent and errors)
        compressed, old_removed = self._remove_old_turns(
            compressed, target_tokens, preserve_recent=10
        )
        turns_removed += old_removed

        current_tokens = sum(turn.token_count for turn in compressed)

        # Record metrics
        metrics = ContextMetrics(
            context_tokens_used=current_tokens,
            context_budget_total=self.budget.total_tokens,
            compression_applied=True,
            unique_information_tokens=calculate_unique_information(compressed),
        )
        self._metrics_history.append(metrics)

        return compressed, self._make_result(
            original_tokens, current_tokens, turns_removed
        )

    def _summarize_tool_outputs(
        self,
        turns: list[ConversationTurn],
    ) -> list[ConversationTurn]:
        """Summarize verbose tool outputs in older turns.

        Args:
            turns: Conversation turns.

        Returns:
            Turns with summarized tool outputs.
        """
        # Only process turns older than the recent 10
        if len(turns) <= 10:
            return turns

        result: list[ConversationTurn] = []
        for i, turn in enumerate(turns):
            if i < len(turns) - 10 and turn.tool_calls:
                # Summarize content for old turns with tool calls
                summarized = summarize_tool_output(turn.content, max_length=300)
                new_turn = ConversationTurn(
                    role=turn.role,
                    content=summarized,
                    token_count=estimate_tokens(summarized),
                    tool_calls=turn.tool_calls,
                    timestamp=turn.timestamp,
                )
                result.append(new_turn)
            else:
                result.append(turn)

        return result

    def _merge_similar(
        self,
        turns: list[ConversationTurn],
    ) -> tuple[list[ConversationTurn], int]:
        """Merge similar conversation turns.

        Args:
            turns: Conversation turns.

        Returns:
            Tuple of (merged turns, count removed).
        """
        # Only merge in older portion
        if len(turns) <= 15:
            return turns, 0

        old_turns = turns[:-10]
        recent_turns = turns[-10:]

        groups = identify_similar_turns(old_turns, similarity_threshold=0.6)

        if not groups:
            return turns, 0

        # Build new turn list, replacing groups with merged versions
        merged_indices: set[int] = set()
        for group in groups:
            merged_indices.update(group)

        result: list[ConversationTurn] = []
        for i, turn in enumerate(old_turns):
            if i in merged_indices:
                # Check if this is the first in a group
                for group in groups:
                    if group[0] == i:
                        merged = merge_similar_turns(old_turns, group)
                        result.append(
                            ConversationTurn(
                                role=merged.role,
                                content=merged.merged_content,
                                token_count=estimate_tokens(merged.merged_content),
                                timestamp=turn.timestamp,
                            )
                        )
                        break
            else:
                result.append(turn)

        result.extend(recent_turns)
        removed = len(turns) - len(result)
        return result, removed

    def _extract_and_compact(
        self,
        turns: list[ConversationTurn],
    ) -> tuple[list[ConversationTurn], int]:
        """Extract key decisions and compact old deliberation.

        Args:
            turns: Conversation turns.

        Returns:
            Tuple of (compacted turns, count removed).
        """
        if len(turns) <= 15:
            return turns, 0

        old_turns = turns[:-10]
        recent_turns = turns[-10:]

        # Extract decisions from old turns
        decisions = extract_key_decisions(old_turns)

        if not decisions:
            return turns, 0

        # Create a summary turn with decisions
        decision_text = "Key decisions made:\n" + "\n".join(
            f"- {d.description}"
            + (f" (because {d.rationale})" if d.rationale else "")
            for d in decisions[:10]  # Limit to 10 most important
        )

        decision_turn = ConversationTurn(
            role="assistant",
            content=decision_text,
            token_count=estimate_tokens(decision_text),
            timestamp=old_turns[-1].timestamp if old_turns else None,
        )

        # Keep only non-deliberation turns from old turns
        # (turns with tool calls or user messages)
        filtered_old: list[ConversationTurn] = []
        for turn in old_turns:
            if turn.role == "user" or turn.tool_calls:
                filtered_old.append(turn)

        result = filtered_old + [decision_turn] + recent_turns
        removed = len(turns) - len(result)
        return result, max(0, removed)

    def _remove_old_turns(
        self,
        turns: list[ConversationTurn],
        target_tokens: int,
        preserve_recent: int = 10,
    ) -> tuple[list[ConversationTurn], int]:
        """Remove oldest turns to reach target.

        Preserves error messages and their resolutions.

        Args:
            turns: Conversation turns.
            target_tokens: Target token count.
            preserve_recent: Number of recent turns to always keep.

        Returns:
            Tuple of (filtered turns, count removed).
        """
        if len(turns) <= preserve_recent:
            return turns, 0

        # Find error resolutions to preserve
        error_resolutions = extract_error_resolutions(turns)
        preserve_indices: set[int] = set()

        # Mark turns with errors/resolutions for preservation
        for error, _ in error_resolutions:
            for i, turn in enumerate(turns):
                if error in turn.content:
                    preserve_indices.add(i)
                    # Also preserve next turn (likely the fix)
                    if i + 1 < len(turns):
                        preserve_indices.add(i + 1)

        # Always preserve recent turns
        for i in range(max(0, len(turns) - preserve_recent), len(turns)):
            preserve_indices.add(i)

        # Remove oldest non-preserved turns until target reached
        result: list[ConversationTurn] = []
        current_tokens = 0
        removed = 0

        # Process from newest to oldest, so we remove oldest first
        for i in range(len(turns) - 1, -1, -1):
            turn = turns[i]
            if i in preserve_indices or current_tokens + turn.token_count <= target_tokens:
                result.insert(0, turn)
                current_tokens += turn.token_count
            else:
                removed += 1

        return result, removed

    def _make_result(
        self,
        original: int,
        compressed: int,
        removed: int,
    ) -> CompressionResult:
        """Create a CompressionResult."""
        return CompressionResult(
            original_tokens=original,
            compressed_tokens=compressed,
            turns_removed=removed,
            summary_generated=removed > 0,
        )

    def get_metrics_history(self) -> list[ContextMetrics]:
        """Get history of context metrics."""
        return list(self._metrics_history)

    def reset_metrics_history(self) -> None:
        """Clear metrics history."""
        self._metrics_history.clear()
