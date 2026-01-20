"""Checkpoint tagging and annotation.

Implements:
- [SPEC-07.12.01] Tag Management
- [SPEC-07.12.02] Annotation Management
- [SPEC-07.12.03] Issue Linking
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from parhelia.session import (
    Checkpoint,
    CheckpointAnnotation,
    LinkedIssue,
)


@dataclass
class ParsedIssueReference:
    """Parsed issue reference from URL or ID."""

    tracker: Literal["github", "linear", "beads"]
    id: str
    url: str | None = None
    owner: str | None = None  # For GitHub
    repo: str | None = None  # For GitHub
    workspace: str | None = None  # For Linear


def parse_issue_reference(reference: str) -> ParsedIssueReference | None:
    """Parse an issue reference from URL or ID.

    Implements [SPEC-07.12.03].

    Supported formats:
    - GitHub: https://github.com/owner/repo/issues/123
    - Linear: https://linear.app/workspace/issue/ID-123
    - Beads: ph-xxx (local prefix)

    Args:
        reference: URL or issue ID to parse.

    Returns:
        ParsedIssueReference or None if not recognized.
    """
    # Try GitHub URL
    github_match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)",
        reference,
    )
    if github_match:
        owner, repo, issue_num = github_match.groups()
        return ParsedIssueReference(
            tracker="github",
            id=issue_num,
            url=reference,
            owner=owner,
            repo=repo,
        )

    # Try Linear URL
    linear_match = re.match(
        r"https?://linear\.app/([^/]+)/issue/([A-Z]+-\d+)",
        reference,
    )
    if linear_match:
        workspace, issue_id = linear_match.groups()
        return ParsedIssueReference(
            tracker="linear",
            id=issue_id,
            url=reference,
            workspace=workspace,
        )

    # Try Beads ID (ph-xxx or similar prefix pattern)
    # Must be lowercase to distinguish from Linear IDs (which are UPPERCASE-123)
    beads_match = re.match(r"^([a-z]{2,4})-([a-z0-9]+)$", reference)
    if beads_match:
        return ParsedIssueReference(
            tracker="beads",
            id=reference,
            url=None,
        )

    # Try GitHub shorthand: owner/repo#123
    github_short_match = re.match(r"([^/]+)/([^#]+)#(\d+)", reference)
    if github_short_match:
        owner, repo, issue_num = github_short_match.groups()
        return ParsedIssueReference(
            tracker="github",
            id=issue_num,
            url=f"https://github.com/{owner}/{repo}/issues/{issue_num}",
            owner=owner,
            repo=repo,
        )

    return None


def validate_tag(tag: str) -> bool:
    """Validate tag format.

    Tags should be:
    - Lowercase alphanumeric with hyphens, dots, and slashes
    - Hierarchical: category/subcategory/name

    Args:
        tag: Tag string to validate.

    Returns:
        True if valid.
    """
    if not tag:
        return False
    # Allow alphanumeric, hyphens, underscores, dots, and slashes for hierarchy
    return bool(re.match(r"^[a-z0-9][a-z0-9\-_./]*[a-z0-9]$|^[a-z0-9]$", tag))


def normalize_tag(tag: str) -> str:
    """Normalize tag to standard format.

    Args:
        tag: Tag string to normalize.

    Returns:
        Normalized lowercase tag.
    """
    return tag.lower().strip()


class TagManager:
    """Manage checkpoint tags, annotations, and issue links.

    Implements [SPEC-07.12].
    """

    def __init__(self):
        """Initialize the tag manager."""
        pass

    # =========================================================================
    # Tag Operations [SPEC-07.12.01]
    # =========================================================================

    def add_tag(self, checkpoint: Checkpoint, tag: str) -> Checkpoint:
        """Add a tag to a checkpoint.

        Args:
            checkpoint: The checkpoint to tag.
            tag: Tag to add (will be normalized).

        Returns:
            Updated checkpoint.

        Raises:
            ValueError: If tag format is invalid.
        """
        normalized = normalize_tag(tag)
        if not validate_tag(normalized):
            raise ValueError(f"Invalid tag format: {tag}")

        if normalized not in checkpoint.tags:
            checkpoint.tags.append(normalized)

        return checkpoint

    def remove_tag(self, checkpoint: Checkpoint, tag: str) -> Checkpoint:
        """Remove a tag from a checkpoint.

        Args:
            checkpoint: The checkpoint to untag.
            tag: Tag to remove.

        Returns:
            Updated checkpoint.
        """
        normalized = normalize_tag(tag)
        if normalized in checkpoint.tags:
            checkpoint.tags.remove(normalized)

        return checkpoint

    def has_tag(self, checkpoint: Checkpoint, tag: str) -> bool:
        """Check if checkpoint has a specific tag.

        Args:
            checkpoint: The checkpoint to check.
            tag: Tag to look for.

        Returns:
            True if checkpoint has the tag.
        """
        return normalize_tag(tag) in checkpoint.tags

    def match_tag_pattern(self, checkpoint: Checkpoint, pattern: str) -> bool:
        """Check if checkpoint has tags matching a pattern.

        Args:
            checkpoint: The checkpoint to check.
            pattern: Glob pattern to match (e.g., "milestone/*").

        Returns:
            True if any tag matches the pattern.
        """
        for tag in checkpoint.tags:
            if fnmatch.fnmatch(tag, pattern.lower()):
                return True
        return False

    def filter_by_tag(
        self,
        checkpoints: list[Checkpoint],
        tag_pattern: str,
    ) -> list[Checkpoint]:
        """Filter checkpoints by tag pattern.

        Args:
            checkpoints: List of checkpoints to filter.
            tag_pattern: Glob pattern to match.

        Returns:
            Filtered list of checkpoints.
        """
        return [
            cp for cp in checkpoints if self.match_tag_pattern(cp, tag_pattern)
        ]

    def get_all_tags(self, checkpoints: list[Checkpoint]) -> set[str]:
        """Get all unique tags from a list of checkpoints.

        Args:
            checkpoints: List of checkpoints.

        Returns:
            Set of all unique tags.
        """
        tags: set[str] = set()
        for cp in checkpoints:
            tags.update(cp.tags)
        return tags

    # =========================================================================
    # Annotation Operations [SPEC-07.12.02]
    # =========================================================================

    def add_annotation(
        self,
        checkpoint: Checkpoint,
        text: str,
        user: str,
    ) -> CheckpointAnnotation:
        """Add an annotation to a checkpoint.

        Annotations are append-only for audit trail.

        Args:
            checkpoint: The checkpoint to annotate.
            text: Annotation text.
            user: Username of annotator.

        Returns:
            The created annotation.
        """
        annotation = CheckpointAnnotation(
            timestamp=datetime.now(),
            user=user,
            text=text,
        )
        checkpoint.annotations.append(annotation)
        return annotation

    def get_annotations(self, checkpoint: Checkpoint) -> list[CheckpointAnnotation]:
        """Get all annotations for a checkpoint.

        Args:
            checkpoint: The checkpoint.

        Returns:
            List of annotations in chronological order.
        """
        return sorted(checkpoint.annotations, key=lambda a: a.timestamp)

    def search_annotations(
        self,
        checkpoints: list[Checkpoint],
        query: str,
    ) -> list[tuple[Checkpoint, CheckpointAnnotation]]:
        """Search annotations across checkpoints.

        Args:
            checkpoints: List of checkpoints to search.
            query: Search query (case-insensitive substring match).

        Returns:
            List of (checkpoint, annotation) tuples matching the query.
        """
        results: list[tuple[Checkpoint, CheckpointAnnotation]] = []
        query_lower = query.lower()

        for cp in checkpoints:
            for annotation in cp.annotations:
                if query_lower in annotation.text.lower():
                    results.append((cp, annotation))

        return results

    # =========================================================================
    # Issue Link Operations [SPEC-07.12.03]
    # =========================================================================

    def link_issue(
        self,
        checkpoint: Checkpoint,
        reference: str,
    ) -> LinkedIssue | None:
        """Link an external issue to a checkpoint.

        Args:
            checkpoint: The checkpoint.
            reference: Issue URL or ID.

        Returns:
            The created LinkedIssue, or None if reference not recognized.
        """
        parsed = parse_issue_reference(reference)
        if parsed is None:
            return None

        # Check if already linked
        for existing in checkpoint.linked_issues:
            if existing.tracker == parsed.tracker and existing.id == parsed.id:
                return existing  # Already linked

        linked_issue = LinkedIssue(
            tracker=parsed.tracker,
            id=parsed.id,
            url=parsed.url,
        )
        checkpoint.linked_issues.append(linked_issue)
        return linked_issue

    def unlink_issue(
        self,
        checkpoint: Checkpoint,
        reference: str,
    ) -> bool:
        """Remove an issue link from a checkpoint.

        Args:
            checkpoint: The checkpoint.
            reference: Issue URL or ID.

        Returns:
            True if link was removed, False if not found.
        """
        parsed = parse_issue_reference(reference)
        if parsed is None:
            # Try direct URL match
            for i, linked in enumerate(checkpoint.linked_issues):
                if linked.url == reference:
                    checkpoint.linked_issues.pop(i)
                    return True
            return False

        for i, linked in enumerate(checkpoint.linked_issues):
            if linked.tracker == parsed.tracker and linked.id == parsed.id:
                checkpoint.linked_issues.pop(i)
                return True

        return False

    def get_linked_issues(self, checkpoint: Checkpoint) -> list[LinkedIssue]:
        """Get all linked issues for a checkpoint.

        Args:
            checkpoint: The checkpoint.

        Returns:
            List of linked issues.
        """
        return list(checkpoint.linked_issues)

    def filter_by_issue(
        self,
        checkpoints: list[Checkpoint],
        tracker: str | None = None,
        issue_id: str | None = None,
    ) -> list[Checkpoint]:
        """Filter checkpoints by linked issue.

        Args:
            checkpoints: List of checkpoints to filter.
            tracker: Optional tracker to filter by (github, linear, beads).
            issue_id: Optional issue ID to filter by.

        Returns:
            Filtered list of checkpoints.
        """
        results: list[Checkpoint] = []

        for cp in checkpoints:
            for linked in cp.linked_issues:
                matches = True
                if tracker and linked.tracker != tracker:
                    matches = False
                if issue_id and linked.id != issue_id:
                    matches = False
                if matches:
                    results.append(cp)
                    break

        return results

    # =========================================================================
    # Combined Search [SPEC-07.12]
    # =========================================================================

    def search(
        self,
        checkpoints: list[Checkpoint],
        query: str,
    ) -> list[Checkpoint]:
        """Search checkpoints by tags and annotation text.

        Args:
            checkpoints: List of checkpoints to search.
            query: Search query.

        Returns:
            List of matching checkpoints.
        """
        query_lower = query.lower()
        results: list[Checkpoint] = []
        seen_ids: set[str] = set()

        for cp in checkpoints:
            if cp.id in seen_ids:
                continue

            # Search in tags
            for tag in cp.tags:
                if query_lower in tag:
                    results.append(cp)
                    seen_ids.add(cp.id)
                    break

            if cp.id in seen_ids:
                continue

            # Search in annotations
            for annotation in cp.annotations:
                if query_lower in annotation.text.lower():
                    results.append(cp)
                    seen_ids.add(cp.id)
                    break

        return results


def extract_issue_references(text: str) -> list[ParsedIssueReference]:
    """Extract issue references from text.

    Looks for patterns like:
    - #123 (GitHub issue in context of default repo)
    - owner/repo#123 (GitHub)
    - Full URLs

    Args:
        text: Text to search for references.

    Returns:
        List of parsed issue references.
    """
    references: list[ParsedIssueReference] = []
    seen: set[str] = set()

    # Find GitHub URLs
    github_urls = re.findall(
        r"https?://github\.com/[^/]+/[^/]+/issues/\d+",
        text,
    )
    for url in github_urls:
        if url not in seen:
            parsed = parse_issue_reference(url)
            if parsed:
                references.append(parsed)
                seen.add(url)

    # Find Linear URLs
    linear_urls = re.findall(
        r"https?://linear\.app/[^/]+/issue/[A-Z]+-\d+",
        text,
    )
    for url in linear_urls:
        if url not in seen:
            parsed = parse_issue_reference(url)
            if parsed:
                references.append(parsed)
                seen.add(url)

    # Find GitHub shorthand (owner/repo#123)
    github_shorts = re.findall(r"([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)#(\d+)", text)
    for owner_repo, num in github_shorts:
        ref = f"{owner_repo}#{num}"
        if ref not in seen:
            parsed = parse_issue_reference(ref)
            if parsed:
                references.append(parsed)
                seen.add(ref)

    # Find Beads IDs (ph-xxx pattern) - lowercase only to avoid matching Linear IDs
    beads_ids = re.findall(r"\b([a-z]{2,4}-[a-z0-9]+)\b", text)
    for beads_id in beads_ids:
        if beads_id not in seen:
            parsed = parse_issue_reference(beads_id)
            if parsed:
                references.append(parsed)
                seen.add(beads_id)

    return references
