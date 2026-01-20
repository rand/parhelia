"""Tests for checkpoint tagging and annotation.

@trace SPEC-07.12.01 - Tag Management
@trace SPEC-07.12.02 - Annotation Management
@trace SPEC-07.12.03 - Issue Linking
"""

from datetime import datetime

import pytest

from parhelia.session import (
    Checkpoint,
    CheckpointAnnotation,
    CheckpointTrigger,
    LinkedIssue,
)
from parhelia.tagging import (
    ParsedIssueReference,
    TagManager,
    extract_issue_references,
    normalize_tag,
    parse_issue_reference,
    validate_tag,
)


class TestParseIssueReference:
    """Tests for parse_issue_reference function."""

    def test_parse_github_url(self):
        """@trace SPEC-07.12.03 - Parser MUST recognize GitHub URLs."""
        url = "https://github.com/owner/repo/issues/123"
        result = parse_issue_reference(url)

        assert result is not None
        assert result.tracker == "github"
        assert result.id == "123"
        assert result.url == url
        assert result.owner == "owner"
        assert result.repo == "repo"

    def test_parse_github_shorthand(self):
        """@trace SPEC-07.12.03 - Parser MUST recognize GitHub shorthand."""
        ref = "owner/repo#456"
        result = parse_issue_reference(ref)

        assert result is not None
        assert result.tracker == "github"
        assert result.id == "456"
        assert result.owner == "owner"
        assert result.repo == "repo"
        assert "github.com" in result.url

    def test_parse_linear_url(self):
        """@trace SPEC-07.12.03 - Parser MUST recognize Linear URLs."""
        url = "https://linear.app/myworkspace/issue/PROJ-789"
        result = parse_issue_reference(url)

        assert result is not None
        assert result.tracker == "linear"
        assert result.id == "PROJ-789"
        assert result.url == url
        assert result.workspace == "myworkspace"

    def test_parse_beads_id(self):
        """@trace SPEC-07.12.03 - Parser MUST recognize Beads IDs."""
        ref = "ph-abc123"
        result = parse_issue_reference(ref)

        assert result is not None
        assert result.tracker == "beads"
        assert result.id == "ph-abc123"
        assert result.url is None

    def test_parse_beads_id_short_prefix(self):
        """@trace SPEC-07.12.03 - Parser MUST handle various Beads prefixes."""
        ref = "bd-xyz"
        result = parse_issue_reference(ref)

        assert result is not None
        assert result.tracker == "beads"
        assert result.id == "bd-xyz"

    def test_parse_invalid_reference(self):
        """@trace SPEC-07.12.03 - Parser MUST return None for invalid refs."""
        result = parse_issue_reference("not-a-valid-reference!")
        assert result is None

    def test_parse_plain_url(self):
        """Parser MUST return None for non-issue URLs."""
        result = parse_issue_reference("https://example.com/page")
        assert result is None


class TestValidateTag:
    """Tests for tag validation."""

    def test_valid_simple_tag(self):
        """@trace SPEC-07.12.01 - Simple tags MUST be valid."""
        assert validate_tag("stable") is True
        assert validate_tag("v1") is True

    def test_valid_hierarchical_tag(self):
        """@trace SPEC-07.12.01 - Hierarchical tags MUST be valid."""
        assert validate_tag("milestone/v1.0") is True
        assert validate_tag("experiment/approach-a") is True
        assert validate_tag("category/sub/name") is True

    def test_valid_hyphenated_tag(self):
        """@trace SPEC-07.12.01 - Hyphenated tags MUST be valid."""
        assert validate_tag("feature-complete") is True

    def test_invalid_empty_tag(self):
        """@trace SPEC-07.12.01 - Empty tags MUST be invalid."""
        assert validate_tag("") is False

    def test_invalid_uppercase_tag(self):
        """@trace SPEC-07.12.01 - Uppercase tags MUST be invalid."""
        assert validate_tag("UPPERCASE") is False

    def test_invalid_special_chars(self):
        """@trace SPEC-07.12.01 - Special character tags MUST be invalid."""
        assert validate_tag("tag with space") is False
        assert validate_tag("tag@special") is False


class TestNormalizeTag:
    """Tests for tag normalization."""

    def test_normalize_lowercase(self):
        """normalize_tag MUST convert to lowercase."""
        assert normalize_tag("UPPERCASE") == "uppercase"
        assert normalize_tag("MixedCase") == "mixedcase"

    def test_normalize_trim(self):
        """normalize_tag MUST trim whitespace."""
        assert normalize_tag("  tag  ") == "tag"


class TestTagManager:
    """Tests for TagManager."""

    @pytest.fixture
    def manager(self) -> TagManager:
        """Create TagManager instance."""
        return TagManager()

    @pytest.fixture
    def checkpoint(self) -> Checkpoint:
        """Create test checkpoint."""
        return Checkpoint(
            id="cp-test-123",
            session_id="session-test",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp/workspace",
        )

    # =========================================================================
    # Tag Operations
    # =========================================================================

    def test_add_tag(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - add_tag MUST add normalized tag."""
        manager.add_tag(checkpoint, "Stable")

        assert "stable" in checkpoint.tags

    def test_add_tag_hierarchical(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - add_tag MUST support hierarchical tags."""
        manager.add_tag(checkpoint, "milestone/v1.0")

        assert "milestone/v1.0" in checkpoint.tags

    def test_add_tag_duplicate(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - add_tag MUST not duplicate tags."""
        manager.add_tag(checkpoint, "stable")
        manager.add_tag(checkpoint, "stable")

        assert checkpoint.tags.count("stable") == 1

    def test_add_tag_invalid(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - add_tag MUST reject invalid tags."""
        with pytest.raises(ValueError):
            manager.add_tag(checkpoint, "invalid tag!")

    def test_remove_tag(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - remove_tag MUST remove tag."""
        manager.add_tag(checkpoint, "stable")
        manager.remove_tag(checkpoint, "stable")

        assert "stable" not in checkpoint.tags

    def test_remove_tag_not_present(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - remove_tag MUST handle missing tag."""
        # Should not raise
        manager.remove_tag(checkpoint, "nonexistent")

    def test_has_tag(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - has_tag MUST check tag presence."""
        manager.add_tag(checkpoint, "stable")

        assert manager.has_tag(checkpoint, "stable") is True
        assert manager.has_tag(checkpoint, "unstable") is False

    def test_match_tag_pattern(self, manager, checkpoint):
        """@trace SPEC-07.12.01 - match_tag_pattern MUST support globs."""
        manager.add_tag(checkpoint, "milestone/v1.0")
        manager.add_tag(checkpoint, "milestone/v2.0")

        assert manager.match_tag_pattern(checkpoint, "milestone/*") is True
        assert manager.match_tag_pattern(checkpoint, "release/*") is False

    def test_filter_by_tag(self, manager):
        """@trace SPEC-07.12.01 - filter_by_tag MUST return matching checkpoints."""
        cp1 = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["stable", "milestone/v1.0"],
        )
        cp2 = Checkpoint(
            id="cp-2",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["experimental"],
        )
        cp3 = Checkpoint(
            id="cp-3",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["stable"],
        )

        checkpoints = [cp1, cp2, cp3]

        stable = manager.filter_by_tag(checkpoints, "stable")
        assert len(stable) == 2

        milestone = manager.filter_by_tag(checkpoints, "milestone/*")
        assert len(milestone) == 1

    def test_get_all_tags(self, manager):
        """@trace SPEC-07.12.01 - get_all_tags MUST return unique tags."""
        cp1 = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["stable", "milestone/v1.0"],
        )
        cp2 = Checkpoint(
            id="cp-2",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["stable", "experimental"],
        )

        all_tags = manager.get_all_tags([cp1, cp2])

        assert all_tags == {"stable", "milestone/v1.0", "experimental"}

    # =========================================================================
    # Annotation Operations
    # =========================================================================

    def test_add_annotation(self, manager, checkpoint):
        """@trace SPEC-07.12.02 - add_annotation MUST create annotation."""
        annotation = manager.add_annotation(
            checkpoint,
            text="This is a note",
            user="testuser",
        )

        assert annotation.text == "This is a note"
        assert annotation.user == "testuser"
        assert annotation.timestamp is not None
        assert len(checkpoint.annotations) == 1

    def test_annotations_append_only(self, manager, checkpoint):
        """@trace SPEC-07.12.02 - Annotations MUST be append-only."""
        manager.add_annotation(checkpoint, "First note", "user1")
        manager.add_annotation(checkpoint, "Second note", "user2")

        assert len(checkpoint.annotations) == 2
        # No method to remove annotations - append-only

    def test_get_annotations_chronological(self, manager, checkpoint):
        """@trace SPEC-07.12.02 - get_annotations MUST return chronologically."""
        manager.add_annotation(checkpoint, "First", "user")
        manager.add_annotation(checkpoint, "Second", "user")
        manager.add_annotation(checkpoint, "Third", "user")

        annotations = manager.get_annotations(checkpoint)

        assert annotations[0].text == "First"
        assert annotations[2].text == "Third"

    def test_search_annotations(self, manager):
        """@trace SPEC-07.12.02 - search_annotations MUST find matches."""
        cp1 = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            annotations=[
                CheckpointAnnotation(
                    timestamp=datetime.now(),
                    user="user",
                    text="Fixed the authentication bug",
                )
            ],
        )
        cp2 = Checkpoint(
            id="cp-2",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            annotations=[
                CheckpointAnnotation(
                    timestamp=datetime.now(),
                    user="user",
                    text="Added unit tests",
                )
            ],
        )

        results = manager.search_annotations([cp1, cp2], "authentication")

        assert len(results) == 1
        assert results[0][0].id == "cp-1"

    # =========================================================================
    # Issue Link Operations
    # =========================================================================

    def test_link_github_issue(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - link_issue MUST link GitHub URLs."""
        linked = manager.link_issue(
            checkpoint,
            "https://github.com/owner/repo/issues/123",
        )

        assert linked is not None
        assert linked.tracker == "github"
        assert linked.id == "123"
        assert len(checkpoint.linked_issues) == 1

    def test_link_linear_issue(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - link_issue MUST link Linear URLs."""
        linked = manager.link_issue(
            checkpoint,
            "https://linear.app/workspace/issue/PROJ-456",
        )

        assert linked is not None
        assert linked.tracker == "linear"
        assert linked.id == "PROJ-456"

    def test_link_beads_issue(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - link_issue MUST link Beads IDs."""
        linked = manager.link_issue(checkpoint, "ph-abc123")

        assert linked is not None
        assert linked.tracker == "beads"
        assert linked.id == "ph-abc123"

    def test_link_issue_duplicate(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - link_issue MUST not duplicate links."""
        manager.link_issue(checkpoint, "ph-abc123")
        manager.link_issue(checkpoint, "ph-abc123")

        assert len(checkpoint.linked_issues) == 1

    def test_link_issue_invalid(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - link_issue MUST return None for invalid."""
        result = manager.link_issue(checkpoint, "not-a-valid-ref!")
        assert result is None

    def test_unlink_issue(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - unlink_issue MUST remove link."""
        manager.link_issue(checkpoint, "ph-abc123")
        result = manager.unlink_issue(checkpoint, "ph-abc123")

        assert result is True
        assert len(checkpoint.linked_issues) == 0

    def test_unlink_issue_not_found(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - unlink_issue MUST return False if not found."""
        result = manager.unlink_issue(checkpoint, "ph-nonexistent")
        assert result is False

    def test_get_linked_issues(self, manager, checkpoint):
        """@trace SPEC-07.12.03 - get_linked_issues MUST return all links."""
        manager.link_issue(checkpoint, "ph-issue1")
        manager.link_issue(checkpoint, "ph-issue2")

        issues = manager.get_linked_issues(checkpoint)

        assert len(issues) == 2

    def test_filter_by_issue(self, manager):
        """@trace SPEC-07.12.03 - filter_by_issue MUST return matching checkpoints."""
        cp1 = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            linked_issues=[LinkedIssue(tracker="github", id="123", url="...")],
        )
        cp2 = Checkpoint(
            id="cp-2",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            linked_issues=[LinkedIssue(tracker="beads", id="ph-abc", url=None)],
        )

        github_cps = manager.filter_by_issue([cp1, cp2], tracker="github")
        assert len(github_cps) == 1
        assert github_cps[0].id == "cp-1"

        beads_cps = manager.filter_by_issue([cp1, cp2], issue_id="ph-abc")
        assert len(beads_cps) == 1
        assert beads_cps[0].id == "cp-2"

    # =========================================================================
    # Combined Search
    # =========================================================================

    def test_search_by_tag(self, manager):
        """search MUST find checkpoints by tag."""
        cp = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["milestone/v1.0"],
        )

        results = manager.search([cp], "milestone")
        assert len(results) == 1

    def test_search_by_annotation(self, manager):
        """search MUST find checkpoints by annotation text."""
        cp = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            annotations=[
                CheckpointAnnotation(
                    timestamp=datetime.now(),
                    user="user",
                    text="Fixed authentication bug",
                )
            ],
        )

        results = manager.search([cp], "authentication")
        assert len(results) == 1

    def test_search_no_duplicates(self, manager):
        """search MUST not return duplicates."""
        cp = Checkpoint(
            id="cp-1",
            session_id="session",
            trigger=CheckpointTrigger.MANUAL,
            working_directory="/tmp",
            tags=["auth"],
            annotations=[
                CheckpointAnnotation(
                    timestamp=datetime.now(),
                    user="user",
                    text="Authentication fix",
                )
            ],
        )

        results = manager.search([cp], "auth")
        assert len(results) == 1  # Not 2


class TestExtractIssueReferences:
    """Tests for extract_issue_references function."""

    def test_extract_github_urls(self):
        """extract_issue_references MUST find GitHub URLs."""
        text = "Fixed https://github.com/owner/repo/issues/123 today"
        refs = extract_issue_references(text)

        assert len(refs) == 1
        assert refs[0].tracker == "github"
        assert refs[0].id == "123"

    def test_extract_github_shorthand(self):
        """extract_issue_references MUST find GitHub shorthand."""
        text = "Related to owner/repo#456"
        refs = extract_issue_references(text)

        assert len(refs) == 1
        assert refs[0].tracker == "github"
        assert refs[0].id == "456"

    def test_extract_linear_urls(self):
        """extract_issue_references MUST find Linear URLs."""
        text = "Tracking at https://linear.app/team/issue/PROJ-789"
        refs = extract_issue_references(text)

        assert len(refs) == 1
        assert refs[0].tracker == "linear"
        assert refs[0].id == "PROJ-789"

    def test_extract_beads_ids(self):
        """extract_issue_references MUST find Beads IDs."""
        text = "Working on ph-abc123 and ph-def456"
        refs = extract_issue_references(text)

        assert len(refs) == 2
        assert all(r.tracker == "beads" for r in refs)

    def test_extract_multiple_types(self):
        """extract_issue_references MUST find all types."""
        text = """
        GitHub: https://github.com/owner/repo/issues/1
        Linear: https://linear.app/team/issue/PROJ-2
        Beads: ph-xyz789
        """
        refs = extract_issue_references(text)

        trackers = {r.tracker for r in refs}
        assert trackers == {"github", "linear", "beads"}

    def test_extract_no_duplicates(self):
        """extract_issue_references MUST not duplicate."""
        text = "See ph-abc123, also ph-abc123 again"
        refs = extract_issue_references(text)

        assert len(refs) == 1
