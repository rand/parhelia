"""Tests for issue tracker integration.

@trace SPEC-07.22.01 - Adapter Interface
@trace SPEC-07.22.02 - GitHub Adapter
@trace SPEC-07.22.03 - Linear Adapter
@trace SPEC-07.22.04 - Beads Adapter
@trace SPEC-07.22.05 - Auto-Linking
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.issue_tracker import (
    BeadsAdapter,
    BeadsConfig,
    GitHubAdapter,
    GitHubConfig,
    Issue,
    IssueTrackerManager,
    LinearAdapter,
    LinearConfig,
    detect_issue_from_git_branch,
    detect_issue_from_prompt,
    detect_issue_from_session_name,
    parse_issue_tracker_config,
)


class AsyncContextManagerMock:
    """Helper for mocking async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_creation(self):
        """Issue MUST capture all fields."""
        issue = Issue(
            tracker="github",
            id="owner/repo#123",
            title="Test Issue",
            status="open",
            url="https://github.com/owner/repo/issues/123",
            description="Test description",
            labels=["bug", "priority:high"],
            assignee="testuser",
        )

        assert issue.tracker == "github"
        assert issue.id == "owner/repo#123"
        assert issue.title == "Test Issue"
        assert issue.status == "open"
        assert len(issue.labels) == 2


class TestGitHubAdapter:
    """Tests for GitHub adapter."""

    @pytest.fixture
    def adapter(self) -> GitHubAdapter:
        """Create GitHub adapter with test config."""
        return GitHubAdapter(
            GitHubConfig(
                token="test-token",
                default_owner="testowner",
                default_repo="testrepo",
            )
        )

    def test_tracker_name(self, adapter):
        """@trace SPEC-07.22.02 - GitHub adapter MUST identify as 'github'."""
        assert adapter.tracker_name == "github"

    def test_parse_issue_id_full(self, adapter):
        """@trace SPEC-07.22.02 - GitHub adapter MUST parse owner/repo#num."""
        owner, repo, num = adapter._parse_issue_id("owner/repo#123")
        assert owner == "owner"
        assert repo == "repo"
        assert num == "123"

    def test_parse_issue_id_default(self, adapter):
        """@trace SPEC-07.22.02 - GitHub adapter MUST use default repo."""
        owner, repo, num = adapter._parse_issue_id("456")
        assert owner == "testowner"
        assert repo == "testrepo"
        assert num == "456"

    def test_parse_issue_id_no_default(self):
        """@trace SPEC-07.22.02 - GitHub adapter MUST error without default."""
        adapter = GitHubAdapter(GitHubConfig(token="test"))
        with pytest.raises(ValueError):
            adapter._parse_issue_id("123")

    @pytest.mark.asyncio
    async def test_get_issue_success(self, adapter):
        """@trace SPEC-07.22.02 - GitHub adapter MUST fetch issues."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "title": "Test Issue",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/123",
                "body": "Description",
                "labels": [{"name": "bug"}],
                "assignee": {"login": "user"},
                "created_at": "2026-01-20T10:00:00Z",
                "updated_at": "2026-01-20T11:00:00Z",
            }
        )

        mock_session_instance = MagicMock()
        mock_session_instance.get = MagicMock(
            return_value=AsyncContextManagerMock(mock_response)
        )

        with patch(
            "parhelia.issue_tracker.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(
                mock_session_instance
            )

            issue = await adapter.get_issue("123")

            assert issue is not None
            assert issue.title == "Test Issue"
            assert issue.status == "open"

    @pytest.mark.asyncio
    async def test_get_issue_not_found(self, adapter):
        """@trace SPEC-07.22.02 - GitHub adapter MUST return None if not found."""
        mock_response = MagicMock()
        mock_response.status = 404

        mock_session_instance = MagicMock()
        mock_session_instance.get = MagicMock(
            return_value=AsyncContextManagerMock(mock_response)
        )

        with patch(
            "parhelia.issue_tracker.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(
                mock_session_instance
            )

            issue = await adapter.get_issue("999")
            assert issue is None


class TestLinearAdapter:
    """Tests for Linear adapter."""

    @pytest.fixture
    def adapter(self) -> LinearAdapter:
        """Create Linear adapter with test config."""
        return LinearAdapter(
            LinearConfig(
                api_key="test-api-key",
                default_team="TEAM-123",
            )
        )

    def test_tracker_name(self, adapter):
        """@trace SPEC-07.22.03 - Linear adapter MUST identify as 'linear'."""
        assert adapter.tracker_name == "linear"

    @pytest.mark.asyncio
    async def test_get_issue_success(self, adapter):
        """@trace SPEC-07.22.03 - Linear adapter MUST fetch issues via GraphQL."""
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "data": {
                    "issue": {
                        "id": "issue-uuid",
                        "identifier": "PROJ-123",
                        "title": "Test Linear Issue",
                        "description": "Description",
                        "url": "https://linear.app/team/issue/PROJ-123",
                        "state": {"name": "In Progress"},
                        "labels": {"nodes": [{"name": "feature"}]},
                        "assignee": {"name": "Test User"},
                        "createdAt": "2026-01-20T10:00:00Z",
                        "updatedAt": "2026-01-20T11:00:00Z",
                    }
                }
            }
        )

        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(
            return_value=AsyncContextManagerMock(mock_response)
        )

        with patch(
            "parhelia.issue_tracker.aiohttp.ClientSession"
        ) as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(
                mock_session_instance
            )

            issue = await adapter.get_issue("PROJ-123")

            assert issue is not None
            assert issue.id == "PROJ-123"
            assert issue.title == "Test Linear Issue"
            assert issue.status == "in progress"


class TestBeadsAdapter:
    """Tests for Beads adapter."""

    @pytest.fixture
    def adapter(self) -> BeadsAdapter:
        """Create Beads adapter."""
        return BeadsAdapter(BeadsConfig())

    def test_tracker_name(self, adapter):
        """@trace SPEC-07.22.04 - Beads adapter MUST identify as 'beads'."""
        assert adapter.tracker_name == "beads"

    @pytest.mark.asyncio
    async def test_get_issue_success(self, adapter):
        """@trace SPEC-07.22.04 - Beads adapter MUST fetch issues via bd CLI."""
        with patch.object(adapter, "_run_bd") as mock_bd:
            mock_bd.return_value = """{
                "id": "ph-abc123",
                "title": "Test Beads Issue",
                "status": "open",
                "description": "Description",
                "labels": ["bug"],
                "assignee": "testuser"
            }"""

            issue = await adapter.get_issue("ph-abc123")

            assert issue is not None
            assert issue.id == "ph-abc123"
            assert issue.title == "Test Beads Issue"
            assert issue.status == "open"

    @pytest.mark.asyncio
    async def test_get_issue_not_found(self, adapter):
        """@trace SPEC-07.22.04 - Beads adapter MUST return None if bd fails."""
        with patch.object(adapter, "_run_bd") as mock_bd:
            mock_bd.return_value = None

            issue = await adapter.get_issue("ph-nonexistent")
            assert issue is None

    @pytest.mark.asyncio
    async def test_update_issue(self, adapter):
        """@trace SPEC-07.22.04 - Beads adapter MUST update issues via bd CLI."""
        with patch.object(adapter, "_run_bd") as mock_bd:
            mock_bd.return_value = ""

            result = await adapter.update_issue(
                "ph-abc123", status="in_progress", comment="Test note"
            )

            assert result is True
            assert mock_bd.call_count == 2  # status and notes


class TestAutoLinking:
    """Tests for auto-linking functions."""

    def test_detect_from_session_name_beads(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find Beads IDs in session name."""
        # Beads IDs are prefix-alphanumeric (e.g., ph-abc123), not long hyphenated strings
        # From "ph-abc123-fix-auth", extract "ph-abc123" (the beads ID part)
        assert detect_issue_from_session_name("ph-abc123-fix-auth") == "ph-abc123"
        # Check it finds beads pattern in middle of string
        result = detect_issue_from_session_name("session-ph-xyz789")
        assert result == "ph-xyz789"

    def test_detect_from_session_name_linear(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find Linear IDs in session name."""
        result = detect_issue_from_session_name("PROJ-123-fix-bug")
        assert result == "PROJ-123"

    def test_detect_from_session_name_issue_number(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find issue numbers."""
        result = detect_issue_from_session_name("fix-issue-456")
        assert result == "456"

        result = detect_issue_from_session_name("fix-issue_789")
        assert result == "789"

    def test_detect_from_session_name_hash(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find #N patterns."""
        result = detect_issue_from_session_name("session-#123-desc")
        assert result == "123"

    def test_detect_from_session_name_none(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST return None if no match."""
        result = detect_issue_from_session_name("random-session-name")
        assert result is None

    def test_detect_from_prompt_fixes(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find 'Fix #N' patterns."""
        refs = detect_issue_from_prompt("Fix #123 and close #456")
        assert "123" in refs
        assert "456" in refs

    def test_detect_from_prompt_beads(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find Beads IDs in prompt."""
        refs = detect_issue_from_prompt("Working on ph-abc123")
        assert "ph-abc123" in refs

    def test_detect_from_prompt_linear(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find Linear IDs in prompt."""
        refs = detect_issue_from_prompt("Implementing PROJ-456")
        assert "PROJ-456" in refs

    def test_detect_from_prompt_github_url(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find GitHub URLs in prompt."""
        refs = detect_issue_from_prompt(
            "See https://github.com/owner/repo/issues/789"
        )
        assert "789" in refs

    def test_detect_from_prompt_multiple(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find all references."""
        refs = detect_issue_from_prompt(
            "Fix #123, relates to ph-abc and PROJ-456"
        )
        assert len(refs) >= 3

    def test_detect_from_prompt_deduplicates(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST deduplicate."""
        refs = detect_issue_from_prompt("Fix #123 and #123 again")
        assert refs.count("123") == 1

    def test_detect_from_git_branch(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST check git branch."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="fix/issue-123\n"
            )

            result = detect_issue_from_git_branch()
            assert result == "123"

    def test_detect_from_git_branch_beads(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST find Beads in branch."""
        with patch("subprocess.run") as mock_run:
            # Branch name includes beads ID followed by description
            mock_run.return_value = MagicMock(
                returncode=0, stdout="ph-abc123-fix-auth\n"
            )

            result = detect_issue_from_git_branch()
            # Beads IDs are prefix-alphanumeric (ph-abc123), extract just that part
            assert result == "ph-abc123"

    def test_detect_from_git_branch_not_git(self):
        """@trace SPEC-07.22.05 - Auto-detect MUST handle non-git dirs."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128, stdout="")

            result = detect_issue_from_git_branch()
            assert result is None


class TestIssueTrackerManager:
    """Tests for IssueTrackerManager."""

    @pytest.fixture
    def manager(self) -> IssueTrackerManager:
        """Create manager with mock adapters."""
        mgr = IssueTrackerManager()

        # Register mock GitHub adapter
        github = MagicMock()
        github.tracker_name = "github"
        mgr.register_adapter(github)

        # Register mock Beads adapter
        beads = MagicMock()
        beads.tracker_name = "beads"
        mgr.register_adapter(beads)

        return mgr

    def test_register_adapter(self, manager):
        """Manager MUST register adapters."""
        assert "github" in manager.adapters
        assert "beads" in manager.adapters

    def test_get_adapter(self, manager):
        """Manager MUST return adapter by name."""
        adapter = manager.get_adapter("github")
        assert adapter is not None
        assert adapter.tracker_name == "github"

    def test_get_adapter_not_found(self, manager):
        """Manager MUST return None for unknown tracker."""
        adapter = manager.get_adapter("unknown")
        assert adapter is None

    def test_detect_tracker_beads(self, manager):
        """Manager MUST detect Beads references."""
        assert manager.detect_tracker("ph-abc123") == "beads"
        assert manager.detect_tracker("bd-xyz") == "beads"

    def test_detect_tracker_linear(self, manager):
        """Manager MUST detect Linear references."""
        assert manager.detect_tracker("PROJ-123") == "linear"
        assert manager.detect_tracker("TEAM-456") == "linear"

    def test_detect_tracker_github(self, manager):
        """Manager MUST detect GitHub references."""
        assert manager.detect_tracker("123") == "github"
        assert manager.detect_tracker("owner/repo#456") == "github"

    def test_detect_tracker_unknown(self, manager):
        """Manager MUST return None for unknown references."""
        assert manager.detect_tracker("not-a-valid-ref!") is None

    @pytest.mark.asyncio
    async def test_get_issue(self, manager):
        """Manager MUST route to correct adapter."""
        manager.adapters["beads"].get_issue = AsyncMock(
            return_value=Issue(
                tracker="beads",
                id="ph-abc123",
                title="Test",
                status="open",
            )
        )

        issue = await manager.get_issue("ph-abc123")

        assert issue is not None
        assert issue.id == "ph-abc123"
        manager.adapters["beads"].get_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_checkpoint_to_issues(self, manager):
        """Manager MUST link checkpoint to multiple issues."""
        manager.adapters["beads"].link_checkpoint = AsyncMock(return_value=True)
        manager.adapters["github"].link_checkpoint = AsyncMock(return_value=True)

        results = await manager.link_checkpoint_to_issues(
            references=["ph-abc123", "123"],
            checkpoint_id="cp-xyz",
            session_id="session-test",
        )

        assert results["ph-abc123"] is True
        assert results["123"] is True

    def test_auto_detect_issues(self, manager):
        """Manager MUST auto-detect from multiple sources."""
        with patch(
            "parhelia.issue_tracker.detect_issue_from_git_branch"
        ) as mock_git:
            mock_git.return_value = "789"

            refs = manager.auto_detect_issues(
                session_name="ph-abc123-session",
                prompt="Fix #456",
                check_git=True,
            )

            # Should find references from all sources
            assert "ph-abc123" in refs
            assert "456" in refs
            assert "789" in refs


class TestParseConfig:
    """Tests for parse_issue_tracker_config."""

    def test_parse_empty_config(self):
        """Parse empty config should return manager with no adapters."""
        manager = parse_issue_tracker_config({})
        # Beads is enabled by default
        assert "beads" in manager.adapters

    def test_parse_github_config(self):
        """Parse GitHub config should register adapter."""
        config = {
            "github": {
                "token": "test-token",
                "default_repo": "owner/repo",
            }
        }

        manager = parse_issue_tracker_config(config)

        assert "github" in manager.adapters
        adapter = manager.adapters["github"]
        assert isinstance(adapter, GitHubAdapter)
        assert adapter.config.default_owner == "owner"
        assert adapter.config.default_repo == "repo"

    def test_parse_linear_config(self):
        """Parse Linear config should register adapter."""
        config = {
            "linear": {
                "api_key": "test-key",
                "default_team": "TEAM-123",
            }
        }

        manager = parse_issue_tracker_config(config)

        assert "linear" in manager.adapters
        adapter = manager.adapters["linear"]
        assert isinstance(adapter, LinearAdapter)

    def test_parse_beads_disabled(self):
        """Parse config can disable Beads."""
        config = {"beads": {"enabled": False}}

        manager = parse_issue_tracker_config(config)

        assert "beads" not in manager.adapters
