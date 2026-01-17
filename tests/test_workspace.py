"""Tests for workspace cloning and sync.

@trace SPEC-01.12 - Volume Mounting (workspaces directory)
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestWorkspaceManager:
    """Tests for WorkspaceManager class."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create temporary workspace root."""
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        return ws_root

    @pytest.fixture
    def manager(self, workspace_root: Path):
        """Create WorkspaceManager instance."""
        from parhelia.workspace import WorkspaceManager

        return WorkspaceManager(workspace_root=str(workspace_root))

    def test_manager_initialization(self, manager, workspace_root):
        """WorkspaceManager MUST initialize with workspace root."""
        assert manager.workspace_root == str(workspace_root)

    def test_generate_workspace_path(self, manager):
        """Workspace path MUST be hash-based for uniqueness."""
        repo_url = "https://github.com/user/repo.git"

        path = manager.generate_workspace_path(repo_url)

        # Should contain hash of repo URL
        assert path is not None
        assert manager.workspace_root in path

    def test_generate_workspace_path_deterministic(self, manager):
        """Same repo URL MUST produce same workspace path."""
        repo_url = "https://github.com/user/repo.git"

        path1 = manager.generate_workspace_path(repo_url)
        path2 = manager.generate_workspace_path(repo_url)

        assert path1 == path2

    def test_generate_workspace_path_different_for_different_repos(self, manager):
        """Different repos MUST produce different paths."""
        path1 = manager.generate_workspace_path("https://github.com/user/repo1.git")
        path2 = manager.generate_workspace_path("https://github.com/user/repo2.git")

        assert path1 != path2

    def test_generate_workspace_path_with_branch(self, manager):
        """Branch name SHOULD affect workspace path."""
        repo_url = "https://github.com/user/repo.git"

        path_main = manager.generate_workspace_path(repo_url, branch="main")
        path_dev = manager.generate_workspace_path(repo_url, branch="develop")

        assert path_main != path_dev


class TestWorkspaceCloning:
    """Tests for workspace cloning functionality."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create temporary workspace root."""
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        return ws_root

    @pytest.fixture
    def manager(self, workspace_root: Path):
        """Create WorkspaceManager instance."""
        from parhelia.workspace import WorkspaceManager

        return WorkspaceManager(workspace_root=str(workspace_root))

    @pytest.mark.asyncio
    async def test_clone_creates_directory(self, manager, workspace_root):
        """clone_repo MUST create workspace directory."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await manager.clone_repo(
                repo_url="https://github.com/user/repo.git"
            )

            assert result.success is True
            assert result.workspace_path is not None

    @pytest.mark.asyncio
    async def test_clone_with_branch(self, manager):
        """clone_repo MUST support branch specification."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"abc123", b""))
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await manager.clone_repo(
                repo_url="https://github.com/user/repo.git",
                branch="feature-branch",
            )

            # Verify git clone was called with branch (first call)
            first_call = mock_exec.call_args_list[0]
            cmd_list = list(first_call[0])
            assert "-b" in cmd_list

    @pytest.mark.asyncio
    async def test_clone_failure_returns_error(self, manager):
        """clone_repo MUST return error on failure."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Repository not found"))
            mock_proc.wait = AsyncMock(return_value=128)  # Git error
            mock_proc.returncode = 128
            mock_exec.return_value = mock_proc

            result = await manager.clone_repo(
                repo_url="https://github.com/user/nonexistent.git"
            )

            assert result.success is False
            assert result.error is not None


class TestWorkspaceSync:
    """Tests for workspace sync functionality."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path) -> Path:
        """Create temporary workspace root."""
        ws_root = tmp_path / "workspaces"
        ws_root.mkdir()
        return ws_root

    @pytest.fixture
    def manager(self, workspace_root: Path):
        """Create WorkspaceManager instance."""
        from parhelia.workspace import WorkspaceManager

        return WorkspaceManager(workspace_root=str(workspace_root))

    @pytest.mark.asyncio
    async def test_sync_pulls_changes(self, manager, workspace_root):
        """sync_workspace MUST pull latest changes."""
        # Create fake workspace
        ws_path = workspace_root / "test-workspace"
        ws_path.mkdir()
        (ws_path / ".git").mkdir()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"Already up to date.", b""))
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await manager.sync_workspace(str(ws_path))

            assert result.success is True

    @pytest.mark.asyncio
    async def test_sync_handles_conflicts(self, manager, workspace_root):
        """sync_workspace MUST handle merge conflicts gracefully."""
        ws_path = workspace_root / "test-workspace"
        ws_path.mkdir()
        (ws_path / ".git").mkdir()

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Merge conflict in file.txt"))
            mock_proc.wait = AsyncMock(return_value=1)  # Conflict
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            result = await manager.sync_workspace(str(ws_path))

            assert result.success is False
            assert result.error is not None


class TestWorkspaceInfo:
    """Tests for WorkspaceInfo dataclass."""

    def test_workspace_info_creation(self):
        """WorkspaceInfo MUST capture workspace metadata."""
        from parhelia.workspace import WorkspaceInfo

        info = WorkspaceInfo(
            path="/vol/parhelia/workspaces/abc123",
            repo_url="https://github.com/user/repo.git",
            branch="main",
            commit_hash="abc123def456",
        )

        assert info.path == "/vol/parhelia/workspaces/abc123"
        assert info.repo_url == "https://github.com/user/repo.git"
        assert info.branch == "main"
        assert info.commit_hash == "abc123def456"


class TestCloneResult:
    """Tests for CloneResult dataclass."""

    def test_clone_result_success(self):
        """CloneResult MUST indicate success."""
        from parhelia.workspace import CloneResult

        result = CloneResult(
            success=True,
            workspace_path="/path/to/workspace",
        )

        assert result.success is True
        assert result.workspace_path == "/path/to/workspace"
        assert result.error is None

    def test_clone_result_failure(self):
        """CloneResult MUST capture error on failure."""
        from parhelia.workspace import CloneResult

        result = CloneResult(
            success=False,
            error="Repository not found",
        )

        assert result.success is False
        assert result.error == "Repository not found"
