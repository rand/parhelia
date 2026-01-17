"""Workspace cloning and synchronization.

Implements workspace management for remote Claude Code execution.
Clones git repositories to /vol/parhelia/workspaces/ with hash-based naming.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class WorkspaceInfo:
    """Information about a workspace.

    Captures metadata for a cloned repository workspace.
    """

    path: str
    repo_url: str
    branch: str
    commit_hash: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_sync: datetime | None = None


@dataclass
class CloneResult:
    """Result from clone operation."""

    success: bool
    workspace_path: str | None = None
    error: str | None = None
    commit_hash: str | None = None


@dataclass
class SyncResult:
    """Result from sync operation."""

    success: bool
    error: str | None = None
    commits_pulled: int = 0


class WorkspaceManager:
    """Manage workspace cloning and synchronization.

    Handles:
    - Cloning git repositories to volume storage
    - Branch checkout
    - Incremental sync (git pull)
    - Hash-based directory naming for uniqueness
    """

    DEFAULT_WORKSPACE_ROOT = "/vol/parhelia/workspaces"

    def __init__(self, workspace_root: str | None = None):
        """Initialize the workspace manager.

        Args:
            workspace_root: Root directory for workspaces.
        """
        self.workspace_root = workspace_root or self.DEFAULT_WORKSPACE_ROOT

    def generate_workspace_path(
        self,
        repo_url: str,
        branch: str | None = None,
    ) -> str:
        """Generate deterministic workspace path from repo URL.

        Uses SHA-256 hash of repo URL (and branch if specified) to create
        a unique, deterministic path for the workspace.

        Args:
            repo_url: The git repository URL.
            branch: Optional branch name to include in hash.

        Returns:
            Full path to workspace directory.
        """
        # Create hash input from URL and optional branch
        hash_input = repo_url
        if branch:
            hash_input = f"{repo_url}@{branch}"

        # Generate short hash for directory name
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

        # Extract repo name for readability
        repo_name = self._extract_repo_name(repo_url)

        # Combine for final path: {root}/{repo-name}-{hash}
        dir_name = f"{repo_name}-{hash_digest}"
        return str(Path(self.workspace_root) / dir_name)

    def _extract_repo_name(self, repo_url: str) -> str:
        """Extract repository name from URL.

        Args:
            repo_url: The git repository URL.

        Returns:
            Repository name (without .git suffix).
        """
        # Handle various URL formats
        # https://github.com/user/repo.git -> repo
        # git@github.com:user/repo.git -> repo
        name = repo_url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name

    async def clone_repo(
        self,
        repo_url: str,
        branch: str | None = None,
        depth: int | None = None,
    ) -> CloneResult:
        """Clone a git repository to workspace.

        Args:
            repo_url: The git repository URL to clone.
            branch: Optional branch to checkout.
            depth: Optional shallow clone depth.

        Returns:
            CloneResult with success status and workspace path.
        """
        workspace_path = self.generate_workspace_path(repo_url, branch)

        # Build git clone command
        cmd = ["git", "clone"]

        if branch:
            cmd.extend(["-b", branch])

        if depth:
            cmd.extend(["--depth", str(depth)])

        cmd.extend([repo_url, workspace_path])

        # Ensure parent directory exists
        Path(workspace_path).parent.mkdir(parents=True, exist_ok=True)

        # Execute clone
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await proc.communicate()
        exit_code = await proc.wait()

        if exit_code != 0:
            error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Clone failed"
            return CloneResult(
                success=False,
                error=error_msg,
            )

        # Get current commit hash
        commit_hash = await self._get_commit_hash(workspace_path)

        return CloneResult(
            success=True,
            workspace_path=workspace_path,
            commit_hash=commit_hash,
        )

    async def sync_workspace(self, workspace_path: str) -> SyncResult:
        """Sync workspace with remote (git pull).

        Args:
            workspace_path: Path to the workspace to sync.

        Returns:
            SyncResult with success status.
        """
        # Verify it's a git repository
        git_dir = Path(workspace_path) / ".git"
        if not git_dir.exists():
            return SyncResult(
                success=False,
                error="Not a git repository",
            )

        # Execute git pull
        proc = await asyncio.create_subprocess_exec(
            "git", "pull", "--ff-only",
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        exit_code = await proc.wait()

        if exit_code != 0:
            error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Sync failed"
            # Check for merge conflict
            if b"conflict" in (stderr or b"").lower() or b"conflict" in (stdout or b"").lower():
                error_msg = "Merge conflict detected"
            return SyncResult(
                success=False,
                error=error_msg,
            )

        return SyncResult(success=True)

    async def checkout_branch(
        self,
        workspace_path: str,
        branch: str,
    ) -> SyncResult:
        """Checkout a specific branch.

        Args:
            workspace_path: Path to the workspace.
            branch: Branch name to checkout.

        Returns:
            SyncResult with success status.
        """
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", branch,
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await proc.communicate()
        exit_code = await proc.wait()

        if exit_code != 0:
            error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Checkout failed"
            return SyncResult(
                success=False,
                error=error_msg,
            )

        return SyncResult(success=True)

    async def get_workspace_info(self, workspace_path: str) -> WorkspaceInfo | None:
        """Get information about a workspace.

        Args:
            workspace_path: Path to the workspace.

        Returns:
            WorkspaceInfo or None if not a valid workspace.
        """
        git_dir = Path(workspace_path) / ".git"
        if not git_dir.exists():
            return None

        # Get remote URL
        repo_url = await self._get_remote_url(workspace_path)

        # Get current branch
        branch = await self._get_current_branch(workspace_path)

        # Get current commit
        commit_hash = await self._get_commit_hash(workspace_path)

        return WorkspaceInfo(
            path=workspace_path,
            repo_url=repo_url or "",
            branch=branch or "unknown",
            commit_hash=commit_hash,
        )

    async def _get_commit_hash(self, workspace_path: str) -> str | None:
        """Get current HEAD commit hash."""
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "HEAD",
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            return stdout.decode().strip()
        return None

    async def _get_current_branch(self, workspace_path: str) -> str | None:
        """Get current branch name."""
        proc = await asyncio.create_subprocess_exec(
            "git", "rev-parse", "--abbrev-ref", "HEAD",
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            return stdout.decode().strip()
        return None

    async def _get_remote_url(self, workspace_path: str) -> str | None:
        """Get remote origin URL."""
        proc = await asyncio.create_subprocess_exec(
            "git", "remote", "get-url", "origin",
            cwd=workspace_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            return stdout.decode().strip()
        return None
