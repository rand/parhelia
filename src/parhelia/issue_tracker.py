"""Issue tracker integration.

Implements:
- [SPEC-07.22.01] Adapter Interface
- [SPEC-07.22.02] GitHub Adapter
- [SPEC-07.22.03] Linear Adapter
- [SPEC-07.22.04] Beads Adapter
- [SPEC-07.22.05] Auto-Linking
"""

from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

import aiohttp


@dataclass
class Issue:
    """Represents an issue from any tracker."""

    tracker: str
    id: str
    title: str
    status: str
    url: str | None = None
    description: str | None = None
    labels: list[str] = field(default_factory=list)
    assignee: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class IssueTrackerAdapter(Protocol):
    """Interface for issue tracker integrations.

    Implements [SPEC-07.22.01].
    """

    @property
    def tracker_name(self) -> str:
        """Get the tracker name."""
        ...

    async def get_issue(self, issue_id: str) -> Issue | None:
        """Fetch issue details.

        Args:
            issue_id: The issue ID to fetch.

        Returns:
            Issue details or None if not found.
        """
        ...

    async def update_issue(
        self,
        issue_id: str,
        status: str | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update issue status or add comment.

        Args:
            issue_id: The issue to update.
            status: New status (if supported).
            comment: Comment to add.

        Returns:
            True if update succeeded.
        """
        ...

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create new issue.

        Args:
            title: Issue title.
            body: Issue body/description.
            labels: Optional labels.

        Returns:
            Created issue.
        """
        ...

    async def link_checkpoint(
        self,
        issue_id: str,
        checkpoint_id: str,
        session_id: str,
    ) -> bool:
        """Add checkpoint reference to issue.

        Args:
            issue_id: The issue to link.
            checkpoint_id: Checkpoint ID.
            session_id: Session ID.

        Returns:
            True if link succeeded.
        """
        ...


@dataclass
class GitHubConfig:
    """Configuration for GitHub adapter."""

    token: str
    default_owner: str | None = None
    default_repo: str | None = None
    api_base: str = "https://api.github.com"


class GitHubAdapter:
    """GitHub issue tracker adapter.

    Implements [SPEC-07.22.02].
    """

    def __init__(self, config: GitHubConfig):
        """Initialize GitHub adapter.

        Args:
            config: GitHub configuration.
        """
        self.config = config

    @property
    def tracker_name(self) -> str:
        return "github"

    def _parse_issue_id(self, issue_id: str) -> tuple[str, str, str]:
        """Parse issue ID into owner, repo, number.

        Formats:
        - "123" (uses default owner/repo)
        - "owner/repo#123"
        """
        # Try owner/repo#number format
        match = re.match(r"([^/]+)/([^#]+)#(\d+)", issue_id)
        if match:
            return match.group(1), match.group(2), match.group(3)

        # Use default owner/repo
        if self.config.default_owner and self.config.default_repo:
            return self.config.default_owner, self.config.default_repo, issue_id

        raise ValueError(f"Cannot parse issue ID without default repo: {issue_id}")

    async def get_issue(self, issue_id: str) -> Issue | None:
        """Fetch issue from GitHub.

        Args:
            issue_id: Issue ID (number or owner/repo#number).

        Returns:
            Issue details or None.
        """
        try:
            owner, repo, number = self._parse_issue_id(issue_id)
        except ValueError:
            return None

        url = f"{self.config.api_base}/repos/{owner}/{repo}/issues/{number}"
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

                    return Issue(
                        tracker="github",
                        id=f"{owner}/{repo}#{number}",
                        title=data["title"],
                        status="closed" if data["state"] == "closed" else "open",
                        url=data["html_url"],
                        description=data.get("body"),
                        labels=[l["name"] for l in data.get("labels", [])],
                        assignee=data.get("assignee", {}).get("login") if data.get("assignee") else None,
                        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
                    )
        except Exception:
            return None

    async def update_issue(
        self,
        issue_id: str,
        status: str | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update GitHub issue.

        Args:
            issue_id: Issue ID.
            status: "open" or "closed".
            comment: Comment to add.

        Returns:
            True if update succeeded.
        """
        try:
            owner, repo, number = self._parse_issue_id(issue_id)
        except ValueError:
            return False

        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        success = True

        try:
            async with aiohttp.ClientSession() as session:
                # Update status if provided
                if status:
                    state = "closed" if status == "closed" else "open"
                    url = f"{self.config.api_base}/repos/{owner}/{repo}/issues/{number}"
                    async with session.patch(
                        url,
                        headers=headers,
                        json={"state": state},
                    ) as response:
                        if response.status != 200:
                            success = False

                # Add comment if provided
                if comment:
                    url = f"{self.config.api_base}/repos/{owner}/{repo}/issues/{number}/comments"
                    async with session.post(
                        url,
                        headers=headers,
                        json={"body": comment},
                    ) as response:
                        if response.status != 201:
                            success = False

        except Exception:
            return False

        return success

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create GitHub issue.

        Args:
            title: Issue title.
            body: Issue body.
            labels: Optional labels.

        Returns:
            Created issue.

        Raises:
            ValueError: If no default repo configured.
            RuntimeError: If creation fails.
        """
        if not self.config.default_owner or not self.config.default_repo:
            raise ValueError("Default owner/repo required for issue creation")

        owner = self.config.default_owner
        repo = self.config.default_repo

        url = f"{self.config.api_base}/repos/{owner}/{repo}/issues"
        headers = {
            "Authorization": f"Bearer {self.config.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 201:
                    raise RuntimeError(f"Failed to create issue: {response.status}")

                data = await response.json()

                return Issue(
                    tracker="github",
                    id=f"{owner}/{repo}#{data['number']}",
                    title=data["title"],
                    status="open",
                    url=data["html_url"],
                    description=data.get("body"),
                    labels=[l["name"] for l in data.get("labels", [])],
                )

    async def link_checkpoint(
        self,
        issue_id: str,
        checkpoint_id: str,
        session_id: str,
    ) -> bool:
        """Link checkpoint to GitHub issue via comment.

        Args:
            issue_id: Issue ID.
            checkpoint_id: Checkpoint ID.
            session_id: Session ID.

        Returns:
            True if link succeeded.
        """
        comment = f"""**Parhelia Checkpoint Linked**

- Session: `{session_id}`
- Checkpoint: `{checkpoint_id}`
- Time: {datetime.now().isoformat()}

To review: `parhelia session review {session_id}`
"""
        return await self.update_issue(issue_id, comment=comment)


@dataclass
class LinearConfig:
    """Configuration for Linear adapter."""

    api_key: str
    default_team: str | None = None
    api_base: str = "https://api.linear.app/graphql"


class LinearAdapter:
    """Linear issue tracker adapter.

    Implements [SPEC-07.22.03].
    """

    def __init__(self, config: LinearConfig):
        """Initialize Linear adapter.

        Args:
            config: Linear configuration.
        """
        self.config = config

    @property
    def tracker_name(self) -> str:
        return "linear"

    async def _graphql(self, query: str, variables: dict | None = None) -> dict:
        """Execute GraphQL query.

        Args:
            query: GraphQL query string.
            variables: Query variables.

        Returns:
            Response data.
        """
        headers = {
            "Authorization": self.config.api_key,
            "Content-Type": "application/json",
        }

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.api_base,
                headers=headers,
                json=payload,
            ) as response:
                data = await response.json()
                return data

    async def get_issue(self, issue_id: str) -> Issue | None:
        """Fetch issue from Linear.

        Args:
            issue_id: Issue ID (e.g., "PROJ-123").

        Returns:
            Issue details or None.
        """
        query = """
        query GetIssue($id: String!) {
            issue(id: $id) {
                id
                identifier
                title
                description
                url
                state {
                    name
                }
                labels {
                    nodes {
                        name
                    }
                }
                assignee {
                    name
                }
                createdAt
                updatedAt
            }
        }
        """

        try:
            result = await self._graphql(query, {"id": issue_id})
            issue_data = result.get("data", {}).get("issue")

            if not issue_data:
                return None

            return Issue(
                tracker="linear",
                id=issue_data["identifier"],
                title=issue_data["title"],
                status=issue_data["state"]["name"].lower(),
                url=issue_data["url"],
                description=issue_data.get("description"),
                labels=[l["name"] for l in issue_data.get("labels", {}).get("nodes", [])],
                assignee=issue_data.get("assignee", {}).get("name") if issue_data.get("assignee") else None,
                created_at=datetime.fromisoformat(issue_data["createdAt"].replace("Z", "+00:00")),
                updated_at=datetime.fromisoformat(issue_data["updatedAt"].replace("Z", "+00:00")),
            )
        except Exception:
            return None

    async def update_issue(
        self,
        issue_id: str,
        status: str | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update Linear issue.

        Args:
            issue_id: Issue ID.
            status: Status name (e.g., "Done", "In Progress").
            comment: Comment to add.

        Returns:
            True if update succeeded.
        """
        success = True

        try:
            # Add comment if provided
            if comment:
                mutation = """
                mutation CreateComment($issueId: String!, $body: String!) {
                    commentCreate(input: {issueId: $issueId, body: $body}) {
                        success
                    }
                }
                """
                result = await self._graphql(mutation, {"issueId": issue_id, "body": comment})
                if not result.get("data", {}).get("commentCreate", {}).get("success"):
                    success = False

            # Note: Status updates in Linear require knowing the state ID,
            # which requires additional queries. Simplified for now.

        except Exception:
            return False

        return success

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create Linear issue.

        Args:
            title: Issue title.
            body: Issue description.
            labels: Optional label names.

        Returns:
            Created issue.
        """
        if not self.config.default_team:
            raise ValueError("Default team required for issue creation")

        mutation = """
        mutation CreateIssue($teamId: String!, $title: String!, $description: String) {
            issueCreate(input: {teamId: $teamId, title: $title, description: $description}) {
                success
                issue {
                    id
                    identifier
                    title
                    url
                }
            }
        }
        """

        result = await self._graphql(
            mutation,
            {
                "teamId": self.config.default_team,
                "title": title,
                "description": body,
            },
        )

        issue_data = result.get("data", {}).get("issueCreate", {}).get("issue")
        if not issue_data:
            raise RuntimeError("Failed to create Linear issue")

        return Issue(
            tracker="linear",
            id=issue_data["identifier"],
            title=issue_data["title"],
            status="open",
            url=issue_data["url"],
            description=body,
        )

    async def link_checkpoint(
        self,
        issue_id: str,
        checkpoint_id: str,
        session_id: str,
    ) -> bool:
        """Link checkpoint to Linear issue via comment.

        Args:
            issue_id: Issue ID.
            checkpoint_id: Checkpoint ID.
            session_id: Session ID.

        Returns:
            True if link succeeded.
        """
        comment = f"""**Parhelia Checkpoint Linked**

- Session: `{session_id}`
- Checkpoint: `{checkpoint_id}`
- Time: {datetime.now().isoformat()}

To review: `parhelia session review {session_id}`
"""
        return await self.update_issue(issue_id, comment=comment)


@dataclass
class BeadsConfig:
    """Configuration for Beads adapter."""

    workspace_root: str | None = None  # Auto-detect .beads/ directory


class BeadsAdapter:
    """Beads issue tracker adapter (native integration).

    Implements [SPEC-07.22.04].
    """

    def __init__(self, config: BeadsConfig):
        """Initialize Beads adapter.

        Args:
            config: Beads configuration.
        """
        self.config = config

    @property
    def tracker_name(self) -> str:
        return "beads"

    def _run_bd(self, *args: str) -> str | None:
        """Run bd command and return output.

        Args:
            args: Command arguments.

        Returns:
            Command output or None on error.
        """
        cmd = ["bd", *args]
        if self.config.workspace_root:
            cmd.extend(["--workspace", self.config.workspace_root])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    async def get_issue(self, issue_id: str) -> Issue | None:
        """Fetch issue from Beads.

        Args:
            issue_id: Beads issue ID (e.g., "ph-abc123").

        Returns:
            Issue details or None.
        """
        output = self._run_bd("show", issue_id, "--json")
        if not output:
            return None

        try:
            import json
            data = json.loads(output)

            return Issue(
                tracker="beads",
                id=data["id"],
                title=data["title"],
                status=data["status"],
                url=None,  # Beads is local
                description=data.get("description"),
                labels=data.get("labels", []),
                assignee=data.get("assignee"),
            )
        except Exception:
            return None

    async def update_issue(
        self,
        issue_id: str,
        status: str | None = None,
        comment: str | None = None,
    ) -> bool:
        """Update Beads issue.

        Args:
            issue_id: Issue ID.
            status: New status.
            comment: Note to add.

        Returns:
            True if update succeeded.
        """
        success = True

        if status:
            result = self._run_bd("update", issue_id, "--status", status)
            if result is None:
                success = False

        if comment:
            result = self._run_bd("update", issue_id, "--notes", comment)
            if result is None:
                success = False

        return success

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create Beads issue.

        Args:
            title: Issue title.
            body: Issue description.
            labels: Optional labels.

        Returns:
            Created issue.
        """
        args = ["create", "--title", title, "--description", body]
        if labels:
            for label in labels:
                args.extend(["--label", label])

        output = self._run_bd(*args, "--json")
        if not output:
            raise RuntimeError("Failed to create Beads issue")

        try:
            import json
            data = json.loads(output)

            return Issue(
                tracker="beads",
                id=data["id"],
                title=data["title"],
                status="open",
                description=body,
                labels=labels or [],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse Beads response: {e}")

    async def link_checkpoint(
        self,
        issue_id: str,
        checkpoint_id: str,
        session_id: str,
    ) -> bool:
        """Link checkpoint to Beads issue via notes.

        Args:
            issue_id: Issue ID.
            checkpoint_id: Checkpoint ID.
            session_id: Session ID.

        Returns:
            True if link succeeded.
        """
        note = f"Checkpoint {checkpoint_id} from session {session_id}"
        return await self.update_issue(issue_id, comment=note)


# =============================================================================
# Auto-Linking [SPEC-07.22.05]
# =============================================================================


def detect_issue_from_session_name(session_name: str) -> str | None:
    """Extract issue reference from session name.

    Formats:
    - "ph-fix-auth-123" -> issue #123
    - "fix-issue-456" -> issue #456
    - "PROJ-789-description" -> PROJ-789

    Args:
        session_name: Session name.

    Returns:
        Issue reference or None.
    """
    # Look for Beads-style ID
    beads_match = re.search(r"(ph-[a-z0-9]+)", session_name, re.IGNORECASE)
    if beads_match:
        return beads_match.group(1).lower()

    # Look for Linear-style ID (PROJ-123)
    linear_match = re.search(r"([A-Z]+-\d+)", session_name)
    if linear_match:
        return linear_match.group(1)

    # Look for issue number
    issue_match = re.search(r"issue[_-]?(\d+)", session_name, re.IGNORECASE)
    if issue_match:
        return issue_match.group(1)

    # Look for #123 pattern
    hash_match = re.search(r"#(\d+)", session_name)
    if hash_match:
        return hash_match.group(1)

    return None


def detect_issue_from_prompt(prompt: str) -> list[str]:
    """Extract issue references from prompt text.

    Args:
        prompt: User prompt text.

    Returns:
        List of issue references found.
    """
    references: list[str] = []

    # Look for "Fix issue #123" or "Closes #123"
    hash_matches = re.findall(r"(?:fix|close|resolve|address)\s+#(\d+)", prompt, re.IGNORECASE)
    references.extend(hash_matches)

    # Look for Beads IDs
    beads_matches = re.findall(r"\b(ph-[a-z0-9]+)\b", prompt, re.IGNORECASE)
    references.extend([m.lower() for m in beads_matches])

    # Look for Linear IDs
    linear_matches = re.findall(r"\b([A-Z]+-\d+)\b", prompt)
    references.extend(linear_matches)

    # Look for GitHub URLs
    github_matches = re.findall(
        r"github\.com/[^/]+/[^/]+/issues/(\d+)",
        prompt,
    )
    references.extend(github_matches)

    return list(set(references))  # Deduplicate


def detect_issue_from_git_branch() -> str | None:
    """Extract issue reference from current git branch.

    Branch formats:
    - fix/issue-123
    - feature/PROJ-456
    - ph-abc123-description

    Returns:
        Issue reference or None.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        branch = result.stdout.strip()

        # Look for issue patterns
        return detect_issue_from_session_name(branch)

    except Exception:
        return None


class IssueTrackerManager:
    """Manage multiple issue tracker adapters.

    Implements [SPEC-07.22].
    """

    def __init__(self):
        """Initialize the manager."""
        self.adapters: dict[str, IssueTrackerAdapter] = {}

    def register_adapter(self, adapter: IssueTrackerAdapter) -> None:
        """Register an issue tracker adapter.

        Args:
            adapter: The adapter to register.
        """
        self.adapters[adapter.tracker_name] = adapter

    def get_adapter(self, tracker: str) -> IssueTrackerAdapter | None:
        """Get adapter by tracker name.

        Args:
            tracker: Tracker name (github, linear, beads).

        Returns:
            Adapter or None if not registered.
        """
        return self.adapters.get(tracker)

    def detect_tracker(self, reference: str) -> str | None:
        """Detect which tracker a reference belongs to.

        Args:
            reference: Issue reference.

        Returns:
            Tracker name or None.
        """
        # Beads (lowercase prefix-xxx)
        if re.match(r"^[a-z]{2,4}-[a-z0-9]+$", reference):
            return "beads"

        # Linear (UPPERCASE-123)
        if re.match(r"^[A-Z]+-\d+$", reference):
            return "linear"

        # GitHub (numeric or owner/repo#num)
        if re.match(r"^\d+$", reference) or re.match(r"^[^/]+/[^#]+#\d+$", reference):
            return "github"

        return None

    async def get_issue(self, reference: str) -> Issue | None:
        """Get issue from appropriate tracker.

        Args:
            reference: Issue reference.

        Returns:
            Issue or None.
        """
        tracker = self.detect_tracker(reference)
        if not tracker:
            return None

        adapter = self.get_adapter(tracker)
        if not adapter:
            return None

        return await adapter.get_issue(reference)

    async def link_checkpoint_to_issues(
        self,
        references: list[str],
        checkpoint_id: str,
        session_id: str,
    ) -> dict[str, bool]:
        """Link checkpoint to multiple issues.

        Args:
            references: List of issue references.
            checkpoint_id: Checkpoint ID.
            session_id: Session ID.

        Returns:
            Dict mapping reference to success status.
        """
        results: dict[str, bool] = {}

        for ref in references:
            tracker = self.detect_tracker(ref)
            if not tracker:
                results[ref] = False
                continue

            adapter = self.get_adapter(tracker)
            if not adapter:
                results[ref] = False
                continue

            success = await adapter.link_checkpoint(ref, checkpoint_id, session_id)
            results[ref] = success

        return results

    def auto_detect_issues(
        self,
        session_name: str | None = None,
        prompt: str | None = None,
        check_git: bool = True,
    ) -> list[str]:
        """Auto-detect issue references from various sources.

        Implements [SPEC-07.22.05].

        Args:
            session_name: Optional session name.
            prompt: Optional user prompt.
            check_git: Whether to check git branch.

        Returns:
            List of detected issue references.
        """
        references: list[str] = []

        if session_name:
            ref = detect_issue_from_session_name(session_name)
            if ref:
                references.append(ref)

        if prompt:
            refs = detect_issue_from_prompt(prompt)
            references.extend(refs)

        if check_git:
            ref = detect_issue_from_git_branch()
            if ref:
                references.append(ref)

        return list(set(references))  # Deduplicate


def parse_issue_tracker_config(data: dict) -> IssueTrackerManager:
    """Parse issue tracker configuration from TOML.

    Args:
        data: The 'integrations' section from parhelia.toml.

    Returns:
        Configured IssueTrackerManager.
    """
    manager = IssueTrackerManager()

    # GitHub adapter
    if "github" in data:
        github_data = data["github"]
        token = github_data.get("token", os.environ.get("GITHUB_TOKEN", ""))
        if token:
            default_repo = github_data.get("default_repo", "")
            owner, repo = None, None
            if "/" in default_repo:
                owner, repo = default_repo.split("/", 1)

            adapter = GitHubAdapter(
                GitHubConfig(
                    token=token,
                    default_owner=owner,
                    default_repo=repo,
                )
            )
            manager.register_adapter(adapter)

    # Linear adapter
    if "linear" in data:
        linear_data = data["linear"]
        api_key = linear_data.get("api_key", os.environ.get("LINEAR_API_KEY", ""))
        if api_key:
            adapter = LinearAdapter(
                LinearConfig(
                    api_key=api_key,
                    default_team=linear_data.get("default_team"),
                )
            )
            manager.register_adapter(adapter)

    # Beads adapter (always available if .beads/ exists)
    if data.get("beads", {}).get("enabled", True):
        adapter = BeadsAdapter(BeadsConfig())
        manager.register_adapter(adapter)

    return manager
