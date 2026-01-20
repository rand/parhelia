# SPEC-07: Durable Sessions

**Version**: 1.0
**Status**: Draft
**ADR**: ADR-002-durable-sessions.md

## Overview

This specification defines the Durable Sessions system for Parhelia, implementing human oversight, environment versioning, intelligent memory management, and external integrations for checkpoint-based session management.

---

## SPEC-07.10: Environment Versioning

### Purpose

Capture the complete environment state at checkpoint time to enable reproducibility and debugging.

### Requirements

#### [SPEC-07.10.01] Claude Code Version Capture

The system MUST capture:
- Claude Code binary version string
- Binary SHA256 hash (for exact version identification)
- Installation path

#### [SPEC-07.10.02] Plugin Version Capture

For each installed plugin, the system MUST capture:
- Git remote URL
- Git commit hash (HEAD at checkpoint time)
- Git branch name
- Installation timestamp
- Plugin manifest version (if available)

#### [SPEC-07.10.03] MCP Server Version Capture

For each configured MCP server, the system MUST capture:
- Source type: `npm`, `git`, `local`, `docker`
- Version identifier (npm version, git commit, local path hash)
- Configuration hash (mcp_config.json section for this server)

#### [SPEC-07.10.04] Python Environment Capture

The system MUST capture:
- Python version
- Key package versions: `anthropic`, `modal`, `aiohttp`, etc.
- Full `pip freeze` output (stored separately, referenced by hash)

#### [SPEC-07.10.05] Environment Diff

The system MUST provide:
- `parhelia env diff <checkpoint-a> <checkpoint-b>` command
- Output shows changes in Claude Code, plugins, MCP servers, packages

### Data Model

```python
@dataclass
class EnvironmentSnapshot:
    """Complete environment state at checkpoint time."""

    claude_code: ClaudeCodeVersion
    plugins: dict[str, PluginVersion]
    mcp_servers: dict[str, MCPServerVersion]
    python_version: str
    python_packages: dict[str, str]  # name -> version
    pip_freeze_hash: str  # Reference to full freeze output
    captured_at: datetime

@dataclass
class ClaudeCodeVersion:
    version: str
    binary_hash: str
    install_path: str

@dataclass
class PluginVersion:
    git_remote: str
    git_commit: str
    git_branch: str
    installed_at: datetime
    manifest_version: str | None

@dataclass
class MCPServerVersion:
    source_type: Literal["npm", "git", "local", "docker"]
    version_id: str  # npm version, git commit, path hash, image digest
    config_hash: str
```

---

## SPEC-07.11: Checkpoint Metadata Schema v1.2

### Purpose

Extend checkpoint manifest to include provenance, approval status, environment, and annotations.

### Requirements

#### [SPEC-07.11.01] Schema Version Migration

- New checkpoints MUST use schema version `1.2`
- System MUST read schema versions `1.0`, `1.1`, and `1.2`
- Migration path: auto-upgrade on read, no backfill required

#### [SPEC-07.11.02] Provenance Fields

New required fields:
- `parent_checkpoint_id`: Previous checkpoint in chain (null for first)
- `checkpoint_chain_depth`: Number of checkpoints in this session

#### [SPEC-07.11.03] Approval Fields

New optional fields:
- `approval.status`: `pending` | `approved` | `rejected` | `auto_approved`
- `approval.user`: Username who approved/rejected
- `approval.timestamp`: When decision was made
- `approval.reason`: Free-text explanation
- `approval.policy`: Which escalation policy was applied

#### [SPEC-07.11.04] Environment Fields

New required field:
- `environment`: EnvironmentSnapshot as defined in SPEC-07.10

#### [SPEC-07.11.05] Annotation Fields

New optional fields:
- `tags`: List of hierarchical tags (e.g., `milestone/v1.0`)
- `annotations`: List of timestamped user annotations
- `linked_issues`: List of external issue references

### Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "version": {"const": "1.2"},
    "id": {"type": "string"},
    "session_id": {"type": "string"},
    "created_at": {"type": "string", "format": "date-time"},
    "trigger": {"enum": ["periodic", "detach", "error", "complete", "shutdown", "manual"]},

    "parent_checkpoint_id": {"type": ["string", "null"]},
    "checkpoint_chain_depth": {"type": "integer", "minimum": 1},

    "approval": {
      "type": "object",
      "properties": {
        "status": {"enum": ["pending", "approved", "rejected", "auto_approved"]},
        "user": {"type": "string"},
        "timestamp": {"type": "string", "format": "date-time"},
        "reason": {"type": "string"},
        "policy": {"type": "string"}
      }
    },

    "environment": {"$ref": "#/$defs/EnvironmentSnapshot"},

    "tags": {"type": "array", "items": {"type": "string"}},
    "annotations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "string", "format": "date-time"},
          "user": {"type": "string"},
          "text": {"type": "string"}
        }
      }
    },
    "linked_issues": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "tracker": {"type": "string"},
          "id": {"type": "string"},
          "url": {"type": "string"}
        }
      }
    }
  }
}
```

---

## SPEC-07.12: Checkpoint Tagging and Annotation

### Purpose

Allow users to add meaningful metadata to checkpoints for organization and searchability.

### Requirements

#### [SPEC-07.12.01] Tag Management

Commands:
- `parhelia checkpoint tag <cp-id> <tag>` - Add tag
- `parhelia checkpoint untag <cp-id> <tag>` - Remove tag
- `parhelia checkpoint list --tag <pattern>` - Filter by tag

Tag format:
- Hierarchical: `category/subcategory/name`
- Examples: `milestone/v1.0`, `experiment/approach-a`, `stable`

#### [SPEC-07.12.02] Annotation Management

Commands:
- `parhelia checkpoint annotate <cp-id> "<text>"` - Add annotation
- `parhelia checkpoint annotations <cp-id>` - List annotations

Annotations are:
- Append-only (audit trail)
- Timestamped with user identity
- Searchable via `parhelia checkpoint search "<query>"`

#### [SPEC-07.12.03] Issue Linking

Commands:
- `parhelia checkpoint link <cp-id> <issue-url>` - Link to external issue
- `parhelia checkpoint unlink <cp-id> <issue-url>` - Remove link

Auto-detection:
- Parse GitHub issue URLs: `github.com/:owner/:repo/issues/:id`
- Parse Linear issue URLs: `linear.app/:workspace/issue/:id`
- Parse Beads issue IDs: `ph-xxx`

---

## SPEC-07.20: Approval Workflow

### Purpose

Enable human oversight of session checkpoints with configurable escalation policies.

### Requirements

#### [SPEC-07.20.01] Approval States

Checkpoints have one of four approval states:
- `pending`: Awaiting human review
- `approved`: Human approved
- `rejected`: Human rejected
- `auto_approved`: Policy auto-approved

#### [SPEC-07.20.02] Escalation Policies

Configurable policies determine when human approval is required:

```toml
# parhelia.toml
[approval]
default_policy = "auto"  # auto | review | strict

[approval.policies.auto]
# Auto-approve most checkpoints
auto_approve = ["periodic", "detach"]
require_review = ["complete", "error"]

[approval.policies.review]
# Review completions and errors
auto_approve = ["periodic"]
require_review = ["complete", "error", "detach"]

[approval.policies.strict]
# Review everything
auto_approve = []
require_review = ["periodic", "detach", "complete", "error"]

[approval.escalation]
# Escalate to review if:
cost_threshold_usd = 5.0      # Session cost exceeds $5
duration_threshold_hours = 4   # Session runs > 4 hours
error_count_threshold = 3      # 3+ errors in session
```

#### [SPEC-07.20.03] Review Interface

Command:
```bash
$ parhelia session review <session-id>

Session: ph-fix-auth-20260120T143022
Status: completed (pending review)

Latest Checkpoint: cp-abc123
  Trigger: complete
  Created: 5 minutes ago

Summary:
  Conversation turns: 15
  Tokens used: 45,000
  Estimated cost: $0.45
  Files modified: 3
    - src/auth.py (+45, -12)
    - tests/test_auth.py (+120, -0)
    - .github/workflows/ci.yml (+5, -0)

Actions:
  [a]pprove  [r]eject  [d]iff  [l]og  [c]ontinue session
>
```

#### [SPEC-07.20.04] Approval Audit

All approval decisions MUST be logged:
- Audit event type: `checkpoint.approve` or `checkpoint.reject`
- Include: checkpoint_id, user, timestamp, reason, policy applied

#### [SPEC-07.20.05] Resume Strategy

When resuming a session, prefer checkpoints in this order:
1. Latest `approved` checkpoint
2. Latest `auto_approved` checkpoint
3. Latest `pending` checkpoint (with warning)
4. Never resume from `rejected` checkpoint (require explicit override)

---

## SPEC-07.21: Notification Service

### Purpose

Alert users when sessions need attention via multiple channels.

### Requirements

#### [SPEC-07.21.01] Notification Channels

Supported channels:
- `slack`: Slack webhook
- `discord`: Discord webhook
- `email`: SMTP or API-based
- `ntfy`: ntfy.sh push notifications
- `webhook`: Generic HTTP webhook

Configuration:
```toml
# parhelia.toml
[notifications]
default_channel = "slack"

[notifications.channels.slack]
webhook_url = "${PARHELIA_SLACK_WEBHOOK}"
username = "Parhelia"

[notifications.channels.ntfy]
server = "https://ntfy.sh"
topic = "${PARHELIA_NTFY_TOPIC}"

[notifications.channels.email]
smtp_host = "smtp.example.com"
smtp_port = 587
from_address = "parhelia@example.com"
to_address = "${PARHELIA_ALERT_EMAIL}"
```

#### [SPEC-07.21.02] Notification Priority

Priority levels:
- `info`: Routine events (session started, checkpoint created)
- `notice`: Actionable events (session completed, approval available)
- `warning`: Attention needed (escalation triggered, long-running session)
- `critical`: Immediate attention (error, budget exceeded, security event)

Routing rules:
```toml
[notifications.routing]
info = []  # Silent by default
notice = ["slack"]
warning = ["slack", "ntfy"]
critical = ["slack", "ntfy", "email"]
```

#### [SPEC-07.21.03] Notification Events

Events that trigger notifications:
- `session.started` (info)
- `session.completed` (notice)
- `session.error` (warning/critical based on severity)
- `checkpoint.needs_review` (notice)
- `checkpoint.approved` (info)
- `checkpoint.rejected` (notice)
- `escalation.triggered` (warning)
- `budget.threshold_reached` (warning)
- `budget.exceeded` (critical)

#### [SPEC-07.21.04] Notification Content

Notifications MUST include:
- Session ID and name
- Event type and timestamp
- Actionable link (e.g., `parhelia session review <id>`)
- Context (cost, duration, files changed)

Example Slack message:
```
Parhelia: Session needs review

Session: ph-fix-auth-20260120T143022
Status: Completed
Cost: $0.45 | Duration: 12m | Files: 3

Review: parhelia session review ph-fix-auth-20260120T143022
```

---

## SPEC-07.22: Issue Tracker Integration

### Purpose

Link Parhelia sessions and checkpoints to external issue trackers.

### Requirements

#### [SPEC-07.22.01] Adapter Interface

```python
class IssueTrackerAdapter(Protocol):
    """Interface for issue tracker integrations."""

    async def get_issue(self, issue_id: str) -> Issue | None:
        """Fetch issue details."""
        ...

    async def update_issue(
        self,
        issue_id: str,
        status: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Update issue status or add comment."""
        ...

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> Issue:
        """Create new issue."""
        ...

    async def link_checkpoint(
        self,
        issue_id: str,
        checkpoint_id: str,
        session_id: str,
    ) -> None:
        """Add checkpoint reference to issue."""
        ...
```

#### [SPEC-07.22.02] GitHub Adapter

Configuration:
```toml
[integrations.github]
token = "${GITHUB_TOKEN}"
default_repo = "owner/repo"
```

Features:
- Link checkpoints via issue comments
- Auto-close issues on approved session completion
- Create issues on session errors
- Parse issue references from session prompts

#### [SPEC-07.22.03] Linear Adapter

Configuration:
```toml
[integrations.linear]
api_key = "${LINEAR_API_KEY}"
default_team = "TEAM-ID"
```

Features:
- Same as GitHub adapter
- Map Linear states to Parhelia session states

#### [SPEC-07.22.04] Beads Adapter

Configuration:
```toml
[integrations.beads]
enabled = true  # Auto-detect .beads/ directory
```

Features:
- Native integration (Beads is local)
- Auto-link sessions to in-progress issues
- Update issue status on session state change
- Add checkpoint references to issue notes

#### [SPEC-07.22.05] Auto-Linking

When starting a session, detect issue references:
- From session name: `ph-fix-auth-123` -> issue #123
- From prompt: "Fix issue #123" -> issue #123
- From git branch: `fix/issue-123` -> issue #123

---

## SPEC-07.30: Session Memory and Summarization

### Purpose

Intelligently manage conversation context to maximize effective session length.

### Requirements

#### [SPEC-07.30.01] Memory Hierarchy

Three levels of memory:
1. **Immediate context**: Full conversation, last N turns
2. **Session summary**: Compressed representation of session progress
3. **Project memory**: Persistent knowledge across sessions

#### [SPEC-07.30.02] Session Summary Generation

At checkpoint time, generate summary including:
- Key decisions made
- Files touched with change summary
- Approaches tried (including failed ones)
- Outstanding questions or blockers
- Important context for resumption

Summary format:
```markdown
## Session Summary: ph-fix-auth-20260120T143022

### Progress
- Implemented JWT authentication in src/auth.py
- Added comprehensive test suite (15 tests, all passing)
- Updated CI workflow to run auth tests

### Decisions
- Chose JWT over session cookies for stateless auth
- Used PyJWT library (v2.8.0)

### Failed Approaches
- Initially tried using authlib, but incompatible with async

### Blockers
- Need to add rate limiting (deferred to future session)

### Context for Resume
- Working on feature branch: feature/jwt-auth
- Tests passing, ready for code review
```

#### [SPEC-07.30.03] Summary Storage

- Store in checkpoint manifest as `session_summary` field
- Also store as separate markdown file for human readability
- Path: `/vol/parhelia/sessions/{session-id}/summary.md`

#### [SPEC-07.30.04] Context Window Management

When approaching context limit:
1. Preserve: System prompt, recent turns, session summary
2. Compress: Older turns into summary
3. Discard: Redundant information, verbose tool outputs

Trigger: Context usage > 80% of model limit

---

## SPEC-07.31: Project-Level Memory

### Purpose

Maintain persistent knowledge across sessions for improved continuity.

### Requirements

#### [SPEC-07.31.01] Project Memory Store

Location: `/vol/parhelia/memory/project.json`

Content:
```json
{
  "project_id": "parhelia",
  "last_updated": "2026-01-20T14:30:00Z",
  "knowledge": {
    "architecture": {
      "summary": "Modal-based remote execution system...",
      "key_files": ["src/parhelia/modal_app.py", "src/parhelia/checkpoint.py"],
      "updated_at": "2026-01-20T10:00:00Z"
    },
    "conventions": {
      "testing": "pytest with async support, 80%+ coverage required",
      "specs": "SPEC-XX.YY format in docs/spec/",
      "updated_at": "2026-01-19T15:00:00Z"
    },
    "gotchas": [
      {
        "description": "Modal sandbox requires dangerouslyDisableSandbox for network",
        "discovered_in": "ph-fix-modal-20260118T090000",
        "added_at": "2026-01-18T10:30:00Z"
      }
    ]
  },
  "session_history": [
    {
      "session_id": "ph-fix-auth-20260120T143022",
      "summary": "Implemented JWT authentication",
      "outcome": "approved",
      "timestamp": "2026-01-20T14:30:00Z"
    }
  ]
}
```

#### [SPEC-07.31.02] Memory Update Triggers

Update project memory when:
- Session approved (add to session_history)
- User explicitly saves knowledge (`parhelia memory save "<key>" "<value>"`)
- System detects repeated patterns (auto-suggest)

#### [SPEC-07.31.03] Memory Retrieval

At session start:
1. Load project memory
2. Include relevant sections in system prompt
3. Provide `parhelia memory recall <query>` tool for mid-session retrieval

---

## SPEC-07.32: Context Window Optimization

### Purpose

Maximize useful information within model context limits.

### Requirements

#### [SPEC-07.32.01] Context Budget Allocation

Allocate context window:
- System prompt: 10%
- Project memory: 10%
- Session summary: 15%
- Recent conversation: 50%
- Tool results buffer: 15%

#### [SPEC-07.32.02] Compression Strategies

When context exceeds budget:
1. Summarize old tool outputs (keep results, compress verbose output)
2. Merge similar conversation turns
3. Extract key decisions, discard deliberation
4. Preserve error messages and their resolutions

#### [SPEC-07.32.03] Context Metrics

Track and report:
- `context_tokens_used`: Current token count
- `context_budget_percent`: Usage as percentage of limit
- `compression_applied`: Whether compression was triggered
- `information_density`: Ratio of unique information to tokens

---

## SPEC-07.40: Coordinated Workspace Rollback

### Purpose

Safely revert workspace to a previous checkpoint state.

### Requirements

#### [SPEC-07.40.01] Rollback Command

```bash
$ parhelia checkpoint rollback <checkpoint-id>

Rolling back to checkpoint cp-abc121...

Pre-rollback safety:
  Creating safety checkpoint... cp-abc125
  Stashing current changes... stash@{0}

Restoring workspace:
  Reverting 3 files to checkpoint state...
  - src/auth.py (restored)
  - tests/test_auth.py (restored)
  - .github/workflows/ci.yml (restored)

Post-rollback:
  Workspace restored to cp-abc121 state
  Safety checkpoint: cp-abc125
  Stashed changes: stash@{0}

To resume from this state:
  parhelia session resume <session-id> --from cp-abc121
```

#### [SPEC-07.40.02] Safety Guarantees

Before any rollback:
1. Create safety checkpoint of current state
2. Stash any uncommitted changes
3. Verify target checkpoint is readable
4. Confirm with user (unless --yes flag)

#### [SPEC-07.40.03] Rollback Scope

Rollback affects ONLY:
- Files in workspace snapshot
- Session state (conversation position)

Rollback does NOT affect:
- Git history (no reset, no force push)
- Committed changes
- Other sessions
- Project memory

#### [SPEC-07.40.04] Rollback Recovery

If rollback fails:
1. Restore from safety checkpoint
2. Pop stashed changes
3. Report detailed error

---

## SPEC-07.41: Checkpoint Comparison

### Purpose

Show differences between checkpoints for informed decision-making.

### Requirements

#### [SPEC-07.41.01] Diff Command

```bash
$ parhelia checkpoint diff <cp-a> <cp-b>

Checkpoint Comparison: cp-abc121 → cp-abc123

Time: 2026-01-20T14:00:00Z → 2026-01-20T14:30:00Z (30 minutes)

Conversation:
  Turns: 10 → 15 (+5)
  Tokens: 30,000 → 45,000 (+15,000)
  Cost: $0.30 → $0.45 (+$0.15)

Files Changed:
  M src/auth.py          +45  -12
  A tests/test_auth.py   +120 -0
  M .github/workflows/ci.yml +5 -0

Environment:
  No changes

Approval:
  cp-abc121: approved (by rand)
  cp-abc123: pending
```

#### [SPEC-07.41.02] File Diff

```bash
$ parhelia checkpoint diff <cp-a> <cp-b> --file src/auth.py

# Shows unified diff of file between checkpoints
```

#### [SPEC-07.41.03] Conversation Diff

```bash
$ parhelia checkpoint diff <cp-a> <cp-b> --conversation

# Shows conversation turns added between checkpoints
```

---

## SPEC-07.42: Recovery Workflows

### Purpose

Define standard workflows for common recovery scenarios.

### Requirements

#### [SPEC-07.42.01] Resume from Failure

When container fails:
1. Detect failure via heartbeat timeout
2. Find latest approved/auto-approved checkpoint
3. Notify user if checkpoint is old (> 5 minutes)
4. Auto-resume or wait for user (based on policy)

#### [SPEC-07.42.02] Resume after Rejection

When user rejects a checkpoint:
1. Find previous approved checkpoint
2. Offer options:
   - Resume from previous approved
   - Start new session from that point
   - Abandon session

#### [SPEC-07.42.03] Manual Recovery

Commands:
- `parhelia session recover <session-id>` - Interactive recovery wizard
- `parhelia checkpoint list <session-id> --detailed` - Show all checkpoints with status
- `parhelia session resume <session-id> --from <cp-id>` - Resume from specific checkpoint

---

## Implementation Notes

### Backward Compatibility

- Existing checkpoints (schema v1.0, v1.1) remain readable
- New fields default to null/empty on read
- No migration required for existing data

### Performance Considerations

- Environment capture should be fast (< 1s)
- Summarization can be async (doesn't block checkpoint)
- Project memory loaded once at session start

### Security Considerations

- Notification webhooks must use HTTPS
- API tokens stored as secrets, not in config
- Audit log captures all approval decisions

---

## Acceptance Criteria

For each SPEC section, implementation is complete when:
1. Functionality works as specified
2. Unit tests achieve 80%+ coverage
3. Integration tests verify end-to-end workflow
4. CLI commands documented in `--help`
5. Configuration options documented in parhelia.toml.example
