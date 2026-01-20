# ADR-002: Durable Sessions with Human Oversight

**Status**: Proposed
**Date**: 2026-01-20
**Deciders**: rand

## Context

Parhelia currently implements excellent checkpoint/resume mechanics for session recovery. However, inspired by Joe Magerramov's "Disposable Environments, Durable Sessions" concept, we identified gaps in:

1. **Human oversight** - No explicit approval workflow for checkpoints
2. **Environment versioning** - No tracking of Claude Code, plugin, or MCP server versions
3. **Provenance tracking** - No DAG of checkpoint relationships with human decisions
4. **Intelligent memory management** - Limited session history summarization
5. **External integration** - No linkage to issue trackers (GitHub, Linear)
6. **Notifications** - No push notifications for approval requests

### Requirements from User

1. Approval should be **optional with intelligent escalation** - users can configure when approval is needed
2. Environment tracking must capture the **full dependency tree**
3. Rollback affects **workspace files only** - git interactions handled separately
4. **Issue tracker integration** is necessary (GitHub, Linear, etc.)
5. **Push notifications** are critical when sessions need human attention
6. Strong emphasis on **durability and intelligent memory management**

## Decision

We will implement a **Durable Sessions** system as SPEC-07, adding human oversight, environment versioning, and intelligent memory management to Parhelia's existing checkpoint infrastructure.

### Core Principles

1. **Sessions are conversations, not just executions** - The human-AI collaboration state is the primary artifact
2. **Checkpoints are decision points** - Each checkpoint represents a potential approval/rejection opportunity
3. **Memory is managed, not just stored** - Intelligent summarization keeps context relevant
4. **Durability means survivability + recoverability** - Sessions survive failures AND can be rolled back

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DURABLE SESSION LAYER                              │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Checkpoint     │───▶│  Approval       │───▶│  Memory         │         │
│  │  Manager        │    │  Workflow       │    │  Manager        │         │
│  │  (existing)     │    │  (NEW)          │    │  (NEW)          │         │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘         │
│                                  │                                          │
│  ┌─────────────────┐    ┌────────▼────────┐    ┌─────────────────┐         │
│  │  Environment    │    │  Notification   │    │  Issue Tracker  │         │
│  │  Tracker        │    │  Service        │    │  Integration    │         │
│  │  (NEW)          │    │  (NEW)          │    │  (NEW)          │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Approval Workflow

**Decision**: Optional approval with intelligent escalation

- Default: Auto-approve for routine checkpoints (periodic, heartbeat)
- Escalate to human when:
  - Session completes (`trigger=complete`)
  - Error occurs (`trigger=error`)
  - Cost threshold exceeded
  - Long-running session (configurable timeout)
  - User explicitly requests review
- User can configure escalation policy per session or globally

**Rationale**: Balance between automation and oversight. Most checkpoints don't need human review, but critical decision points should involve humans.

#### 2. Environment Versioning

**Decision**: Full dependency tree capture

Capture at checkpoint time:
```json
{
  "environment": {
    "claude_code": {
      "version": "2026.01.16",
      "binary_hash": "sha256:abc123..."
    },
    "plugins": {
      "cc-polymath": {
        "git_remote": "https://github.com/...",
        "git_commit": "abc123def456",
        "git_branch": "main",
        "installed_at": "2026-01-16T10:00:00Z"
      }
    },
    "mcp_servers": {
      "playwright": {
        "source": "npm:@anthropic/mcp-playwright",
        "version": "1.45.0"
      },
      "beads": {
        "source": "git+https://github.com/...",
        "git_commit": "def456ghi789"
      }
    },
    "python_packages": {
      "anthropic": "0.40.0",
      "modal": "0.73.0"
    }
  }
}
```

**Rationale**: Full dependency tree enables exact reproduction and debugging of "it worked yesterday" scenarios.

#### 3. Rollback Scope

**Decision**: Workspace files only, git handled separately

- Rollback restores workspace files from checkpoint archive
- Creates stash of current changes before rollback
- Does NOT modify git history (no force push, no reset)
- User must explicitly commit/discard after rollback

**Rationale**: Git operations are dangerous and context-dependent. Separating workspace rollback from git operations gives users control.

#### 4. Issue Tracker Integration

**Decision**: Pluggable adapter pattern

Support multiple backends:
- GitHub Issues (native)
- Linear (via API)
- Beads (local, already integrated)
- Custom webhook (for other systems)

Integration points:
- Link checkpoint to issue (annotation)
- Auto-close issue on approved completion
- Create issue on error/escalation
- Update issue status on session state change

**Rationale**: Different teams use different tools. Pluggable pattern avoids lock-in.

#### 5. Notification Service

**Decision**: Multi-channel with priority routing

Channels:
- Slack webhook
- Discord webhook
- Email (via SMTP or SendGrid)
- Desktop notification (via ntfy.sh or similar)
- SMS (via Twilio, for critical alerts)

Priority levels:
- `info`: Session started, checkpoint created (usually silent)
- `notice`: Session completed, approval available
- `warning`: Session needs human attention
- `critical`: Error, budget exceeded, security event

**Rationale**: Notifications must reach users reliably. Multi-channel with priority routing ensures critical alerts aren't missed.

#### 6. Memory Management

**Decision**: Hierarchical summarization with semantic compression

Layers:
1. **Immediate context** - Last N turns, full fidelity
2. **Session summary** - Key decisions, files touched, approaches tried
3. **Project memory** - Persistent knowledge across sessions (what worked, what didn't)

Summarization triggers:
- Context window approaching limit
- Session checkpoint
- User-requested snapshot

Storage:
- Immediate context in checkpoint (existing)
- Session summary in checkpoint metadata (new)
- Project memory in dedicated volume path (new)

**Rationale**: LLM context windows are finite. Intelligent summarization preserves the *important* context while discarding noise.

## Consequences

### Positive

- Sessions become truly durable with human oversight
- Full auditability via provenance chain
- Environment reproducibility for debugging
- Better integration with existing development workflows
- Intelligent memory management extends effective session length

### Negative

- Increased complexity in checkpoint system
- Additional storage for environment metadata
- Notification infrastructure required
- Users must learn new approval workflow

### Risks

- Over-notification could cause alert fatigue
- Environment versioning might not capture all dependencies
- Summarization could lose important context

### Mitigations

- Configurable notification policies with sensible defaults
- Extensible environment capture (users can add custom trackers)
- Preserve full conversation in checkpoint, summary is supplementary

## Implementation Plan

### Phase A: Foundation
- [SPEC-07.10] Environment versioning and capture
- [SPEC-07.11] Checkpoint metadata schema v1.2
- [SPEC-07.12] Checkpoint tagging and annotation

### Phase B: Human Oversight
- [SPEC-07.20] Approval workflow and escalation policies
- [SPEC-07.21] Notification service architecture
- [SPEC-07.22] Issue tracker integration adapters

### Phase C: Memory Management
- [SPEC-07.30] Session memory and summarization
- [SPEC-07.31] Project-level persistent memory
- [SPEC-07.32] Context window management

### Phase D: Rollback and Recovery
- [SPEC-07.40] Coordinated workspace rollback
- [SPEC-07.41] Checkpoint comparison and diff
- [SPEC-07.42] Recovery workflows

## References

- [Disposable Environments, Durable Sessions](https://blog.joemag.dev/2026/01/disposable-environments-durable.html) by Joe Magerramov
- ADR-001: Remote Claude Code Execution System Architecture
- SPEC-03: Checkpoint and Resume
