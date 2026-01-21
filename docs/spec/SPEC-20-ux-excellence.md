# SPEC-20: Parhelia UX Excellence

**Status**: Draft
**Created**: 2026-01-21
**Epic**: World-Class UX for Humans and Agents

## Executive Summary

This specification defines the UX improvement roadmap for Parhelia, targeting both human operators and AI agents as first-class users. Based on comprehensive research of CLI best practices, AI agent interaction patterns, and analysis of Parhelia's current state, this plan identifies high-impact improvements across five pillars:

1. **Command Coherence** - Unified naming, aliases, and discoverability
2. **Feedback Excellence** - Progress, status, and error communication
3. **Agent Optimization** - Machine-readable interfaces and context efficiency
4. **Interactive Intelligence** - Smart prompts, suggestions, and recovery
5. **Streaming & Real-Time** - Event-driven architecture for monitoring

## Background Research Findings

### What Makes Great Terminal UX

**Progressive Disclosure**: Best CLIs reveal complexity gradually:
- Running with no args provides helpful hints
- Common operations need zero flags
- Advanced features hidden behind `--help` or secondary menus
- GitHub CLI exemplifies this: commands prompt for missing inputs

**Feedback Timing Standards**:
| Duration | Pattern | Example |
|----------|---------|---------|
| <1s | No indicator | Direct execution |
| 1-4s | Spinner | Animated dots/bars |
| 4-10s | Progress bar | "3 of 10 files" |
| >10s | Progress + ETA | "~2 min remaining" |

**Exemplary Tools Analysis**:
- **ripgrep/fd/fzf**: Blazing speed, smart defaults, composable
- **GitHub CLI**: Rich prompting, accessibility-first, opinionated
- **lazygit**: Full TUI for complex workflows
- **Vercel/Railway**: Framework-aware, minimal config

**Accessibility Requirements**:
- Never rely on color alone (8% of males have color blindness)
- Support screen readers with structured output
- Provide high-contrast themes
- Detect TTY and disable formatting when piped

### What Makes Great Agent UX

**Status Communication**:
- Replace "mystery" with "momentum" - make work visible
- Get confirmation mid-way through long operations
- Show planning → executing → waiting → complete states
- The more autonomous the agent, the more UI for approve/steer/undo

**Machine-Readable Interfaces**:
- JSON Schema as contract language
- JSONL for streaming (one JSON object per line)
- Server-Sent Events for real-time push
- OpenAPI/MCP for tool discovery

**Context Efficiency**:
- Active context compression preserves learnings
- Semantic caching reuses LLM results
- RAG integration over document stuffing
- Token budget awareness (TALE framework: 68% reduction possible)

**Cost Awareness**:
- Model routing: simple→cheap, complex→premium
- Budget tracking at each iteration
- Adaptive consumption scaling with complexity

**Retry & Recovery**:
- Exponential backoff with jitter
- Durable execution (skip completed steps)
- Fallback chains across providers
- Circuit breakers for failure detection

## Current State Analysis

### Strengths

1. **Dual-Mode Output** (SPEC-11 compliant)
   - OutputFormatter handles JSON/human modes
   - Structured error responses with suggestions
   - next_actions for agent navigation

2. **Event Streaming Foundation**
   - 11 event types (status, progress, heartbeat, etc.)
   - EventFormatter with progress bars
   - Watch commands for real-time monitoring

3. **Interactive Workflows**
   - Session recovery wizard with numbered choices
   - Checkpoint rollback confirmations
   - Category selection for memory operations

4. **MCP Server**
   - 5 core tools with JSON-RPC 2.0
   - Schema-driven capability declaration
   - Proper error codes and handling

### Gaps

| Area | Issue | Impact |
|------|-------|--------|
| Command Naming | `submit` vs `dispatch`, root vs grouped | User confusion |
| Defaults | Async by default (industry uses sync) | Unexpected behavior |
| Error Recovery | Generic suggestions, no "did you mean" | Support burden |
| Progress | No feedback for <1s ops | Feels unresponsive |
| Output Format | Inconsistent formatting across commands | Unprofessional |
| MCP Coverage | Only 5 of 40+ operations exposed | Limited agent autonomy |
| Event Architecture | 2s polling, no push notifications | Latency, server load |
| Discoverability | No aliases, no tab completion | Slow adoption |

## Specification

### [SPEC-20.10] Command Coherence

#### [SPEC-20.11] Unified Command Hierarchy

All commands MUST follow a consistent `<noun> <verb>` or `<group> <action>` structure:

```bash
# Task Operations
parhelia task create <prompt>    # Was: submit
parhelia task list [--status]    # Was: list
parhelia task show <id>          # Unchanged
parhelia task run <prompt>       # New: sync execution
parhelia task watch <id>         # Unchanged

# Session Operations
parhelia session attach <id>     # Was: attach (root)
parhelia session detach <id>     # Was: detach (root)
parhelia session recover <id>    # Unchanged
parhelia session list            # New: list sessions

# Checkpoint Operations
parhelia checkpoint create <id>  # Unchanged
parhelia checkpoint list         # Unchanged
parhelia checkpoint restore <id> # Was: rollback

# Budget Operations
parhelia budget show             # Unchanged
parhelia budget set <amount>     # Unchanged
```

**Rationale**: Matches kubectl, docker, git patterns. Reduces cognitive load.

#### [SPEC-20.12] Command Aliases

All common commands MUST have short aliases:

```bash
# Single-letter aliases
parhelia t create    # task create
parhelia t l         # task list
parhelia t s <id>    # task show
parhelia s a <id>    # session attach

# Word aliases (backward compat)
parhelia submit      # → task create
parhelia attach      # → session attach
parhelia list        # → task list
parhelia status      # → task show
```

**Implementation**: Click's `@cli.command(name='create', aliases=['c', 'new'])`

#### [SPEC-20.13] Shell Completion

Parhelia MUST support shell completion for bash, zsh, and fish:

```bash
parhelia completion install      # Auto-detect shell
parhelia completion bash         # Output bash script
parhelia completion zsh          # Output zsh script

# Usage after install
parhelia task [TAB]              # create, list, show, watch...
parhelia task show task-[TAB]    # task-abc123, task-def456...
```

**Implementation**: Click's shell completion + dynamic task ID completion

#### [SPEC-20.14] Fuzzy Command Matching

Unknown commands MUST suggest similar valid commands:

```bash
$ parhelia submti "test"
Unknown command: submti

Did you mean?
  • parhelia submit "test"
  • parhelia status

Run 'parhelia --help' for all commands.
```

**Implementation**: Levenshtein distance for suggestions

### [SPEC-20.20] Feedback Excellence

#### [SPEC-20.21] Progress Indication Standards

All operations MUST provide appropriate feedback:

| Duration | Indicator | Format |
|----------|-----------|--------|
| <100ms | None | Direct output |
| 100ms-1s | Inline status | `Connecting...` |
| 1-4s | Spinner | `⠋ Creating sandbox...` |
| 4-10s | Progress | `[████░░░░░░] 40% - Building image` |
| >10s | Progress+ETA | `[████░░░░░░] 40% - ~30s remaining` |

**Spinner Styles** (configurable via `PARHELIA_SPINNER`):
- `dots`: ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏
- `line`: -\|/
- `classic`: |/-\
- `none`: Disabled (for CI/pipes)

#### [SPEC-20.22] Structured Error System

All errors MUST use the extended error code system:

```python
class ErrorCode(str, Enum):
    # Validation (1xx)
    VALIDATION_ERROR = "E100"
    INVALID_ARGUMENT = "E101"
    MISSING_REQUIRED = "E102"

    # Resource (2xx)
    SESSION_NOT_FOUND = "E200"
    TASK_NOT_FOUND = "E201"
    CHECKPOINT_NOT_FOUND = "E202"
    WORKER_NOT_FOUND = "E203"

    # Budget (3xx)
    BUDGET_EXCEEDED = "E300"
    BUDGET_WARNING = "E301"

    # Auth (4xx)
    UNAUTHORIZED = "E400"
    FORBIDDEN = "E401"
    TOKEN_EXPIRED = "E402"

    # Infrastructure (5xx)
    RESOURCE_UNAVAILABLE = "E500"
    TIMEOUT = "E501"
    NETWORK_ERROR = "E502"
    MODAL_ERROR = "E503"

    # Operations (6xx)
    CHECKPOINT_FAILED = "E600"
    ATTACH_FAILED = "E601"
    DISPATCH_FAILED = "E602"
    DISPATCH_REJECTED = "E603"  # Hook rejection

    # Internal (9xx)
    INTERNAL_ERROR = "E900"
    NOT_IMPLEMENTED = "E901"
```

Error responses MUST include:
- Error code (Exxx)
- Human-readable message
- Contextual details (what was attempted)
- Recovery suggestions (2-4 specific commands)
- Documentation link (if applicable)

#### [SPEC-20.23] Success Response Standards

All successful operations MUST include:

```json
{
  "success": true,
  "data": { /* operation-specific */ },
  "metadata": {
    "timestamp": "2026-01-21T20:15:30Z",
    "duration_ms": 1234,
    "cost_usd": 0.05
  },
  "next_actions": [
    {
      "action": "watch",
      "description": "Monitor task progress",
      "command": "parhelia task watch task-abc123",
      "priority": "primary"
    },
    {
      "action": "attach",
      "description": "Attach for interactive work",
      "command": "parhelia session attach task-abc123",
      "priority": "secondary"
    }
  ]
}
```

**Priority Levels**: primary (highlighted), secondary (listed), tertiary (in --verbose only)

### [SPEC-20.30] Agent Optimization

#### [SPEC-20.31] Expanded MCP Tool Coverage

The MCP server MUST expose all core operations:

**Task Tools**:
- `parhelia_task_create` - Create and optionally dispatch task
- `parhelia_task_list` - List tasks with filters
- `parhelia_task_show` - Get task details
- `parhelia_task_cancel` - Cancel running task
- `parhelia_task_retry` - Retry failed task

**Session Tools**:
- `parhelia_session_attach_info` - Get SSH connection info
- `parhelia_session_list` - List active sessions
- `parhelia_session_kill` - Terminate session

**Checkpoint Tools**:
- `parhelia_checkpoint_create` - Create checkpoint
- `parhelia_checkpoint_list` - List checkpoints
- `parhelia_checkpoint_restore` - Restore from checkpoint
- `parhelia_checkpoint_diff` - Compare checkpoints

**Budget Tools**:
- `parhelia_budget_status` - Get budget status
- `parhelia_budget_estimate` - Estimate task cost

**System Tools**:
- `parhelia_status` - System health check
- `parhelia_events_subscribe` - Subscribe to event stream

#### [SPEC-20.32] Streaming MCP Support

MCP server MUST support event streaming via JSON-RPC notifications:

```json
// Client subscribes
{"jsonrpc": "2.0", "method": "events/subscribe", "params": {"task_id": "task-123"}}

// Server pushes events
{"jsonrpc": "2.0", "method": "events/notification", "params": {
  "type": "status_change",
  "task_id": "task-123",
  "old_status": "running",
  "new_status": "completed"
}}
```

#### [SPEC-20.33] Context-Efficient Responses

All MCP responses MUST be optimized for token efficiency:

- Use short, consistent field names
- Omit null/empty fields
- Provide summary fields for large data
- Support `fields` parameter for selective retrieval

```json
// Request with field selection
{"method": "tools/call", "params": {
  "name": "parhelia_task_show",
  "arguments": {"task_id": "task-123", "fields": ["id", "status", "cost"]}
}}

// Minimal response
{"id": "task-123", "status": "completed", "cost": 0.05}
```

#### [SPEC-20.34] Cost Awareness Integration

All task operations MUST include cost information:

```json
{
  "data": {
    "task_id": "task-123",
    "estimated_cost_usd": 0.15,
    "budget_remaining_usd": 9.85,
    "budget_after_task_usd": 9.70,
    "cost_breakdown": {
      "compute": 0.10,
      "gpu": 0.05,
      "storage": 0.00
    }
  },
  "warnings": [
    {"type": "budget_low", "message": "Budget will be at 3% after this task"}
  ]
}
```

### [SPEC-20.40] Interactive Intelligence

#### [SPEC-20.41] Smart Prompts

Interactive prompts MUST use intelligent defaults:

```bash
$ parhelia task create
Prompt: Run the test suite
GPU needed? [y/N]: n
Timeout (hours) [4]:
Priority [medium]:

Creating task with:
  • Prompt: "Run the test suite"
  • GPU: None
  • Timeout: 4 hours
  • Priority: Medium

Proceed? [Y/n]:
```

**Defaults Source Priority**:
1. Project config (parhelia.toml)
2. User config (~/.config/parhelia/config.toml)
3. Last used values (cached in ~/.cache/parhelia/)
4. System defaults

#### [SPEC-20.42] Proactive Validation

Operations MUST validate and warn before execution:

```bash
$ parhelia task create "train large model" --timeout 1
⚠ Warning: Training tasks typically need 4+ hours
  Your timeout of 1 hour may cause incomplete results.

Options:
  1. Continue with 1 hour timeout
  2. Use recommended 4 hour timeout
  3. Cancel

Choice [2]:
```

**Validation Rules**:
- GPU tasks without GPU flag → suggest GPU
- Large prompts with short timeout → suggest longer
- Expensive operations near budget → confirm
- Destructive operations → always confirm

#### [SPEC-20.43] Recovery Suggestions

All errors MUST provide actionable recovery paths:

```bash
$ parhelia session attach task-xyz
✗ Error: Session not found (E200)

The session 'task-xyz' was not found. This could mean:
  • The task ID is incorrect
  • The session has already completed
  • The session failed to start

Try these:
  1. List all tasks:     parhelia task list
  2. Search by prefix:   parhelia task list --query xyz
  3. Check recent:       parhelia task list --recent --limit 5
  4. View completed:     parhelia task list --status completed

For more help: parhelia help session-not-found
```

#### [SPEC-20.44] Contextual Help System

Help MUST be contextual and progressive:

```bash
# Topic-based help
parhelia help tasks          # All task commands
parhelia help budget         # Budget management
parhelia help errors         # Common errors
parhelia help workflow       # Typical workflows

# Error-specific help
parhelia help E200           # SESSION_NOT_FOUND details
parhelia help E300           # BUDGET_EXCEEDED details

# Example-based help
parhelia examples submit     # Task submission examples
parhelia examples gpu        # GPU task examples
```

### [SPEC-20.50] Streaming & Real-Time

#### [SPEC-20.51] Event-Driven Architecture

Replace polling with push-based events:

**Current** (polling):
```python
while True:
    status = await get_status(task_id)
    if status != last_status:
        yield StatusEvent(...)
    await asyncio.sleep(2)  # 2 second latency
```

**Target** (push):
```python
async for event in modal_events.subscribe(task_id):
    yield transform_event(event)  # Immediate delivery
```

**Fallback**: If push unavailable, use adaptive polling:
- Start at 500ms
- Increase to 2s after 30s of no changes
- Increase to 5s after 2 min of no changes

#### [SPEC-20.52] Event Aggregation

High-frequency events MUST be aggregated:

```python
# Instead of 100 individual PROGRESS events:
{"type": "progress", "percent": 1}
{"type": "progress", "percent": 2}
...

# Aggregate into batched updates:
{"type": "progress_batch", "updates": [
  {"percent": 1, "ts": "..."},
  {"percent": 50, "ts": "..."},
  {"percent": 100, "ts": "..."}
]}
```

**Aggregation Rules**:
- Batch progress events (emit at 5% intervals or 5s minimum)
- Dedupe heartbeats (one per 10s maximum)
- Never aggregate: status changes, errors, completions

#### [SPEC-20.53] Event Filtering

Watch commands MUST support filtering:

```bash
# Filter by event type
parhelia task watch --events status,completion

# Filter by severity
parhelia task watch --level warning  # warning + error only

# Quiet mode (completion only)
parhelia task watch --quiet
```

#### [SPEC-20.54] Event Persistence

Events SHOULD be persisted for replay:

```bash
# Replay session events
parhelia events replay task-123

# Export events
parhelia events export task-123 --format jsonl > events.jsonl

# Stream from history
parhelia task watch task-123 --from-start
```

## Implementation Phases

### Phase 1: Command Foundation (Priority 1)

**Goal**: Establish consistent, discoverable command structure

| ID | Task | Effort |
|----|------|--------|
| 20.11-A | Restructure commands under noun groups | M |
| 20.12-A | Implement alias system | S |
| 20.13-A | Add shell completion generation | M |
| 20.14-A | Add fuzzy command matching | S |
| 20.22-A | Expand error code system | S |

**Success Criteria**:
- All commands follow `<noun> <verb>` pattern
- Tab completion works in bash/zsh/fish
- Typos suggest correct commands
- 20+ error codes with specific suggestions

### Phase 2: Feedback & Progress (Priority 1)

**Goal**: Every operation provides clear, timely feedback

| ID | Task | Effort |
|----|------|--------|
| 20.21-A | Implement spinner/progress system | M |
| 20.21-B | Add progress callbacks to all long operations | M |
| 20.23-A | Standardize success responses | S |
| 20.42-A | Add proactive validation warnings | M |
| 20.43-A | Enhance error recovery suggestions | S |

**Success Criteria**:
- No operation >1s without visual feedback
- All errors include 2+ recovery commands
- Dangerous operations require confirmation

### Phase 3: Agent Excellence (Priority 2)

**Goal**: Full programmatic access with optimal efficiency

| ID | Task | Effort |
|----|------|--------|
| 20.31-A | Expand MCP tools to 15+ operations | L |
| 20.32-A | Implement streaming MCP notifications | M |
| 20.33-A | Add field selection to responses | S |
| 20.34-A | Integrate cost estimation everywhere | M |

**Success Criteria**:
- All CLI operations available via MCP
- Agents receive real-time event notifications
- Token-efficient responses with field selection
- Cost visible before every task execution

### Phase 4: Real-Time Architecture (Priority 2)

**Goal**: Push-based event delivery with smart aggregation

| ID | Task | Effort |
|----|------|--------|
| 20.51-A | Implement Modal webhook integration | L |
| 20.51-B | Add adaptive polling fallback | S |
| 20.52-A | Implement event aggregation | M |
| 20.53-A | Add event filtering to watch commands | S |
| 20.54-A | Add event persistence and replay | M |

**Success Criteria**:
- Events delivered <500ms from occurrence
- High-frequency events aggregated
- Watch supports filtering and replay

### Phase 5: Interactive Intelligence (Priority 3)

**Goal**: Smart, helpful interactive experience

| ID | Task | Effort |
|----|------|--------|
| 20.41-A | Implement smart prompts with defaults | M |
| 20.44-A | Build contextual help system | M |
| 20.44-B | Add example-based documentation | S |

**Success Criteria**:
- Interactive mode remembers preferences
- Help available for every error code
- Examples cover common workflows

## Metrics & Validation

### Human UX Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Command discoverability (% found via completion) | 0% | >80% |
| Error recovery success (% resolved without docs) | ~30% | >70% |
| Time to first successful task | ~5 min | <2 min |
| Operations with progress feedback | ~40% | 100% |

### Agent UX Metrics

| Metric | Current | Target |
|--------|---------|--------|
| MCP operation coverage | 12% (5/40) | >90% |
| Event delivery latency | 2000ms | <500ms |
| Avg response tokens | ~500 | <200 |
| Cost visibility | Partial | 100% |

## Appendix: Research Sources

### CLI UX
- [CLI Guidelines](https://clig.dev/)
- [10 Design Principles for Delightful CLIs](https://www.atlassian.com/blog/it-teams/10-design-principles-for-delightful-clis)
- [Building a More Accessible GitHub CLI](https://github.blog/engineering/user-experience/building-a-more-accessible-github-cli/)

### Agent UX
- [Agentic LLMs in 2025](https://datasciencedojo.com/blog/agentic-llm-in-2025/)
- [Secrets of Agentic UX](https://uxmag.com/articles/secrets-of-agentic-ux-emerging-design-patterns-for-human-interaction-with-ai-agents)
- [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture)

### Progress & Feedback
- [Progress Bars vs Spinners](https://uxmovement.com/navigation/progress-bars-vs-spinners-when-to-use-which/)
- [CLI UX Best Practices: Progress Displays](https://evilmartians.com/chronicles/cli-ux-best-practices-3-patterns-for-improving-progress-displays)
