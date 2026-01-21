# SPEC-10: Parhelia Plugin Structure

**Status**: Draft
**Author**: Claude + rand
**Date**: 2026-01-21

## Overview

This specification defines how Parhelia exposes itself as a Claude Code plugin with commands, skills, and hooks optimized for both human operators and AI agents.

## Goals

- [SPEC-10.01] Create `.claude-plugin/` manifest for Claude Code discovery
- [SPEC-10.02] Expose slash commands for task dispatch and session management
- [SPEC-10.03] Provide progressive-loading skills for remote execution patterns
- [SPEC-10.04] Implement hooks for pre/post dispatch validation

## Non-Goals

- Web UI (v1 is CLI-focused)
- IDE extensions (future work)

---

## Plugin Manifest

### [SPEC-10.10] Plugin Location

The plugin MUST be located at `src/parhelia/.claude-plugin/` with symlink support:

```bash
# Development: symlink into Claude plugins
ln -s /path/to/parhelia/.claude-plugin ~/.claude/plugins/parhelia

# Production: install via plugin manager
/plugin install https://github.com/user/parhelia
```

### [SPEC-10.11] plugin.json Schema

```json
{
  "name": "parhelia",
  "version": "1.0.0",
  "description": "Remote Claude Code execution on Modal.com with GPU support, checkpoint/resume, and interactive sessions",
  "author": {
    "name": "Parhelia Contributors"
  },
  "homepage": "https://github.com/user/parhelia",
  "repository": "https://github.com/user/parhelia",
  "license": "MIT",
  "keywords": [
    "modal", "remote", "containers", "gpu", "checkpoint",
    "cloud", "distributed", "session-persistence"
  ],
  "commands": "./commands",
  "skills": "../skills"
}
```

---

## Directory Structure

### [SPEC-10.20] Plugin Layout

```
src/parhelia/
├── .claude-plugin/
│   ├── plugin.json              # Plugin manifest
│   ├── commands/
│   │   ├── parhelia-submit.md   # /parhelia-submit command
│   │   ├── parhelia-status.md   # /parhelia-status command
│   │   ├── parhelia-attach.md   # /parhelia-attach command
│   │   ├── parhelia-session.md  # /parhelia-session command
│   │   ├── parhelia-checkpoint.md
│   │   └── parhelia-budget.md
│   └── hooks/
│       ├── pre-dispatch.py      # Budget/resource validation
│       └── post-dispatch.py     # Audit and tracking
└── skills/
    ├── discover-parhelia/
    │   └── SKILL.md             # Gateway skill (auto-activation)
    ├── parhelia/
    │   ├── INDEX.md             # Category index
    │   ├── modal-deployment.md
    │   ├── task-dispatch.md
    │   ├── checkpoint-resume.md
    │   ├── gpu-configuration.md
    │   ├── interactive-attach.md
    │   ├── budget-management.md
    │   └── troubleshooting.md
    └── remote-execution/
        ├── INDEX.md
        ├── parallel-dispatch.md
        ├── session-recovery.md
        └── resource-optimization.md
```

---

## Command Specifications

### [SPEC-10.30] Command Format

Commands are Markdown files with YAML frontmatter:

```yaml
---
name: parhelia-submit
description: Submit a task for remote execution on Modal
argument-hint: <prompt> [--gpu TYPE] [--sync] [--budget USD]
---

# Submit Task to Modal

[Command instructions and phases...]
```

### [SPEC-10.31] Core Commands

| Command | Purpose | Arguments |
|---------|---------|-----------|
| `/parhelia-submit` | Submit task for remote execution | `<prompt> [--gpu] [--sync] [--budget]` |
| `/parhelia-status` | Show system/session status | `[session-id] [--json]` |
| `/parhelia-attach` | Attach to running session via SSH | `<session-id>` |
| `/parhelia-session` | Session management (list, show, kill, logs) | `<subcommand> [args]` |
| `/parhelia-checkpoint` | Checkpoint operations | `<subcommand> [args]` |
| `/parhelia-budget` | Budget management | `<subcommand> [args]` |

---

## Skills Architecture

### [SPEC-10.40] Three-Tier Progressive Loading

**Tier 1: Gateway (~200 lines)**
- `discover-parhelia/SKILL.md` - Auto-activates on remote execution keywords
- Minimal context overhead
- Routes to specific skills

**Tier 2: Category Index (~500 lines)**
- `parhelia/INDEX.md` - Lists all Parhelia skills
- `remote-execution/INDEX.md` - Cross-cutting remote patterns

**Tier 3: Atomic Skills (~300 lines each)**
- Deep, actionable guidance
- Code examples and workflows
- Anti-patterns and troubleshooting

### [SPEC-10.41] Gateway Skill Triggers

The gateway skill MUST auto-activate on these keywords:
- "remote", "modal", "parhelia"
- "GPU", "A10G", "A100", "H100"
- "checkpoint", "resume session"
- "attach", "detach", "interactive"
- "dispatch", "offload", "cloud execution"

---

## Hook Specifications

### [SPEC-10.50] Pre-Dispatch Hook

Location: `.claude-plugin/hooks/pre-dispatch.py`

**Responsibilities**:
1. Validate budget availability
2. Estimate resource costs
3. Check approval gates for high-cost operations
4. Validate prompt safety (no secrets in prompt)

**Input Schema**:
```json
{
  "prompt": "string",
  "gpu": "A10G | A100 | H100 | T4 | null",
  "sync": "boolean",
  "budget_usd": "number | null",
  "cwd": "string"
}
```

**Output Schema**:
```json
{
  "allow": true,
  "warnings": ["Budget at 80% of ceiling"],
  "estimated_cost_usd": 0.50
}
```

Or rejection:
```json
{
  "allow": false,
  "reason": "Task would exceed budget ceiling",
  "suggestion": "Reduce scope or increase budget: parhelia budget set 20.0"
}
```

### [SPEC-10.51] Post-Dispatch Hook

Location: `.claude-plugin/hooks/post-dispatch.py`

**Responsibilities**:
1. Log dispatch to local audit trail
2. Update linked beads issue (if any)
3. Trigger notifications (if configured)
4. Record cost for budget tracking

---

## Acceptance Criteria

- [ ] [SPEC-10.AC1] Plugin discoverable after symlink to `~/.claude/plugins/`
- [ ] [SPEC-10.AC2] All commands appear in `/help` output
- [ ] [SPEC-10.AC3] Gateway skill auto-activates on "run this on GPU" prompt
- [ ] [SPEC-10.AC4] Pre-dispatch hook blocks tasks that would exceed budget
- [ ] [SPEC-10.AC5] Commands work identically for humans and agents (with --json flag)

---

## References

- [Claude Code Plugin Guide](https://docs.anthropic.com/en/docs/claude-code/plugins)
- [SPEC-01: Remote Environment](./SPEC-01-remote-environment.md)
