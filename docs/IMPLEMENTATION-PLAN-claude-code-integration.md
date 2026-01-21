# Implementation Plan: Claude Code First-Class Integration

**Epic**: ph-81k
**Date**: 2026-01-21
**Status**: Awaiting Approval

## Executive Summary

This plan makes Claude Code a first-class user of Parhelia with:
1. **Plugin structure** - Commands, skills, and hooks for Claude Code integration
2. **Agent-optimized interfaces** - Dual output mode (human/JSON) and MCP server
3. **World-class interactive UX** - Seamless attach/detach with automatic checkpoints

## Architecture Decisions

### ADR-003: Plugin Architecture
Parhelia exposes itself as a Claude Code plugin with:
- Slash commands (`/parhelia-submit`, `/parhelia-status`, `/parhelia-attach`)
- Progressive-loading skills (gateway → index → atomic)
- Pre/post dispatch hooks for validation

### ADR-004: Dual Output Mode
All commands support `--json` flag for agent consumption:
- Human mode: Colored, formatted, interactive prompts
- JSON mode: Structured with `next_actions` for agent guidance

### ADR-005: Event Streaming
Real-time status via Server-Sent Events with polling fallback.

## Specifications Created

| Spec | Title | Key Requirements |
|------|-------|------------------|
| SPEC-10 | Plugin Structure | Commands, skills, hooks layout |
| SPEC-11 | Agent Interfaces | Dual output, MCP server, error schemas |
| SPEC-12 | Interactive Sessions | Attach/detach, checkpoints, recovery |

## Implementation Phases

```
Phase 1: Plugin Foundation (ph-20k) [P1]
    │
    ├──► Phase 2: Skills Library (ph-jr7) [P2]
    │
    ├──► Phase 3: Agent Output (ph-lra) [P2]
    │         │
    │         └──► Phase 6: MCP Server (ph-30o) [P2]
    │
    ├──► Phase 4: Interactive Sessions (ph-qtv) [P1]
    │         │
    │         └──► Phase 7: Event Streaming (ph-66a) [P3]
    │
    └──► Phase 5: Hook Framework (ph-4ig) [P2]
```

### Phase 1: Plugin Foundation [ph-20k] - Priority 1
- Create `.claude-plugin/plugin.json` manifest
- Command stubs: `parhelia-submit.md`, `parhelia-status.md`, `parhelia-attach.md`
- Wire commands to existing CLI

### Phase 2: Skills Library [ph-jr7] - Priority 2
- Gateway skill: `discover-parhelia/SKILL.md`
- Category index: `parhelia/INDEX.md`
- Atomic skills: modal-deployment, task-dispatch, checkpoint-resume, gpu-configuration

### Phase 3: Agent-Optimized Output [ph-lra] - Priority 2
- `OutputFormatter` class with JSON/human modes
- `--json` flag on all CLI commands
- Structured error responses with suggestions
- `next_actions` in success responses

### Phase 4: Interactive Sessions [ph-qtv] - Priority 1
- Complete `attach` command with SSH tunnel
- Automatic checkpoint on detach
- Session recovery wizard

### Phase 5: Hook Framework [ph-4ig] - Priority 2
- Hook executor for pre/post dispatch
- Budget validation hook
- Audit logging hook

### Phase 6: MCP Server [ph-30o] - Priority 2
- MCP server entry point
- Tools: submit, status, attach_info, checkpoint

### Phase 7: Event Streaming [ph-66a] - Priority 3
- SSE endpoint in Modal sandbox
- `parhelia status --watch` command
- Progress events for long operations

## Files to Create

```
src/parhelia/
├── .claude-plugin/
│   ├── plugin.json
│   └── commands/
│       ├── parhelia-submit.md
│       ├── parhelia-status.md
│       ├── parhelia-attach.md
│       ├── parhelia-session.md
│       ├── parhelia-checkpoint.md
│       └── parhelia-budget.md
├── skills/
│   ├── discover-parhelia/SKILL.md
│   ├── parhelia/
│   │   ├── INDEX.md
│   │   ├── modal-deployment.md
│   │   ├── task-dispatch.md
│   │   ├── checkpoint-resume.md
│   │   └── gpu-configuration.md
│   └── remote-execution/
│       ├── INDEX.md
│       ├── parallel-dispatch.md
│       └── session-recovery.md
├── output_formatter.py     # Dual-mode output
├── hook_executor.py        # Hook execution framework
├── mcp_server.py           # MCP server
└── event_stream.py         # SSE streaming
```

## Files to Modify

- `cli.py` - Add `--json` flags, improve output formatting
- `ssh.py` - Complete attach implementation
- `dispatch.py` - Add hook integration points
- `entrypoint.py` - Add MCP server startup

## Success Criteria

1. **Agent Usability**: AI agent can dispatch task and retrieve results using only structured output
2. **Human UX**: Attach to session in <5 seconds, checkpoint on detach
3. **Error Recovery**: Failed sessions recoverable via wizard with <3 interactions
4. **Budget Safety**: No task can exceed configured budget ceiling
5. **Skill Discovery**: Parhelia skills auto-load on "remote execution" keywords

## Recommended Starting Point

Begin with **Phase 1** (plugin foundation) and **Phase 4** (interactive sessions) in parallel since they're both P1 priority and independent.

---

## Approval

To approve and begin implementation, run:
```bash
bd update ph-81k --status in_progress
```
