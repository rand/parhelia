# SPEC-11: Agent-Optimized Interfaces

**Status**: Draft
**Author**: Claude + rand
**Date**: 2026-01-21

## Overview

This specification defines interfaces optimized for both human operators and AI agents, ensuring Parhelia provides world-class UX regardless of the caller type.

## Goals

- [SPEC-11.01] Support dual output modes: human-readable and JSON-structured
- [SPEC-11.02] Provide actionable error messages with suggestions
- [SPEC-11.03] Include `next_actions` in all responses for agent guidance
- [SPEC-11.04] Expose MCP server for programmatic access

## Non-Goals

- REST API (MCP is the programmatic interface)
- GraphQL interface

---

## Dual Output Mode

### [SPEC-11.10] OutputFormatter Class

All CLI commands MUST use `OutputFormatter` for output:

```python
class OutputFormatter:
    def __init__(self, json_mode: bool = False):
        self.json_mode = json_mode

    def success(self, data: dict, message: str = None) -> str:
        if self.json_mode:
            return json.dumps({
                "success": True,
                "data": data,
                "metadata": self._metadata(),
                "next_actions": self._suggest_actions(data)
            })
        return self._human_format(data, message)

    def error(self, code: str, message: str, details: dict = None) -> str:
        if self.json_mode:
            return json.dumps({
                "success": False,
                "error": {
                    "code": code,
                    "message": message,
                    "details": details or {}
                },
                "suggestions": self._error_suggestions(code)
            })
        return self._human_error(code, message, details)
```

### [SPEC-11.11] JSON Flag Convention

All commands MUST support `--json` flag:

```bash
# Human output (default)
parhelia status
# Session: ph-abc123-20260121
# Status: RUNNING (12m 34s)
# Container: us-east, CPU, 4 cores

# Agent output
parhelia status --json
# {"success": true, "data": {"session_id": "ph-abc123-20260121", ...}}
```

---

## Response Schemas

### [SPEC-11.20] Success Response

```json
{
  "success": true,
  "data": {
    "task_id": "task-abc123",
    "session_id": "ph-task-abc123-20260121T143022",
    "status": "running",
    "worker": {
      "id": "worker-def456",
      "region": "us-east",
      "gpu": null,
      "started_at": "2026-01-21T14:30:22Z"
    }
  },
  "metadata": {
    "timestamp": "2026-01-21T14:30:25Z",
    "duration_ms": 2847,
    "cost_usd": 0.02
  },
  "next_actions": [
    {
      "action": "attach",
      "description": "Attach to session interactively",
      "command": "parhelia attach task-abc123"
    },
    {
      "action": "status",
      "description": "Check task status",
      "command": "parhelia status task-abc123"
    },
    {
      "action": "logs",
      "description": "View session logs",
      "command": "parhelia session logs task-abc123"
    }
  ]
}
```

### [SPEC-11.21] Error Response

```json
{
  "success": false,
  "error": {
    "code": "BUDGET_EXCEEDED",
    "message": "Task would exceed budget ceiling ($10.00)",
    "details": {
      "requested_usd": 5.00,
      "remaining_usd": 2.50,
      "ceiling_usd": 10.00
    }
  },
  "suggestions": [
    "Reduce task scope or use CPU instead of GPU",
    "Increase budget: parhelia budget set 20.0",
    "Check current usage: parhelia budget show"
  ]
}
```

### [SPEC-11.22] Progress Events

For long-running operations, emit progress events (one per line):

```json
{"type": "progress", "phase": "validating", "percent": 10, "message": "Checking budget..."}
{"type": "progress", "phase": "creating_sandbox", "percent": 30, "message": "Provisioning container..."}
{"type": "progress", "phase": "initializing", "percent": 60, "message": "Starting Claude Code..."}
{"type": "progress", "phase": "running", "percent": 80, "message": "Executing task..."}
{"type": "complete", "result": {...}}
```

---

## Error Codes

### [SPEC-11.30] Standard Error Codes

| Code | HTTP-like | Description |
|------|-----------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input parameters |
| `SESSION_NOT_FOUND` | 404 | Session ID doesn't exist |
| `BUDGET_EXCEEDED` | 402 | Would exceed budget ceiling |
| `UNAUTHORIZED` | 401 | Missing or invalid credentials |
| `RESOURCE_UNAVAILABLE` | 503 | No workers available |
| `TIMEOUT` | 408 | Operation timed out |
| `INTERNAL_ERROR` | 500 | Unexpected internal error |
| `CHECKPOINT_FAILED` | 500 | Failed to create checkpoint |
| `ATTACH_FAILED` | 500 | Failed to establish SSH tunnel |

### [SPEC-11.31] Error Suggestions

Each error code MUST have associated suggestions:

```python
ERROR_SUGGESTIONS = {
    "BUDGET_EXCEEDED": [
        "Reduce task scope or use CPU instead of GPU",
        "Increase budget: parhelia budget set <amount>",
        "Check current usage: parhelia budget show",
    ],
    "SESSION_NOT_FOUND": [
        "List active sessions: parhelia session list",
        "The session may have completed or timed out",
        "Check task status: parhelia status",
    ],
    "RESOURCE_UNAVAILABLE": [
        "Try a different region: --region us-west",
        "Try CPU instead of GPU",
        "Wait and retry in a few minutes",
    ],
}
```

---

## MCP Server

### [SPEC-11.40] MCP Server Entry Point

Parhelia MUST expose an MCP server for programmatic access:

```bash
# Start MCP server
parhelia mcp-server

# Or via mcp_config.json
{
  "mcpServers": {
    "parhelia": {
      "command": "parhelia",
      "args": ["mcp-server"],
      "description": "Parhelia remote execution"
    }
  }
}
```

### [SPEC-11.41] MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `parhelia_submit` | Submit task for remote execution | `prompt`, `gpu?`, `sync?`, `budget_usd?` |
| `parhelia_status` | Get session/system status | `session_id?` |
| `parhelia_attach_info` | Get SSH connection info | `session_id` |
| `parhelia_checkpoint` | Create checkpoint | `session_id`, `name?` |
| `parhelia_session_list` | List all sessions | `status?`, `limit?` |
| `parhelia_session_kill` | Terminate session | `session_id` |
| `parhelia_budget` | Get budget status | - |

### [SPEC-11.42] MCP Tool Schema Example

```json
{
  "name": "parhelia_submit",
  "description": "Submit a task for remote execution on Modal infrastructure",
  "inputSchema": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "Task description or prompt for Claude Code"
      },
      "gpu": {
        "type": "string",
        "enum": ["A10G", "A100", "H100", "T4"],
        "description": "GPU type (omit for CPU-only)"
      },
      "sync": {
        "type": "boolean",
        "default": false,
        "description": "Wait for completion (true) or return immediately (false)"
      },
      "budget_usd": {
        "type": "number",
        "description": "Maximum cost for this task in USD"
      }
    },
    "required": ["prompt"]
  }
}
```

---

## Human UX Enhancements

### [SPEC-11.50] Color and Formatting

Human output MUST use:
- Colors for status (green=success, yellow=warning, red=error)
- Spinners for in-progress operations
- Tables for list data
- Box drawing for structured output

### [SPEC-11.51] Interactive Prompts

When information is missing, prompt interactively (human mode only):

```bash
$ parhelia submit "run tests"

No GPU specified. Select compute type:
  [1] CPU only (default, $0.05/hr)
  [2] A10G GPU ($1.10/hr)
  [3] A100 GPU ($2.50/hr)
  [4] H100 GPU ($4.50/hr)

Select [1]:
```

In JSON mode, return error with required fields instead.

---

## Acceptance Criteria

- [ ] [SPEC-11.AC1] All commands support `--json` flag
- [ ] [SPEC-11.AC2] Error responses include actionable suggestions
- [ ] [SPEC-11.AC3] Success responses include `next_actions`
- [ ] [SPEC-11.AC4] MCP server exposes all core operations
- [ ] [SPEC-11.AC5] Progress events emitted for operations > 2 seconds
- [ ] [SPEC-11.AC6] Human output uses color and formatting

---

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [SPEC-10: Plugin Structure](./SPEC-10-plugin-structure.md)
