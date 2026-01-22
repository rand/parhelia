# MCP Authentication

Parhelia's MCP server supports two transport modes with different authentication approaches.

## Local Transport (stdio)

For local Claude Code integration, the MCP server runs over stdio. No authentication is required since the connection is process-local.

```bash
# Start MCP server (stdio)
parhelia mcp-server
```

Add to `~/.claude/mcp_config.json`:
```json
{
  "mcpServers": {
    "parhelia": {
      "command": "parhelia",
      "args": ["mcp-server"],
      "description": "Parhelia - Remote Claude Code execution"
    }
  }
}
```

## Remote Transport (HTTP)

For remote MCP connections, Parhelia follows the MCP authorization specification using OAuth 2.1.

### Server Configuration

```bash
# Start MCP server with HTTP transport
parhelia mcp-server --transport http --port 8080

# With authentication enabled
parhelia mcp-server --transport http --port 8080 --auth
```

### Authentication Flow

1. **Token-based authentication**: Clients provide a Bearer token in the Authorization header
2. **OAuth 2.1 with PKCE**: For interactive authorization flows

### Environment Variables

For remote connections, configure authentication via environment:

```bash
# Server-side: Set allowed tokens
export PARHELIA_AUTH_TOKENS="token1,token2"

# Client-side: Set token for requests
export PARHELIA_MCP_TOKEN="your-token-here"
```

### Security Considerations

- **HTTPS required** for remote connections in production
- **Token rotation**: Rotate tokens periodically
- **Scope limiting**: Future versions will support granular scopes (read-only, task-create, admin)
- **Audit logging**: All MCP tool calls are logged with caller identity

## Tools Available

The MCP server exposes 29 tools regardless of transport:

| Category | Tools |
|----------|-------|
| Tasks | `parhelia_task_create`, `parhelia_task_list`, `parhelia_task_show`, `parhelia_task_cancel` |
| Containers | `parhelia_containers`, `parhelia_container_show`, `parhelia_container_terminate` |
| Sessions | `parhelia_session_list`, `parhelia_session_attach_info`, `parhelia_session_kill` |
| Checkpoints | `parhelia_checkpoint_create`, `parhelia_checkpoint_list`, `parhelia_checkpoint_restore` |
| Budget | `parhelia_budget_status`, `parhelia_budget_estimate` |
| Health | `parhelia_health`, `parhelia_reconciler_status` |

## Claude Code Integration

After adding Parhelia to your MCP config, restart Claude Code. You can then use natural language to:

- "Create a parhelia task to run the test suite"
- "Show me the budget status"
- "List running containers"
- "Check system health with parhelia doctor"

The MCP tools provide programmatic access to all Parhelia functionality.
