"""Parhelia MCP server for programmatic access.

Implements [SPEC-11.40] MCP Server, [SPEC-21.60] MCP Integration,
and [SPEC-20.30] Agent Excellence.

Exposes 15+ MCP tools for Claude Code and other agents with secure authentication.

Security:
- Token-based authentication (Bearer tokens)
- Scope-based authorization (read/write/admin/budget)
- Audit logging of all tool calls
- HTTP transport with proper auth headers

Container tools:
- parhelia_containers: List containers with optional filters
- parhelia_container_show: Get container details
- parhelia_container_terminate: Terminate a container
- parhelia_container_events: Get container event history

Health tools:
- parhelia_health: Control plane health status
- parhelia_reconciler_status: Reconciler status and last run

Task tools:
- parhelia_task_create: Create task with cost estimate
- parhelia_task_list: List tasks
- parhelia_task_show: Show task details
- parhelia_task_cancel: Cancel running task
- parhelia_task_retry: Retry failed task

Session tools:
- parhelia_session_list: List sessions
- parhelia_session_attach_info: Get attach info
- parhelia_session_kill: Kill a session

Checkpoint tools:
- parhelia_checkpoint_create: Create checkpoint
- parhelia_checkpoint_list: List checkpoints
- parhelia_checkpoint_restore: Restore checkpoint

Budget tools:
- parhelia_budget_status: Budget status
- parhelia_budget_estimate: Estimate cost for task

Usage:
    # Local stdio (no auth required by default)
    parhelia mcp-server

    # HTTP with auth
    parhelia mcp-server --transport http --port 8080

    # Environment variables for auth:
    PARHELIA_AUTH_TOKENS=token1,token2  # Valid tokens
    PARHELIA_AUTH_REQUIRED=true         # Require auth
    PARHELIA_AUDIT_LOG=/path/to/audit.jsonl  # Audit log path
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from parhelia.auth import AuthManager, AuthenticationError, AuthorizationError
from parhelia.budget import BudgetManager
from parhelia.checkpoint import CheckpointManager
from parhelia.config import load_config
from parhelia.mcp import ParheliaMCPTools
from parhelia.orchestrator import Task, TaskRequirements, TaskType
from parhelia.persistence import PersistentOrchestrator


@dataclass
class MCPTool:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]]


@dataclass
class MCPRequest:
    """Incoming MCP request."""

    jsonrpc: str
    id: int | str | None
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Outgoing MCP response."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


class ParheliaMCPServer:
    """MCP server exposing Parhelia functionality.

    Implements JSON-RPC 2.0 over stdio or HTTP for MCP protocol.
    Includes token-based authentication and audit logging.

    Provides 15+ tools for control plane introspection and task management.
    See module docstring for full tool list.
    """

    def __init__(self, require_auth: bool | None = None):
        """Initialize the MCP server.

        Args:
            require_auth: Whether to require authentication. If None, uses
                         PARHELIA_AUTH_REQUIRED env var (default: False for stdio).
        """
        self.config = load_config()
        self.orchestrator = PersistentOrchestrator()
        self.budget_manager = BudgetManager(ceiling_usd=self.config.budget.default_ceiling_usd)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_root=self.config.paths.volume_root + "/checkpoints"
        )

        # Initialize auth manager
        self.auth = AuthManager(require_auth=require_auth)

        # Initialize comprehensive MCP tools
        self.mcp_tools = ParheliaMCPTools(
            orchestrator=self.orchestrator,
            budget_manager=self.budget_manager,
            checkpoint_manager=self.checkpoint_manager,
        )

        # Legacy tools dict for backward compatibility
        self.tools: dict[str, MCPTool] = {}
        self._register_tools()

        # Track current auth token for request context
        self._current_token: str | None = None

    def _register_tools(self) -> None:
        """Register all MCP tools."""
        self.tools["parhelia_submit"] = MCPTool(
            name="parhelia_submit",
            description="Submit a task for remote execution on Modal. Returns task ID for tracking.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task prompt/instructions for Claude Code",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["generic", "code_fix", "test_run", "build", "lint", "refactor"],
                        "description": "Type of task (default: generic)",
                    },
                    "gpu": {
                        "type": "string",
                        "enum": ["A10G", "A100", "H100"],
                        "description": "GPU type if needed (optional)",
                    },
                    "memory_gb": {
                        "type": "integer",
                        "description": "Minimum memory in GB (default: 4)",
                    },
                    "timeout_hours": {
                        "type": "integer",
                        "description": "Max execution time in hours (default: 4)",
                    },
                },
                "required": ["prompt"],
            },
            handler=self._handle_submit,
        )

        self.tools["parhelia_status"] = MCPTool(
            name="parhelia_status",
            description="Get status of a task or list recent tasks.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to check (optional, lists all if not provided)",
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": ["all", "pending", "running", "completed", "failed"],
                        "description": "Filter tasks by status",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max tasks to return (default: 10)",
                    },
                },
            },
            handler=self._handle_status,
        )

        self.tools["parhelia_attach_info"] = MCPTool(
            name="parhelia_attach_info",
            description="Get SSH connection info for attaching to a running session.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session/task ID to get connection info for",
                    },
                },
                "required": ["session_id"],
            },
            handler=self._handle_attach_info,
        )

        self.tools["parhelia_checkpoint"] = MCPTool(
            name="parhelia_checkpoint",
            description="Create a checkpoint or list existing checkpoints for a session.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["create", "list"],
                        "description": "Action to perform (default: list)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Checkpoint message (for create action)",
                    },
                },
                "required": ["session_id"],
            },
            handler=self._handle_checkpoint,
        )

        self.tools["parhelia_budget"] = MCPTool(
            name="parhelia_budget",
            description="Check budget status including remaining balance and usage.",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_budget,
        )

    async def _handle_submit(
        self,
        prompt: str,
        task_type: str = "generic",
        gpu: str | None = None,
        memory_gb: int = 4,
        timeout_hours: int = 4,
    ) -> dict[str, Any]:
        """Handle parhelia_submit tool call."""
        import uuid

        task_id = f"task-{uuid.uuid4().hex[:12]}"

        requirements = TaskRequirements(
            needs_gpu=gpu is not None,
            gpu_type=gpu,
            min_memory_gb=memory_gb,
            estimated_duration_minutes=timeout_hours * 60,
        )

        task = Task(
            id=task_id,
            prompt=prompt,
            task_type=TaskType(task_type),
            requirements=requirements,
        )

        await self.orchestrator.submit_task(task)

        return {
            "success": True,
            "task_id": task_id,
            "session_id": f"ph-{task_id}",
            "message": f"Task submitted successfully",
            "next_actions": [
                {"action": "status", "command": f"parhelia task show {task_id}"},
                {"action": "attach", "command": f"parhelia attach {task_id}"},
            ],
        }

    async def _handle_status(
        self,
        task_id: str | None = None,
        status_filter: str = "all",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Handle parhelia_status tool call."""
        if task_id:
            task = await self.orchestrator.get_task(task_id)
            if not task:
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}",
                }

            worker = self.orchestrator.worker_store.get_by_task(task_id)

            return {
                "success": True,
                "task": {
                    "id": task.id,
                    "prompt_preview": task.prompt[:100],
                    "type": task.task_type.value,
                    "status": self.orchestrator.task_store.get_status(task_id),
                    "created_at": task.created_at.isoformat(),
                },
                "worker": {
                    "id": worker.id,
                    "state": worker.state.value,
                    "target_type": worker.target_type,
                } if worker else None,
            }

        # List tasks
        tasks = await self.orchestrator.list_tasks(
            status=status_filter if status_filter != "all" else None,
            limit=limit,
        )

        return {
            "success": True,
            "tasks": [
                {
                    "id": t.id,
                    "prompt_preview": t.prompt[:50],
                    "type": t.task_type.value,
                    "status": self.orchestrator.task_store.get_status(t.id),
                }
                for t in tasks
            ],
            "count": len(tasks),
        }

    async def _handle_attach_info(self, session_id: str) -> dict[str, Any]:
        """Handle parhelia_attach_info tool call."""
        worker = self.orchestrator.worker_store.get_by_task(session_id)
        if not worker:
            worker = self.orchestrator.get_worker(session_id)

        if not worker:
            return {
                "success": False,
                "error": f"Session not found: {session_id}",
            }

        if worker.state.value not in ("running", "idle"):
            return {
                "success": False,
                "error": f"Session not running (state: {worker.state.value})",
                "suggestion": f"parhelia session recover {session_id}",
            }

        tunnel_host = worker.metrics.get("tunnel_host", "localhost")
        tunnel_port = worker.metrics.get("tunnel_port", 2222)
        tmux_session = f"ph-{session_id}"

        return {
            "success": True,
            "session_id": session_id,
            "connection": {
                "host": tunnel_host,
                "port": tunnel_port,
                "user": "root",
                "tmux_session": tmux_session,
            },
            "ssh_command": f"ssh -p {tunnel_port} root@{tunnel_host} -t 'tmux attach -t {tmux_session}'",
            "attach_command": f"parhelia attach {session_id}",
        }

    async def _handle_checkpoint(
        self,
        session_id: str,
        action: str = "list",
        message: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_checkpoint tool call."""
        from parhelia.session import CheckpointTrigger, Session, SessionState

        if action == "create":
            session = Session(
                id=session_id,
                task_id=session_id,
                state=SessionState.RUNNING,
                working_directory=f"/vol/parhelia/workspaces/{session_id}",
            )

            try:
                cp = await self.checkpoint_manager.create_checkpoint(
                    session=session,
                    trigger=CheckpointTrigger.MANUAL,
                    message=message,
                )
                return {
                    "success": True,
                    "checkpoint_id": cp.id,
                    "message": "Checkpoint created successfully",
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create checkpoint: {e}",
                }

        # List checkpoints
        checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)

        return {
            "success": True,
            "session_id": session_id,
            "checkpoints": [
                {
                    "id": cp.id,
                    "trigger": cp.trigger.value,
                    "created_at": cp.created_at.isoformat() if cp.created_at else None,
                }
                for cp in checkpoints
            ],
            "count": len(checkpoints),
        }

    async def _handle_budget(self) -> dict[str, Any]:
        """Handle parhelia_budget tool call."""
        status = self.budget_manager.check_budget(raise_on_exceeded=False)

        return {
            "success": True,
            "budget": {
                "ceiling_usd": status.ceiling_usd,
                "used_usd": status.used_usd,
                "remaining_usd": status.remaining_usd,
                "usage_percent": status.usage_percent,
                "task_count": status.task_count,
            },
            "warning": "Budget running low" if status.warning_threshold_reached else None,
        }

    async def handle_request(self, request: MCPRequest, token: str | None = None) -> MCPResponse:
        """Handle an incoming MCP request with authentication.

        Args:
            request: The MCP request
            token: Optional auth token (from header or params)

        Returns:
            MCP response
        """
        import time
        start_time = time.time()

        # Store token for this request context
        self._current_token = token

        if request.method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "parhelia",
                        "version": "0.2.0",
                    },
                },
            )

        if request.method == "tools/list":
            # Combine legacy tools with new comprehensive tools
            all_tools = self.mcp_tools.get_tools()

            # Add legacy tools that aren't duplicated
            legacy_names = {t["name"] for t in all_tools}
            for tool in self.tools.values():
                if tool.name not in legacy_names:
                    all_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.input_schema,
                    })

            return MCPResponse(
                id=request.id,
                result={"tools": all_tools},
            )

        if request.method == "tools/call":
            tool_name = request.params.get("name")
            tool_args = request.params.get("arguments", {})

            # Authenticate and authorize
            try:
                auth_token = self.auth.check_tool_auth(token, tool_name)
            except AuthenticationError as e:
                self.auth.log_request(tool_name, None, False, str(e))
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32001,  # Auth error
                        "message": f"Authentication failed: {e}",
                    },
                )
            except AuthorizationError as e:
                self.auth.log_request(tool_name, None, False, str(e))
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32002,  # Authz error
                        "message": f"Authorization failed: {e}",
                    },
                )

            # Try new tools first
            try:
                result = await self.mcp_tools.call_tool(tool_name, tool_args)
                duration_ms = (time.time() - start_time) * 1000
                self.auth.log_request(tool_name, auth_token, True, duration_ms=duration_ms)
                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2),
                            }
                        ],
                    },
                )
            except KeyError:
                # Tool not found in new tools, try legacy
                pass
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                self.auth.log_request(tool_name, auth_token, False, str(e), duration_ms)
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32603,
                        "message": f"Tool execution failed: {e}",
                    },
                )

            # Try legacy tools
            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name].handler(**tool_args)
                    duration_ms = (time.time() - start_time) * 1000
                    self.auth.log_request(tool_name, auth_token, True, duration_ms=duration_ms)
                    return MCPResponse(
                        id=request.id,
                        result={
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2),
                                }
                            ],
                        },
                    )
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.auth.log_request(tool_name, auth_token, False, str(e), duration_ms)
                    return MCPResponse(
                        id=request.id,
                        error={
                            "code": -32603,
                            "message": f"Tool execution failed: {e}",
                        },
                    )

            self.auth.log_request(tool_name, auth_token, False, "Unknown tool")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            )

        return MCPResponse(
            id=request.id,
            error={
                "code": -32601,
                "message": f"Method not found: {request.method}",
            },
        )

    async def run(self) -> None:
        """Run the MCP server on stdio.

        For stdio transport, authentication is validated once at startup using
        the PARHELIA_MCP_TOKEN environment variable if auth is required.
        """
        import os

        # For stdio, validate token once at startup if auth required
        stdio_token = os.environ.get("PARHELIA_MCP_TOKEN")
        if self.auth.require_auth:
            try:
                self.auth.check_auth(stdio_token)
            except AuthenticationError as e:
                sys.stderr.write(f"Authentication failed: {e}\n")
                sys.stderr.write("Set PARHELIA_MCP_TOKEN environment variable with a valid token\n")
                sys.exit(1)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request_data = json.loads(line.decode())
                request = MCPRequest(
                    jsonrpc=request_data.get("jsonrpc", "2.0"),
                    id=request_data.get("id"),
                    method=request_data.get("method", ""),
                    params=request_data.get("params", {}),
                )

                # Pass stdio token for each request
                response = await self.handle_request(request, stdio_token)
                response_json = json.dumps(response.to_dict()) + "\n"
                writer.write(response_json.encode())
                await writer.drain()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                error_response = MCPResponse(
                    error={"code": -32603, "message": str(e)}
                )
                writer.write((json.dumps(error_response.to_dict()) + "\n").encode())
                await writer.drain()


async def run_http_server(server: ParheliaMCPServer, host: str, port: int) -> None:
    """Run MCP server over HTTP with authentication.

    Args:
        server: The MCP server instance
        host: Host to bind to
        port: Port to listen on
    """
    from aiohttp import web

    async def handle_mcp(request: web.Request) -> web.Response:
        """Handle MCP request over HTTP."""
        # Extract auth token from header
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

        try:
            body = await request.json()
            mcp_request = MCPRequest(
                jsonrpc=body.get("jsonrpc", "2.0"),
                id=body.get("id"),
                method=body.get("method", ""),
                params=body.get("params", {}),
            )

            response = await server.handle_request(mcp_request, token)
            return web.json_response(response.to_dict())

        except json.JSONDecodeError:
            return web.json_response(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
                status=400,
            )

    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "auth_enabled": server.auth.is_auth_enabled})

    app = web.Application()
    app.router.add_post("/mcp", handle_mcp)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    print(f"MCP server listening on http://{host}:{port}")
    print(f"Auth enabled: {server.auth.is_auth_enabled}")
    if server.auth.is_auth_enabled:
        print("Set Authorization: Bearer <token> header for authenticated requests")

    # Keep running
    await asyncio.Event().wait()


def run_mcp_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8080,
    require_auth: bool | None = None,
) -> None:
    """Entry point for MCP server.

    Args:
        transport: Transport type ("stdio" or "http")
        host: Host for HTTP transport
        port: Port for HTTP transport
        require_auth: Require authentication (default: True for HTTP, False for stdio)
    """
    # Default to requiring auth for HTTP
    if require_auth is None:
        require_auth = transport == "http"

    server = ParheliaMCPServer(require_auth=require_auth)

    if transport == "http":
        asyncio.run(run_http_server(server, host, port))
    else:
        asyncio.run(server.run())


if __name__ == "__main__":
    run_mcp_server()
