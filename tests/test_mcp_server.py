"""Tests for Parhelia MCP server."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.mcp_server import (
    MCPRequest,
    MCPResponse,
    MCPTool,
    ParheliaMCPServer,
)


# =============================================================================
# MCPRequest/Response Tests
# =============================================================================


class TestMCPRequest:
    """Tests for MCPRequest class."""

    def test_request_with_params(self):
        """MCPRequest MUST accept params."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=1,
            method="tools/call",
            params={"name": "test", "arguments": {}},
        )

        assert request.method == "tools/call"
        assert request.params["name"] == "test"


class TestMCPResponse:
    """Tests for MCPResponse class."""

    def test_response_success(self):
        """MCPResponse MUST serialize success result."""
        response = MCPResponse(
            id=1,
            result={"tools": []},
        )

        d = response.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["id"] == 1
        assert d["result"] == {"tools": []}
        assert "error" not in d

    def test_response_error(self):
        """MCPResponse MUST serialize error."""
        response = MCPResponse(
            id=2,
            error={"code": -32601, "message": "Method not found"},
        )

        d = response.to_dict()
        assert d["error"]["code"] == -32601
        # When error is present, result is omitted from output


# =============================================================================
# ParheliaMCPServer Tests
# =============================================================================


class TestParheliaMCPServer:
    """Tests for ParheliaMCPServer class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.budget.default_ceiling_usd = 10.0
        config.paths.volume_root = "/vol/parhelia"
        return config

    @pytest.fixture
    def server(self, mock_config, tmp_path):
        """Create server with mocked dependencies."""
        with patch("parhelia.mcp_server.load_config", return_value=mock_config):
            with patch("parhelia.mcp_server.PersistentOrchestrator") as mock_orch:
                mock_orch.return_value = MagicMock()
                with patch("parhelia.mcp_server.CheckpointManager") as mock_cp:
                    mock_cp.return_value = MagicMock()
                    server = ParheliaMCPServer()
                    return server

    @pytest.mark.asyncio
    async def test_handle_initialize(self, server):
        """handle_request MUST respond to initialize."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=1,
            method="initialize",
        )

        response = await server.handle_request(request)

        assert response.result is not None
        assert response.result["protocolVersion"] == "2024-11-05"
        assert "tools" in response.result["capabilities"]
        assert response.result["serverInfo"]["name"] == "parhelia"

    @pytest.mark.asyncio
    async def test_handle_tools_list(self, server):
        """handle_request MUST list available tools."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=2,
            method="tools/list",
        )

        response = await server.handle_request(request)

        assert response.result is not None
        tools = response.result["tools"]
        tool_names = [t["name"] for t in tools]

        assert "parhelia_submit" in tool_names
        assert "parhelia_status" in tool_names
        assert "parhelia_attach_info" in tool_names
        assert "parhelia_checkpoint" in tool_names
        assert "parhelia_budget" in tool_names

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, server):
        """handle_request MUST return error for unknown method."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=3,
            method="unknown/method",
        )

        response = await server.handle_request(request)

        assert response.error is not None
        assert response.error["code"] == -32601
        assert "not found" in response.error["message"].lower()

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self, server):
        """handle_request MUST return error for unknown tool."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=4,
            method="tools/call",
            params={"name": "unknown_tool", "arguments": {}},
        )

        response = await server.handle_request(request)

        assert response.error is not None
        assert "unknown_tool" in response.error["message"].lower()

    @pytest.mark.asyncio
    async def test_handle_budget_tool(self, server):
        """handle_request MUST handle parhelia_budget tool."""
        request = MCPRequest(
            jsonrpc="2.0",
            id=5,
            method="tools/call",
            params={"name": "parhelia_budget", "arguments": {}},
        )

        response = await server.handle_request(request)

        assert response.result is not None
        content = response.result["content"][0]["text"]
        data = json.loads(content)

        assert data["success"] is True
        assert "budget" in data
        assert "ceiling_usd" in data["budget"]

    @pytest.mark.asyncio
    async def test_handle_submit_tool(self, server):
        """handle_request MUST handle parhelia_submit tool."""
        server.orchestrator.submit_task = AsyncMock()

        request = MCPRequest(
            jsonrpc="2.0",
            id=6,
            method="tools/call",
            params={
                "name": "parhelia_submit",
                "arguments": {
                    "prompt": "Run the tests",
                    "task_type": "test_run",
                },
            },
        )

        response = await server.handle_request(request)

        assert response.result is not None
        content = response.result["content"][0]["text"]
        data = json.loads(content)

        assert data["success"] is True
        assert "task_id" in data
        assert data["task_id"].startswith("task-")
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_handle_status_tool_list(self, server):
        """handle_request MUST handle parhelia_status tool for listing."""
        server.orchestrator.list_tasks = AsyncMock(return_value=[])

        request = MCPRequest(
            jsonrpc="2.0",
            id=7,
            method="tools/call",
            params={
                "name": "parhelia_status",
                "arguments": {"limit": 5},
            },
        )

        response = await server.handle_request(request)

        assert response.result is not None
        content = response.result["content"][0]["text"]
        data = json.loads(content)

        assert data["success"] is True
        assert "tasks" in data
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_handle_attach_info_not_found(self, server):
        """handle_request MUST return error when session not found."""
        server.orchestrator.worker_store.get_by_task = MagicMock(return_value=None)
        server.orchestrator.get_worker = MagicMock(return_value=None)

        request = MCPRequest(
            jsonrpc="2.0",
            id=8,
            method="tools/call",
            params={
                "name": "parhelia_attach_info",
                "arguments": {"session_id": "nonexistent"},
            },
        )

        response = await server.handle_request(request)

        assert response.result is not None
        content = response.result["content"][0]["text"]
        data = json.loads(content)

        assert data["success"] is False
        assert "not found" in data["error"].lower()


class TestMCPToolDefinitions:
    """Tests for MCP tool definitions."""

    @pytest.fixture
    def server(self, tmp_path):
        """Create server with mocked dependencies."""
        mock_config = MagicMock()
        mock_config.budget.default_ceiling_usd = 10.0
        mock_config.paths.volume_root = "/vol/parhelia"

        with patch("parhelia.mcp_server.load_config", return_value=mock_config):
            with patch("parhelia.mcp_server.PersistentOrchestrator"):
                with patch("parhelia.mcp_server.CheckpointManager"):
                    return ParheliaMCPServer()

    def test_submit_tool_schema(self, server):
        """parhelia_submit tool MUST have required prompt field."""
        tool = server.tools["parhelia_submit"]

        assert "prompt" in tool.input_schema["required"]
        assert tool.input_schema["properties"]["prompt"]["type"] == "string"

    def test_status_tool_schema(self, server):
        """parhelia_status tool MUST have optional task_id."""
        tool = server.tools["parhelia_status"]

        assert "task_id" in tool.input_schema["properties"]
        assert "required" not in tool.input_schema or "task_id" not in tool.input_schema.get("required", [])

    def test_attach_info_tool_schema(self, server):
        """parhelia_attach_info tool MUST require session_id."""
        tool = server.tools["parhelia_attach_info"]

        assert "session_id" in tool.input_schema["required"]

    def test_all_tools_have_descriptions(self, server):
        """All tools MUST have descriptions."""
        for name, tool in server.tools.items():
            assert tool.description, f"Tool {name} missing description"
            assert len(tool.description) > 10, f"Tool {name} description too short"
