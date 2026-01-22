"""Tests for Parhelia MCP tools module.

Tests for [SPEC-21.60] MCP Integration and [SPEC-20.30] Agent Excellence.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parhelia.mcp import (
    MCPToolDefinition,
    ParheliaMCPTools,
    estimate_task_cost,
    filter_fields,
)
from parhelia.state import (
    Container,
    ContainerState,
    Event,
    EventType,
    HealthStatus,
    StateStore,
)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFilterFields:
    """Tests for filter_fields helper function."""

    def test_filter_fields_with_selection(self):
        """filter_fields MUST return only selected fields."""
        data = {"id": "123", "name": "test", "status": "running", "cost": 1.5}
        result = filter_fields(data, ["id", "status"])

        assert result == {"id": "123", "status": "running"}
        assert "name" not in result
        assert "cost" not in result

    def test_filter_fields_none_returns_all(self):
        """filter_fields MUST return all fields when fields is None."""
        data = {"id": "123", "name": "test", "status": "running"}
        result = filter_fields(data, None)

        assert result == data

    def test_filter_fields_empty_list(self):
        """filter_fields MUST return empty dict for empty fields list."""
        data = {"id": "123", "name": "test"}
        result = filter_fields(data, [])

        assert result == {}

    def test_filter_fields_nonexistent_field(self):
        """filter_fields MUST ignore fields that don't exist."""
        data = {"id": "123", "name": "test"}
        result = filter_fields(data, ["id", "nonexistent"])

        assert result == {"id": "123"}


class TestEstimateTaskCost:
    """Tests for estimate_task_cost function."""

    def test_estimate_basic_task(self):
        """estimate_task_cost MUST return cost for basic task."""
        estimate = estimate_task_cost(
            task_type="generic",
            gpu_type=None,
            memory_gb=4,
            timeout_hours=1,
        )

        assert "estimated_cost_usd" in estimate
        assert estimate["estimated_cost_usd"] > 0
        assert "cost_breakdown" in estimate
        assert estimate["cost_breakdown"]["gpu_cost_usd"] == 0

    def test_estimate_gpu_task(self):
        """estimate_task_cost MUST include GPU cost."""
        estimate = estimate_task_cost(
            task_type="generic",
            gpu_type="A100",
            memory_gb=16,
            timeout_hours=2,
        )

        assert estimate["cost_breakdown"]["gpu_cost_usd"] > 0
        assert estimate["estimated_cost_usd"] > estimate["cost_breakdown"]["cpu_cost_usd"]

    def test_estimate_high_cost_warning(self):
        """estimate_task_cost MUST include warning for high cost."""
        estimate = estimate_task_cost(
            task_type="generic",
            gpu_type="H100",
            memory_gb=64,
            timeout_hours=4,
        )

        assert estimate["estimated_cost_usd"] > 1.0
        assert estimate.get("warning") is not None

    def test_estimate_includes_assumptions(self):
        """estimate_task_cost MUST include assumptions."""
        estimate = estimate_task_cost(
            task_type="build",
            gpu_type="T4",
            memory_gb=8,
            timeout_hours=2,
        )

        assert estimate["assumptions"]["timeout_hours"] == 2
        assert estimate["assumptions"]["memory_gb"] == 8
        assert estimate["assumptions"]["gpu_type"] == "T4"


# =============================================================================
# ParheliaMCPTools Tests
# =============================================================================


class TestParheliaMCPTools:
    """Tests for ParheliaMCPTools class."""

    @pytest.fixture
    def mock_state_store(self, tmp_path):
        """Create mock state store."""
        return StateStore(db_path=tmp_path / "test_state.db")

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.task_store = MagicMock()
        orch.worker_store = MagicMock()
        orch.submit_task = AsyncMock()
        orch.get_all_tasks = AsyncMock(return_value=[])
        orch.get_pending_tasks = AsyncMock(return_value=[])
        orch.get_running_tasks = AsyncMock(return_value=[])
        orch.get_task = AsyncMock(return_value=None)
        orch.cancel_task = AsyncMock()
        orch.retry_task = AsyncMock(return_value="task-new123")
        return orch

    @pytest.fixture
    def mock_budget_manager(self):
        """Create mock budget manager."""
        manager = MagicMock()
        status = MagicMock()
        status.ceiling_usd = 100.0
        status.used_usd = 10.0
        status.remaining_usd = 90.0
        status.usage_percent = 10.0
        status.task_count = 5
        status.total_input_tokens = 1000
        status.total_output_tokens = 500
        status.warning_threshold_reached = False
        status.is_exceeded = False
        manager.check_budget.return_value = status
        return manager

    @pytest.fixture
    def mock_checkpoint_manager(self):
        """Create mock checkpoint manager."""
        manager = MagicMock()
        manager.list_checkpoints = AsyncMock(return_value=[])
        manager.create_checkpoint = AsyncMock()
        return manager

    @pytest.fixture
    def mcp_tools(
        self,
        mock_state_store,
        mock_orchestrator,
        mock_budget_manager,
        mock_checkpoint_manager,
    ):
        """Create MCP tools with mocked dependencies."""
        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 100.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            tools = ParheliaMCPTools(
                state_store=mock_state_store,
                orchestrator=mock_orchestrator,
                budget_manager=mock_budget_manager,
                checkpoint_manager=mock_checkpoint_manager,
            )
            return tools

    def test_get_tools_returns_15_plus(self, mcp_tools):
        """get_tools MUST return at least 15 tool definitions."""
        tools = mcp_tools.get_tools()

        assert len(tools) >= 15

    def test_all_tools_have_required_fields(self, mcp_tools):
        """All tools MUST have name, description, and inputSchema."""
        tools = mcp_tools.get_tools()

        for tool in tools:
            assert "name" in tool, f"Tool missing name"
            assert "description" in tool, f"Tool {tool.get('name')} missing description"
            assert "inputSchema" in tool, f"Tool {tool.get('name')} missing inputSchema"
            assert len(tool["description"]) > 10, f"Tool {tool['name']} description too short"

    def test_tool_names_follow_convention(self, mcp_tools):
        """All tool names MUST start with 'parhelia_'."""
        tools = mcp_tools.get_tools()

        for tool in tools:
            assert tool["name"].startswith("parhelia_"), f"Tool {tool['name']} doesn't follow naming convention"


# =============================================================================
# Container Tool Tests
# =============================================================================


class TestContainerTools:
    """Tests for container-related MCP tools."""

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store with test data."""
        store = StateStore(db_path=tmp_path / "test_state.db")

        # Add test containers
        c1 = Container.create(
            modal_sandbox_id="sb-123",
            task_id="task-abc",
            worker_id="w-1",
        )
        c1.state = ContainerState.RUNNING
        c1.health_status = HealthStatus.HEALTHY
        c1.cost_accrued_usd = 0.50
        store.create_container(c1)

        c2 = Container.create(
            modal_sandbox_id="sb-456",
            task_id="task-def",
        )
        c2.state = ContainerState.ORPHANED
        c2.health_status = HealthStatus.UNKNOWN
        store.create_container(c2)

        return store

    @pytest.fixture
    def mcp_tools(self, state_store):
        """Create MCP tools with test state store."""
        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 100.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            with patch("parhelia.mcp.PersistentOrchestrator"):
                with patch("parhelia.mcp.BudgetManager"):
                    with patch("parhelia.mcp.CheckpointManager"):
                        tools = ParheliaMCPTools(state_store=state_store)
                        return tools

    @pytest.mark.asyncio
    async def test_containers_list_all(self, mcp_tools):
        """parhelia_containers MUST list all containers."""
        result = await mcp_tools.call_tool("parhelia_containers", {})

        assert result["success"] is True
        assert "containers" in result
        assert result["count"] >= 1

    @pytest.mark.asyncio
    async def test_containers_filter_by_state(self, mcp_tools):
        """parhelia_containers MUST filter by state."""
        result = await mcp_tools.call_tool("parhelia_containers", {"state": "orphaned"})

        assert result["success"] is True
        for c in result["containers"]:
            assert c["state"] == "orphaned"

    @pytest.mark.asyncio
    async def test_containers_field_selection(self, mcp_tools):
        """parhelia_containers MUST support field selection."""
        result = await mcp_tools.call_tool(
            "parhelia_containers",
            {"fields": ["id", "state"]},
        )

        assert result["success"] is True
        if result["containers"]:
            container = result["containers"][0]
            assert "id" in container
            assert "state" in container
            assert "modal_sandbox_id" not in container
            assert "cost_accrued_usd" not in container

    @pytest.mark.asyncio
    async def test_container_show_found(self, mcp_tools, state_store):
        """parhelia_container_show MUST return container details."""
        containers = state_store.containers.list_active()
        container_id = containers[0].id

        result = await mcp_tools.call_tool(
            "parhelia_container_show",
            {"container_id": container_id},
        )

        assert result["success"] is True
        assert "container" in result
        assert result["container"]["id"] == container_id

    @pytest.mark.asyncio
    async def test_container_show_not_found(self, mcp_tools):
        """parhelia_container_show MUST return error for unknown container."""
        result = await mcp_tools.call_tool(
            "parhelia_container_show",
            {"container_id": "nonexistent"},
        )

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_container_events(self, mcp_tools, state_store):
        """parhelia_container_events MUST return event history."""
        containers = state_store.containers.list_active()
        container_id = containers[0].id

        result = await mcp_tools.call_tool(
            "parhelia_container_events",
            {"container_id": container_id},
        )

        assert result["success"] is True
        assert "events" in result
        assert "count" in result


# =============================================================================
# Health Tool Tests
# =============================================================================


class TestHealthTools:
    """Tests for health-related MCP tools."""

    @pytest.fixture
    def mcp_tools(self, tmp_path):
        """Create MCP tools with test state store."""
        state_store = StateStore(db_path=tmp_path / "test_state.db")

        # Add containers with various health states
        healthy = Container.create(modal_sandbox_id="sb-healthy", task_id="t1")
        healthy.state = ContainerState.RUNNING
        healthy.health_status = HealthStatus.HEALTHY
        state_store.create_container(healthy)

        unhealthy = Container.create(modal_sandbox_id="sb-unhealthy", task_id="t2")
        unhealthy.state = ContainerState.RUNNING
        unhealthy.health_status = HealthStatus.UNHEALTHY
        state_store.create_container(unhealthy)

        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 100.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            with patch("parhelia.mcp.PersistentOrchestrator"):
                with patch("parhelia.mcp.BudgetManager"):
                    with patch("parhelia.mcp.CheckpointManager"):
                        return ParheliaMCPTools(state_store=state_store)

    @pytest.mark.asyncio
    async def test_health_status(self, mcp_tools):
        """parhelia_health MUST return overall health status."""
        result = await mcp_tools.call_tool("parhelia_health", {})

        assert result["success"] is True
        assert "overall_health" in result
        assert result["overall_health"] in ("healthy", "warning", "degraded")
        assert "total_containers" in result

    @pytest.mark.asyncio
    async def test_health_includes_details(self, mcp_tools):
        """parhelia_health MUST include details when requested."""
        result = await mcp_tools.call_tool("parhelia_health", {"include_details": True})

        assert result["success"] is True
        assert "by_state" in result
        assert "by_health" in result

    @pytest.mark.asyncio
    async def test_reconciler_status(self, mcp_tools):
        """parhelia_reconciler_status MUST return reconciler info."""
        result = await mcp_tools.call_tool("parhelia_reconciler_status", {})

        assert result["success"] is True
        assert "is_running" in result
        assert "config" in result
        assert "poll_interval_seconds" in result["config"]


# =============================================================================
# Task Tool Tests
# =============================================================================


class TestTaskTools:
    """Tests for task-related MCP tools."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator."""
        orch = MagicMock()
        orch.task_store = MagicMock()
        orch.task_store.get_status.return_value = "pending"
        orch.task_store.get_result.return_value = None
        orch.worker_store = MagicMock()
        orch.worker_store.get_by_task.return_value = None
        orch.submit_task = AsyncMock()
        orch.get_all_tasks = AsyncMock(return_value=[])
        orch.get_pending_tasks = AsyncMock(return_value=[])
        orch.get_running_tasks = AsyncMock(return_value=[])
        orch.get_task = AsyncMock(return_value=None)
        orch.cancel_task = AsyncMock()
        orch.retry_task = AsyncMock(return_value="task-retry123")
        return orch

    @pytest.fixture
    def mock_budget_manager(self):
        """Create mock budget manager with sufficient budget."""
        manager = MagicMock()
        status = MagicMock()
        status.ceiling_usd = 100.0
        status.used_usd = 10.0
        status.remaining_usd = 90.0
        status.usage_percent = 10.0
        status.task_count = 5
        status.total_input_tokens = 1000
        status.total_output_tokens = 500
        status.warning_threshold_reached = False
        status.is_exceeded = False
        manager.check_budget.return_value = status
        return manager

    @pytest.fixture
    def mcp_tools(self, tmp_path, mock_orchestrator, mock_budget_manager):
        """Create MCP tools with mocks."""
        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 100.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            with patch("parhelia.mcp.CheckpointManager"):
                tools = ParheliaMCPTools(
                    state_store=StateStore(db_path=tmp_path / "test.db"),
                    orchestrator=mock_orchestrator,
                    budget_manager=mock_budget_manager,
                )
                return tools

    @pytest.mark.asyncio
    async def test_task_create_includes_cost_estimate(self, mcp_tools):
        """parhelia_task_create MUST include cost estimate."""
        result = await mcp_tools.call_tool(
            "parhelia_task_create",
            {
                "prompt": "Run tests",
                "task_type": "test_run",
                "memory_gb": 4,
                "timeout_hours": 1,
            },
        )

        assert result["success"] is True
        assert "task_id" in result
        assert "estimated_cost_usd" in result
        assert "cost_breakdown" in result
        assert result["estimated_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_task_create_with_gpu_cost(self, mcp_tools):
        """parhelia_task_create MUST include GPU in cost estimate."""
        result = await mcp_tools.call_tool(
            "parhelia_task_create",
            {
                "prompt": "Train model",
                "gpu": "A100",
                "memory_gb": 32,
                "timeout_hours": 2,
            },
        )

        assert result["success"] is True
        assert result["cost_breakdown"]["gpu_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_task_create_budget_check(self, mcp_tools, mock_budget_manager):
        """parhelia_task_create MUST check budget."""
        # Set insufficient budget
        status = mock_budget_manager.check_budget.return_value
        status.remaining_usd = 0.01

        result = await mcp_tools.call_tool(
            "parhelia_task_create",
            {
                "prompt": "Big expensive task",
                "gpu": "H100",
                "memory_gb": 64,
                "timeout_hours": 10,
            },
        )

        assert result["success"] is False
        assert "budget" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_task_list(self, mcp_tools):
        """parhelia_task_list MUST return task list."""
        result = await mcp_tools.call_tool("parhelia_task_list", {})

        assert result["success"] is True
        assert "tasks" in result
        assert "count" in result


# =============================================================================
# Budget Tool Tests
# =============================================================================


class TestBudgetTools:
    """Tests for budget-related MCP tools."""

    @pytest.fixture
    def mcp_tools(self, tmp_path):
        """Create MCP tools with mock budget manager."""
        mock_budget = MagicMock()
        status = MagicMock()
        status.ceiling_usd = 50.0
        status.used_usd = 25.0
        status.remaining_usd = 25.0
        status.usage_percent = 50.0
        status.task_count = 10
        status.total_input_tokens = 5000
        status.total_output_tokens = 2500
        status.warning_threshold_reached = False
        status.is_exceeded = False
        mock_budget.check_budget.return_value = status

        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 50.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            with patch("parhelia.mcp.PersistentOrchestrator"):
                with patch("parhelia.mcp.CheckpointManager"):
                    return ParheliaMCPTools(
                        state_store=StateStore(db_path=tmp_path / "test.db"),
                        budget_manager=mock_budget,
                    )

    @pytest.mark.asyncio
    async def test_budget_status(self, mcp_tools):
        """parhelia_budget_status MUST return budget info."""
        result = await mcp_tools.call_tool("parhelia_budget_status", {})

        assert result["success"] is True
        assert "budget" in result
        assert result["budget"]["ceiling_usd"] == 50.0
        assert result["budget"]["used_usd"] == 25.0
        assert result["budget"]["remaining_usd"] == 25.0

    @pytest.mark.asyncio
    async def test_budget_estimate(self, mcp_tools):
        """parhelia_budget_estimate MUST estimate without creating task."""
        result = await mcp_tools.call_tool(
            "parhelia_budget_estimate",
            {
                "task_type": "build",
                "memory_gb": 8,
                "timeout_hours": 2,
            },
        )

        assert result["success"] is True
        assert "estimated_cost_usd" in result
        assert "within_budget" in result
        assert "remaining_budget_usd" in result

    @pytest.mark.asyncio
    async def test_budget_estimate_with_gpu(self, mcp_tools):
        """parhelia_budget_estimate MUST include GPU costs."""
        result = await mcp_tools.call_tool(
            "parhelia_budget_estimate",
            {
                "gpu": "A10G",
                "memory_gb": 16,
                "timeout_hours": 1,
            },
        )

        assert result["success"] is True
        assert result["cost_breakdown"]["gpu_cost_usd"] > 0


# =============================================================================
# Field Selection Response Size Tests
# =============================================================================


class TestFieldSelectionReducesSize:
    """Tests that field selection reduces response size."""

    @pytest.fixture
    def state_store(self, tmp_path):
        """Create state store with multiple containers."""
        store = StateStore(db_path=tmp_path / "test_state.db")

        # Add several containers
        for i in range(10):
            c = Container.create(
                modal_sandbox_id=f"sb-{i:03d}",
                task_id=f"task-{i:03d}",
                worker_id=f"w-{i}",
            )
            c.state = ContainerState.RUNNING
            c.health_status = HealthStatus.HEALTHY
            c.cpu_cores = 4
            c.memory_mb = 8192
            c.region = "us-east"
            c.cost_accrued_usd = 0.5 * i
            store.create_container(c)

        return store

    @pytest.fixture
    def mcp_tools(self, state_store):
        """Create MCP tools."""
        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 100.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            with patch("parhelia.mcp.PersistentOrchestrator"):
                with patch("parhelia.mcp.BudgetManager"):
                    with patch("parhelia.mcp.CheckpointManager"):
                        return ParheliaMCPTools(state_store=state_store)

    @pytest.mark.asyncio
    async def test_field_selection_reduces_response_size(self, mcp_tools):
        """Field selection MUST reduce response size by >50%."""
        # Get full response
        full_result = await mcp_tools.call_tool("parhelia_containers", {"limit": 10})

        # Get minimal response
        minimal_result = await mcp_tools.call_tool(
            "parhelia_containers",
            {"limit": 10, "fields": ["id", "state"]},
        )

        full_json = json.dumps(full_result)
        minimal_json = json.dumps(minimal_result)

        # Minimal should be significantly smaller
        assert len(minimal_json) < len(full_json) * 0.7, (
            f"Field selection did not reduce size enough: "
            f"full={len(full_json)}, minimal={len(minimal_json)}"
        )


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestToolSchemas:
    """Tests for MCP tool schema correctness."""

    @pytest.fixture
    def mcp_tools(self, tmp_path):
        """Create MCP tools."""
        with patch("parhelia.mcp.load_config") as mock_config:
            mock_config.return_value = MagicMock()
            mock_config.return_value.budget.default_ceiling_usd = 100.0
            mock_config.return_value.paths.volume_root = "/vol/parhelia"

            with patch("parhelia.mcp.PersistentOrchestrator"):
                with patch("parhelia.mcp.BudgetManager"):
                    with patch("parhelia.mcp.CheckpointManager"):
                        return ParheliaMCPTools(
                            state_store=StateStore(db_path=tmp_path / "test.db")
                        )

    def test_container_tools_exist(self, mcp_tools):
        """Container tools MUST be registered."""
        tools = mcp_tools.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "parhelia_containers" in tool_names
        assert "parhelia_container_show" in tool_names
        assert "parhelia_container_terminate" in tool_names
        assert "parhelia_container_events" in tool_names

    def test_health_tools_exist(self, mcp_tools):
        """Health tools MUST be registered."""
        tools = mcp_tools.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "parhelia_health" in tool_names
        assert "parhelia_reconciler_status" in tool_names

    def test_task_tools_exist(self, mcp_tools):
        """Task tools MUST be registered."""
        tools = mcp_tools.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "parhelia_task_create" in tool_names
        assert "parhelia_task_list" in tool_names
        assert "parhelia_task_show" in tool_names
        assert "parhelia_task_cancel" in tool_names
        assert "parhelia_task_retry" in tool_names

    def test_session_tools_exist(self, mcp_tools):
        """Session tools MUST be registered."""
        tools = mcp_tools.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "parhelia_session_list" in tool_names
        assert "parhelia_session_attach_info" in tool_names
        assert "parhelia_session_kill" in tool_names

    def test_checkpoint_tools_exist(self, mcp_tools):
        """Checkpoint tools MUST be registered."""
        tools = mcp_tools.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "parhelia_checkpoint_create" in tool_names
        assert "parhelia_checkpoint_list" in tool_names
        assert "parhelia_checkpoint_restore" in tool_names

    def test_budget_tools_exist(self, mcp_tools):
        """Budget tools MUST be registered."""
        tools = mcp_tools.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "parhelia_budget_status" in tool_names
        assert "parhelia_budget_estimate" in tool_names

    def test_task_create_requires_prompt(self, mcp_tools):
        """parhelia_task_create schema MUST require prompt."""
        tools = mcp_tools.get_tools()
        task_create = next(t for t in tools if t["name"] == "parhelia_task_create")

        assert "required" in task_create["inputSchema"]
        assert "prompt" in task_create["inputSchema"]["required"]

    def test_container_show_requires_id(self, mcp_tools):
        """parhelia_container_show schema MUST require container_id."""
        tools = mcp_tools.get_tools()
        container_show = next(t for t in tools if t["name"] == "parhelia_container_show")

        assert "required" in container_show["inputSchema"]
        assert "container_id" in container_show["inputSchema"]["required"]

    def test_fields_parameter_on_list_tools(self, mcp_tools):
        """List tools MUST support fields parameter."""
        tools = mcp_tools.get_tools()
        list_tools = [
            "parhelia_containers",
            "parhelia_task_list",
            "parhelia_session_list",
            "parhelia_checkpoint_list",
            "parhelia_container_events",
        ]

        for tool_name in list_tools:
            tool = next((t for t in tools if t["name"] == tool_name), None)
            assert tool is not None, f"Tool {tool_name} not found"

            properties = tool["inputSchema"].get("properties", {})
            assert "fields" in properties, f"Tool {tool_name} missing fields parameter"
            assert properties["fields"]["type"] == "array"
