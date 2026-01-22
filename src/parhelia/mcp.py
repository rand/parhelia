"""Parhelia MCP tools module.

Implements [SPEC-21.60] MCP Integration and [SPEC-20.30] Agent Excellence.

Provides 15+ MCP tools for control plane introspection and task management:

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
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine

from parhelia.budget import BudgetManager
from parhelia.checkpoint import CheckpointManager
from parhelia.config import load_config
from parhelia.orchestrator import Task, TaskRequirements, TaskType
from parhelia.persistence import PersistentOrchestrator
from parhelia.reconciler import ContainerReconciler, RealModalClient, ReconcilerConfig
from parhelia.state import (
    Container,
    ContainerState,
    EventType,
    HealthStatus,
    StateStore,
)


@dataclass
class MCPToolDefinition:
    """MCP tool definition with schema and handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Coroutine[Any, Any, dict[str, Any]]]


def filter_fields(data: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    """Filter dictionary to only include specified fields.

    Args:
        data: Source dictionary
        fields: List of field names to include, or None for all fields

    Returns:
        Filtered dictionary with only requested fields
    """
    if fields is None:
        return data
    return {k: v for k, v in data.items() if k in fields}


def estimate_task_cost(
    task_type: str,
    gpu_type: str | None,
    memory_gb: int,
    timeout_hours: int,
) -> dict[str, Any]:
    """Estimate cost for a task based on resources.

    Returns cost breakdown and total estimated cost in USD.
    """
    # Base rates per hour (example rates, would be configurable in production)
    cpu_rate_per_hour = 0.05
    memory_rate_per_gb_hour = 0.01
    gpu_rates = {
        "T4": 0.50,
        "A10G": 1.00,
        "A100": 2.50,
        "H100": 4.00,
    }

    # Calculate component costs
    cpu_cost = cpu_rate_per_hour * timeout_hours
    memory_cost = memory_rate_per_gb_hour * memory_gb * timeout_hours
    gpu_cost = gpu_rates.get(gpu_type, 0) * timeout_hours if gpu_type else 0

    total = cpu_cost + memory_cost + gpu_cost

    return {
        "estimated_cost_usd": round(total, 4),
        "cost_breakdown": {
            "cpu_cost_usd": round(cpu_cost, 4),
            "memory_cost_usd": round(memory_cost, 4),
            "gpu_cost_usd": round(gpu_cost, 4),
        },
        "assumptions": {
            "timeout_hours": timeout_hours,
            "memory_gb": memory_gb,
            "gpu_type": gpu_type,
        },
        "warning": "This is an estimate. Actual costs may vary." if total > 1.0 else None,
    }


class ParheliaMCPTools:
    """Collection of MCP tools for Parhelia control plane.

    Provides stateless tools that query StateStore for all data.
    Each tool returns structured JSON responses suitable for agent consumption.
    """

    def __init__(
        self,
        state_store: StateStore | None = None,
        orchestrator: PersistentOrchestrator | None = None,
        budget_manager: BudgetManager | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        """Initialize MCP tools with dependencies.

        Args:
            state_store: Container/event state store (created if None)
            orchestrator: Task orchestrator (created if None)
            budget_manager: Budget manager (created if None)
            checkpoint_manager: Checkpoint manager (created if None)
        """
        config = load_config()

        self.state_store = state_store or StateStore()
        self.orchestrator = orchestrator or PersistentOrchestrator()
        self.budget_manager = budget_manager or BudgetManager(
            ceiling_usd=config.budget.default_ceiling_usd
        )
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            storage_root=config.paths.volume_root + "/checkpoints"
        )
        self.reconciler = ContainerReconciler(
            state_store=self.state_store,
            modal_client=RealModalClient(),
            config=ReconcilerConfig(),
        )

        self._tools: dict[str, MCPToolDefinition] = {}
        self._register_all_tools()

    def _register_all_tools(self) -> None:
        """Register all MCP tools."""
        # Container tools
        self._register_container_tools()
        # Health tools
        self._register_health_tools()
        # Task tools
        self._register_task_tools()
        # Session tools
        self._register_session_tools()
        # Checkpoint tools
        self._register_checkpoint_tools()
        # Budget tools
        self._register_budget_tools()
        # Event streaming tools (Wave 5)
        self._register_event_tools()

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for MCP protocol.

        Returns:
            List of tool definitions with name, description, and inputSchema.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result as dictionary

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return await self._tools[name].handler(**arguments)

    # =========================================================================
    # Container Tools
    # =========================================================================

    def _register_container_tools(self) -> None:
        """Register container-related MCP tools."""

        self._tools["parhelia_containers"] = MCPToolDefinition(
            name="parhelia_containers",
            description="List containers tracked by the control plane with optional filtering by state or health.",
            input_schema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["all", "running", "created", "stopped", "terminated", "orphaned"],
                        "description": "Filter by container state (default: all)",
                    },
                    "health": {
                        "type": "string",
                        "enum": ["all", "healthy", "degraded", "unhealthy", "dead", "unknown"],
                        "description": "Filter by health status (default: all)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum containers to return (default: 20)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return for each container (reduces response size)",
                    },
                },
            },
            handler=self._handle_containers,
        )

        self._tools["parhelia_container_show"] = MCPToolDefinition(
            name="parhelia_container_show",
            description="Get detailed information about a specific container by ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID (c-xxxxx) or Modal sandbox ID",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return (reduces response size)",
                    },
                },
                "required": ["container_id"],
            },
            handler=self._handle_container_show,
        )

        self._tools["parhelia_container_terminate"] = MCPToolDefinition(
            name="parhelia_container_terminate",
            description="Terminate a running container. Use with caution.",
            input_schema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID to terminate",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force termination even if container is busy (default: false)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for termination (for audit)",
                    },
                },
                "required": ["container_id"],
            },
            handler=self._handle_container_terminate,
        )

        self._tools["parhelia_container_events"] = MCPToolDefinition(
            name="parhelia_container_events",
            description="Get event history for a container.",
            input_schema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID to get events for",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 50)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return per event",
                    },
                },
                "required": ["container_id"],
            },
            handler=self._handle_container_events,
        )

    async def _handle_containers(
        self,
        state: str = "all",
        health: str = "all",
        limit: int = 20,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_containers tool call."""
        store = self.state_store

        if state != "all":
            containers = store.get_containers_by_state(ContainerState(state))
        elif health != "all":
            containers = store.get_containers_by_health(HealthStatus(health))
        else:
            containers = store.containers.list_active(limit)

        containers = containers[:limit]

        container_data = []
        for c in containers:
            item = {
                "id": c.id,
                "modal_sandbox_id": c.modal_sandbox_id,
                "state": c.state.value,
                "health_status": c.health_status.value,
                "task_id": c.task_id,
                "worker_id": c.worker_id,
                "created_at": c.created_at.isoformat(),
                "cost_accrued_usd": c.cost_accrued_usd,
            }
            container_data.append(filter_fields(item, fields))

        return {
            "success": True,
            "containers": container_data,
            "count": len(container_data),
            "filters": {"state": state, "health": health},
        }

    async def _handle_container_show(
        self,
        container_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_container_show tool call."""
        store = self.state_store

        container = store.get_container(container_id)
        if not container:
            container = store.get_container_by_modal_id(container_id)

        if not container:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }

        data = {
            "id": container.id,
            "modal_sandbox_id": container.modal_sandbox_id,
            "state": container.state.value,
            "health_status": container.health_status.value,
            "worker_id": container.worker_id,
            "task_id": container.task_id,
            "session_id": container.session_id,
            "created_at": container.created_at.isoformat(),
            "started_at": container.started_at.isoformat() if container.started_at else None,
            "terminated_at": container.terminated_at.isoformat() if container.terminated_at else None,
            "last_heartbeat_at": container.last_heartbeat_at.isoformat() if container.last_heartbeat_at else None,
            "exit_code": container.exit_code,
            "termination_reason": container.termination_reason,
            "consecutive_failures": container.consecutive_failures,
            "cpu_cores": container.cpu_cores,
            "memory_mb": container.memory_mb,
            "gpu_type": container.gpu_type,
            "region": container.region,
            "cost_accrued_usd": container.cost_accrued_usd,
            "cost_rate_per_hour": container.cost_rate_per_hour,
        }

        return {
            "success": True,
            "container": filter_fields(data, fields),
        }

    async def _handle_container_terminate(
        self,
        container_id: str,
        force: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_container_terminate tool call."""
        store = self.state_store

        container = store.get_container(container_id)
        if not container:
            container = store.get_container_by_modal_id(container_id)

        if not container:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }

        if container.state == ContainerState.TERMINATED:
            return {
                "success": False,
                "error": "Container already terminated",
            }

        try:
            success = await self.reconciler.modal_client.terminate_sandbox(
                container.modal_sandbox_id
            )

            if success:
                store.update_container_state(
                    container.id,
                    ContainerState.TERMINATED,
                    reason=reason or "Terminated via MCP tool",
                )
                return {
                    "success": True,
                    "container_id": container.id,
                    "message": "Container terminated successfully",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to terminate container in Modal",
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Termination failed: {str(e)}",
            }

    async def _handle_container_events(
        self,
        container_id: str,
        event_type: str | None = None,
        limit: int = 50,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_container_events tool call."""
        store = self.state_store

        evt_type = EventType(event_type) if event_type else None
        events = store.get_events(
            container_id=container_id,
            event_type=evt_type,
            limit=limit,
        )

        event_data = []
        for e in events:
            item = {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "message": e.message,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "source": e.source,
            }
            event_data.append(filter_fields(item, fields))

        return {
            "success": True,
            "container_id": container_id,
            "events": event_data,
            "count": len(event_data),
        }

    # =========================================================================
    # Health Tools
    # =========================================================================

    def _register_health_tools(self) -> None:
        """Register health-related MCP tools."""

        self._tools["parhelia_health"] = MCPToolDefinition(
            name="parhelia_health",
            description="Get control plane health status including container statistics and system health.",
            input_schema={
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed breakdown (default: true)",
                    },
                },
            },
            handler=self._handle_health,
        )

        self._tools["parhelia_reconciler_status"] = MCPToolDefinition(
            name="parhelia_reconciler_status",
            description="Get reconciler status including configuration and recent activity.",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_reconciler_status,
        )

    async def _handle_health(
        self,
        include_details: bool = True,
    ) -> dict[str, Any]:
        """Handle parhelia_health tool call."""
        store = self.state_store
        stats = store.get_container_stats()

        # Determine overall health
        orphaned = stats.by_state.get("orphaned", 0)
        unhealthy = stats.by_health.get("unhealthy", 0) + stats.by_health.get("dead", 0)

        if orphaned > 0 or unhealthy > 5:
            overall_health = "degraded"
        elif unhealthy > 0:
            overall_health = "warning"
        else:
            overall_health = "healthy"

        result = {
            "success": True,
            "overall_health": overall_health,
            "total_containers": stats.total,
            "total_cost_usd": stats.total_cost_usd,
            "orphaned_containers": orphaned,
            "unhealthy_containers": unhealthy,
        }

        if include_details:
            result["by_state"] = stats.by_state
            result["by_health"] = stats.by_health
            result["oldest_running"] = stats.oldest_running.isoformat() if stats.oldest_running else None

        # Add warnings
        warnings = []
        if orphaned > 0:
            warnings.append(f"{orphaned} orphaned container(s) detected - run cleanup")
        if unhealthy > 3:
            warnings.append(f"{unhealthy} unhealthy container(s) - investigate health issues")
        if stats.total_cost_usd > 10.0:
            warnings.append(f"High total cost: ${stats.total_cost_usd:.2f}")

        if warnings:
            result["warnings"] = warnings

        return result

    async def _handle_reconciler_status(self) -> dict[str, Any]:
        """Handle parhelia_reconciler_status tool call."""
        store = self.state_store

        # Get recent reconciliation events
        drift_events = store.get_events(
            event_type=EventType.STATE_DRIFT_CORRECTED,
            limit=10,
        )
        orphan_events = store.get_events(
            event_type=EventType.ORPHAN_DETECTED,
            limit=10,
        )
        error_events = store.get_events(
            event_type=EventType.RECONCILE_FAILED,
            limit=5,
        )

        return {
            "success": True,
            "is_running": self.reconciler.is_running,
            "config": {
                "poll_interval_seconds": self.reconciler.config.poll_interval_seconds,
                "stale_threshold_seconds": self.reconciler.config.stale_threshold_seconds,
                "orphan_grace_period_seconds": self.reconciler.config.orphan_grace_period_seconds,
                "auto_terminate_orphans": self.reconciler.config.auto_terminate_orphans,
            },
            "recent_activity": {
                "drift_corrections": len(drift_events),
                "orphans_detected": len(orphan_events),
                "errors": len(error_events),
            },
            "last_error": error_events[0].message if error_events else None,
        }

    # =========================================================================
    # Task Tools
    # =========================================================================

    def _register_task_tools(self) -> None:
        """Register task-related MCP tools."""

        self._tools["parhelia_task_create"] = MCPToolDefinition(
            name="parhelia_task_create",
            description="Create a new task for remote execution. Returns task ID and cost estimate.",
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Task prompt/instructions for Claude Code",
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["generic", "code_fix", "test_run", "build", "lint", "refactor"],
                        "description": "Type of task (default: generic)",
                    },
                    "gpu": {
                        "type": "string",
                        "enum": ["T4", "A10G", "A100", "H100"],
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
                    "dispatch": {
                        "type": "boolean",
                        "description": "Immediately dispatch to Modal (default: true)",
                    },
                },
                "required": ["prompt"],
            },
            handler=self._handle_task_create,
        )

        self._tools["parhelia_task_list"] = MCPToolDefinition(
            name="parhelia_task_list",
            description="List tasks with optional status filter.",
            input_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "pending", "running", "completed", "failed"],
                        "description": "Filter by status (default: all)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum tasks to return (default: 20)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return",
                    },
                },
            },
            handler=self._handle_task_list,
        )

        self._tools["parhelia_task_show"] = MCPToolDefinition(
            name="parhelia_task_show",
            description="Get detailed information about a specific task.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to retrieve",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_task_show,
        )

        self._tools["parhelia_task_cancel"] = MCPToolDefinition(
            name="parhelia_task_cancel",
            description="Cancel a running task.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to cancel",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for cancellation",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_task_cancel,
        )

        self._tools["parhelia_task_retry"] = MCPToolDefinition(
            name="parhelia_task_retry",
            description="Retry a failed task.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID to retry",
                    },
                },
                "required": ["task_id"],
            },
            handler=self._handle_task_retry,
        )

    async def _handle_task_create(
        self,
        prompt: str,
        task_type: str = "generic",
        gpu: str | None = None,
        memory_gb: int = 4,
        timeout_hours: int = 4,
        dispatch: bool = True,
    ) -> dict[str, Any]:
        """Handle parhelia_task_create tool call."""
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

        # Calculate cost estimate
        cost_estimate = estimate_task_cost(task_type, gpu, memory_gb, timeout_hours)

        # Check budget
        budget_status = self.budget_manager.check_budget(raise_on_exceeded=False)
        if cost_estimate["estimated_cost_usd"] > budget_status.remaining_usd:
            return {
                "success": False,
                "error": "Insufficient budget",
                "estimated_cost_usd": cost_estimate["estimated_cost_usd"],
                "remaining_budget_usd": budget_status.remaining_usd,
                "suggestion": "Reduce timeout_hours or memory_gb, or increase budget with parhelia budget set",
            }

        await self.orchestrator.submit_task(task)

        result = {
            "success": True,
            "task_id": task_id,
            "session_id": f"ph-{task_id}",
            "status": "pending",
            "estimated_cost_usd": cost_estimate["estimated_cost_usd"],
            "cost_breakdown": cost_estimate["cost_breakdown"],
            "message": "Task created successfully",
            "next_actions": [
                {"action": "status", "command": f"parhelia task show {task_id}"},
                {"action": "attach", "command": f"parhelia attach {task_id}"},
            ],
        }

        if cost_estimate.get("warning"):
            result["cost_warning"] = cost_estimate["warning"]

        return result

    async def _handle_task_list(
        self,
        status: str = "all",
        limit: int = 20,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_task_list tool call."""
        tasks = await self.orchestrator.list_tasks(
            status=status if status != "all" else None,
            limit=limit,
        )

        task_data = []
        for t in tasks:
            item = {
                "id": t.id,
                "prompt_preview": t.prompt[:100] + "..." if len(t.prompt) > 100 else t.prompt,
                "task_type": t.task_type.value,
                "status": self.orchestrator.task_store.get_status(t.id),
                "created_at": t.created_at.isoformat(),
            }
            task_data.append(filter_fields(item, fields))

        return {
            "success": True,
            "tasks": task_data,
            "count": len(task_data),
        }

    async def _handle_task_show(
        self,
        task_id: str,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_task_show tool call."""
        task = await self.orchestrator.get_task(task_id)

        if not task:
            return {
                "success": False,
                "error": f"Task not found: {task_id}",
            }

        worker = self.orchestrator.worker_store.get_by_task(task_id)
        result = self.orchestrator.task_store.get_result(task_id)

        data = {
            "id": task.id,
            "prompt": task.prompt,
            "task_type": task.task_type.value,
            "status": self.orchestrator.task_store.get_status(task_id),
            "created_at": task.created_at.isoformat(),
            "requirements": {
                "needs_gpu": task.requirements.needs_gpu,
                "gpu_type": task.requirements.gpu_type,
                "min_memory_gb": task.requirements.min_memory_gb,
            },
        }

        if worker:
            data["worker"] = {
                "id": worker.id,
                "state": worker.state.value,
                "target_type": worker.target_type,
            }

        if result:
            data["result"] = {
                "status": result.status,
                "output_preview": result.output[:500] if result.output else None,
                "cost_usd": result.cost_usd,
            }

        return {
            "success": True,
            "task": filter_fields(data, fields),
        }

    async def _handle_task_cancel(
        self,
        task_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_task_cancel tool call."""
        task = await self.orchestrator.get_task(task_id)

        if not task:
            return {
                "success": False,
                "error": f"Task not found: {task_id}",
            }

        status = self.orchestrator.task_store.get_status(task_id)
        if status not in ("pending", "running"):
            return {
                "success": False,
                "error": f"Task cannot be cancelled (status: {status})",
            }

        # Cancel the task
        try:
            await self.orchestrator.cancel_task(task_id, reason=reason)
            return {
                "success": True,
                "task_id": task_id,
                "message": "Task cancelled",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to cancel task: {str(e)}",
            }

    async def _handle_task_retry(self, task_id: str) -> dict[str, Any]:
        """Handle parhelia_task_retry tool call."""
        task = await self.orchestrator.get_task(task_id)

        if not task:
            return {
                "success": False,
                "error": f"Task not found: {task_id}",
            }

        status = self.orchestrator.task_store.get_status(task_id)
        if status != "failed":
            return {
                "success": False,
                "error": f"Only failed tasks can be retried (status: {status})",
            }

        try:
            new_task_id = await self.orchestrator.retry_task(task_id)
            return {
                "success": True,
                "original_task_id": task_id,
                "new_task_id": new_task_id,
                "message": "Task retry initiated",
                "next_actions": [
                    {"action": "status", "command": f"parhelia task show {new_task_id}"},
                ],
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to retry task: {str(e)}",
            }

    # =========================================================================
    # Session Tools
    # =========================================================================

    def _register_session_tools(self) -> None:
        """Register session-related MCP tools."""

        self._tools["parhelia_session_list"] = MCPToolDefinition(
            name="parhelia_session_list",
            description="List active sessions.",
            input_schema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["all", "running", "idle", "completed"],
                        "description": "Filter by session state",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum sessions to return (default: 20)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return",
                    },
                },
            },
            handler=self._handle_session_list,
        )

        self._tools["parhelia_session_attach_info"] = MCPToolDefinition(
            name="parhelia_session_attach_info",
            description="Get SSH connection information for attaching to a session.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID or task ID",
                    },
                },
                "required": ["session_id"],
            },
            handler=self._handle_session_attach_info,
        )

        self._tools["parhelia_session_kill"] = MCPToolDefinition(
            name="parhelia_session_kill",
            description="Kill a running session and its associated container.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to kill",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for killing the session",
                    },
                },
                "required": ["session_id"],
            },
            handler=self._handle_session_kill,
        )

    async def _handle_session_list(
        self,
        state: str = "all",
        limit: int = 20,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_session_list tool call."""
        # Sessions are workers with active tasks
        workers = self.orchestrator.worker_store.get_all()

        if state != "all":
            workers = [w for w in workers if w.state.value == state]

        workers = workers[:limit]

        session_data = []
        for w in workers:
            item = {
                "session_id": w.id,
                "task_id": w.assigned_task_id,
                "state": w.state.value,
                "target_type": w.target_type,
                "created_at": w.created_at.isoformat() if w.created_at else None,
            }
            session_data.append(filter_fields(item, fields))

        return {
            "success": True,
            "sessions": session_data,
            "count": len(session_data),
        }

    async def _handle_session_attach_info(self, session_id: str) -> dict[str, Any]:
        """Handle parhelia_session_attach_info tool call."""
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
            "cli_command": f"parhelia attach {session_id}",
        }

    async def _handle_session_kill(
        self,
        session_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_session_kill tool call."""
        worker = self.orchestrator.worker_store.get_by_task(session_id)
        if not worker:
            worker = self.orchestrator.get_worker(session_id)

        if not worker:
            return {
                "success": False,
                "error": f"Session not found: {session_id}",
            }

        # Find associated container
        containers = self.state_store.get_containers_for_task(worker.assigned_task_id or session_id)

        try:
            # Terminate containers
            for container in containers:
                if container.state != ContainerState.TERMINATED:
                    await self.reconciler.modal_client.terminate_sandbox(
                        container.modal_sandbox_id
                    )
                    self.state_store.update_container_state(
                        container.id,
                        ContainerState.TERMINATED,
                        reason=reason or "Session killed via MCP",
                    )

            return {
                "success": True,
                "session_id": session_id,
                "containers_terminated": len(containers),
                "message": "Session killed",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to kill session: {str(e)}",
            }

    # =========================================================================
    # Checkpoint Tools
    # =========================================================================

    def _register_checkpoint_tools(self) -> None:
        """Register checkpoint-related MCP tools."""

        self._tools["parhelia_checkpoint_create"] = MCPToolDefinition(
            name="parhelia_checkpoint_create",
            description="Create a checkpoint for a session.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to checkpoint",
                    },
                    "message": {
                        "type": "string",
                        "description": "Checkpoint message/description",
                    },
                },
                "required": ["session_id"],
            },
            handler=self._handle_checkpoint_create,
        )

        self._tools["parhelia_checkpoint_list"] = MCPToolDefinition(
            name="parhelia_checkpoint_list",
            description="List checkpoints for a session.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to list checkpoints for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum checkpoints to return (default: 20)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return",
                    },
                },
                "required": ["session_id"],
            },
            handler=self._handle_checkpoint_list,
        )

        self._tools["parhelia_checkpoint_restore"] = MCPToolDefinition(
            name="parhelia_checkpoint_restore",
            description="Restore a session from a checkpoint.",
            input_schema={
                "type": "object",
                "properties": {
                    "checkpoint_id": {
                        "type": "string",
                        "description": "Checkpoint ID to restore",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (if checkpoint ID is ambiguous)",
                    },
                },
                "required": ["checkpoint_id"],
            },
            handler=self._handle_checkpoint_restore,
        )

    async def _handle_checkpoint_create(
        self,
        session_id: str,
        message: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_checkpoint_create tool call."""
        from parhelia.checkpoint import CheckpointTrigger
        from parhelia.session import Session, SessionState

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
                "session_id": session_id,
                "created_at": cp.created_at.isoformat() if cp.created_at else None,
                "message": "Checkpoint created successfully",
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create checkpoint: {str(e)}",
            }

    async def _handle_checkpoint_list(
        self,
        session_id: str,
        limit: int = 20,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_checkpoint_list tool call."""
        checkpoints = await self.checkpoint_manager.list_checkpoints(session_id)
        checkpoints = checkpoints[:limit]

        cp_data = []
        for cp in checkpoints:
            item = {
                "id": cp.id,
                "session_id": cp.session_id,
                "trigger": cp.trigger.value,
                "created_at": cp.created_at.isoformat() if cp.created_at else None,
                "tokens_used": cp.tokens_used,
                "cost_estimate": cp.cost_estimate,
            }
            cp_data.append(filter_fields(item, fields))

        return {
            "success": True,
            "session_id": session_id,
            "checkpoints": cp_data,
            "count": len(cp_data),
        }

    async def _handle_checkpoint_restore(
        self,
        checkpoint_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_checkpoint_restore tool call."""
        # This is a placeholder - full implementation would use ResumeManager
        return {
            "success": False,
            "error": "Checkpoint restore via MCP not yet implemented",
            "suggestion": f"Use CLI: parhelia checkpoint rollback {checkpoint_id}",
        }

    # =========================================================================
    # Budget Tools
    # =========================================================================

    def _register_budget_tools(self) -> None:
        """Register budget-related MCP tools."""

        self._tools["parhelia_budget_status"] = MCPToolDefinition(
            name="parhelia_budget_status",
            description="Get current budget status and usage.",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_budget_status,
        )

        self._tools["parhelia_budget_estimate"] = MCPToolDefinition(
            name="parhelia_budget_estimate",
            description="Estimate cost for a potential task without creating it.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["generic", "code_fix", "test_run", "build", "lint", "refactor"],
                        "description": "Type of task",
                    },
                    "gpu": {
                        "type": "string",
                        "enum": ["T4", "A10G", "A100", "H100"],
                        "description": "GPU type if needed",
                    },
                    "memory_gb": {
                        "type": "integer",
                        "description": "Memory in GB (default: 4)",
                    },
                    "timeout_hours": {
                        "type": "integer",
                        "description": "Estimated runtime in hours (default: 1)",
                    },
                },
            },
            handler=self._handle_budget_estimate,
        )

    async def _handle_budget_status(self) -> dict[str, Any]:
        """Handle parhelia_budget_status tool call."""
        status = self.budget_manager.check_budget(raise_on_exceeded=False)

        result = {
            "success": True,
            "budget": {
                "ceiling_usd": status.ceiling_usd,
                "used_usd": status.used_usd,
                "remaining_usd": status.remaining_usd,
                "usage_percent": status.usage_percent,
                "task_count": status.task_count,
                "total_input_tokens": status.total_input_tokens,
                "total_output_tokens": status.total_output_tokens,
            },
        }

        if status.warning_threshold_reached:
            result["warning"] = "Budget running low"
        if status.is_exceeded:
            result["error"] = "Budget exceeded"

        return result

    async def _handle_budget_estimate(
        self,
        task_type: str = "generic",
        gpu: str | None = None,
        memory_gb: int = 4,
        timeout_hours: int = 1,
    ) -> dict[str, Any]:
        """Handle parhelia_budget_estimate tool call."""
        estimate = estimate_task_cost(task_type, gpu, memory_gb, timeout_hours)

        # Check against budget
        status = self.budget_manager.check_budget(raise_on_exceeded=False)
        within_budget = estimate["estimated_cost_usd"] <= status.remaining_usd

        return {
            "success": True,
            **estimate,
            "within_budget": within_budget,
            "remaining_budget_usd": status.remaining_usd,
        }

    # =========================================================================
    # Event Streaming Tools (Wave 5)
    # =========================================================================

    def _register_event_tools(self) -> None:
        """Register event streaming MCP tools."""

        self._tools["parhelia_events_subscribe"] = MCPToolDefinition(
            name="parhelia_events_subscribe",
            description="Subscribe to real-time event stream. Returns subscription ID for use with parhelia_events_stream.",
            input_schema={
                "type": "object",
                "properties": {
                    "event_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to specific event types (e.g., ['container_started', 'error'])",
                    },
                    "levels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["debug", "info", "warning", "error"]},
                        "description": "Filter by severity levels",
                    },
                    "container_id": {
                        "type": "string",
                        "description": "Filter to specific container",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Filter to specific task",
                    },
                },
            },
            handler=self._handle_events_subscribe,
        )

        self._tools["parhelia_events_stream"] = MCPToolDefinition(
            name="parhelia_events_stream",
            description="Get next events from a subscription. Returns events with <500ms latency.",
            input_schema={
                "type": "object",
                "properties": {
                    "subscription_id": {
                        "type": "string",
                        "description": "Subscription ID from parhelia_events_subscribe",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "How long to wait for events (default: 30, max: 60)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 10)",
                    },
                },
                "required": ["subscription_id"],
            },
            handler=self._handle_events_stream,
        )

        self._tools["parhelia_events_unsubscribe"] = MCPToolDefinition(
            name="parhelia_events_unsubscribe",
            description="Cancel an event subscription.",
            input_schema={
                "type": "object",
                "properties": {
                    "subscription_id": {
                        "type": "string",
                        "description": "Subscription ID to cancel",
                    },
                },
                "required": ["subscription_id"],
            },
            handler=self._handle_events_unsubscribe,
        )

        self._tools["parhelia_events_list"] = MCPToolDefinition(
            name="parhelia_events_list",
            description="Query historical events with filtering.",
            input_schema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Filter by container ID",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Filter by task ID",
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type",
                    },
                    "since_minutes": {
                        "type": "integer",
                        "description": "Only events from last N minutes",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 50)",
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific fields to return (reduces response size)",
                    },
                },
            },
            handler=self._handle_events_list,
        )

        self._tools["parhelia_events_replay"] = MCPToolDefinition(
            name="parhelia_events_replay",
            description="Replay historical events for a container in chronological order.",
            input_schema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "Container ID to replay events for",
                    },
                    "from_start": {
                        "type": "boolean",
                        "description": "Start from container creation (default: true)",
                    },
                    "since_minutes": {
                        "type": "integer",
                        "description": "Only events from last N minutes (ignored if from_start)",
                    },
                },
                "required": ["container_id"],
            },
            handler=self._handle_events_replay,
        )

    async def _handle_events_subscribe(
        self,
        event_types: list[str] | None = None,
        levels: list[str] | None = None,
        container_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_events_subscribe tool call."""
        from parhelia.events import EventFilter, EventLogger, SubscriptionManager

        # Create filter
        parsed_event_types = None
        if event_types:
            parsed_event_types = [EventType(t) for t in event_types]

        filter = EventFilter(
            event_types=parsed_event_types,
            levels=levels,
            container_id=container_id,
            task_id=task_id,
        )

        # Get or create subscription manager
        if not hasattr(self, "_subscription_manager"):
            logger = EventLogger(self.state_store)
            self._subscription_manager = SubscriptionManager(logger)

        # Create subscription
        subscription = self._subscription_manager.subscribe(filter)

        return {
            "success": True,
            "subscription_id": subscription.id,
            "filter": filter.to_dict(),
            "created_at": subscription.created_at.isoformat(),
            "message": "Subscription created. Use parhelia_events_stream to receive events.",
            "next_actions": [
                {
                    "action": "stream",
                    "description": "Get next events",
                    "tool": "parhelia_events_stream",
                    "args": {"subscription_id": subscription.id},
                },
            ],
        }

    async def _handle_events_stream(
        self,
        subscription_id: str,
        timeout_seconds: float = 30.0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Handle parhelia_events_stream tool call."""
        if not hasattr(self, "_subscription_manager"):
            return {
                "success": False,
                "error": "No subscriptions exist. Use parhelia_events_subscribe first.",
            }

        subscription = self._subscription_manager.get_subscription(subscription_id)
        if not subscription:
            return {
                "success": False,
                "error": f"Subscription not found: {subscription_id}",
                "suggestion": "Create a new subscription with parhelia_events_subscribe",
            }

        # Clamp timeout
        timeout_seconds = min(max(timeout_seconds, 1.0), 60.0)

        # Collect events
        events = []
        start_time = datetime.utcnow()

        while len(events) < limit:
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            remaining = timeout_seconds - elapsed

            if remaining <= 0:
                break

            event = await self._subscription_manager.get_next_event(
                subscription_id,
                timeout=min(remaining, 0.5),  # Check every 500ms for <500ms latency
            )

            if event:
                events.append({
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "container_id": event.container_id,
                    "task_id": event.task_id,
                    "message": event.message,
                    "source": event.source,
                })

        return {
            "success": True,
            "subscription_id": subscription_id,
            "events": events,
            "count": len(events),
            "has_more": not subscription.listener.empty(),
        }

    async def _handle_events_unsubscribe(
        self,
        subscription_id: str,
    ) -> dict[str, Any]:
        """Handle parhelia_events_unsubscribe tool call."""
        if not hasattr(self, "_subscription_manager"):
            return {
                "success": False,
                "error": "No subscriptions exist.",
            }

        success = self._subscription_manager.unsubscribe(subscription_id)

        if success:
            return {
                "success": True,
                "subscription_id": subscription_id,
                "message": "Subscription cancelled",
            }
        else:
            return {
                "success": False,
                "error": f"Subscription not found: {subscription_id}",
            }

    async def _handle_events_list(
        self,
        container_id: str | None = None,
        task_id: str | None = None,
        event_type: str | None = None,
        since_minutes: int | None = None,
        limit: int = 50,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_events_list tool call."""
        from datetime import timedelta

        # Build query
        since = None
        if since_minutes:
            since = datetime.utcnow() - timedelta(minutes=since_minutes)

        evt_type = EventType(event_type) if event_type else None

        events = self.state_store.get_events(
            container_id=container_id,
            task_id=task_id,
            event_type=evt_type,
            since=since,
            limit=limit,
        )

        event_data = []
        for e in events:
            item = {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "container_id": e.container_id,
                "task_id": e.task_id,
                "message": e.message,
                "source": e.source,
            }
            event_data.append(filter_fields(item, fields))

        return {
            "success": True,
            "events": event_data,
            "count": len(event_data),
        }

    async def _handle_events_replay(
        self,
        container_id: str,
        from_start: bool = True,
        since_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Handle parhelia_events_replay tool call."""
        from datetime import timedelta
        from parhelia.events import EventLogger

        logger = EventLogger(self.state_store)

        since = None
        if not from_start and since_minutes:
            since = datetime.utcnow() - timedelta(minutes=since_minutes)

        events = logger.replay(
            container_id=container_id,
            from_start=from_start,
            since=since,
        )

        event_data = [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "message": e.message,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "source": e.source,
            }
            for e in events
        ]

        return {
            "success": True,
            "container_id": container_id,
            "events": event_data,
            "count": len(event_data),
            "from_start": from_start,
        }
