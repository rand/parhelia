"""Parhelia CLI for session management.

Implements CLI commands for managing Claude Code sessions in Modal containers.

Usage:
    parhelia status              # Show system status
    parhelia list                # List active sessions
    parhelia submit <prompt>     # Submit a new task
    parhelia attach <session>    # Attach to a session
    parhelia detach <session>    # Detach from a session
    parhelia checkpoint <session> # Create checkpoint
    parhelia resume <session>    # Resume from checkpoint
    parhelia logs <session>      # View session logs
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from parhelia.budget import BudgetManager
from parhelia.checkpoint import CheckpointManager
from parhelia.config import load_config
from parhelia.environment import (
    EnvironmentCapture,
    diff_environments,
    format_environment_diff,
)
from parhelia.heartbeat import HeartbeatMonitor
from parhelia.orchestrator import Task, TaskRequirements, TaskType
from parhelia.persistence import PersistentOrchestrator
from parhelia.resume import ResumeManager
from parhelia.session import Session, SessionState


# =============================================================================
# CLI Context
# =============================================================================


class CLIContext:
    """Context object for CLI commands."""

    def __init__(self, config_path: str | None = None, verbose: bool = False):
        """Initialize CLI context.

        Args:
            config_path: Path to config file.
            verbose: Enable verbose output.
        """
        self.verbose = verbose
        self.config = load_config(config_path)

        # Lazy-loaded components
        self._orchestrator: PersistentOrchestrator | None = None
        self._checkpoint_manager: CheckpointManager | None = None
        self._resume_manager: ResumeManager | None = None
        self._budget_manager: BudgetManager | None = None
        self._heartbeat_monitor: HeartbeatMonitor | None = None

    @property
    def orchestrator(self) -> PersistentOrchestrator:
        """Get or create orchestrator with SQLite persistence."""
        if self._orchestrator is None:
            self._orchestrator = PersistentOrchestrator()
        return self._orchestrator

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Get or create checkpoint manager."""
        if self._checkpoint_manager is None:
            self._checkpoint_manager = CheckpointManager(
                checkpoint_root=str(
                    Path(self.config.paths.volume_root) / "checkpoints"
                )
            )
        return self._checkpoint_manager

    @property
    def resume_manager(self) -> ResumeManager:
        """Get or create resume manager."""
        if self._resume_manager is None:
            self._resume_manager = ResumeManager(
                checkpoint_manager=self.checkpoint_manager,
                workspace_root=str(
                    Path(self.config.paths.volume_root) / "workspaces"
                ),
            )
        return self._resume_manager

    @property
    def budget_manager(self) -> BudgetManager:
        """Get or create budget manager."""
        if self._budget_manager is None:
            self._budget_manager = BudgetManager(
                ceiling_usd=self.config.budget.default_ceiling_usd,
            )
        return self._budget_manager

    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose or level == "error":
            prefix = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✗"}
            click.echo(f"{prefix.get(level, '•')} {message}")


pass_context = click.make_pass_decorator(CLIContext)


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group()
@click.option(
    "-c", "--config",
    type=click.Path(exists=True),
    help="Path to config file",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.version_option(version="0.1.0", prog_name="parhelia")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """Parhelia - Remote Claude Code execution system.

    Manage Claude Code sessions in Modal containers with checkpoint/resume,
    parallel dispatch, and interactive attachment.
    """
    ctx.obj = CLIContext(config_path=config, verbose=verbose)


# =============================================================================
# Status Command
# =============================================================================


@cli.command()
@pass_context
def status(ctx: CLIContext) -> None:
    """Show system status and health."""
    click.echo("Parhelia System Status")
    click.echo("=" * 40)

    # Config info
    click.echo(f"\nConfiguration:")
    click.echo(f"  Volume: {ctx.config.modal.volume_name}")
    click.echo(f"  CPU: {ctx.config.modal.cpu_count} cores, {ctx.config.modal.memory_mb}MB")
    click.echo(f"  Default timeout: {ctx.config.modal.default_timeout_hours}h")

    # Orchestrator status
    orch = ctx.orchestrator
    click.echo(f"\nOrchestrator:")
    click.echo(f"  Pending tasks: {orch.get_pending_count()}")
    click.echo(f"  Active workers: {orch.get_active_worker_count()}")

    # Budget status
    budget = ctx.budget_manager
    budget_status = budget.check_budget(raise_on_exceeded=False)
    click.echo(f"\nBudget:")
    click.echo(f"  Ceiling: ${budget_status.ceiling_usd:.2f}")
    click.echo(f"  Used: ${budget_status.used_usd:.2f} ({budget_status.usage_percent:.1f}%)")
    click.echo(f"  Remaining: ${budget_status.remaining_usd:.2f}")

    if budget_status.warning_threshold_reached:
        click.secho("  ⚠ Warning threshold reached!", fg="yellow")
    if budget_status.is_exceeded:
        click.secho("  ✗ Budget exceeded!", fg="red")


# =============================================================================
# List Command
# =============================================================================


@cli.command("list")
@click.option(
    "-s", "--status",
    type=click.Choice(["all", "pending", "running", "completed", "failed"]),
    default="all",
    help="Filter by status",
)
@click.option(
    "-n", "--limit",
    type=int,
    default=20,
    help="Maximum number of items to show",
)
@pass_context
def list_sessions(ctx: CLIContext, status: str, limit: int) -> None:
    """List tasks and sessions."""

    async def _list():
        # Get tasks from orchestrator
        if status == "all":
            tasks = await ctx.orchestrator.get_all_tasks(limit)
        elif status == "pending":
            tasks = await ctx.orchestrator.get_pending_tasks()
        elif status == "running":
            tasks = await ctx.orchestrator.get_running_tasks()
        else:
            tasks = await ctx.orchestrator.get_all_tasks(limit)
            tasks = [t for t in tasks if ctx.orchestrator.task_store.get_status(t.id) == status]

        if not tasks:
            click.echo("No tasks found.")
            return

        click.echo(f"{'ID':<20} {'Status':<12} {'Type':<12} {'Created':<20}")
        click.echo("-" * 70)

        for task in tasks[:limit]:
            task_status = ctx.orchestrator.task_store.get_status(task.id) or "unknown"
            status_color = {
                "pending": "yellow",
                "running": "green",
                "completed": "blue",
                "failed": "red",
            }.get(task_status, "white")

            click.echo(
                f"{task.id:<20} "
                f"{click.style(task_status, fg=status_color):<12} "
                f"{task.task_type.value:<12} "
                f"{task.created_at.strftime('%Y-%m-%d %H:%M'):<20}"
            )

    asyncio.run(_list())


# =============================================================================
# Submit Command
# =============================================================================


@cli.command()
@click.argument("prompt")
@click.option(
    "-t", "--type",
    "task_type",
    type=click.Choice(["generic", "code_fix", "test", "build", "lint", "refactor"]),
    default="generic",
    help="Task type",
)
@click.option(
    "--gpu",
    type=click.Choice(["none", "A10G", "A100", "H100", "T4"]),
    default="none",
    help="GPU type required",
)
@click.option(
    "--memory",
    type=int,
    default=4,
    help="Minimum memory in GB",
)
@click.option(
    "-w", "--workspace",
    type=click.Path(exists=True),
    help="Working directory for the task",
)
@pass_context
def submit(
    ctx: CLIContext,
    prompt: str,
    task_type: str,
    gpu: str,
    memory: int,
    workspace: str | None,
) -> None:
    """Submit a new task for execution."""
    import uuid

    task_type_map = {
        "generic": TaskType.GENERIC,
        "code_fix": TaskType.CODE_FIX,
        "test": TaskType.TEST_RUN,
        "build": TaskType.BUILD,
        "lint": TaskType.LINT,
        "refactor": TaskType.REFACTOR,
    }

    task = Task(
        id=f"task-{uuid.uuid4().hex[:8]}",
        prompt=prompt,
        task_type=task_type_map[task_type],
        requirements=TaskRequirements(
            needs_gpu=gpu != "none",
            gpu_type=gpu if gpu != "none" else None,
            min_memory_gb=memory,
            working_directory=workspace,
        ),
    )

    async def _submit():
        task_id = await ctx.orchestrator.submit_task(task)
        return task_id

    task_id = asyncio.run(_submit())

    click.echo(f"Task submitted: {task_id}")
    ctx.log(f"Type: {task_type}, GPU: {gpu}, Memory: {memory}GB", "info")


# =============================================================================
# Attach Command
# =============================================================================


@cli.command()
@click.argument("session_id")
@pass_context
def attach(ctx: CLIContext, session_id: str) -> None:
    """Attach to a running session via SSH/tmux."""
    click.echo(f"Attaching to session: {session_id}")

    # Check if session exists
    worker = ctx.orchestrator.get_worker(session_id)
    if not worker:
        click.secho(f"Session not found: {session_id}", fg="red")
        sys.exit(1)

    if worker.state.value != "running":
        click.secho(f"Session is not running (status: {worker.state.value})", fg="yellow")

    # In a full implementation, this would:
    # 1. Get SSH connection info from Modal
    # 2. Start SSH tunnel
    # 3. Attach to tmux session
    click.echo("SSH tunnel would be established here...")
    click.echo(f"tmux attach-session -t {session_id}")


# =============================================================================
# Detach Command
# =============================================================================


@cli.command()
@click.argument("session_id")
@pass_context
def detach(ctx: CLIContext, session_id: str) -> None:
    """Detach from a session (keeps it running)."""
    click.echo(f"Detaching from session: {session_id}")

    # In a full implementation, this would:
    # 1. Send tmux detach command
    # 2. Close SSH tunnel
    click.echo("Session continues running in background.")


# =============================================================================
# Checkpoint Command Group [SPEC-07.40, SPEC-07.41]
# =============================================================================


class CheckpointGroup(click.Group):
    """Custom group that handles legacy 'checkpoint SESSION_ID' syntax."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Handle legacy checkpoint command syntax."""
        # If first arg doesn't look like a subcommand and isn't an option,
        # treat it as the legacy 'checkpoint SESSION_ID [-m MSG]' syntax
        if args and not args[0].startswith("-"):
            # Check if it's a known subcommand
            if args[0] not in self.commands:
                # Legacy mode: prepend 'create' subcommand
                args = ["create"] + args
        return super().parse_args(ctx, args)


@cli.group(cls=CheckpointGroup)
def checkpoint() -> None:
    """Checkpoint management commands.

    Examples:
        parhelia checkpoint my-session           # Create checkpoint (legacy)
        parhelia checkpoint create my-session    # Create checkpoint
        parhelia checkpoint rollback cp-abc123   # Rollback to checkpoint
        parhelia checkpoint diff cp-a cp-b       # Compare checkpoints
        parhelia checkpoint list                 # List checkpoints
    """
    pass


@checkpoint.command("create")
@click.argument("session_id")
@click.option(
    "-m", "--message",
    help="Checkpoint message/description",
)
@pass_context
def checkpoint_create(ctx: CLIContext, session_id: str, message: str | None) -> None:
    """Create a checkpoint for a session."""
    from parhelia.checkpoint import CheckpointTrigger

    async def _checkpoint():
        # Create a mock session for checkpointing
        # In production, this would get the actual session state
        session = Session(
            id=session_id,
            task_id=f"task-{session_id}",
            state=SessionState.RUNNING,
            working_directory="/vol/parhelia/workspaces/{session_id}",
        )

        cp = await ctx.checkpoint_manager.create_checkpoint(
            session=session,
            trigger=CheckpointTrigger.MANUAL,
            conversation={"message": message} if message else None,
        )

        return cp

    cp = asyncio.run(_checkpoint())
    click.echo(f"Checkpoint created: {cp.id}")
    click.echo(f"  Session: {session_id}")
    click.echo(f"  Time: {cp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# Resume Command
# =============================================================================


@cli.command()
@click.argument("session_id")
@click.option(
    "--checkpoint-id",
    help="Specific checkpoint to resume from (default: latest)",
)
@click.option(
    "--target",
    type=click.Path(),
    help="Target directory for restored workspace",
)
@pass_context
def resume(
    ctx: CLIContext,
    session_id: str,
    checkpoint_id: str | None,
    target: str | None,
) -> None:
    """Resume a session from checkpoint."""

    async def _resume():
        # Check if resumable
        if not await ctx.resume_manager.can_resume(session_id):
            click.secho(f"No checkpoint found for session: {session_id}", fg="red")
            sys.exit(1)

        # Get resume info
        info = await ctx.resume_manager.get_resume_info(session_id)
        if info:
            click.echo(f"Resuming from checkpoint: {info['checkpoint_id']}")
            click.echo(f"  Created: {info['created_at']}")

        # Perform resume
        result = await ctx.resume_manager.resume_session(
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            target_directory=target,
            run_claude=False,  # Don't auto-run in CLI
        )

        return result

    result = asyncio.run(_resume())

    if result.success:
        click.secho("✓ Session resumed successfully", fg="green")
        click.echo(f"  Workspace: {result.restored_working_directory}")
        if result.conversation_turn:
            click.echo(f"  Conversation turn: {result.conversation_turn}")
    else:
        click.secho(f"✗ Resume failed: {result.error}", fg="red")
        sys.exit(1)


# =============================================================================
# Logs Command
# =============================================================================


@cli.command()
@click.argument("session_id")
@click.option(
    "-f", "--follow",
    is_flag=True,
    help="Follow log output",
)
@click.option(
    "-n", "--lines",
    type=int,
    default=50,
    help="Number of lines to show",
)
@pass_context
def logs(ctx: CLIContext, session_id: str, follow: bool, lines: int) -> None:
    """View session logs."""
    click.echo(f"Logs for session: {session_id}")
    click.echo("-" * 40)

    # In a full implementation, this would:
    # 1. Connect to Modal and get container logs
    # 2. Stream if --follow is set
    click.echo(f"[Would show last {lines} lines of logs]")
    if follow:
        click.echo("[Following log output... Press Ctrl+C to stop]")


# =============================================================================
# Budget Command
# =============================================================================


@cli.group()
def budget() -> None:
    """Manage budget settings."""
    pass


@budget.command("show")
@pass_context
def budget_show(ctx: CLIContext) -> None:
    """Show current budget status."""
    status = ctx.budget_manager.check_budget(raise_on_exceeded=False)

    click.echo("Budget Status")
    click.echo("=" * 30)
    click.echo(f"Ceiling:     ${status.ceiling_usd:.2f}")
    click.echo(f"Used:        ${status.used_usd:.2f}")
    click.echo(f"Remaining:   ${status.remaining_usd:.2f}")
    click.echo(f"Usage:       {status.usage_percent:.1f}%")
    click.echo(f"Tasks:       {status.task_count}")
    click.echo(f"Tokens in:   {status.total_input_tokens:,}")
    click.echo(f"Tokens out:  {status.total_output_tokens:,}")

    if status.is_exceeded:
        click.secho("\n⚠ Budget exceeded!", fg="red", bold=True)
    elif status.warning_threshold_reached:
        click.secho("\n⚠ Warning threshold reached", fg="yellow")


@budget.command("set")
@click.argument("amount", type=float)
@pass_context
def budget_set(ctx: CLIContext, amount: float) -> None:
    """Set budget ceiling in USD."""
    ctx.budget_manager.set_ceiling(amount)
    click.echo(f"Budget ceiling set to ${amount:.2f}")


@budget.command("reset")
@click.confirmation_option(prompt="Are you sure you want to reset budget tracking?")
@pass_context
def budget_reset(ctx: CLIContext) -> None:
    """Reset budget tracking."""
    ctx.budget_manager.reset()
    click.echo("Budget tracking reset.")


# =============================================================================
# Config Command
# =============================================================================


@cli.command()
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@pass_context
def config(ctx: CLIContext, as_json: bool) -> None:
    """Show current configuration."""
    if as_json:
        config_dict = {
            "modal": {
                "volume_name": ctx.config.modal.volume_name,
                "cpu_count": ctx.config.modal.cpu_count,
                "memory_mb": ctx.config.modal.memory_mb,
                "default_timeout_hours": ctx.config.modal.default_timeout_hours,
            },
            "budget": {
                "default_ceiling_usd": ctx.config.budget.default_ceiling_usd,
            },
            "paths": {
                "volume_root": ctx.config.paths.volume_root,
            },
        }
        click.echo(json.dumps(config_dict, indent=2))
    else:
        click.echo("Parhelia Configuration")
        click.echo("=" * 30)
        click.echo(f"\n[modal]")
        click.echo(f"  volume_name = {ctx.config.modal.volume_name}")
        click.echo(f"  cpu_count = {ctx.config.modal.cpu_count}")
        click.echo(f"  memory_mb = {ctx.config.modal.memory_mb}")
        click.echo(f"  default_timeout_hours = {ctx.config.modal.default_timeout_hours}")
        click.echo(f"\n[budget]")
        click.echo(f"  default_ceiling_usd = {ctx.config.budget.default_ceiling_usd}")
        click.echo(f"\n[paths]")
        click.echo(f"  volume_root = {ctx.config.paths.volume_root}")


# =============================================================================
# Session Command Group
# =============================================================================


@cli.group()
def session() -> None:
    """Session management commands.

    Implements [SPEC-07.20].
    """
    pass


@session.command("review")
@click.argument("session_id")
@click.option(
    "--approve",
    "action",
    flag_value="approve",
    help="Approve the checkpoint",
)
@click.option(
    "--reject",
    "action",
    flag_value="reject",
    help="Reject the checkpoint",
)
@click.option(
    "--reason",
    help="Reason for approval/rejection",
)
@click.option(
    "--user",
    default="cli-user",
    help="Username for audit",
)
@pass_context
def session_review(
    ctx: CLIContext,
    session_id: str,
    action: str | None,
    reason: str | None,
    user: str,
) -> None:
    """Review a session's latest checkpoint for approval.

    Implements [SPEC-07.20.03].

    Example:
        parhelia session review my-session
        parhelia session review my-session --approve --reason "Looks good"
        parhelia session review my-session --reject --reason "Needs more tests"
    """
    from parhelia.approval import ApprovalConfig, ApprovalManager
    from parhelia.session import ApprovalStatus

    async def _review():
        # Get latest checkpoint for session
        checkpoints = await ctx.checkpoint_manager.list_checkpoints(session_id)
        if not checkpoints:
            click.secho(f"No checkpoints found for session: {session_id}", fg="red")
            return

        # Sort by created_at and get latest
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        checkpoint = checkpoints[0]

        # Initialize approval manager
        approval_config = ctx.config.approval or ApprovalConfig.default()
        manager = ApprovalManager(config=approval_config)

        # If action specified, perform it
        if action == "approve":
            approval = await manager.approve(checkpoint, user=user, reason=reason)
            checkpoint.approval = approval
            # TODO: Save checkpoint with updated approval
            click.secho(f"Checkpoint {checkpoint.id} approved", fg="green")
            return

        if action == "reject":
            if not reason:
                click.secho("Rejection requires --reason", fg="red")
                return
            approval = await manager.reject(checkpoint, user=user, reason=reason)
            checkpoint.approval = approval
            # TODO: Save checkpoint with updated approval
            click.secho(f"Checkpoint {checkpoint.id} rejected", fg="red")
            return

        # Display review interface
        click.echo(f"\nSession: {session_id}")
        if checkpoint.approval:
            status = checkpoint.approval.status.value
        else:
            status = "pending"
        click.echo(f"Status: {status}")
        click.echo()

        click.echo(f"Latest Checkpoint: {checkpoint.id}")
        click.echo(f"  Trigger: {checkpoint.trigger.value}")
        click.echo(f"  Created: {checkpoint.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo()

        click.echo("Summary:")
        click.echo(f"  Tokens used: {checkpoint.tokens_used:,}")
        click.echo(f"  Estimated cost: ${checkpoint.cost_estimate:.2f}")
        if checkpoint.uncommitted_changes:
            click.echo(f"  Files modified: {len(checkpoint.uncommitted_changes)}")
            for f in checkpoint.uncommitted_changes[:5]:
                click.echo(f"    - {f}")
            if len(checkpoint.uncommitted_changes) > 5:
                click.echo(f"    ... and {len(checkpoint.uncommitted_changes) - 5} more")
        click.echo()

        # Show approval status
        if checkpoint.approval:
            click.echo("Approval:")
            click.echo(f"  Status: {checkpoint.approval.status.value}")
            if checkpoint.approval.user:
                click.echo(f"  User: {checkpoint.approval.user}")
            if checkpoint.approval.timestamp:
                click.echo(f"  Time: {checkpoint.approval.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if checkpoint.approval.reason:
                click.echo(f"  Reason: {checkpoint.approval.reason}")
            if checkpoint.approval.policy:
                click.echo(f"  Policy: {checkpoint.approval.policy}")

        click.echo()
        click.echo("Actions:")
        click.echo("  parhelia session review {session_id} --approve --reason '...'")
        click.echo("  parhelia session review {session_id} --reject --reason '...'")

    asyncio.run(_review())


@session.command("pending")
@click.option(
    "--limit",
    default=20,
    help="Maximum number of checkpoints to show",
)
@pass_context
def session_pending(ctx: CLIContext, limit: int) -> None:
    """List checkpoints awaiting review.

    Implements [SPEC-07.20.03].
    """
    from parhelia.approval import ApprovalConfig, ApprovalManager
    from parhelia.session import ApprovalStatus

    async def _list_pending():
        # Get all checkpoints from all sessions
        checkpoint_root = ctx.checkpoint_manager.checkpoint_root
        if not checkpoint_root.exists():
            click.echo("No checkpoints found.")
            return

        pending = []
        for session_dir in checkpoint_root.iterdir():
            if session_dir.is_dir():
                checkpoints = await ctx.checkpoint_manager.list_checkpoints(
                    session_dir.name
                )
                for cp in checkpoints:
                    if cp.approval is None or cp.approval.status == ApprovalStatus.PENDING:
                        pending.append(cp)

        if not pending:
            click.echo("No pending checkpoints.")
            return

        # Sort by created_at
        pending.sort(key=lambda c: c.created_at, reverse=True)

        click.echo(f"Pending Checkpoints ({len(pending)}):")
        click.echo("=" * 60)
        for cp in pending[:limit]:
            click.echo(
                f"  {cp.id}  {cp.session_id}  {cp.trigger.value}  "
                f"{cp.created_at.strftime('%Y-%m-%d %H:%M')}"
            )

    asyncio.run(_list_pending())


# =============================================================================
# Environment Command
# =============================================================================


@cli.group()
def env() -> None:
    """Environment versioning commands.

    Implements [SPEC-07.10.05].
    """
    pass


@env.command("show")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@pass_context
def env_show(ctx: CLIContext, as_json: bool) -> None:
    """Show current environment state."""

    async def _capture():
        capture = EnvironmentCapture()
        return await capture.capture()

    env_snapshot = asyncio.run(_capture())

    if as_json:
        click.echo(json.dumps(env_snapshot.to_dict(), indent=2))
    else:
        click.echo("Environment Snapshot")
        click.echo("=" * 40)

        click.echo(f"\nClaude Code:")
        click.echo(f"  Version: {env_snapshot.claude_code.version}")
        click.echo(f"  Binary hash: {env_snapshot.claude_code.binary_hash[:16]}...")
        click.echo(f"  Path: {env_snapshot.claude_code.install_path}")

        if env_snapshot.plugins:
            click.echo(f"\nPlugins ({len(env_snapshot.plugins)}):")
            for name, plugin in env_snapshot.plugins.items():
                click.echo(f"  {name}:")
                click.echo(f"    Commit: {plugin.git_commit[:8]}")
                click.echo(f"    Branch: {plugin.git_branch}")

        if env_snapshot.mcp_servers:
            click.echo(f"\nMCP Servers ({len(env_snapshot.mcp_servers)}):")
            for name, server in env_snapshot.mcp_servers.items():
                click.echo(f"  {name}:")
                click.echo(f"    Type: {server.source_type}")
                click.echo(f"    Version: {server.version_id}")

        click.echo(f"\nPython:")
        click.echo(f"  Version: {env_snapshot.python_version}")
        if env_snapshot.python_packages:
            click.echo(f"  Key packages:")
            for name, version in sorted(env_snapshot.python_packages.items()):
                click.echo(f"    {name}=={version}")

        click.echo(f"\nCaptured at: {env_snapshot.captured_at.strftime('%Y-%m-%d %H:%M:%S')}")


@env.command("diff")
@click.argument("checkpoint_a")
@click.argument("checkpoint_b")
@click.option(
    "--session",
    "session_id",
    help="Session ID (required if checkpoint IDs are ambiguous)",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@pass_context
def env_diff(
    ctx: CLIContext,
    checkpoint_a: str,
    checkpoint_b: str,
    session_id: str | None,
    as_json: bool,
) -> None:
    """Compare environments between two checkpoints.

    Implements [SPEC-07.10.05].

    Example:
        parhelia env diff cp-abc123 cp-def456 --session my-session
    """

    async def _diff():
        # Find checkpoints
        cp_a = None
        cp_b = None

        if session_id:
            # Look in specific session
            checkpoints = await ctx.checkpoint_manager.list_checkpoints(session_id)
            for cp in checkpoints:
                if cp.id == checkpoint_a:
                    cp_a = cp
                if cp.id == checkpoint_b:
                    cp_b = cp
        else:
            # Search all sessions in checkpoint root
            checkpoint_root = ctx.checkpoint_manager.checkpoint_root
            if checkpoint_root.exists():
                for session_dir in checkpoint_root.iterdir():
                    if session_dir.is_dir():
                        checkpoints = await ctx.checkpoint_manager.list_checkpoints(
                            session_dir.name
                        )
                        for cp in checkpoints:
                            if cp.id == checkpoint_a:
                                cp_a = cp
                            if cp.id == checkpoint_b:
                                cp_b = cp
                        if cp_a and cp_b:
                            break

        return cp_a, cp_b

    cp_a, cp_b = asyncio.run(_diff())

    if not cp_a:
        click.secho(f"Checkpoint not found: {checkpoint_a}", fg="red")
        sys.exit(1)

    if not cp_b:
        click.secho(f"Checkpoint not found: {checkpoint_b}", fg="red")
        sys.exit(1)

    if not cp_a.environment_snapshot:
        click.secho(
            f"Checkpoint {checkpoint_a} has no environment snapshot", fg="yellow"
        )
        sys.exit(1)

    if not cp_b.environment_snapshot:
        click.secho(
            f"Checkpoint {checkpoint_b} has no environment snapshot", fg="yellow"
        )
        sys.exit(1)

    # Compute diff
    diff = diff_environments(cp_a.environment_snapshot, cp_b.environment_snapshot)

    if as_json:
        diff_dict = {
            "checkpoint_a": checkpoint_a,
            "checkpoint_b": checkpoint_b,
            "claude_code_changed": diff.claude_code_changed,
            "plugins_added": diff.plugins_added,
            "plugins_removed": diff.plugins_removed,
            "plugins_changed": diff.plugins_changed,
            "mcp_servers_added": diff.mcp_servers_added,
            "mcp_servers_removed": diff.mcp_servers_removed,
            "mcp_servers_changed": diff.mcp_servers_changed,
            "python_version_changed": diff.python_version_changed,
            "packages_added": diff.packages_added,
            "packages_removed": diff.packages_removed,
            "packages_changed": {k: list(v) for k, v in diff.packages_changed.items()},
        }
        click.echo(json.dumps(diff_dict, indent=2))
    else:
        click.echo(f"Environment Diff: {checkpoint_a} → {checkpoint_b}")
        click.echo(f"Time: {cp_a.created_at} → {cp_b.created_at}")
        click.echo()
        click.echo(format_environment_diff(diff))


# =============================================================================
# Memory Command Group [SPEC-07.31]
# =============================================================================


@cli.group()
def memory() -> None:
    """Project memory commands.

    Implements [SPEC-07.31].
    """
    pass


@memory.command("save")
@click.argument("key")
@click.argument("value")
@click.option(
    "--category",
    type=click.Choice(["architecture", "convention", "gotcha"]),
    default="convention",
    help="Memory category",
)
@pass_context
def memory_save(ctx: CLIContext, key: str, value: str, category: str) -> None:
    """Save knowledge to project memory.

    Implements [SPEC-07.31].

    Example:
        parhelia memory save "testing" "Always run pytest with -v flag"
        parhelia memory save "auth" "Uses JWT tokens" --category convention
    """
    from parhelia.project_memory import ProjectMemoryManager

    memory_path = Path(ctx.config.paths.volume_root) / "memory" / "project.json"
    manager = ProjectMemoryManager(memory_path)

    if category == "architecture":
        manager.set_architecture(value, key_files=[key])
        click.secho(f"✓ Architecture knowledge saved", fg="green")
    elif category == "convention":
        manager.set_convention(key, value)
        click.secho(f"✓ Convention saved: {key}", fg="green")
    elif category == "gotcha":
        manager.add_gotcha(value, session_id="cli")
        click.secho(f"✓ Gotcha recorded", fg="green")

    manager.save()


@memory.command("recall")
@click.argument("query")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@pass_context
def memory_recall(ctx: CLIContext, query: str, as_json: bool) -> None:
    """Recall knowledge from project memory.

    Implements [SPEC-07.31].

    Example:
        parhelia memory recall "testing"
        parhelia memory recall "auth patterns" --json
    """
    from parhelia.project_memory import ProjectMemoryManager

    memory_path = Path(ctx.config.paths.volume_root) / "memory" / "project.json"
    manager = ProjectMemoryManager(memory_path)

    results = manager.recall(query)

    if as_json:
        click.echo(json.dumps(results, indent=2, default=str))
    else:
        click.echo(f"Memory Recall: '{query}'")
        click.echo("=" * 40)

        if results.get("architecture"):
            arch = results["architecture"]
            click.echo(f"\nArchitecture:")
            click.echo(f"  {arch.get('summary', 'No summary')}")
            if arch.get("key_files"):
                click.echo(f"  Key files: {', '.join(arch['key_files'][:5])}")

        if results.get("conventions"):
            click.echo(f"\nConventions:")
            for key, value in results["conventions"].items():
                click.echo(f"  {key}: {value}")

        if results.get("gotchas"):
            click.echo(f"\nGotchas:")
            for gotcha in results["gotchas"][:5]:
                click.echo(f"  • {gotcha.get('description', gotcha)}")

        if results.get("recent_sessions"):
            click.echo(f"\nRecent Sessions:")
            for session in results["recent_sessions"][:3]:
                click.echo(f"  • {session.get('summary', session)}")


@memory.command("show")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@pass_context
def memory_show(ctx: CLIContext, as_json: bool) -> None:
    """Show all project memory."""
    from parhelia.project_memory import ProjectMemoryManager

    memory_path = Path(ctx.config.paths.volume_root) / "memory" / "project.json"
    manager = ProjectMemoryManager(memory_path)

    if as_json:
        click.echo(json.dumps(manager.memory.to_dict(), indent=2, default=str))
    else:
        mem = manager.memory
        knowledge = mem.knowledge
        click.echo("Project Memory")
        click.echo("=" * 40)

        if knowledge.architecture:
            click.echo(f"\nArchitecture:")
            click.echo(f"  {knowledge.architecture.summary}")

        if knowledge.conventions:
            conventions = knowledge.conventions.conventions
            click.echo(f"\nConventions ({len(conventions)}):")
            for key, value in list(conventions.items())[:10]:
                click.echo(f"  {key}: {value}")
        else:
            click.echo("\nConventions (0):")

        click.echo(f"\nGotchas ({len(knowledge.gotchas)}):")
        for gotcha in knowledge.gotchas[:5]:
            click.echo(f"  • {gotcha.description}")

        click.echo(f"\nSession History ({len(mem.session_history)}):")
        for session in mem.session_history[-5:]:
            click.echo(f"  • {session.session_id}: {session.summary}")


# =============================================================================
# Checkpoint Subcommands [SPEC-07.40, SPEC-07.41]
# =============================================================================


@checkpoint.command("rollback")
@click.argument("checkpoint_id")
@click.option(
    "--session",
    "session_id",
    help="Session ID (if checkpoint ID is ambiguous)",
)
@click.option(
    "-y", "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
@pass_context
def checkpoint_rollback(
    ctx: CLIContext,
    checkpoint_id: str,
    session_id: str | None,
    yes: bool,
) -> None:
    """Rollback workspace to a checkpoint state.

    Implements [SPEC-07.40].

    Safety guarantees:
    - Creates safety checkpoint before rollback
    - Stashes uncommitted changes
    - Verifies target checkpoint is readable
    - Provides recovery on failure

    Example:
        parhelia checkpoint rollback cp-abc123
        parhelia checkpoint rollback cp-abc123 --session my-session -y
    """
    from parhelia.rollback import WorkspaceRollback

    async def _rollback():
        # Find checkpoint
        target_cp = None
        if session_id:
            checkpoints = await ctx.checkpoint_manager.list_checkpoints(session_id)
            for cp in checkpoints:
                if cp.id == checkpoint_id:
                    target_cp = cp
                    break
        else:
            # Search all sessions
            checkpoint_root = ctx.checkpoint_manager.checkpoint_root
            if checkpoint_root.exists():
                for sess_dir in checkpoint_root.iterdir():
                    if sess_dir.is_dir():
                        checkpoints = await ctx.checkpoint_manager.list_checkpoints(
                            sess_dir.name
                        )
                        for cp in checkpoints:
                            if cp.id == checkpoint_id:
                                target_cp = cp
                                break
                        if target_cp:
                            break

        if not target_cp:
            click.secho(f"Checkpoint not found: {checkpoint_id}", fg="red")
            sys.exit(1)

        # Show rollback plan
        click.echo(f"Rollback Plan")
        click.echo("=" * 40)
        click.echo(f"Target: {target_cp.id}")
        click.echo(f"Session: {target_cp.session_id}")
        click.echo(f"Created: {target_cp.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if target_cp.uncommitted_changes:
            click.echo(f"Files ({len(target_cp.uncommitted_changes)}):")
            for f in target_cp.uncommitted_changes[:5]:
                click.echo(f"  - {f}")
        click.echo()

        # Confirm unless -y flag
        if not yes:
            if not click.confirm("Proceed with rollback?"):
                click.echo("Rollback cancelled.")
                return

        # Perform rollback
        rollback = WorkspaceRollback(
            checkpoint_manager=ctx.checkpoint_manager,
            workspace_dir=target_cp.working_directory,
            session_id=target_cp.session_id,
        )

        def progress_callback(msg: str, phase) -> None:
            click.echo(f"  {msg}")

        rollback.set_progress_callback(progress_callback)

        result = await rollback.rollback(
            checkpoint_id,
            skip_confirmation=True,  # Already confirmed above
        )

        click.echo()
        if result.success:
            click.secho("✓ Rollback completed successfully", fg="green")
            click.echo(f"  Safety checkpoint: {result.safety_checkpoint_id}")
            if result.stash_ref:
                click.echo(f"  Changes stashed: {result.stash_ref}")
        else:
            click.secho(f"✗ Rollback failed: {result.error_message}", fg="red")
            if result.recovery_performed:
                click.echo("  Recovery was performed to restore previous state.")
            sys.exit(1)

    asyncio.run(_rollback())


@checkpoint.command("diff")
@click.argument("checkpoint_a")
@click.argument("checkpoint_b")
@click.option(
    "--session",
    "session_id",
    help="Session ID (if checkpoint IDs are ambiguous)",
)
@click.option(
    "--file",
    "file_path",
    help="Show diff for specific file",
)
@click.option(
    "--conversation",
    is_flag=True,
    help="Show conversation diff",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@pass_context
def checkpoint_diff(
    ctx: CLIContext,
    checkpoint_a: str,
    checkpoint_b: str,
    session_id: str | None,
    file_path: str | None,
    conversation: bool,
    as_json: bool,
) -> None:
    """Compare two checkpoints.

    Implements [SPEC-07.41].

    Example:
        parhelia checkpoint diff cp-abc123 cp-def456
        parhelia checkpoint diff cp-a cp-b --file src/auth.py
        parhelia checkpoint diff cp-a cp-b --conversation
    """
    from parhelia.checkpoint_diff import CheckpointDiffer

    async def _diff():
        # Find checkpoints
        cp_a = None
        cp_b = None

        if session_id:
            checkpoints = await ctx.checkpoint_manager.list_checkpoints(session_id)
            for cp in checkpoints:
                if cp.id == checkpoint_a:
                    cp_a = cp
                if cp.id == checkpoint_b:
                    cp_b = cp
        else:
            checkpoint_root = ctx.checkpoint_manager.checkpoint_root
            if checkpoint_root.exists():
                for sess_dir in checkpoint_root.iterdir():
                    if sess_dir.is_dir():
                        checkpoints = await ctx.checkpoint_manager.list_checkpoints(
                            sess_dir.name
                        )
                        for cp in checkpoints:
                            if cp.id == checkpoint_a:
                                cp_a = cp
                            if cp.id == checkpoint_b:
                                cp_b = cp
                        if cp_a and cp_b:
                            break

        return cp_a, cp_b

    cp_a, cp_b = asyncio.run(_diff())

    if not cp_a:
        click.secho(f"Checkpoint not found: {checkpoint_a}", fg="red")
        sys.exit(1)

    if not cp_b:
        click.secho(f"Checkpoint not found: {checkpoint_b}", fg="red")
        sys.exit(1)

    differ = CheckpointDiffer()

    if file_path:
        # File-specific diff
        # Would need to read file contents from checkpoint archives
        click.echo(f"File Diff: {file_path}")
        click.echo(f"Checkpoints: {checkpoint_a} → {checkpoint_b}")
        click.echo()
        click.secho(
            "Note: File diff requires checkpoint archive access (not yet implemented)",
            fg="yellow",
        )
    elif conversation:
        # Conversation diff
        # Would need conversation data from checkpoints
        click.echo(f"Conversation Diff: {checkpoint_a} → {checkpoint_b}")
        click.echo()
        click.secho(
            "Note: Conversation diff requires checkpoint conversation data",
            fg="yellow",
        )
    else:
        # Overview comparison
        comparison = differ.compare(cp_a, cp_b)

        if as_json:
            click.echo(json.dumps(comparison.to_dict(), indent=2, default=str))
        else:
            click.echo(differ.format_comparison(comparison))


@checkpoint.command("list")
@click.option(
    "--session",
    "session_id",
    help="Filter by session ID",
)
@click.option(
    "--limit",
    default=20,
    help="Maximum checkpoints to show",
)
@pass_context
def checkpoint_list(ctx: CLIContext, session_id: str | None, limit: int) -> None:
    """List checkpoints."""

    async def _list():
        all_checkpoints = []

        checkpoint_root = ctx.checkpoint_manager.checkpoint_root
        if not checkpoint_root.exists():
            return []

        if session_id:
            checkpoints = await ctx.checkpoint_manager.list_checkpoints(session_id)
            all_checkpoints.extend(checkpoints)
        else:
            for sess_dir in checkpoint_root.iterdir():
                if sess_dir.is_dir():
                    checkpoints = await ctx.checkpoint_manager.list_checkpoints(
                        sess_dir.name
                    )
                    all_checkpoints.extend(checkpoints)

        return all_checkpoints

    checkpoints = asyncio.run(_list())

    if not checkpoints:
        click.echo("No checkpoints found.")
        return

    # Sort by created_at
    checkpoints.sort(key=lambda c: c.created_at, reverse=True)

    click.echo(f"{'ID':<16} {'Session':<20} {'Trigger':<10} {'Created':<20}")
    click.echo("-" * 70)

    for cp in checkpoints[:limit]:
        click.echo(
            f"{cp.id:<16} {cp.session_id:<20} {cp.trigger.value:<10} "
            f"{cp.created_at.strftime('%Y-%m-%d %H:%M'):<20}"
        )


# =============================================================================
# Session Recover Command [SPEC-07.42]
# =============================================================================


@session.command("recover")
@click.argument("session_id")
@click.option(
    "--from",
    "from_checkpoint",
    help="Specific checkpoint to recover from",
)
@click.option(
    "--action",
    type=click.Choice(["resume", "new", "wait"]),
    help="Recovery action (skip interactive selection)",
)
@pass_context
def session_recover(
    ctx: CLIContext,
    session_id: str,
    from_checkpoint: str | None,
    action: str | None,
) -> None:
    """Interactive recovery wizard for a session.

    Implements [SPEC-07.42].

    Handles common recovery scenarios:
    - Resume from failure (crash, timeout)
    - Resume after rejection
    - Manual checkpoint selection

    Example:
        parhelia session recover my-session
        parhelia session recover my-session --from cp-abc123
        parhelia session recover my-session --action resume
    """
    from parhelia.recovery import RecoveryAction, RecoveryManager

    async def _recover():
        manager = RecoveryManager(
            checkpoint_manager=ctx.checkpoint_manager,
        )

        # Get recovery plan
        if from_checkpoint:
            plan = await manager.plan_manual_recovery(
                session_id=session_id,
                from_checkpoint_id=from_checkpoint,
            )
        else:
            plan = await manager.plan_failure_recovery(session_id=session_id)

        if not plan:
            click.secho(f"No recovery options found for session: {session_id}", fg="red")
            return

        # Display plan
        click.echo(manager.format_recovery_plan(plan))
        click.echo()

        # Determine action
        if action:
            action_map = {
                "resume": RecoveryAction.RESUME,
                "new": RecoveryAction.NEW_SESSION,
                "wait": RecoveryAction.WAIT_FOR_USER,
            }
            selected_action = action_map.get(action)
        else:
            # Interactive selection
            click.echo("Available actions:")
            for i, opt in enumerate(plan.options, 1):
                click.echo(f"  [{i}] {opt.action.value}: {opt.description}")
            click.echo()

            choice = click.prompt(
                "Select action",
                type=int,
                default=1,
            )

            if 1 <= choice <= len(plan.options):
                selected_action = plan.options[choice - 1].action
            else:
                click.secho("Invalid selection", fg="red")
                return

        # Execute recovery
        result = await manager.execute_failure_recovery(plan, selected_action)

        click.echo()
        click.echo(manager.format_recovery_result(result))

        if not result.success:
            sys.exit(1)

    asyncio.run(_recover())


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
