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
from parhelia.orchestrator import LocalOrchestrator, Task, TaskRequirements, TaskType
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
        self._orchestrator: LocalOrchestrator | None = None
        self._checkpoint_manager: CheckpointManager | None = None
        self._resume_manager: ResumeManager | None = None
        self._budget_manager: BudgetManager | None = None
        self._heartbeat_monitor: HeartbeatMonitor | None = None

    @property
    def orchestrator(self) -> LocalOrchestrator:
        """Get or create orchestrator."""
        if self._orchestrator is None:
            self._orchestrator = LocalOrchestrator()
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
    type=click.Choice(["all", "running", "completed", "failed"]),
    default="all",
    help="Filter by status",
)
@click.option(
    "-n", "--limit",
    type=int,
    default=20,
    help="Maximum number of sessions to show",
)
@pass_context
def list_sessions(ctx: CLIContext, status: str, limit: int) -> None:
    """List sessions."""

    async def _list():
        # Get sessions from orchestrator
        workers = await ctx.orchestrator.get_workers()

        if not workers:
            click.echo("No active sessions.")
            return

        click.echo(f"{'ID':<20} {'Status':<12} {'Type':<12} {'Created':<20}")
        click.echo("-" * 70)

        for worker in workers[:limit]:
            status_color = {
                "idle": "white",
                "running": "green",
                "completed": "blue",
                "failed": "red",
            }.get(worker.state.value, "white")

            click.echo(
                f"{worker.id:<20} "
                f"{click.style(worker.state.value, fg=status_color):<12} "
                f"{worker.target_type:<12} "
                f"{worker.created_at.strftime('%Y-%m-%d %H:%M'):<20}"
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
# Checkpoint Command
# =============================================================================


@cli.command()
@click.argument("session_id")
@click.option(
    "-m", "--message",
    help="Checkpoint message/description",
)
@pass_context
def checkpoint(ctx: CLIContext, session_id: str, message: str | None) -> None:
    """Create a checkpoint for a session."""
    from parhelia.session import CheckpointTrigger

    async def _checkpoint():
        # Check if session can be checkpointed
        can_resume = await ctx.resume_manager.can_resume(session_id)

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
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
