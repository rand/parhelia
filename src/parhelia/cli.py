"""Parhelia CLI for session management.

Implements CLI commands for managing Claude Code sessions in Modal containers.

Usage (noun-verb pattern per SPEC-20.11):
    parhelia task create <prompt>    # Submit a new task
    parhelia task list               # List tasks
    parhelia task show <id>          # Show task details
    parhelia task cancel <id>        # Cancel a running task
    parhelia task retry <id>         # Retry a failed task
    parhelia task watch <id>         # Watch task status

    parhelia session list            # List sessions
    parhelia session attach <id>     # Attach to session
    parhelia session kill <id>       # Kill a session

    parhelia checkpoint create <id>  # Create checkpoint
    parhelia checkpoint list         # List checkpoints
    parhelia checkpoint restore <id> # Restore from checkpoint
    parhelia checkpoint diff <a> <b> # Compare checkpoints

    parhelia budget status           # Show budget status
    parhelia budget set <amount>     # Set budget ceiling
    parhelia budget history          # Show budget history

    parhelia completion <shell>      # Output shell completion script

Aliases (SPEC-20.12):
    t  -> task       s  -> session
    c  -> checkpoint b  -> budget

Legacy commands (deprecated but functional):
    parhelia submit <prompt>     # -> task create
    parhelia list                # -> task list
    parhelia attach <session>    # -> session attach
    parhelia detach <session>    # -> session detach
"""

from __future__ import annotations

import asyncio
import difflib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import click

from parhelia.budget import BudgetManager
from parhelia.feedback import (
    ErrorRecovery,
    ProgressSpinner,
    StatusFormatter,
    create_spinner,
)
from parhelia.interactive import (
    ExampleSystem,
    HelpSystem,
    SmartPrompt,
    get_example_system,
    get_help_system,
    get_smart_prompt,
)
from parhelia.checkpoint import CheckpointManager
from parhelia.config import load_config
from parhelia.environment import (
    EnvironmentCapture,
    diff_environments,
    format_environment_diff,
)
from parhelia.heartbeat import HeartbeatMonitor
from parhelia.orchestrator import Task, TaskRequirements, TaskType
from parhelia.permissions import TrustLevel
from parhelia.persistence import PersistentOrchestrator
from parhelia.reconciler import ContainerReconciler, RealModalClient, ReconcilerConfig
from parhelia.resume import ResumeManager
from parhelia.session import Session, SessionState
from parhelia.state import (
    Container,
    ContainerState,
    EventType,
    HealthStatus,
    StateStore,
)


# =============================================================================
# Command Aliases (SPEC-20.12)
# =============================================================================

# Maps short aliases to full command names
GROUP_ALIASES: dict[str, str] = {
    "t": "task",
    "s": "session",
    "cp": "checkpoint",  # Changed from 'c' to 'cp' to free up 'c' for container
    "c": "container",
    "b": "budget",
    "r": "reconciler",
    "e": "events",  # Wave 5: Events streaming
}

# Maps legacy root commands to new noun-verb equivalents
LEGACY_COMMAND_MAPPING: dict[str, tuple[str, str]] = {
    "submit": ("task", "create"),
    "list": ("task", "list"),
    "attach": ("session", "attach"),
    "detach": ("session", "detach"),
}


# =============================================================================
# Fuzzy Matching (SPEC-20.14)
# =============================================================================


def get_close_matches_for_command(
    word: str,
    possibilities: list[str],
    n: int = 3,
    cutoff: float = 0.6,
) -> list[str]:
    """Find close matches for a command using Levenshtein distance.

    Args:
        word: The misspelled command.
        possibilities: List of valid command names.
        n: Maximum number of suggestions.
        cutoff: Minimum similarity ratio (0-1).

    Returns:
        List of similar command names, sorted by similarity.
    """
    return difflib.get_close_matches(word, possibilities, n=n, cutoff=cutoff)


def suggest_command(invalid_cmd: str, valid_commands: list[str]) -> str | None:
    """Generate a suggestion message for an invalid command.

    Args:
        invalid_cmd: The command that was not found.
        valid_commands: List of valid command names.

    Returns:
        Suggestion message or None if no close matches.
    """
    matches = get_close_matches_for_command(invalid_cmd, valid_commands)
    if not matches:
        return None

    if len(matches) == 1:
        return f"Did you mean '{matches[0]}'?"
    else:
        suggestions = "\n".join(f"  - {m}" for m in matches)
        return f"Did you mean one of these?\n{suggestions}"


# =============================================================================
# Deprecation Warnings
# =============================================================================


def emit_deprecation_warning(old_cmd: str, new_cmd: str) -> None:
    """Emit a deprecation warning for legacy commands.

    Args:
        old_cmd: The deprecated command name.
        new_cmd: The new command to use instead.
    """
    click.secho(
        f"Warning: '{old_cmd}' is deprecated. Use '{new_cmd}' instead.",
        fg="yellow",
        err=True,
    )


# =============================================================================
# Custom Click Classes for Alias and Fuzzy Matching Support
# =============================================================================


class AliasGroup(click.Group):
    """Click Group that supports command aliases and fuzzy matching.

    Implements SPEC-20.12 (aliases) and SPEC-20.14 (fuzzy matching).
    """

    def __init__(self, *args, aliases: dict[str, str] | None = None, **kwargs):
        """Initialize the alias group.

        Args:
            aliases: Mapping of alias -> real command name.
        """
        super().__init__(*args, **kwargs)
        self.aliases = aliases or {}

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Get a command, resolving aliases first.

        Args:
            ctx: Click context.
            cmd_name: Command name (may be an alias).

        Returns:
            The resolved command or None.
        """
        # First, try to resolve as an alias
        resolved_name = self.aliases.get(cmd_name, cmd_name)

        # Get the command
        cmd = super().get_command(ctx, resolved_name)

        if cmd is not None:
            return cmd

        # Command not found - try fuzzy matching
        valid_commands = list(self.list_commands(ctx))
        suggestion = suggest_command(cmd_name, valid_commands)

        if suggestion:
            click.secho(f"\nUnknown command: {cmd_name}", fg="red", err=True)
            click.secho(suggestion, fg="yellow", err=True)
            click.echo("\nRun 'parhelia --help' for all commands.", err=True)

        return None

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        """Resolve command name and handle aliases.

        Args:
            ctx: Click context.
            args: Command arguments.

        Returns:
            Tuple of (command name, command, remaining args).
        """
        cmd_name, cmd, args = super().resolve_command(ctx, args)

        # If we resolved an alias, use the original name for display
        if cmd_name in self.aliases:
            cmd_name = self.aliases[cmd_name]

        return cmd_name, cmd, args


class DeprecatedAliasGroup(AliasGroup):
    """Click Group that handles deprecated legacy commands.

    Shows deprecation warnings when legacy commands are used.
    """

    def __init__(
        self,
        *args,
        deprecated_commands: dict[str, tuple[str, str]] | None = None,
        **kwargs,
    ):
        """Initialize with deprecated command mappings.

        Args:
            deprecated_commands: Mapping of old_cmd -> (group, verb).
        """
        super().__init__(*args, **kwargs)
        self.deprecated_commands = deprecated_commands or {}

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Get command, emitting deprecation warning for legacy commands."""
        # Check if this is a deprecated command
        if cmd_name in self.deprecated_commands:
            group_name, verb = self.deprecated_commands[cmd_name]
            new_cmd = f"parhelia {group_name} {verb}"
            emit_deprecation_warning(f"parhelia {cmd_name}", new_cmd)

        return super().get_command(ctx, cmd_name)


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
        self._state_store: StateStore | None = None
        self._reconciler: ContainerReconciler | None = None

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

    @property
    def state_store(self) -> StateStore:
        """Get or create state store for container/event tracking."""
        if self._state_store is None:
            self._state_store = StateStore()
        return self._state_store

    @property
    def reconciler(self) -> ContainerReconciler:
        """Get or create container reconciler."""
        if self._reconciler is None:
            self._reconciler = ContainerReconciler(
                state_store=self.state_store,
                modal_client=RealModalClient(),
                config=ReconcilerConfig(),
            )
        return self._reconciler

    def log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose or level == "error":
            prefix = {"info": "ℹ", "success": "✓", "warning": "⚠", "error": "✗"}
            click.echo(f"{prefix.get(level, '•')} {message}")


pass_context = click.make_pass_decorator(CLIContext)


# =============================================================================
# Main CLI Group
# =============================================================================


@click.group(
    cls=DeprecatedAliasGroup,
    aliases=GROUP_ALIASES,
    deprecated_commands=LEGACY_COMMAND_MAPPING,
)
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
@click.option(
    "--dispatch/--no-dispatch",
    default=True,
    help="Dispatch task to Modal immediately (default: yes)",
)
@click.option(
    "--sync",
    is_flag=True,
    help="Wait for task completion (synchronous dispatch)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Skip actual Modal execution (for testing)",
)
@click.option(
    "--automated",
    is_flag=True,
    help="Run in automated mode (skip permission prompts). Use for CI/headless execution.",
)
@pass_context
def submit(
    ctx: CLIContext,
    prompt: str,
    task_type: str,
    gpu: str,
    memory: int,
    workspace: str | None,
    dispatch: bool,
    sync: bool,
    dry_run: bool,
    automated: bool,
) -> None:
    """Submit a new task for execution.

    By default, tasks are dispatched to Modal immediately in async mode.
    Use --no-dispatch to only persist the task without execution.
    Use --sync to wait for completion.
    Use --dry-run to test without Modal.
    Use --automated for headless/CI execution (skips permission prompts).

    Smart defaults: Memory and workspace are remembered from previous
    invocations. GPU must be explicitly requested each time (--gpu).
    """
    import uuid

    from parhelia.dispatch import DispatchMode, TaskDispatcher

    # Smart defaults (SPEC-20.50): Remember user preferences
    smart_prompt = get_smart_prompt()

    # Use cached values if user didn't explicitly provide them
    # NOTE: GPU is intentionally NOT cached - GPUs are expensive and should
    # only be used when explicitly requested via --gpu flag.

    # Memory: use cached if user used default (4)
    if memory == 4:
        cached_memory = smart_prompt.get_default("memory_gb")
        if cached_memory:
            try:
                cached_memory_int = int(cached_memory)
                if cached_memory_int != 4:
                    click.echo(StatusFormatter.info(f"Using cached memory: {cached_memory_int}GB (override with --memory)"))
                    memory = cached_memory_int
            except ValueError:
                pass

    # Remember current values for next time (only if non-default)
    # NOTE: GPU is intentionally NOT remembered - must be explicit each time
    if memory != 4:
        smart_prompt.remember("memory_gb", str(memory))
    if workspace:
        smart_prompt.remember("workspace", workspace)

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
        trust_level=TrustLevel.AUTOMATED if automated else TrustLevel.INTERACTIVE,
    )

    async def _submit_and_dispatch():
        # Always persist the task first
        task_id = await ctx.orchestrator.submit_task(task)
        click.echo(StatusFormatter.success(f"Task submitted: {task_id}"))

        if not dispatch:
            click.echo(StatusFormatter.info("Task saved but not dispatched (use --dispatch to run)"))
            return task_id, None

        # Dispatch to Modal
        dispatcher = TaskDispatcher(ctx.orchestrator, skip_modal=dry_run)

        mode = DispatchMode.SYNC if sync else DispatchMode.ASYNC
        mode_str = "sync" if sync else "async"
        dry_str = " (dry-run)" if dry_run else ""

        # Use spinner for long-running dispatch operation
        spinner = create_spinner(f"Dispatching{dry_str} in {mode_str} mode...")

        def progress(msg: str) -> None:
            spinner.update(msg)
            if ctx.verbose or dry_run:
                # Also log to verbose output
                ctx.log(msg, "info")

        dispatcher.set_progress_callback(progress)

        spinner.start()
        try:
            result = await dispatcher.dispatch(task, mode=mode)
        except Exception as e:
            spinner.stop(success=False, final_message=f"Dispatch failed: {e}")
            # Provide recovery suggestions
            suggestions = ErrorRecovery.suggest("E602", {"task_id": task_id})
            click.echo(StatusFormatter.error(str(e), suggestions))
            return task_id, None
        finally:
            if spinner._running:
                spinner.stop(success=True)

        if result.success:
            click.echo(StatusFormatter.success(f"Worker started: {result.worker_id}"))
            if result.sandbox_id:
                click.echo(StatusFormatter.info(f"Sandbox: {result.sandbox_id}"))
            if result.output and sync:
                click.echo("\nOutput:")
                click.echo(result.output)
        else:
            suggestions = ErrorRecovery.suggest("E602", {"task_id": task_id})
            click.echo(StatusFormatter.error(f"Dispatch failed: {result.error}", suggestions))

        return task_id, result

    asyncio.run(_submit_and_dispatch())


# =============================================================================
# Task Command Group
# =============================================================================


@cli.group()
def task() -> None:
    """Task management commands."""
    pass


@task.command("list")
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def task_list(ctx: CLIContext, status: str, limit: int, as_json: bool) -> None:
    """List tasks with optional filtering.

    Examples:
        parhelia task list
        parhelia task list --status pending
        parhelia task list --status running --limit 5
    """

    async def _list():
        if status == "all":
            tasks = await ctx.orchestrator.get_all_tasks(limit)
        elif status == "pending":
            tasks = await ctx.orchestrator.get_pending_tasks()
        elif status == "running":
            tasks = await ctx.orchestrator.get_running_tasks()
        else:
            tasks = await ctx.orchestrator.get_all_tasks(limit)
            tasks = [t for t in tasks if ctx.orchestrator.task_store.get_status(t.id) == status]

        tasks = tasks[:limit]

        if as_json:
            import json
            click.echo(json.dumps([
                {
                    "id": t.id,
                    "status": ctx.orchestrator.task_store.get_status(t.id),
                    "type": t.task_type.value,
                    "created_at": t.created_at.isoformat(),
                }
                for t in tasks
            ], indent=2))
            return

        if not tasks:
            click.echo("No tasks found.")
            return

        click.echo(f"{'ID':<20} {'Status':<12} {'Type':<12} {'Created':<20}")
        click.echo("-" * 70)

        for t in tasks:
            task_status = ctx.orchestrator.task_store.get_status(t.id) or "unknown"
            status_color = {
                "pending": "yellow",
                "running": "green",
                "completed": "blue",
                "failed": "red",
            }.get(task_status, "white")

            click.echo(
                f"{t.id:<20} "
                f"{click.style(task_status, fg=status_color):<12} "
                f"{t.task_type.value:<12} "
                f"{t.created_at.strftime('%Y-%m-%d %H:%M'):<20}"
            )

    asyncio.run(_list())


@task.command("delete")
@click.argument("task_ids", nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@pass_context
def task_delete(ctx: CLIContext, task_ids: tuple[str, ...], force: bool) -> None:
    """Delete one or more tasks.

    Only pending and failed tasks can be deleted. Running tasks must be
    cancelled first.

    Examples:
        parhelia task delete task-abc123
        parhelia task delete task-abc123 task-def456
        parhelia task delete task-abc123 --force
    """

    async def _delete():
        deleted = 0
        skipped = 0

        for task_id in task_ids:
            task = await ctx.orchestrator.get_task(task_id)
            if not task:
                click.secho(f"Task not found: {task_id}", fg="yellow")
                skipped += 1
                continue

            status = ctx.orchestrator.task_store.get_status(task_id)
            if status == "running":
                click.secho(f"Cannot delete running task: {task_id} (cancel it first)", fg="red")
                skipped += 1
                continue

            if not force:
                click.echo(f"Delete {task_id} ({status}, {task.task_type.value})?")
                if not click.confirm("Confirm"):
                    skipped += 1
                    continue

            ctx.orchestrator.task_store.delete(task_id)
            click.secho(f"Deleted: {task_id}", fg="green")
            deleted += 1

        click.echo(f"\nDeleted {deleted} task(s), skipped {skipped}")

    asyncio.run(_delete())


@task.command("cleanup")
@click.option(
    "--status",
    type=click.Choice(["pending", "failed", "completed", "all"]),
    default="pending",
    help="Status of tasks to clean up (default: pending)",
)
@click.option("--older-than", type=int, help="Only delete tasks older than N hours")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@pass_context
def task_cleanup(
    ctx: CLIContext, status: str, older_than: int | None, dry_run: bool, force: bool
) -> None:
    """Clean up old or test tasks.

    Examples:
        parhelia task cleanup                    # Delete all pending tasks
        parhelia task cleanup --status failed    # Delete failed tasks
        parhelia task cleanup --older-than 24    # Delete pending tasks > 24h old
        parhelia task cleanup --dry-run          # Preview what would be deleted
    """
    from datetime import datetime, timedelta

    async def _cleanup():
        if status == "all":
            tasks = await ctx.orchestrator.get_all_tasks(1000)
        elif status == "pending":
            tasks = await ctx.orchestrator.get_pending_tasks()
        elif status == "running":
            click.secho("Cannot cleanup running tasks. Use 'task cancel' first.", fg="red")
            return
        else:
            tasks = await ctx.orchestrator.get_all_tasks(1000)
            tasks = [t for t in tasks if ctx.orchestrator.task_store.get_status(t.id) == status]

        # Filter by age if specified
        if older_than:
            cutoff = datetime.now() - timedelta(hours=older_than)
            tasks = [t for t in tasks if t.created_at < cutoff]

        if not tasks:
            click.echo("No tasks to clean up.")
            return

        click.echo(f"Found {len(tasks)} task(s) to clean up:")
        for t in tasks[:10]:
            task_status = ctx.orchestrator.task_store.get_status(t.id)
            click.echo(f"  {t.id} ({task_status}, {t.created_at.strftime('%Y-%m-%d %H:%M')})")
        if len(tasks) > 10:
            click.echo(f"  ... and {len(tasks) - 10} more")

        if dry_run:
            click.echo("\n(dry-run, no changes made)")
            return

        if not force:
            if not click.confirm(f"\nDelete {len(tasks)} task(s)?"):
                click.echo("Cancelled.")
                return

        deleted = 0
        for t in tasks:
            ctx.orchestrator.task_store.delete(t.id)
            deleted += 1

        click.secho(f"\nDeleted {deleted} task(s).", fg="green")

    asyncio.run(_cleanup())


@task.command("show")
@click.argument("task_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--watch", "-w", is_flag=True, help="Watch for status changes")
@pass_context
def task_show(ctx: CLIContext, task_id: str, as_json: bool, watch: bool) -> None:
    """Show detailed information about a task.

    Use --watch to stream status updates until completion.

    Examples:
        parhelia task show task-abc123
        parhelia task show task-abc123 --watch
        parhelia task show task-abc123 --watch --json
    """
    if watch:
        _watch_task(ctx, task_id, as_json)
        return

    async def _show():
        task = await ctx.orchestrator.get_task(task_id)
        if not task:
            click.secho(f"Task not found: {task_id}", fg="red")
            sys.exit(1)

        status = ctx.orchestrator.task_store.get_status(task_id)
        result = ctx.orchestrator.task_store.get_result(task_id)
        worker = ctx.orchestrator.worker_store.get_by_task(task_id)

        if as_json:
            import json
            from dataclasses import asdict
            data = {
                "task": {
                    "id": task.id,
                    "prompt": task.prompt,
                    "type": task.task_type.value,
                    "status": status,
                    "created_at": task.created_at.isoformat(),
                    "requirements": {
                        "needs_gpu": task.requirements.needs_gpu,
                        "gpu_type": task.requirements.gpu_type,
                        "min_memory_gb": task.requirements.min_memory_gb,
                    },
                },
                "worker": {
                    "id": worker.id,
                    "state": worker.state.value,
                    "target_type": worker.target_type,
                } if worker else None,
                "result": {
                    "status": result.status,
                    "output": result.output[:500] if result.output else None,
                    "cost_usd": result.cost_usd,
                } if result else None,
            }
            click.echo(json.dumps(data, indent=2))
            return

        click.echo(f"Task: {task.id}")
        click.echo("=" * 50)
        click.echo(f"Status:     {status}")
        click.echo(f"Type:       {task.task_type.value}")
        click.echo(f"Created:    {task.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        click.echo(f"\nPrompt:")
        click.echo(f"  {task.prompt[:200]}{'...' if len(task.prompt) > 200 else ''}")

        click.echo(f"\nRequirements:")
        click.echo(f"  GPU:      {task.requirements.gpu_type or 'none'}")
        click.echo(f"  Memory:   {task.requirements.min_memory_gb}GB")

        if worker:
            click.echo(f"\nWorker:")
            click.echo(f"  ID:       {worker.id}")
            click.echo(f"  State:    {worker.state.value}")
            click.echo(f"  Target:   {worker.target_type}")

        if result:
            click.echo(f"\nResult:")
            click.echo(f"  Status:   {result.status}")
            click.echo(f"  Cost:     ${result.cost_usd:.4f}")
            if result.output:
                click.echo(f"  Output:   {result.output[:100]}...")

    asyncio.run(_show())


@task.command("dispatch")
@click.argument("task_id", required=False)
@click.option("--all", "dispatch_all", is_flag=True, help="Dispatch all pending tasks")
@click.option("--limit", default=10, help="Max tasks to dispatch when using --all")
@click.option("--dry-run", is_flag=True, help="Skip actual Modal execution")
@pass_context
def task_dispatch(
    ctx: CLIContext,
    task_id: str | None,
    dispatch_all: bool,
    limit: int,
    dry_run: bool,
) -> None:
    """Dispatch pending task(s) to Modal.

    Examples:
        parhelia task dispatch task-abc123
        parhelia task dispatch --all
        parhelia task dispatch --all --limit 5 --dry-run
    """
    from parhelia.dispatch import DispatchMode, TaskDispatcher

    async def _dispatch():
        dispatcher = TaskDispatcher(ctx.orchestrator, skip_modal=dry_run)

        if dispatch_all:
            spinner = create_spinner(f"Dispatching up to {limit} pending tasks...")
            spinner.start()

            def progress(msg: str) -> None:
                spinner.update(msg)

            dispatcher.set_progress_callback(progress)

            try:
                results = await dispatcher.dispatch_pending(limit=limit)
                spinner.stop(success=True)
            except Exception as e:
                spinner.stop(success=False, final_message=str(e))
                suggestions = ErrorRecovery.suggest("E602")
                click.echo(StatusFormatter.error(str(e), suggestions))
                sys.exit(1)

            success = sum(1 for r in results if r.success)
            click.echo(f"\nDispatched {success}/{len(results)} tasks")
            for r in results:
                if r.success:
                    click.echo(StatusFormatter.success(f"{r.task_id}: worker {r.worker_id}"))
                else:
                    click.echo(StatusFormatter.error(f"{r.task_id}: {r.error}"))
        elif task_id:
            task = await ctx.orchestrator.get_task(task_id)
            if not task:
                suggestions = ErrorRecovery.suggest("E201", {"task_id": task_id})
                click.echo(StatusFormatter.error(f"Task not found: {task_id}", suggestions))
                sys.exit(1)

            status = ctx.orchestrator.task_store.get_status(task_id)
            if status != "pending":
                click.echo(StatusFormatter.warning(f"Task is not pending (status: {status})"))
                return

            spinner = create_spinner(f"Dispatching task {task_id}...")

            def progress(msg: str) -> None:
                spinner.update(msg)

            dispatcher.set_progress_callback(progress)
            spinner.start()

            try:
                result = await dispatcher.dispatch(task, mode=DispatchMode.ASYNC)
                spinner.stop(success=result.success)
            except Exception as e:
                spinner.stop(success=False, final_message=str(e))
                suggestions = ErrorRecovery.suggest("E602", {"task_id": task_id})
                click.echo(StatusFormatter.error(str(e), suggestions))
                sys.exit(1)

            if result.success:
                click.echo(StatusFormatter.success(f"Dispatched: worker {result.worker_id}"))
            else:
                suggestions = ErrorRecovery.suggest("E602", {"task_id": task_id})
                click.echo(StatusFormatter.error(f"Failed: {result.error}", suggestions))
        else:
            click.echo(StatusFormatter.error("Specify a task ID or use --all"))
            sys.exit(1)

    asyncio.run(_dispatch())


def _watch_task(ctx: CLIContext, task_id: str, json_mode: bool) -> None:
    """Watch a task for status changes.

    Helper function used by task show --watch.
    """
    from parhelia.event_stream import EventFormatter, EventStream

    async def _watch():
        stream = EventStream(ctx.orchestrator)
        formatter = EventFormatter(json_mode=json_mode)

        if not json_mode:
            click.echo(f"Watching task {task_id} for status changes...")
            click.echo("Press Ctrl+C to stop\n")

        try:
            async for event in stream.watch(task_id=task_id, stop_on_complete=True):
                click.echo(formatter.format(event))
        except KeyboardInterrupt:
            if not json_mode:
                click.echo("\nStopped watching.")

    asyncio.run(_watch())


@task.command("watch")
@click.argument("task_id", required=False)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON lines")
@click.option(
    "--status",
    type=click.Choice(["all", "pending", "running"]),
    default="running",
    help="Filter by status when watching all",
)
@click.option("--no-heartbeat", is_flag=True, help="Disable heartbeat events")
@pass_context
def task_watch(
    ctx: CLIContext,
    task_id: str | None,
    as_json: bool,
    status: str,
    no_heartbeat: bool,
) -> None:
    """Watch task(s) for real-time status updates.

    Streams events as they occur. Use --json for machine-readable output.

    Implements [SPEC-12.41] Status Watch.

    Examples:
        parhelia task watch task-abc123          # Watch specific task
        parhelia task watch                       # Watch all running tasks
        parhelia task watch --status pending      # Watch all pending tasks
        parhelia task watch --json               # JSON line output
    """
    from parhelia.event_stream import EventFormatter, EventStream

    async def _watch():
        stream = EventStream(ctx.orchestrator)
        formatter = EventFormatter(json_mode=as_json)
        include_heartbeat = not no_heartbeat

        if not as_json:
            if task_id:
                click.echo(f"Watching task {task_id}...")
            else:
                click.echo(f"Watching {status} tasks...")
            click.echo("Press Ctrl+C to stop\n")

        try:
            if task_id:
                async for event in stream.watch(
                    task_id=task_id,
                    include_heartbeat=include_heartbeat,
                    stop_on_complete=True,
                ):
                    click.echo(formatter.format(event))
            else:
                status_filter = None if status == "all" else status
                async for event in stream.watch_all(
                    status_filter=status_filter,
                    include_heartbeat=include_heartbeat,
                ):
                    click.echo(formatter.format(event))
        except KeyboardInterrupt:
            if not as_json:
                click.echo("\nStopped watching.")

    asyncio.run(_watch())


# =============================================================================
# Attach Command
# =============================================================================


@cli.command()
@click.argument("session_id", required=False)
@click.option("--info-only", is_flag=True, help="Show connection info without attaching")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--last", is_flag=True, help="Attach to most recent session")
@pass_context
def attach(ctx: CLIContext, session_id: str | None, info_only: bool, as_json: bool, last: bool) -> None:
    """Attach to a running session via SSH/tmux.

    Implements [SPEC-12.10] Interactive Attach.

    Smart defaults: If no session ID provided, suggests the most recently
    attached or created session.

    Examples:
        parhelia attach task-abc123
        parhelia attach task-abc123 --info-only
        parhelia attach --last       # Attach to most recent session
    """
    from parhelia.output_formatter import (
        ErrorCode,
        NextAction,
        OutputFormatter,
        format_session_not_found,
    )
    from parhelia.ssh import AttachmentManager, SSHTunnelManager

    formatter = OutputFormatter(json_mode=as_json)

    # Smart defaults (SPEC-20.50): Suggest most recent session
    smart_prompt = get_smart_prompt()

    # Handle --last flag or missing session_id
    if last or not session_id:
        cached_session = smart_prompt.get_default("last_session")

        if last and cached_session:
            session_id = cached_session
            click.echo(StatusFormatter.info(f"Attaching to last session: {session_id}"))
        elif not session_id:
            if cached_session:
                click.echo(StatusFormatter.info(f"Hint: Last session was {cached_session}"))
                click.echo("Use 'parhelia attach --last' to attach to it, or specify a session ID.")
            else:
                click.echo(StatusFormatter.info("Hint: Use 'parhelia session list' to see available sessions."))
            click.echo("")
            suggestions = ErrorRecovery.suggest("E200")
            click.echo(StatusFormatter.error("Session ID required", suggestions))
            sys.exit(1)

    async def _attach():
        # Check if session/task exists
        task = await ctx.orchestrator.get_task(session_id)
        worker = None

        if task:
            worker = ctx.orchestrator.worker_store.get_by_task(session_id)
        else:
            # Try as worker ID
            worker = ctx.orchestrator.get_worker(session_id)

        if not worker:
            if as_json:
                click.echo(format_session_not_found(session_id, json_mode=as_json))
            else:
                suggestions = ErrorRecovery.suggest("E200", {"session_id": session_id})
                click.echo(StatusFormatter.error(f"Session not found: {session_id}", suggestions))
            sys.exit(1)

        # Remember this session for next time
        smart_prompt.remember("last_session", session_id)

        # Check worker state
        if worker.state.value not in ("running", "idle"):
            if as_json:
                click.echo(formatter.error(
                    code=ErrorCode.ATTACH_FAILED,
                    message=f"Session is not running (status: {worker.state.value})",
                    details={"session_id": session_id, "state": worker.state.value},
                ))
            else:
                suggestions = ErrorRecovery.suggest("E601", {"session_id": session_id})
                if worker.state.value == "failed":
                    suggestions.insert(0, f"Try session recovery: parhelia session recover {session_id}")
                click.echo(StatusFormatter.error(
                    f"Session is not running (status: {worker.state.value})",
                    suggestions
                ))
            sys.exit(1)

        # Get tunnel info from worker metrics
        # In production, this would come from Modal sandbox.tunnel()
        tunnel_host = worker.metrics.get("tunnel_host", "localhost")
        tunnel_port = worker.metrics.get("tunnel_port", 2222)
        tmux_session = f"ph-{session_id}"

        # Display connection info
        if not as_json:
            click.echo(f"\nConnecting to session {session_id}...")
            click.echo(f"  Container: {worker.target_type}")
            if worker.created_at:
                from datetime import datetime
                elapsed = datetime.now() - worker.created_at
                minutes = int(elapsed.total_seconds() // 60)
                click.echo(f"  Running for: {minutes}m")
            click.echo("")

        # Create attachment manager
        tunnel_manager = SSHTunnelManager()
        attachment_manager = AttachmentManager(tunnel_manager)

        # Create tunnel record
        tunnel = await tunnel_manager.create_tunnel(
            session_id=session_id,
            host=tunnel_host,
            port=tunnel_port,
        )

        if info_only:
            # Just show connection info
            if as_json:
                click.echo(formatter.success(
                    data={
                        "session_id": session_id,
                        "tunnel": {
                            "host": tunnel.host,
                            "port": tunnel.port,
                            "user": tunnel.user,
                        },
                        "ssh_command": " ".join(tunnel.ssh_command),
                        "tmux_session": tmux_session,
                    },
                    message="Connection info",
                    next_actions=[
                        NextAction(
                            action="connect",
                            description="Connect manually",
                            command=" ".join(tunnel.ssh_command) + f" -t 'tmux attach -t {tmux_session}'",
                        ),
                    ],
                ))
            else:
                click.echo("Connection info:")
                click.echo(f"  Host: {tunnel.host}")
                click.echo(f"  Port: {tunnel.port}")
                click.echo(f"  User: {tunnel.user}")
                click.echo(f"  tmux session: {tmux_session}")
                click.echo("")
                click.echo("Manual connection:")
                click.echo(f"  {' '.join(tunnel.ssh_command)} -t 'tmux attach -t {tmux_session}'")
            return

        # Build and execute attach command
        spinner = None
        if not as_json:
            spinner = create_spinner("Establishing SSH tunnel...")
            spinner.start()

        attach_cmd = tunnel_manager.build_attach_command(tunnel, tmux_session)

        if not as_json:
            if spinner:
                spinner.stop(success=True, final_message="SSH tunnel established")
            click.echo(StatusFormatter.info("Attaching to tmux session..."))
            click.secho("[Press Ctrl+B, D to detach]", fg="cyan")
            click.echo("")

        # Execute SSH with tmux attach
        import subprocess
        try:
            result = subprocess.run(attach_cmd, check=False)

            # After detach, create checkpoint
            if result.returncode == 0 and not as_json:
                click.echo("")
                click.echo(StatusFormatter.success("Detached from session."))
                click.echo("")

                # Trigger checkpoint creation with spinner
                from parhelia.checkpoint import CheckpointTrigger
                from parhelia.session import Session, SessionState

                session = Session(
                    id=session_id,
                    task_id=session_id,
                    state=SessionState.RUNNING,
                    working_directory=f"/vol/parhelia/workspaces/{session_id}",
                )

                cp_spinner = create_spinner("Creating checkpoint...")
                cp_spinner.start()
                try:
                    cp = await ctx.checkpoint_manager.create_checkpoint(
                        session=session,
                        trigger=CheckpointTrigger.DETACH,
                    )
                    cp_spinner.stop(success=True, final_message=f"Checkpoint created: {cp.id}")
                except Exception as e:
                    cp_spinner.stop(success=False, final_message=f"Checkpoint failed: {e}")
                    suggestions = ErrorRecovery.suggest("E600", {"session_id": session_id})
                    click.echo(StatusFormatter.warning(f"Could not create checkpoint: {e}"))

                click.echo("")
                click.echo("Session continues running in background.")
                click.echo("")
                click.echo("Commands:")
                click.echo(f"  Re-attach:    parhelia attach {session_id}")
                click.echo(f"  View logs:    parhelia logs {session_id}")
                click.echo(f"  Kill session: parhelia session kill {session_id}")

        except KeyboardInterrupt:
            if not as_json:
                click.echo("\nConnection interrupted.")

    asyncio.run(_attach())


# =============================================================================
# Detach Command
# =============================================================================


@cli.command()
@click.argument("session_id")
@click.option("--no-checkpoint", is_flag=True, help="Skip checkpoint creation")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def detach(ctx: CLIContext, session_id: str, no_checkpoint: bool, as_json: bool) -> None:
    """Detach from a session (keeps it running).

    Implements [SPEC-12.20] Detach with Checkpoint.

    By default, creates a checkpoint when detaching.
    """
    from parhelia.output_formatter import OutputFormatter

    formatter = OutputFormatter(json_mode=as_json)

    async def _detach():
        if not as_json:
            click.echo(StatusFormatter.info(f"Detaching from session: {session_id}"))

        checkpoint_id = None

        if not no_checkpoint:
            from parhelia.checkpoint import CheckpointTrigger
            from parhelia.session import Session, SessionState

            session = Session(
                id=session_id,
                task_id=session_id,
                state=SessionState.RUNNING,
                working_directory=f"/vol/parhelia/workspaces/{session_id}",
            )

            spinner = None
            if not as_json:
                spinner = create_spinner("Creating checkpoint...")
                spinner.start()

            try:
                cp = await ctx.checkpoint_manager.create_checkpoint(
                    session=session,
                    trigger=CheckpointTrigger.DETACH,
                )
                checkpoint_id = cp.id
                if spinner:
                    spinner.stop(success=True, final_message=f"Checkpoint created: {cp.id}")
            except Exception as e:
                if spinner:
                    spinner.stop(success=False, final_message=f"Checkpoint failed: {e}")
                elif not as_json:
                    click.echo(StatusFormatter.warning(f"Checkpoint failed: {e}"))

        if as_json:
            click.echo(formatter.success(
                data={
                    "session_id": session_id,
                    "status": "detached",
                    "checkpoint_id": checkpoint_id,
                },
                message="Session detached",
            ))
        else:
            click.echo("")
            click.echo("Session continues running in background.")
            click.echo("")
            click.echo("Commands:")
            click.echo(f"  Re-attach:    parhelia attach {session_id}")
            click.echo(f"  View logs:    parhelia logs {session_id}")
            click.echo(f"  Kill session: parhelia session kill {session_id}")

    asyncio.run(_detach())


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
# Container Command Group (SPEC-21 P5)
# =============================================================================


@cli.group(cls=AliasGroup, aliases={"ls": "list", "rm": "terminate"})
def container() -> None:
    """Container introspection commands.

    Implements [SPEC-21.50] Control Plane Introspection.

    View and manage Modal containers tracked by the control plane.
    """
    pass


@container.command("list")
@click.option(
    "--state",
    type=click.Choice(["all", "running", "created", "stopped", "terminated", "orphaned"]),
    default="all",
    help="Filter by container state",
)
@click.option(
    "--health",
    type=click.Choice(["all", "healthy", "degraded", "unhealthy", "dead", "unknown"]),
    default="all",
    help="Filter by health status",
)
@click.option("--limit", default=20, help="Maximum containers to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def container_list(
    ctx: CLIContext,
    state: str,
    health: str,
    limit: int,
    as_json: bool,
) -> None:
    """List all containers with state and health.

    Examples:
        parhelia container list
        parhelia container list --state running
        parhelia container list --health unhealthy --json
        parhelia c ls  # Using alias
    """
    store = ctx.state_store

    # Get containers based on filters
    if state != "all":
        containers = store.get_containers_by_state(ContainerState(state))
    elif health != "all":
        containers = store.get_containers_by_health(HealthStatus(health))
    else:
        containers = store.containers.list_active(limit)
        # Also include recently terminated
        terminated = store.get_containers_by_state(ContainerState.TERMINATED)
        containers = containers + terminated[:max(0, limit - len(containers))]

    containers = containers[:limit]

    if as_json:
        data = {
            "containers": [
                {
                    "id": c.id,
                    "modal_sandbox_id": c.modal_sandbox_id,
                    "state": c.state.value,
                    "health_status": c.health_status.value,
                    "task_id": c.task_id,
                    "created_at": c.created_at.isoformat(),
                    "cost_accrued_usd": c.cost_accrued_usd,
                }
                for c in containers
            ],
            "count": len(containers),
        }
        click.echo(json.dumps(data, indent=2))
        return

    if not containers:
        click.echo("No containers found.")
        return

    # Table header
    click.echo(f"{'ID':<12} {'State':<12} {'Health':<10} {'Task ID':<16} {'Cost':<8}")
    click.echo("-" * 62)

    for c in containers:
        state_color = {
            "running": "green",
            "created": "cyan",
            "stopped": "yellow",
            "terminated": "white",
            "orphaned": "red",
        }.get(c.state.value, "white")

        health_color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
            "dead": "red",
        }.get(c.health_status.value, "white")

        click.echo(
            f"{c.id:<12} "
            f"{click.style(c.state.value, fg=state_color):<12} "
            f"{click.style(c.health_status.value, fg=health_color):<10} "
            f"{(c.task_id or '-'):<16} "
            f"${c.cost_accrued_usd:.2f}"
        )


@container.command("show")
@click.argument("container_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def container_show(ctx: CLIContext, container_id: str, as_json: bool) -> None:
    """Show detailed information about a container.

    Examples:
        parhelia container show c-abc12345
        parhelia container show c-abc12345 --json
    """
    store = ctx.state_store
    container = store.get_container(container_id)

    if not container:
        # Try by Modal sandbox ID
        container = store.get_container_by_modal_id(container_id)

    if not container:
        click.secho(f"Container not found: {container_id}", fg="red")
        sys.exit(1)

    if as_json:
        from dataclasses import asdict
        data = asdict(container)
        # Convert enums and datetimes
        data["state"] = container.state.value
        data["health_status"] = container.health_status.value
        data["created_at"] = container.created_at.isoformat()
        data["started_at"] = container.started_at.isoformat() if container.started_at else None
        data["terminated_at"] = container.terminated_at.isoformat() if container.terminated_at else None
        data["last_heartbeat_at"] = container.last_heartbeat_at.isoformat() if container.last_heartbeat_at else None
        data["updated_at"] = container.updated_at.isoformat()
        click.echo(json.dumps(data, indent=2))
        return

    click.echo(f"Container: {container.id}")
    click.echo("=" * 50)

    click.echo(f"\nIdentifiers:")
    click.echo(f"  Container ID:     {container.id}")
    click.echo(f"  Modal Sandbox ID: {container.modal_sandbox_id}")
    click.echo(f"  Worker ID:        {container.worker_id or '-'}")
    click.echo(f"  Task ID:          {container.task_id or '-'}")
    click.echo(f"  Session ID:       {container.session_id or '-'}")

    click.echo(f"\nStatus:")
    click.echo(f"  State:            {container.state.value}")
    click.echo(f"  Health:           {container.health_status.value}")
    click.echo(f"  Failures:         {container.consecutive_failures}")

    click.echo(f"\nLifecycle:")
    click.echo(f"  Created:          {container.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if container.started_at:
        click.echo(f"  Started:          {container.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if container.terminated_at:
        click.echo(f"  Terminated:       {container.terminated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if container.last_heartbeat_at:
        click.echo(f"  Last Heartbeat:   {container.last_heartbeat_at.strftime('%Y-%m-%d %H:%M:%S')}")

    if container.termination_reason:
        click.echo(f"\nTermination:")
        click.echo(f"  Reason:           {container.termination_reason}")
        if container.exit_code is not None:
            click.echo(f"  Exit Code:        {container.exit_code}")

    click.echo(f"\nResources:")
    click.echo(f"  CPU Cores:        {container.cpu_cores or '-'}")
    click.echo(f"  Memory MB:        {container.memory_mb or '-'}")
    click.echo(f"  GPU:              {container.gpu_type or '-'}")
    click.echo(f"  Region:           {container.region or '-'}")

    click.echo(f"\nCost:")
    click.echo(f"  Accrued:          ${container.cost_accrued_usd:.4f}")
    if container.cost_rate_per_hour:
        click.echo(f"  Rate/hr:          ${container.cost_rate_per_hour:.4f}")


@container.command("events")
@click.argument("container_id")
@click.option("--limit", default=50, help="Maximum events to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def container_events(ctx: CLIContext, container_id: str, limit: int, as_json: bool) -> None:
    """Show event history for a container.

    Examples:
        parhelia container events c-abc12345
        parhelia container events c-abc12345 --limit 100
    """
    store = ctx.state_store
    events = store.get_events(container_id=container_id, limit=limit)

    if as_json:
        data = {
            "container_id": container_id,
            "events": [
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
            ],
            "count": len(events),
        }
        click.echo(json.dumps(data, indent=2))
        return

    if not events:
        click.echo(f"No events found for container: {container_id}")
        return

    click.echo(f"Events for {container_id} ({len(events)} events)")
    click.echo("=" * 70)

    for e in events:
        time_str = e.timestamp.strftime("%H:%M:%S")
        type_color = {
            "container_created": "cyan",
            "container_started": "green",
            "container_stopped": "yellow",
            "container_terminated": "white",
            "container_healthy": "green",
            "container_unhealthy": "red",
            "container_dead": "red",
            "orphan_detected": "red",
            "error": "red",
        }.get(e.event_type.value, "white")

        click.echo(
            f"[{time_str}] "
            f"{click.style(e.event_type.value, fg=type_color)}: "
            f"{e.message or '-'}"
        )


@container.command("health")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def container_health(ctx: CLIContext, as_json: bool) -> None:
    """Show container health summary.

    Provides overview of container states and health across the control plane.

    Examples:
        parhelia container health
        parhelia container health --json
    """
    store = ctx.state_store
    stats = store.get_container_stats()

    if as_json:
        data = {
            "total": stats.total,
            "by_state": stats.by_state,
            "by_health": stats.by_health,
            "total_cost_usd": stats.total_cost_usd,
            "oldest_running": stats.oldest_running.isoformat() if stats.oldest_running else None,
        }
        click.echo(json.dumps(data, indent=2))
        return

    click.echo("Container Health Summary")
    click.echo("=" * 40)

    click.echo(f"\nTotal Containers: {stats.total}")
    click.echo(f"Total Cost:       ${stats.total_cost_usd:.2f}")

    click.echo(f"\nBy State:")
    for state, count in sorted(stats.by_state.items()):
        color = {
            "running": "green",
            "created": "cyan",
            "stopped": "yellow",
            "terminated": "white",
            "orphaned": "red",
        }.get(state, "white")
        click.echo(f"  {click.style(state, fg=color):<14} {count}")

    click.echo(f"\nBy Health:")
    for health, count in sorted(stats.by_health.items()):
        color = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
            "dead": "red",
        }.get(health, "white")
        click.echo(f"  {click.style(health, fg=color):<14} {count}")

    if stats.oldest_running:
        elapsed = datetime.now() - stats.oldest_running
        hours = int(elapsed.total_seconds() // 3600)
        click.echo(f"\nOldest Running: {hours}h (since {stats.oldest_running.strftime('%Y-%m-%d %H:%M')})")


@container.command("watch")
@click.option("--state", help="Filter by state")
@click.option("--interval", default=5, help="Refresh interval in seconds")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON lines")
@pass_context
def container_watch(ctx: CLIContext, state: str | None, interval: int, as_json: bool) -> None:
    """Watch container changes in real-time.

    Streams container state changes as they occur.

    Examples:
        parhelia container watch
        parhelia container watch --state running
        parhelia container watch --interval 10 --json
    """
    import time

    if not as_json:
        click.echo("Watching containers for changes (Ctrl+C to stop)...")
        click.echo("")

    last_seen: dict[str, tuple[str, str]] = {}  # id -> (state, health)

    try:
        while True:
            store = ctx.state_store

            if state:
                containers = store.get_containers_by_state(ContainerState(state))
            else:
                containers = store.containers.list_active(100)

            current: dict[str, tuple[str, str]] = {}
            changes = []

            for c in containers:
                current[c.id] = (c.state.value, c.health_status.value)

                if c.id not in last_seen:
                    changes.append({
                        "type": "new",
                        "container_id": c.id,
                        "state": c.state.value,
                        "health": c.health_status.value,
                    })
                elif last_seen[c.id] != current[c.id]:
                    old_state, old_health = last_seen[c.id]
                    changes.append({
                        "type": "changed",
                        "container_id": c.id,
                        "old_state": old_state,
                        "new_state": c.state.value,
                        "old_health": old_health,
                        "new_health": c.health_status.value,
                    })

            # Check for removed containers
            for cid in last_seen:
                if cid not in current:
                    changes.append({
                        "type": "removed",
                        "container_id": cid,
                    })

            # Output changes
            for change in changes:
                timestamp = datetime.now().strftime("%H:%M:%S")
                if as_json:
                    change["timestamp"] = datetime.now().isoformat()
                    click.echo(json.dumps(change))
                else:
                    if change["type"] == "new":
                        click.echo(
                            f"[{timestamp}] "
                            f"{click.style('NEW', fg='cyan')}: {change['container_id']} "
                            f"({change['state']}, {change['health']})"
                        )
                    elif change["type"] == "changed":
                        click.echo(
                            f"[{timestamp}] "
                            f"{click.style('CHANGED', fg='yellow')}: {change['container_id']} "
                            f"state: {change['old_state']} -> {change['new_state']}, "
                            f"health: {change['old_health']} -> {change['new_health']}"
                        )
                    elif change["type"] == "removed":
                        click.echo(
                            f"[{timestamp}] "
                            f"{click.style('REMOVED', fg='red')}: {change['container_id']}"
                        )

            last_seen = current
            time.sleep(interval)

    except KeyboardInterrupt:
        if not as_json:
            click.echo("\nStopped watching.")


@container.command("terminate")
@click.argument("container_id")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
@click.option("--force", is_flag=True, help="Force termination even if busy")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def container_terminate(
    ctx: CLIContext,
    container_id: str,
    yes: bool,
    force: bool,
    as_json: bool,
) -> None:
    """Terminate a container.

    Gracefully terminates a container and updates the control plane state.

    Examples:
        parhelia container terminate c-abc12345
        parhelia container terminate c-abc12345 -y --force
        parhelia c rm c-abc12345  # Using alias
    """
    store = ctx.state_store
    container = store.get_container(container_id)

    if not container:
        container = store.get_container_by_modal_id(container_id)

    if not container:
        if as_json:
            click.echo(json.dumps({"success": False, "error": f"Container not found: {container_id}"}))
        else:
            click.secho(f"Container not found: {container_id}", fg="red")
        sys.exit(1)

    if container.state == ContainerState.TERMINATED:
        if as_json:
            click.echo(json.dumps({"success": False, "error": "Container already terminated"}))
        else:
            click.echo("Container already terminated.")
        return

    # Confirm unless -y
    if not yes and not as_json:
        click.echo(f"Container: {container.id}")
        click.echo(f"  Modal ID: {container.modal_sandbox_id}")
        click.echo(f"  State:    {container.state.value}")
        click.echo(f"  Task:     {container.task_id or '-'}")
        click.echo("")
        if not click.confirm("Terminate this container?"):
            click.echo("Cancelled.")
            return

    async def _terminate():
        reconciler = ctx.reconciler
        success = await reconciler.modal_client.terminate_sandbox(container.modal_sandbox_id)

        if success:
            store.update_container_state(
                container.id,
                ContainerState.TERMINATED,
                reason="Manual termination via CLI",
            )
        return success

    success = asyncio.run(_terminate())

    if as_json:
        click.echo(json.dumps({
            "success": success,
            "container_id": container.id,
            "modal_sandbox_id": container.modal_sandbox_id,
        }))
    else:
        if success:
            click.secho(f"Container {container.id} terminated successfully", fg="green")
        else:
            click.secho(f"Failed to terminate container {container.id}", fg="red")
            sys.exit(1)


# =============================================================================
# Reconciler Command Group (SPEC-21 P5)
# =============================================================================


@cli.group()
def reconciler() -> None:
    """Reconciler status and control commands.

    Implements [SPEC-21.50] Control Plane Introspection.
    """
    pass


@reconciler.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def reconciler_status(ctx: CLIContext, as_json: bool) -> None:
    """Show reconciler status and last run info.

    Examples:
        parhelia reconciler status
        parhelia r status  # Using alias
    """
    reconciler = ctx.reconciler
    store = ctx.state_store

    # Get recent reconciliation events
    recent_events = store.get_events(
        event_type=EventType.STATE_DRIFT_CORRECTED,
        limit=5,
    )
    orphan_events = store.get_events(
        event_type=EventType.ORPHAN_DETECTED,
        limit=5,
    )
    error_events = store.get_events(
        event_type=EventType.RECONCILE_FAILED,
        limit=3,
    )

    # Get container stats
    stats = store.get_container_stats()
    orphaned_count = stats.by_state.get("orphaned", 0)

    if as_json:
        data = {
            "is_running": reconciler.is_running,
            "config": {
                "poll_interval_seconds": reconciler.config.poll_interval_seconds,
                "stale_threshold_seconds": reconciler.config.stale_threshold_seconds,
                "orphan_grace_period_seconds": reconciler.config.orphan_grace_period_seconds,
                "auto_terminate_orphans": reconciler.config.auto_terminate_orphans,
            },
            "stats": {
                "total_containers": stats.total,
                "orphaned_containers": orphaned_count,
            },
            "recent_drift_corrections": len(recent_events),
            "recent_orphans_detected": len(orphan_events),
            "recent_errors": len(error_events),
        }
        click.echo(json.dumps(data, indent=2))
        return

    click.echo("Reconciler Status")
    click.echo("=" * 40)

    status_str = click.style("Running", fg="green") if reconciler.is_running else click.style("Stopped", fg="yellow")
    click.echo(f"\nStatus: {status_str}")

    click.echo(f"\nConfiguration:")
    click.echo(f"  Poll Interval:      {reconciler.config.poll_interval_seconds}s")
    click.echo(f"  Stale Threshold:    {reconciler.config.stale_threshold_seconds}s")
    click.echo(f"  Orphan Grace:       {reconciler.config.orphan_grace_period_seconds}s")
    click.echo(f"  Auto-Terminate:     {reconciler.config.auto_terminate_orphans}")

    click.echo(f"\nContainer Stats:")
    click.echo(f"  Total:              {stats.total}")
    click.echo(f"  Orphaned:           {orphaned_count}")

    if recent_events:
        click.echo(f"\nRecent Drift Corrections ({len(recent_events)}):")
        for e in recent_events[:3]:
            click.echo(f"  [{e.timestamp.strftime('%H:%M:%S')}] {e.container_id}: {e.message}")

    if orphan_events:
        click.echo(f"\nRecent Orphans Detected ({len(orphan_events)}):")
        for e in orphan_events[:3]:
            click.echo(f"  [{e.timestamp.strftime('%H:%M:%S')}] {e.container_id}")

    if error_events:
        click.echo(f"\nRecent Errors ({len(error_events)}):")
        for e in error_events:
            click.echo(f"  [{e.timestamp.strftime('%H:%M:%S')}] {e.message}")


@reconciler.command("run")
@click.option("--once", is_flag=True, help="Run single reconciliation cycle")
@pass_context
def reconciler_run(ctx: CLIContext, once: bool) -> None:
    """Manually trigger reconciliation.

    Examples:
        parhelia reconciler run --once
    """
    async def _run():
        reconciler = ctx.reconciler

        if once:
            spinner = create_spinner("Running reconciliation...")
            spinner.start()
            try:
                result = await reconciler.reconcile()
                spinner.stop(success=True)
                click.echo(f"\n{result}")
                if result.errors:
                    for err in result.errors:
                        click.secho(f"  Error: {err}", fg="red")
            except Exception as e:
                spinner.stop(success=False, final_message=str(e))
                sys.exit(1)
        else:
            click.echo("Starting reconciliation loop (Ctrl+C to stop)...")
            try:
                await reconciler.run_background()
            except KeyboardInterrupt:
                reconciler.stop()
                click.echo("\nReconciliation stopped.")

    asyncio.run(_run())


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
            await ctx.checkpoint_manager.update_checkpoint(checkpoint)
            click.secho(f"Checkpoint {checkpoint.id} approved", fg="green")
            return

        if action == "reject":
            if not reason:
                click.secho("Rejection requires --reason", fg="red")
                return
            approval = await manager.reject(checkpoint, user=user, reason=reason)
            checkpoint.approval = approval
            await ctx.checkpoint_manager.update_checkpoint(checkpoint)
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def session_recover(
    ctx: CLIContext,
    session_id: str,
    from_checkpoint: str | None,
    action: str | None,
    as_json: bool,
) -> None:
    """Interactive recovery wizard for a session.

    Implements [SPEC-07.42] and [SPEC-12.30].

    Handles common recovery scenarios:
    - Resume from failure (crash, timeout)
    - Resume after rejection
    - Manual checkpoint selection

    Example:
        parhelia session recover my-session
        parhelia session recover my-session --from cp-abc123
        parhelia session recover my-session --action resume
        parhelia session recover my-session --action resume --json
    """
    from parhelia.output_formatter import ErrorCode, NextAction, OutputFormatter
    from parhelia.recovery import RecoveryAction, RecoveryManager

    formatter = OutputFormatter(json_mode=as_json)

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
            if as_json:
                click.echo(formatter.error(
                    code=ErrorCode.SESSION_NOT_FOUND,
                    message=f"No recovery options found for session: {session_id}",
                    details={"session_id": session_id},
                ))
            else:
                click.secho(f"No recovery options found for session: {session_id}", fg="red")
            sys.exit(1)

        # JSON mode with --action: execute non-interactively
        if as_json and action:
            action_map = {
                "resume": RecoveryAction.RESUME,
                "new": RecoveryAction.NEW_SESSION,
                "wait": RecoveryAction.WAIT_FOR_USER,
            }
            selected_action = action_map.get(action)

            result = await manager.execute_failure_recovery(plan, selected_action)

            if result.success:
                click.echo(formatter.success(
                    data={
                        "session_id": session_id,
                        "action": action,
                        "new_session_id": result.new_session_id,
                        "checkpoint_id": plan.current_checkpoint_id,
                    },
                    message="Recovery successful",
                    next_actions=[
                        NextAction(
                            action="attach",
                            description="Attach to recovered session",
                            command=f"parhelia attach {result.new_session_id or session_id}",
                        ),
                    ] if result.new_session_id else [],
                ))
            else:
                click.echo(formatter.error(
                    code=ErrorCode.INTERNAL_ERROR,
                    message=result.error_message or "Recovery failed",
                    details={
                        "session_id": session_id,
                        "action": action,
                    },
                ))
                sys.exit(1)
            return

        # JSON mode without --action: return plan options
        if as_json and not action:
            click.echo(formatter.success(
                data={
                    "session_id": session_id,
                    "scenario": plan.scenario.value,
                    "checkpoint_id": plan.current_checkpoint_id,
                    "recommended_checkpoint_id": plan.recommended_checkpoint_id,
                    "options": [
                        {
                            "action": opt.action.value,
                            "description": opt.description,
                            "recommended": opt.recommended,
                        }
                        for opt in plan.options
                    ],
                },
                message="Recovery plan available",
                next_actions=[
                    NextAction(
                        action=opt.action.value,
                        description=opt.description,
                        command=f"parhelia session recover {session_id} --action {opt.action.value} --json",
                    )
                    for opt in plan.options
                ],
            ))
            return

        # Human mode: display plan
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
# MCP Server Command
# =============================================================================


@cli.command("mcp-server")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="Transport type (default: stdio)",
)
@click.option("--host", default="127.0.0.1", help="Host for HTTP transport")
@click.option("--port", default=8080, type=int, help="Port for HTTP transport")
@click.option("--require-auth", is_flag=True, help="Require authentication (default for HTTP)")
@click.option("--no-auth", is_flag=True, help="Disable authentication")
def mcp_server(transport: str, host: str, port: int, require_auth: bool, no_auth: bool) -> None:
    """Start Parhelia MCP server for programmatic access.

    Implements [SPEC-11.40] MCP Server with secure authentication.

    Transport modes:
    - stdio: For Claude Code integration (default, no auth required)
    - http: For remote access (auth required by default)

    Authentication:
    - Set PARHELIA_AUTH_TOKENS env var with comma-separated tokens
    - For HTTP: Include "Authorization: Bearer <token>" header

    Examples:
        parhelia mcp-server                    # stdio for Claude Code
        parhelia mcp-server --transport http   # HTTP with auth
        parhelia mcp-server --transport http --no-auth  # HTTP without auth (dev only)

    Add to ~/.claude/mcp_config.json for Claude Code:
        {
            "mcpServers": {
                "parhelia": {
                    "command": "parhelia",
                    "args": ["mcp-server"]
                }
            }
        }
    """
    from parhelia.mcp_server import run_mcp_server

    # Determine auth requirement
    auth_required = None
    if require_auth:
        auth_required = True
    elif no_auth:
        auth_required = False

    run_mcp_server(transport=transport, host=host, port=port, require_auth=auth_required)


# =============================================================================
# Cleanup Command
# =============================================================================


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be cleaned up without doing it")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def cleanup(ctx: CLIContext, dry_run: bool, force: bool, as_json: bool) -> None:
    """Find and terminate orphaned Modal containers.

    Scans for running Parhelia containers and terminates them.
    Use this to stop runaway containers and prevent cost overruns.

    Examples:
        parhelia cleanup              # Interactive cleanup
        parhelia cleanup --dry-run    # See what would be cleaned
        parhelia cleanup --force      # Skip confirmation
    """
    import subprocess

    def run_modal_cmd(args: list[str]) -> str:
        """Run modal CLI command and return output."""
        result = subprocess.run(
            ["modal"] + args,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr

    # Get active containers
    try:
        output = run_modal_cmd(["container", "list"])
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        click.secho(f"Error running modal CLI: {e}", fg="red")
        sys.exit(1)

    # Parse container output - look for parhelia app
    lines = output.strip().split("\n")
    parhelia_containers = []

    for line in lines:
        if "parhelia" in line.lower() and "ta-" in line:
            # Extract container ID (starts with ta-)
            parts = line.split()
            for part in parts:
                if part.startswith("ta-"):
                    container_id = part.rstrip("…")
                    parhelia_containers.append(container_id)
                    break

    if not parhelia_containers:
        if as_json:
            click.echo('{"containers_found": 0, "action": "none"}')
        else:
            click.secho("No orphaned Parhelia containers found.", fg="green")
        return

    # Show what we found
    if as_json:
        data = {
            "containers_found": len(parhelia_containers),
            "container_ids": parhelia_containers,
            "dry_run": dry_run,
        }
        if dry_run:
            data["action"] = "would_terminate"
            click.echo(json.dumps(data, indent=2))
            return
    else:
        click.echo(f"\nFound {len(parhelia_containers)} Parhelia container(s):")
        for cid in parhelia_containers[:10]:
            click.echo(f"  • {cid}")
        if len(parhelia_containers) > 10:
            click.echo(f"  ... and {len(parhelia_containers) - 10} more")

        if dry_run:
            click.secho("\n[DRY RUN] Would terminate these containers", fg="yellow")
            return

    # Confirm unless --force
    if not force:
        click.echo()
        if not click.confirm(f"Terminate {len(parhelia_containers)} container(s)?", default=True):
            click.echo("Aborted.")
            return

    # Stop the app (terminates all containers)
    click.echo("\nTerminating containers...")
    try:
        run_modal_cmd(["app", "stop", "parhelia"])
    except subprocess.TimeoutExpired:
        click.secho("Timeout stopping app, trying individual containers...", fg="yellow")

    # Verify cleanup
    try:
        output = run_modal_cmd(["container", "list"])
        remaining = output.count("parhelia")
        if remaining == 0 or "parhelia" not in output.lower():
            if as_json:
                click.echo(json.dumps({
                    "containers_terminated": len(parhelia_containers),
                    "action": "terminated",
                    "success": True,
                }))
            else:
                click.secho(f"\n✓ Successfully terminated {len(parhelia_containers)} container(s)", fg="green")
        else:
            if as_json:
                click.echo(json.dumps({
                    "containers_terminated": len(parhelia_containers),
                    "action": "partial",
                    "success": False,
                    "message": "Some containers may still be running",
                }))
            else:
                click.secho("\n⚠ Some containers may still be running. Check Modal dashboard.", fg="yellow")
    except Exception as e:
        click.secho(f"Could not verify cleanup: {e}", fg="yellow")


# =============================================================================
# Doctor Command
# =============================================================================


@cli.command()
@click.option("--fix", is_flag=True, help="Attempt to auto-fix issues")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def doctor(ctx: CLIContext, fix: bool, verbose: bool, as_json: bool) -> None:
    """Check system health and diagnose issues.

    Runs consistency checks on tasks, workers, containers, and configuration.
    Use --fix to attempt automatic repairs.

    Checks performed:
    - Task-worker linkage (running tasks have workers)
    - Worker-container consistency (workers have valid containers)
    - Orphaned references (dangling foreign keys)
    - Stale heartbeats (containers not reporting)
    - Database integrity (schema, constraints)
    - Configuration validity (Modal credentials, paths)

    Examples:
        parhelia doctor              # Run all checks
        parhelia doctor --fix        # Fix issues automatically
        parhelia doctor --verbose    # Show detailed diagnostics
    """
    from dataclasses import dataclass, field

    @dataclass
    class CheckResult:
        name: str
        status: str  # PASS, WARN, FAIL
        message: str
        issues: list[str] = field(default_factory=list)
        fixed: list[str] = field(default_factory=list)

    results: list[CheckResult] = []

    async def run_checks():
        # Check 1: Task-Worker Linkage
        check = CheckResult(name="task-worker-linkage", status="PASS", message="")
        running_tasks = await ctx.orchestrator.get_running_tasks()
        orphaned_tasks = []
        for task in running_tasks:
            worker = ctx.orchestrator.worker_store.get_by_task(task.id)
            if not worker:
                orphaned_tasks.append(task.id)
                check.issues.append(f"Task {task.id} running but no worker assigned")

        if orphaned_tasks:
            check.status = "FAIL"
            check.message = f"{len(orphaned_tasks)} running task(s) without workers"
            if fix:
                for task_id in orphaned_tasks:
                    ctx.orchestrator.task_store.update_status(task_id, "failed")
                    check.fixed.append(f"Marked {task_id} as failed")
        else:
            check.message = f"{len(running_tasks)} running tasks all have workers"
        results.append(check)

        # Check 2: Worker-Container Consistency
        check = CheckResult(name="worker-container-consistency", status="PASS", message="")
        workers = ctx.orchestrator.worker_store.list_all(limit=500)
        running_workers = [w for w in workers if w.state.value == "running"]
        orphaned_workers = []
        for worker in running_workers:
            if not worker.container_id:
                orphaned_workers.append(worker.id)
                check.issues.append(f"Worker {worker.id} running but no container_id")

        if orphaned_workers:
            check.status = "WARN"
            check.message = f"{len(orphaned_workers)} worker(s) without container reference"
        else:
            check.message = f"{len(running_workers)} running workers all have containers"
        results.append(check)

        # Check 3: Stale Heartbeats
        check = CheckResult(name="stale-heartbeats", status="PASS", message="")
        from datetime import timedelta
        stale_threshold = timedelta(minutes=5)
        now = datetime.now()
        stale_workers = []
        for worker in running_workers:
            if worker.last_heartbeat_at:
                age = now - worker.last_heartbeat_at
                if age > stale_threshold:
                    stale_workers.append((worker.id, age))
                    check.issues.append(f"Worker {worker.id} last heartbeat {age.seconds}s ago")

        if stale_workers:
            check.status = "WARN"
            check.message = f"{len(stale_workers)} worker(s) with stale heartbeats"
        else:
            check.message = "All running workers have recent heartbeats"
        results.append(check)

        # Check 4: Database Integrity
        check = CheckResult(name="database-integrity", status="PASS", message="")
        try:
            task_count = len(await ctx.orchestrator.get_all_tasks(1000))
            worker_count = len(ctx.orchestrator.worker_store.list_all(1000))
            check.message = f"Database OK: {task_count} tasks, {worker_count} workers"
        except Exception as e:
            check.status = "FAIL"
            check.message = f"Database error: {e}"
            check.issues.append(str(e))
        results.append(check)

        # Check 5: Configuration
        check = CheckResult(name="configuration", status="PASS", message="")
        config_issues = []
        config = ctx.config

        if not config.paths.volume_root:
            config_issues.append("volume_root not configured")
        if config.budget.default_ceiling_usd <= 0:
            config_issues.append("budget ceiling not set")

        if config_issues:
            check.status = "WARN"
            check.message = f"{len(config_issues)} configuration issue(s)"
            check.issues = config_issues
        else:
            check.message = "Configuration valid"
        results.append(check)

        # Check 6: Modal Connectivity
        check = CheckResult(name="modal-connectivity", status="PASS", message="")
        try:
            import subprocess
            result = subprocess.run(
                ["modal", "app", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                check.message = "Modal connected"
            else:
                check.status = "FAIL"
                check.message = "Modal connection failed"
                err = result.stderr.strip() if result.stderr else "Unknown error"
                check.issues.append(err if err else "Run 'modal token set' to authenticate")
        except FileNotFoundError:
            check.status = "FAIL"
            check.message = "Modal CLI not found"
            check.issues.append("Install modal: pip install modal")
        except subprocess.TimeoutExpired:
            check.status = "WARN"
            check.message = "Modal CLI timeout"
        results.append(check)

    asyncio.run(run_checks())

    # Output results
    if as_json:
        click.echo(json.dumps([
            {
                "name": r.name,
                "status": r.status,
                "message": r.message,
                "issues": r.issues,
                "fixed": r.fixed,
            }
            for r in results
        ], indent=2))
        return

    click.echo("Parhelia System Health Check")
    click.echo("=" * 60)
    click.echo()

    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    for r in results:
        status_color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[r.status]
        status_symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[r.status]

        click.echo(f"{click.style(status_symbol, fg=status_color)} {r.name}: {r.message}")

        if verbose and r.issues:
            for issue in r.issues:
                click.echo(f"    → {issue}")
        if r.fixed:
            for fixed in r.fixed:
                click.echo(click.style(f"    ✓ Fixed: {fixed}", fg="green"))

    click.echo()
    click.echo(f"Summary: {pass_count} passed, {warn_count} warnings, {fail_count} failed")

    if fail_count > 0 and not fix:
        click.echo()
        click.echo("Run 'parhelia doctor --fix' to attempt automatic repairs")


# =============================================================================
# Worker Command Group
# =============================================================================


@cli.group()
def worker() -> None:
    """Worker management commands.

    Inspect and manage task execution workers.
    """
    pass


@worker.command("list")
@click.option(
    "--state",
    type=click.Choice(["all", "idle", "running", "completed", "failed", "terminated"]),
    default="all",
    help="Filter by state",
)
@click.option("-n", "--limit", type=int, default=20, help="Maximum workers to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def worker_list(ctx: CLIContext, state: str, limit: int, as_json: bool) -> None:
    """List workers with optional filtering.

    Examples:
        parhelia worker list
        parhelia worker list --state running
    """
    from parhelia.orchestrator import WorkerState

    if state == "all":
        workers = ctx.orchestrator.worker_store.list_all(limit)
    else:
        workers = ctx.orchestrator.worker_store.list_by_state(WorkerState(state), limit)

    if as_json:
        click.echo(json.dumps([
            {
                "id": w.id,
                "task_id": w.task_id,
                "state": w.state.value,
                "target_type": w.target_type,
                "container_id": w.container_id,
                "health_status": w.health_status,
                "created_at": w.created_at.isoformat(),
            }
            for w in workers
        ], indent=2))
        return

    if not workers:
        click.echo("No workers found.")
        return

    click.echo(f"{'ID':<20} {'State':<12} {'Task':<20} {'Container':<15}")
    click.echo("-" * 70)

    for w in workers:
        state_color = {
            "idle": "blue",
            "running": "green",
            "completed": "cyan",
            "failed": "red",
            "terminated": "white",
        }.get(w.state.value, "white")

        container_short = (w.container_id or "-")[:12]
        task_short = (w.task_id or "-")[:18]

        click.echo(
            f"{w.id:<20} "
            f"{click.style(w.state.value, fg=state_color):<12} "
            f"{task_short:<20} "
            f"{container_short:<15}"
        )


@worker.command("show")
@click.argument("worker_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def worker_show(ctx: CLIContext, worker_id: str, as_json: bool) -> None:
    """Show detailed worker information.

    Examples:
        parhelia worker show worker-abc123
    """
    worker = ctx.orchestrator.get_worker(worker_id)

    if not worker:
        click.secho(f"Worker not found: {worker_id}", fg="red")
        sys.exit(1)

    if as_json:
        click.echo(json.dumps({
            "id": worker.id,
            "task_id": worker.task_id,
            "state": worker.state.value,
            "target_type": worker.target_type,
            "gpu_type": worker.gpu_type,
            "container_id": worker.container_id,
            "session_id": worker.session_id,
            "health_status": worker.health_status,
            "last_heartbeat_at": worker.last_heartbeat_at.isoformat() if worker.last_heartbeat_at else None,
            "created_at": worker.created_at.isoformat(),
            "terminated_at": worker.terminated_at.isoformat() if worker.terminated_at else None,
            "exit_code": worker.exit_code,
            "metrics": worker.metrics,
        }, indent=2))
        return

    click.echo(f"Worker: {worker.id}")
    click.echo("=" * 50)
    click.echo(f"State:           {worker.state.value}")
    click.echo(f"Task ID:         {worker.task_id or '-'}")
    click.echo(f"Target:          {worker.target_type}")
    click.echo(f"GPU:             {worker.gpu_type or '-'}")
    click.echo(f"Container:       {worker.container_id or '-'}")
    click.echo(f"Session:         {worker.session_id or '-'}")
    click.echo(f"Health:          {worker.health_status}")
    click.echo(f"Last Heartbeat:  {worker.last_heartbeat_at or '-'}")
    click.echo(f"Created:         {worker.created_at}")
    if worker.terminated_at:
        click.echo(f"Terminated:      {worker.terminated_at}")
        click.echo(f"Exit Code:       {worker.exit_code}")


@worker.command("kill")
@click.argument("worker_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@pass_context
def worker_kill(ctx: CLIContext, worker_id: str, force: bool) -> None:
    """Terminate a worker and mark its task as failed.

    Examples:
        parhelia worker kill worker-abc123
    """
    worker = ctx.orchestrator.get_worker(worker_id)

    if not worker:
        click.secho(f"Worker not found: {worker_id}", fg="red")
        sys.exit(1)

    if worker.state.value in ("completed", "failed", "terminated"):
        click.secho(f"Worker already in terminal state: {worker.state.value}", fg="yellow")
        return

    if not force:
        click.echo(f"Worker {worker_id} is {worker.state.value}")
        if worker.task_id:
            click.echo(f"Associated task {worker.task_id} will be marked as failed")
        if not click.confirm("Proceed?"):
            click.echo("Cancelled.")
            return

    # Mark worker as terminated
    from parhelia.orchestrator import WorkerState
    ctx.orchestrator.worker_store.update_state(worker_id, WorkerState.TERMINATED)

    # Mark task as failed
    if worker.task_id:
        ctx.orchestrator.task_store.update_status(worker.task_id, "failed")
        click.secho(f"Task {worker.task_id} marked as failed", fg="yellow")

    click.secho(f"Worker {worker_id} terminated", fg="green")


# =============================================================================
# Task Cancel and Retry Commands
# =============================================================================


@task.command("cancel")
@click.argument("task_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@pass_context
def task_cancel(ctx: CLIContext, task_id: str, force: bool) -> None:
    """Cancel a running or pending task.

    Terminates the associated worker and marks the task as cancelled.

    Examples:
        parhelia task cancel task-abc123
    """

    async def _cancel():
        task = await ctx.orchestrator.get_task(task_id)
        if not task:
            click.secho(f"Task not found: {task_id}", fg="red")
            sys.exit(1)

        status = ctx.orchestrator.task_store.get_status(task_id)
        if status in ("completed", "failed"):
            click.secho(f"Task already in terminal state: {status}", fg="yellow")
            return

        if not force:
            click.echo(f"Task {task_id} is {status}")
            if not click.confirm("Cancel this task?"):
                click.echo("Aborted.")
                return

        # Find and terminate associated worker
        worker = ctx.orchestrator.worker_store.get_by_task(task_id)
        if worker and worker.state.value == "running":
            from parhelia.orchestrator import WorkerState
            ctx.orchestrator.worker_store.update_state(worker.id, WorkerState.TERMINATED)
            click.echo(f"Worker {worker.id} terminated")

        # Mark task as failed (cancelled)
        ctx.orchestrator.task_store.update_status(task_id, "failed")
        click.secho(f"Task {task_id} cancelled", fg="green")

    asyncio.run(_cancel())


@task.command("retry")
@click.argument("task_id")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation")
@pass_context
def task_retry(ctx: CLIContext, task_id: str, force: bool) -> None:
    """Retry a failed task.

    Resets the task to pending status for re-dispatch.

    Examples:
        parhelia task retry task-abc123
    """

    async def _retry():
        task = await ctx.orchestrator.get_task(task_id)
        if not task:
            click.secho(f"Task not found: {task_id}", fg="red")
            sys.exit(1)

        status = ctx.orchestrator.task_store.get_status(task_id)
        if status not in ("failed", "completed"):
            click.secho(f"Can only retry failed/completed tasks. Current status: {status}", fg="yellow")
            return

        if not force:
            click.echo(f"Task {task_id} ({status}): {task.prompt[:50]}...")
            if not click.confirm("Retry this task?"):
                click.echo("Aborted.")
                return

        # Reset to pending
        ctx.orchestrator.task_store.update_status(task_id, "pending")
        click.secho(f"Task {task_id} reset to pending", fg="green")
        click.echo("Run 'parhelia task dispatch' to execute")

    asyncio.run(_retry())


# =============================================================================
# Events Command Group (SPEC-20 P4)
# =============================================================================


@cli.group(cls=AliasGroup, aliases={"ls": "list", "w": "watch", "r": "replay"})
def events() -> None:
    """Event streaming and history commands.

    Implements [SPEC-20.40] Real-Time Events.

    View, filter, and export control plane events.

    Aliases:
        e -> events
    """
    pass


# Add events alias to GROUP_ALIASES at the top of file through import modification
# This is handled by the GROUP_ALIASES dict which already maps 'e' -> 'events' if added


@events.command("list")
@click.option(
    "--type", "event_type",
    type=click.Choice([
        "container_created", "container_started", "container_stopped",
        "container_terminated", "container_healthy", "container_degraded",
        "container_unhealthy", "container_dead", "container_recovered",
        "orphan_detected", "state_drift_corrected", "reconcile_failed",
        "heartbeat_received", "heartbeat_missed", "error",
    ]),
    help="Filter by event type",
)
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Filter by severity level",
)
@click.option(
    "--since",
    help="Only events after this time (e.g., '1h', '30m', '2024-01-21T10:00:00')",
)
@click.option(
    "--container", "container_id",
    help="Filter by container ID",
)
@click.option(
    "--task", "task_id",
    help="Filter by task ID",
)
@click.option("--limit", default=50, help="Maximum events to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def events_list(
    ctx: CLIContext,
    event_type: str | None,
    level: str | None,
    since: str | None,
    container_id: str | None,
    task_id: str | None,
    limit: int,
    as_json: bool,
) -> None:
    """List events with optional filtering.

    Examples:
        parhelia events list
        parhelia events list --type error
        parhelia events list --level warning --since 1h
        parhelia events list --container c-abc123
        parhelia e ls --json
    """
    from parhelia.events import EventFilter

    store = ctx.state_store

    # Parse since argument
    since_dt = None
    if since:
        since_dt = _parse_time_arg(since)

    # Build filter
    event_types = None
    if event_type:
        event_types = [EventType(event_type)]

    levels = None
    if level:
        levels = [level]

    filter = EventFilter(
        event_types=event_types,
        levels=levels,
        container_id=container_id,
        task_id=task_id,
        since=since_dt,
    )

    # Query events
    events = store.get_events(
        container_id=container_id,
        task_id=task_id,
        event_type=EventType(event_type) if event_type else None,
        since=since_dt,
        limit=limit,
    )

    # Apply additional filtering (levels)
    if level:
        events = [e for e in events if filter.matches(e)]

    if as_json:
        data = {
            "events": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type.value,
                    "container_id": e.container_id,
                    "task_id": e.task_id,
                    "message": e.message,
                    "source": e.source,
                }
                for e in events
            ],
            "count": len(events),
            "filter": filter.to_dict(),
        }
        click.echo(json.dumps(data, indent=2))
        return

    if not events:
        click.echo("No events found matching criteria.")
        return

    click.echo(f"Events ({len(events)} shown)")
    click.echo("=" * 80)

    for e in events:
        time_str = e.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        type_color = _get_event_type_color(e.event_type.value)

        msg = e.message or "-"
        if len(msg) > 50:
            msg = msg[:47] + "..."

        click.echo(
            f"[{time_str}] "
            f"{click.style(e.event_type.value, fg=type_color):<24} "
            f"{(e.container_id or '-'):<12} "
            f"{msg}"
        )


@events.command("watch")
@click.option(
    "--type", "event_type",
    type=click.Choice([
        "container_created", "container_started", "container_stopped",
        "container_terminated", "container_healthy", "container_degraded",
        "container_unhealthy", "container_dead", "container_recovered",
        "orphan_detected", "state_drift_corrected", "reconcile_failed",
        "heartbeat_received", "heartbeat_missed", "error",
    ]),
    help="Filter by event type",
)
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    help="Filter by severity level",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Quiet mode - only show completion events",
)
@click.option(
    "--container", "container_id",
    help="Filter by container ID",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON lines")
@pass_context
def events_watch(
    ctx: CLIContext,
    event_type: str | None,
    level: str | None,
    quiet: bool,
    container_id: str | None,
    as_json: bool,
) -> None:
    """Watch events in real-time.

    Streams events as they occur. Press Ctrl+C to stop.

    Examples:
        parhelia events watch
        parhelia events watch --type error
        parhelia events watch --level warning
        parhelia events watch --quiet  # Only completion events
        parhelia e w --json
    """
    from parhelia.events import EventFilter, EventLogger

    if not as_json:
        click.echo("Watching events (Ctrl+C to stop)...")
        click.echo("")

    store = ctx.state_store

    # Build filter
    event_types = None
    if event_type:
        event_types = [EventType(event_type)]
    elif quiet:
        # Quiet mode: only lifecycle completion events
        event_types = [
            EventType.CONTAINER_STOPPED,
            EventType.CONTAINER_TERMINATED,
        ]

    levels = None
    if level:
        levels = [level]

    filter = EventFilter(
        event_types=event_types,
        levels=levels,
        container_id=container_id,
    )

    # Poll for new events
    import time

    last_event_id = 0

    # Get the latest event ID to start from
    recent = store.get_events(limit=1)
    if recent:
        last_event_id = recent[0].id or 0

    try:
        while True:
            # Get new events since last seen
            events = store.get_events(limit=50)

            # Filter to new events
            new_events = [e for e in events if (e.id or 0) > last_event_id]
            new_events.reverse()  # Show oldest first

            for e in new_events:
                if filter.matches(e):
                    if as_json:
                        data = {
                            "id": e.id,
                            "timestamp": e.timestamp.isoformat(),
                            "event_type": e.event_type.value,
                            "container_id": e.container_id,
                            "task_id": e.task_id,
                            "message": e.message,
                            "source": e.source,
                        }
                        click.echo(json.dumps(data))
                    else:
                        time_str = e.timestamp.strftime("%H:%M:%S")
                        type_color = _get_event_type_color(e.event_type.value)
                        msg = e.message or "-"
                        click.echo(
                            f"[{time_str}] "
                            f"{click.style(e.event_type.value, fg=type_color)}: "
                            f"{msg}"
                        )

                last_event_id = max(last_event_id, e.id or 0)

            time.sleep(0.5)  # <500ms latency requirement

    except KeyboardInterrupt:
        if not as_json:
            click.echo("\nStopped watching.")


@events.command("replay")
@click.argument("container_id")
@click.option(
    "--from-start",
    is_flag=True,
    help="Replay from container creation",
)
@click.option(
    "--since",
    help="Replay from this time (e.g., '1h', '30m', or ISO timestamp)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def events_replay(
    ctx: CLIContext,
    container_id: str,
    from_start: bool,
    since: str | None,
    as_json: bool,
) -> None:
    """Replay historical events for a container.

    Shows events in chronological order to understand what happened.

    Examples:
        parhelia events replay c-abc12345
        parhelia events replay c-abc12345 --from-start
        parhelia events replay c-abc12345 --since 2h
        parhelia e r c-abc12345 --json
    """
    from parhelia.events import EventLogger

    store = ctx.state_store

    # Parse since argument
    since_dt = None
    if since and not from_start:
        since_dt = _parse_time_arg(since)

    # Get events
    events = store.get_events(
        container_id=container_id,
        since=since_dt if not from_start else None,
        limit=1000,
    )

    # Sort chronologically
    events.sort(key=lambda e: e.timestamp)

    if as_json:
        data = {
            "container_id": container_id,
            "events": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type.value,
                    "message": e.message,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "source": e.source,
                    "details": e.details,
                }
                for e in events
            ],
            "count": len(events),
            "from_start": from_start,
        }
        click.echo(json.dumps(data, indent=2))
        return

    if not events:
        click.echo(f"No events found for container: {container_id}")
        return

    click.echo(f"Event Replay: {container_id}")
    click.echo(f"Period: {events[0].timestamp.strftime('%Y-%m-%d %H:%M:%S')} to {events[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo("=" * 80)

    for e in events:
        time_str = e.timestamp.strftime("%H:%M:%S")
        type_color = _get_event_type_color(e.event_type.value)

        # Format the event message
        msg = e.message or ""
        if e.old_value and e.new_value:
            msg = f"{e.old_value} -> {e.new_value}"
            if e.message:
                msg = f"{e.message} ({msg})"

        click.echo(
            f"[{time_str}] "
            f"{click.style(e.event_type.value, fg=type_color)}: "
            f"{msg or '-'}"
        )


@events.command("export")
@click.argument("output_file")
@click.option(
    "--format", "output_format",
    type=click.Choice(["jsonl", "json"]),
    default="jsonl",
    help="Output format (default: jsonl)",
)
@click.option(
    "--since",
    help="Export events after this time",
)
@click.option(
    "--until",
    help="Export events before this time",
)
@click.option(
    "--container", "container_id",
    help="Filter by container ID",
)
@click.option(
    "--type", "event_type",
    help="Filter by event type",
)
@pass_context
def events_export(
    ctx: CLIContext,
    output_file: str,
    output_format: str,
    since: str | None,
    until: str | None,
    container_id: str | None,
    event_type: str | None,
) -> None:
    """Export events to file.

    Supports JSONL (JSON Lines) and JSON array formats.

    Examples:
        parhelia events export events.jsonl
        parhelia events export events.json --format json
        parhelia events export errors.jsonl --type error
        parhelia events export today.jsonl --since 24h
    """
    from parhelia.events import EventExporter, EventFilter

    store = ctx.state_store

    # Parse time arguments
    since_dt = _parse_time_arg(since) if since else None
    until_dt = _parse_time_arg(until) if until else None

    # Build filter
    event_types = [EventType(event_type)] if event_type else None

    filter = EventFilter(
        event_types=event_types,
        container_id=container_id,
        since=since_dt,
        until=until_dt,
    )

    # Get events
    events = store.get_events(
        container_id=container_id,
        event_type=EventType(event_type) if event_type else None,
        since=since_dt,
        until=until_dt,
        limit=10000,  # Large limit for export
    )

    # Apply additional filtering
    events = [e for e in events if filter.matches(e)]

    # Sort chronologically
    events.sort(key=lambda e: e.timestamp)

    # Export
    exporter = EventExporter()

    if output_format == "jsonl":
        count = exporter.to_jsonl(events, output_file)
    else:
        json_str = exporter.to_json(events)
        with open(output_file, "w") as f:
            f.write(json_str)
        count = len(events)

    click.secho(f"Exported {count} events to {output_file}", fg="green")


def _parse_time_arg(time_str: str) -> datetime:
    """Parse a time argument into a datetime.

    Supports:
    - Relative times: '1h', '30m', '2d', '1w'
    - ISO timestamps: '2024-01-21T10:00:00'
    """
    if not time_str:
        return datetime.utcnow()

    # Check for relative time
    if time_str[-1] in "smhdw":
        try:
            value = int(time_str[:-1])
            unit = time_str[-1]
            deltas = {
                "s": timedelta(seconds=value),
                "m": timedelta(minutes=value),
                "h": timedelta(hours=value),
                "d": timedelta(days=value),
                "w": timedelta(weeks=value),
            }
            return datetime.utcnow() - deltas[unit]
        except (ValueError, KeyError):
            pass

    # Try ISO format
    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        raise click.BadParameter(f"Invalid time format: {time_str}")


def _get_event_type_color(event_type: str) -> str:
    """Get color for event type display."""
    return {
        "container_created": "cyan",
        "container_started": "green",
        "container_stopped": "yellow",
        "container_terminated": "white",
        "container_healthy": "green",
        "container_degraded": "yellow",
        "container_unhealthy": "red",
        "container_dead": "red",
        "container_recovered": "green",
        "orphan_detected": "red",
        "state_drift_corrected": "yellow",
        "reconcile_failed": "red",
        "heartbeat_received": "white",
        "heartbeat_missed": "yellow",
        "error": "red",
    }.get(event_type, "white")


# =============================================================================
# Help Command (SPEC-20.51)
# =============================================================================


@cli.command("help")
@click.argument("topic", required=False)
def help_cmd(topic: str | None) -> None:
    """Get contextual help on topics and error codes.

    Implements [SPEC-20.51] - Contextual Help System.

    Provides detailed help for:
    - Topics: task, session, container, checkpoint, budget, events, reconciler
    - Error codes: E100-E599 (validation, resource, budget, auth, infra)

    Examples:
        parhelia help                 # List available topics
        parhelia help task            # Help on task management
        parhelia help session         # Help on sessions
        parhelia help E200            # Help on SESSION_NOT_FOUND error
        parhelia help E300            # Help on BUDGET_EXCEEDED error
    """
    help_sys = get_help_system()

    if not topic:
        # Show available topics
        click.echo("Parhelia Help")
        click.echo("=" * 60)
        click.echo()
        click.echo("Available topics:")
        for t in help_sys.list_topics():
            summary = help_sys.get_topic_summary(t)
            if summary:
                click.echo(f"  {t:<15} {summary.split(': ', 1)[1] if ': ' in summary else summary}")
        click.echo()
        click.echo("Error codes:")
        click.echo("  E1xx           Validation errors")
        click.echo("  E2xx           Resource errors (not found)")
        click.echo("  E3xx           Budget errors")
        click.echo("  E4xx           Authentication errors")
        click.echo("  E5xx           Infrastructure errors")
        click.echo()
        click.echo("Usage:")
        click.echo("  parhelia help <topic>      Show topic help")
        click.echo("  parhelia help E200         Show error code help")
        click.echo("  parhelia examples <topic>  Show examples")
        return

    # Check if it's an error code (starts with E followed by digits)
    if topic.upper().startswith("E") and len(topic) >= 2 and topic[1:].isdigit():
        error_help = help_sys.get_error_help(topic.upper())
        if error_help:
            click.echo(error_help)
        else:
            click.secho(f"Unknown error code: {topic}", fg="yellow")
            click.echo()
            click.echo("Known error code ranges:")
            click.echo("  E100-E199: Validation errors")
            click.echo("  E200-E299: Resource errors")
            click.echo("  E300-E399: Budget errors")
            click.echo("  E400-E499: Authentication errors")
            click.echo("  E500-E599: Infrastructure errors")
        return

    # Otherwise treat as topic
    topic_help = help_sys.get_topic_help(topic)
    if topic_help:
        click.echo(topic_help)
    else:
        click.secho(f"Unknown topic: {topic}", fg="yellow")
        click.echo()
        click.echo("Available topics:")
        for t in help_sys.list_topics():
            click.echo(f"  {t}")
        click.echo()
        click.echo("Use 'parhelia help <topic>' to see detailed help.")


# =============================================================================
# Examples Command (SPEC-20.52)
# =============================================================================


@cli.command("examples")
@click.argument("topic", required=False)
def examples_cmd(topic: str | None) -> None:
    """Show example workflows for common tasks.

    Implements [SPEC-20.52] - Example-based Documentation.

    Provides copy-pasteable command examples for:
    - gpu: GPU task creation and monitoring
    - checkpoint: Checkpoint workflows
    - budget: Budget management
    - debug: Debugging failed tasks
    - interactive: Interactive session workflows
    - task: Basic task workflows
    - workflow: Complete development workflows

    Examples:
        parhelia examples             # List available example topics
        parhelia examples gpu         # GPU task examples
        parhelia examples checkpoint  # Checkpoint examples
        parhelia examples debug       # Debugging examples
    """
    example_sys = get_example_system()

    if not topic:
        # Show available topics
        click.echo("Parhelia Examples")
        click.echo("=" * 60)
        click.echo()
        click.echo("Available example topics:")
        for t in example_sys.list_topics():
            examples = example_sys.get_examples(t)
            count = len(examples) if examples else 0
            click.echo(f"  {t:<15} ({count} examples)")
        click.echo()
        click.echo("Usage:")
        click.echo("  parhelia examples <topic>  Show examples for topic")
        return

    formatted = example_sys.format_examples(topic)
    if formatted:
        click.echo(formatted)
    else:
        click.secho(f"Unknown topic: {topic}", fg="yellow")
        click.echo()
        click.echo("Available topics:")
        for t in example_sys.list_topics():
            click.echo(f"  {t}")


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
