"""Interactive intelligence for Parhelia CLI.

Implements:
- [SPEC-20.50] Smart Prompts with intelligent defaults
- [SPEC-20.51] Contextual Help System
- [SPEC-20.52] Example-based Documentation

Provides smart defaults, contextual help, and example workflows to improve
CLI usability and reduce time to first successful task.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parhelia.feedback import ErrorRecovery


# =============================================================================
# Constants
# =============================================================================

# XDG-compliant cache directory
DEFAULT_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "parhelia"


# =============================================================================
# Smart Prompts (SPEC-20.50)
# =============================================================================


class SmartPrompt:
    """Prompts with intelligent defaults based on history.

    Implements [SPEC-20.50].

    Caches user preferences for common values like GPU type, timeout, etc.
    Uses XDG-compliant ~/.cache/parhelia/ directory.

    Usage:
        prompt = SmartPrompt()

        # First time: no default
        gpu = prompt.prompt_with_default("gpu_type", "GPU type", default="none")
        prompt.remember("gpu_type", gpu)

        # Next time: uses cached value as default
        gpu = prompt.prompt_with_default("gpu_type", "GPU type")  # default: previous value
    """

    CACHE_FILE = "preferences.json"

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the smart prompt system.

        Args:
            cache_dir: Directory for preference cache. Defaults to ~/.cache/parhelia/
        """
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_file = self._cache_dir / self.CACHE_FILE
        self._cache: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached preferences from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Invalid cache file, start fresh
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cached preferences to disk."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError:
            # Silently fail on cache write errors
            pass

    def get_default(self, name: str) -> str | None:
        """Get the cached default value for a preference.

        Args:
            name: Preference name (e.g., "gpu_type", "timeout").

        Returns:
            Cached value or None if not cached.
        """
        return self._cache.get(name)

    def prompt_with_default(
        self,
        name: str,
        message: str,
        default: str | None = None,
    ) -> str:
        """Prompt user with intelligent default from cache.

        Args:
            name: Preference name for caching.
            message: Prompt message to display.
            default: Explicit default (overrides cached value).

        Returns:
            User-provided or default value.
        """
        import click

        # Use cached value if no explicit default provided
        effective_default = default or self._cache.get(name)

        if effective_default:
            value = click.prompt(message, default=effective_default)
        else:
            value = click.prompt(message)

        return value

    def remember(self, name: str, value: str) -> None:
        """Cache a value for future use as default.

        Args:
            name: Preference name.
            value: Value to cache.
        """
        self._cache[name] = value
        self._save_cache()

    def forget(self, name: str) -> None:
        """Remove a cached preference.

        Args:
            name: Preference name to remove.
        """
        if name in self._cache:
            del self._cache[name]
            self._save_cache()

    def clear_all(self) -> None:
        """Clear all cached preferences."""
        self._cache = {}
        self._save_cache()

    def get_all(self) -> dict[str, str]:
        """Get all cached preferences.

        Returns:
            Dict of all cached name-value pairs.
        """
        return dict(self._cache)


# =============================================================================
# Help System (SPEC-20.51)
# =============================================================================


@dataclass
class HelpTopic:
    """A help topic with detailed content."""

    name: str
    title: str
    summary: str
    content: str
    related: list[str] = field(default_factory=list)


class HelpSystem:
    """Contextual help with topic and error code support.

    Implements [SPEC-20.51].

    Provides detailed help for:
    - Topics: task, session, container, checkpoint, budget, events, reconciler
    - Error codes: E100-E599 (validation, resource, budget, auth, infra)

    Usage:
        help_sys = HelpSystem()

        # Get topic help
        help_text = help_sys.get_topic_help("task")

        # Get error-specific help
        error_help = help_sys.get_error_help("E200")

        # List all topics
        topics = help_sys.list_topics()
    """

    # Topic help content
    _TOPICS: dict[str, HelpTopic] = {
        "task": HelpTopic(
            name="task",
            title="Task Management",
            summary="Creating, monitoring, and managing compute tasks",
            content="""
Task Management in Parhelia
===========================

Tasks are the primary unit of work in Parhelia. Each task runs in an isolated
Modal container with Claude Code.

CREATING TASKS
--------------
  parhelia task create <prompt>     Create a new task
  parhelia submit <prompt>          Legacy alias for task create

Options:
  --gpu <type>      GPU type (none, A10G, A100, H100, T4)
  --memory <GB>     Minimum memory in GB (default: 4)
  --timeout <min>   Task timeout in minutes (default: 60)
  --workspace <dir> Working directory for the task
  --sync            Wait for completion
  --dry-run         Test without Modal execution

MONITORING TASKS
----------------
  parhelia task list               List all tasks
  parhelia task show <id>          Show task details
  parhelia task watch <id>         Watch task in real-time

MANAGING TASKS
--------------
  parhelia task cancel <id>        Cancel a running task
  parhelia task retry <id>         Retry a failed task

TASK STATES
-----------
  pending    -> Task queued, not yet started
  running    -> Task executing in Modal
  completed  -> Task finished successfully
  failed     -> Task failed (check logs)
  cancelled  -> Task was cancelled

TIPS
----
- Use --gpu A10G for most AI workloads (best cost/performance)
- Use --sync for short tasks to see output directly
- Use --dry-run to test configuration without costs
""",
            related=["session", "checkpoint", "budget"],
        ),
        "session": HelpTopic(
            name="session",
            title="Session Management",
            summary="Interactive sessions and attachment",
            content="""
Session Management in Parhelia
==============================

Sessions provide interactive access to running containers.

LISTING SESSIONS
----------------
  parhelia session list            List active sessions
  parhelia session list --all      Include completed sessions

SESSION ATTACHMENT
------------------
  parhelia session attach <id>     Attach to a session (SSH)
  parhelia session attach --web    Open in browser (coming soon)

Attachment connects your terminal to the running container via SSH,
giving you full shell access to the Claude Code environment.

MANAGING SESSIONS
-----------------
  parhelia session kill <id>       Terminate a session
  parhelia session detach <id>     Detach from a session

SESSION STATES
--------------
  starting   -> Container initializing
  active     -> Session ready for attachment
  attached   -> Currently attached
  terminated -> Session ended

TIPS
----
- Sessions persist across SSH disconnections
- Use Ctrl+D or 'exit' to cleanly detach
- Session state is preserved in checkpoints
- Kill sessions when done to avoid charges
""",
            related=["task", "checkpoint", "container"],
        ),
        "container": HelpTopic(
            name="container",
            title="Container Management",
            summary="View and manage Modal containers",
            content="""
Container Management in Parhelia
================================

Containers are the actual Modal sandboxes running your tasks.

VIEWING CONTAINERS
------------------
  parhelia container list          List all containers
  parhelia container show <id>     Show container details
  parhelia container events <id>   Show container events

CONTAINER HEALTH
----------------
  parhelia container health <id>   Check container health status
  parhelia container watch <id>    Watch container in real-time

Health states:
  healthy    -> Container running normally
  degraded   -> Container experiencing issues
  unhealthy  -> Container failing health checks
  unknown    -> Health status not yet determined

MANAGING CONTAINERS
-------------------
  parhelia container terminate <id>  Force terminate a container

RECONCILIATION
--------------
  parhelia reconciler status       Show reconciler state
  parhelia reconciler run          Force immediate reconciliation

The reconciler syncs container state with Modal every 60 seconds,
detecting orphaned containers and state drift.

TIPS
----
- Container IDs are Modal sandbox IDs
- Use 'container events' to debug issues
- The reconciler auto-cleans orphaned containers
""",
            related=["task", "session", "reconciler"],
        ),
        "checkpoint": HelpTopic(
            name="checkpoint",
            title="Checkpoint Management",
            summary="Creating and restoring session state",
            content="""
Checkpoint Management in Parhelia
=================================

Checkpoints save complete session state for later restoration.

CREATING CHECKPOINTS
--------------------
  parhelia checkpoint create <session-id>   Create from session
  parhelia checkpoint create --auto         Auto-name checkpoint

Options:
  --name <name>     Custom checkpoint name
  --description     Add description

LISTING CHECKPOINTS
-------------------
  parhelia checkpoint list         List all checkpoints
  parhelia checkpoint list --tag   Filter by tag

RESTORING CHECKPOINTS
---------------------
  parhelia checkpoint restore <id>  Restore to new session

Options:
  --session-id      Restore to specific session
  --dry-run         Show what would be restored

COMPARING CHECKPOINTS
---------------------
  parhelia checkpoint diff <a> <b>  Compare two checkpoints

Shows differences in:
  - File changes (added, modified, deleted)
  - Environment variables
  - Working directory state

CHECKPOINT CONTENTS
-------------------
Checkpoints include:
  - All workspace files
  - Environment variables
  - Claude Code state
  - Current working directory
  - Shell history

TIPS
----
- Create checkpoints before risky operations
- Use diff to compare before/after states
- Checkpoints are stored on Modal volumes
- Auto-checkpoints occur on graceful shutdown
""",
            related=["session", "task"],
        ),
        "budget": HelpTopic(
            name="budget",
            title="Budget Management",
            summary="Cost tracking and spending limits",
            content="""
Budget Management in Parhelia
=============================

Budget controls prevent unexpected compute costs.

CHECKING BUDGET
---------------
  parhelia budget status           Show current usage
  parhelia budget history          Show spending history

Status shows:
  - Current ceiling
  - Amount used
  - Remaining budget
  - Warning threshold status

SETTING BUDGET
--------------
  parhelia budget set <amount>     Set ceiling in USD
  parhelia budget set 50           Set $50 ceiling

Options:
  --warning <percent>  Warning threshold (default: 80%)

BUDGET WARNINGS
---------------
When budget reaches warning threshold:
  - CLI shows warning on all commands
  - New tasks include cost estimates
  - MCP responses include budget status

COST ESTIMATION
---------------
Before task submission:
  - Estimated cost shown based on GPU/duration
  - Warning if near budget

GPU costs (approximate per hour):
  - CPU only: ~$0.10
  - T4:       ~$0.50
  - A10G:     ~$1.00
  - A100:     ~$3.00
  - H100:     ~$5.00

TIPS
----
- Set budget before first task
- Use CPU for non-GPU workloads
- Monitor with 'budget status' regularly
- Set warning at 80% to avoid surprises
""",
            related=["task"],
        ),
        "events": HelpTopic(
            name="events",
            title="Event System",
            summary="Real-time events and notifications",
            content="""
Event System in Parhelia
========================

Events provide real-time visibility into task and container state.

WATCHING EVENTS
---------------
  parhelia events watch            Watch all events
  parhelia events watch <task-id>  Watch specific task

Options:
  --events <types>   Filter by type (status,progress,error)
  --level <level>    Filter by severity (info,warning,error)
  --quiet            Only show completion events
  --from-start       Replay from beginning

EVENT HISTORY
-------------
  parhelia events list             List recent events
  parhelia events replay <id>      Replay task events

EVENT TYPES
-----------
  task_created       Task submitted
  task_started       Execution began
  task_progress      Progress update
  task_completed     Task finished
  task_failed        Task errored
  container_started  Container launched
  container_healthy  Health check passed
  container_stopped  Container terminated
  checkpoint_created Checkpoint saved

STREAMING
---------
Events are pushed via MCP notifications (<500ms latency).
Fallback polling available when streaming unavailable.

TIPS
----
- Use --events to filter noise
- Use --from-start for debugging
- Events are persisted for 7 days
- Export events: events list --export
""",
            related=["task", "container"],
        ),
        "reconciler": HelpTopic(
            name="reconciler",
            title="Container Reconciler",
            summary="State synchronization with Modal",
            content="""
Container Reconciler in Parhelia
================================

The reconciler ensures local state matches Modal reality.

STATUS
------
  parhelia reconciler status       Show reconciler state

Shows:
  - Last sync time
  - Sync interval
  - Orphan count
  - Drift detections

MANUAL RECONCILIATION
---------------------
  parhelia reconciler run          Force immediate sync

WHAT IT DOES
------------
Every 60 seconds, the reconciler:
1. Queries Modal for all sandboxes
2. Compares with local container records
3. Detects orphans (Modal has, we don't know)
4. Detects missing (we track, Modal doesn't have)
5. Updates health status
6. Logs state drift

ORPHAN HANDLING
---------------
Orphaned containers are:
1. Logged as warnings
2. Marked for investigation
3. Optionally auto-terminated (configurable)

DRIFT DETECTION
---------------
State drift occurs when:
- Container running but local says stopped
- Local says running but Modal disagrees
- Health status mismatch

CONFIGURATION
-------------
In parhelia.toml:
  [reconciler]
  interval_seconds = 60
  auto_terminate_orphans = false
  drift_threshold = 2

TIPS
----
- Run manually after Modal dashboard changes
- Check status after connectivity issues
- Orphans may indicate failed dispatches
""",
            related=["container", "task"],
        ),
    }

    # Error code help content
    _ERROR_HELP: dict[str, tuple[str, str, list[str]]] = {
        # Validation errors (1xx)
        "E100": (
            "Invalid Command Syntax",
            "The command syntax is incorrect or missing required arguments.",
            [
                "parhelia <command> --help",
                "parhelia examples <command>",
            ],
        ),
        "E101": (
            "Invalid Argument Value",
            "An argument has an invalid type or value. Check the expected format.",
            [
                "parhelia <command> --help",
            ],
        ),
        "E102": (
            "Missing Required Argument",
            "A required argument was not provided.",
            [
                "parhelia <command> --help",
            ],
        ),
        # Resource errors (2xx)
        "E200": (
            "Session Not Found",
            "The specified session ID does not exist or has expired. Sessions are "
            "automatically cleaned up after completion or timeout.",
            [
                "parhelia session list --all",
                "parhelia task list",
            ],
        ),
        "E201": (
            "Task Not Found",
            "The specified task ID does not exist. Tasks are persisted but may have "
            "been cleaned up after long periods of inactivity.",
            [
                "parhelia task list",
                "parhelia task list --recent",
            ],
        ),
        "E202": (
            "Checkpoint Not Found",
            "The specified checkpoint does not exist or has been deleted.",
            [
                "parhelia checkpoint list",
            ],
        ),
        "E203": (
            "Worker Not Found",
            "The worker for this task has terminated or cannot be found. The task may "
            "have completed, failed, or been cancelled.",
            [
                "parhelia task show <task-id>",
                "parhelia task retry <task-id>",
            ],
        ),
        # Budget errors (3xx)
        "E300": (
            "Budget Exceeded",
            "The budget ceiling has been reached. No new tasks can be dispatched until "
            "the budget is increased or reset.",
            [
                "parhelia budget status",
                "parhelia budget set <amount>",
            ],
        ),
        "E301": (
            "Budget Warning Threshold",
            "Budget usage has exceeded the warning threshold. Consider increasing the "
            "budget or reducing usage.",
            [
                "parhelia budget status",
                "parhelia budget history",
            ],
        ),
        # Auth errors (4xx)
        "E400": (
            "Authentication Failed",
            "Modal credentials are missing or invalid. Ensure you have authenticated "
            "with Modal and have valid credentials.",
            [
                "modal token show",
                "modal token set",
            ],
        ),
        "E401": (
            "Permission Denied",
            "You don't have permission for this operation. Check your Modal workspace "
            "and team permissions.",
            [
                "modal config show",
            ],
        ),
        "E402": (
            "Token Expired",
            "Your Modal authentication token has expired. Re-authenticate to continue.",
            [
                "modal token set",
            ],
        ),
        # Infrastructure errors (5xx)
        "E500": (
            "Resource Unavailable",
            "The requested compute resource (GPU type, region) is currently unavailable. "
            "Modal may be experiencing capacity constraints.",
            [
                "Try CPU instead of GPU",
                "Try a different GPU type",
                "Wait and retry in a few minutes",
            ],
        ),
        "E501": (
            "Operation Timeout",
            "The operation timed out. This may be due to network issues or Modal delays.",
            [
                "parhelia task show <task-id>",
                "Retry the operation",
            ],
        ),
        "E502": (
            "Network Error",
            "Unable to connect to Modal API. Check your internet connection and Modal's "
            "service status.",
            [
                "Check https://status.modal.com",
                "Verify network connectivity",
            ],
        ),
        "E503": (
            "Container Startup Failed",
            "The container failed to start. This may be due to resource constraints, "
            "image issues, or configuration problems.",
            [
                "parhelia task show <task-id>",
                "Check budget: parhelia budget status",
                "Try with --dry-run to test",
            ],
        ),
    }

    def get_topic_help(self, topic: str) -> str | None:
        """Get help for a topic.

        Args:
            topic: Topic name (task, session, container, etc.).

        Returns:
            Formatted help text or None if topic not found.
        """
        topic_lower = topic.lower()
        help_topic = self._TOPICS.get(topic_lower)

        if not help_topic:
            return None

        lines = [
            help_topic.content.strip(),
            "",
            "Related topics: " + ", ".join(help_topic.related),
            "  Run: parhelia help <topic>",
        ]

        return "\n".join(lines)

    def get_error_help(self, error_code: str) -> str | None:
        """Get detailed help for an error code.

        Args:
            error_code: Error code (E200, E300, etc.).

        Returns:
            Formatted help text or None if error code not found.
        """
        error_code_upper = error_code.upper()
        error_info = self._ERROR_HELP.get(error_code_upper)

        if not error_info:
            return None

        title, description, commands = error_info

        # Get recovery suggestions from ErrorRecovery
        suggestions = ErrorRecovery.suggest(error_code_upper)

        lines = [
            f"Error {error_code_upper}: {title}",
            "=" * (len(title) + len(error_code_upper) + 8),
            "",
            description,
            "",
            "Suggested Actions:",
        ]

        for suggestion in suggestions:
            lines.append(f"  - {suggestion}")

        if commands:
            lines.append("")
            lines.append("Useful Commands:")
            for cmd in commands:
                lines.append(f"  {cmd}")

        return "\n".join(lines)

    def list_topics(self) -> list[str]:
        """List all available help topics.

        Returns:
            List of topic names.
        """
        return list(self._TOPICS.keys())

    def list_error_codes(self) -> list[str]:
        """List all documented error codes.

        Returns:
            List of error codes.
        """
        return list(self._ERROR_HELP.keys())

    def get_topic_summary(self, topic: str) -> str | None:
        """Get a brief summary for a topic.

        Args:
            topic: Topic name.

        Returns:
            One-line summary or None if topic not found.
        """
        help_topic = self._TOPICS.get(topic.lower())
        if help_topic:
            return f"{help_topic.title}: {help_topic.summary}"
        return None


# =============================================================================
# Example System (SPEC-20.52)
# =============================================================================


@dataclass
class Example:
    """An example command or workflow."""

    title: str
    description: str
    commands: list[str]
    notes: str | None = None


class ExampleSystem:
    """Example-based documentation.

    Implements [SPEC-20.52].

    Provides copy-pasteable examples for common workflows.

    Usage:
        examples = ExampleSystem()

        # Get examples for a topic
        gpu_examples = examples.get_examples("gpu")

        # List all topics
        topics = examples.list_topics()
    """

    # Example content by topic
    _EXAMPLES: dict[str, list[Example]] = {
        "gpu": [
            Example(
                title="Create a GPU Task",
                description="Run a compute-intensive task with GPU acceleration.",
                commands=[
                    '# Create task with A10G GPU',
                    'parhelia task create "Train the model on the dataset" --gpu A10G',
                    '',
                    '# Create with A100 for large models',
                    'parhelia task create "Fine-tune LLM" --gpu A100 --memory 16',
                    '',
                    '# Create with H100 for maximum performance',
                    'parhelia task create "Run inference benchmark" --gpu H100',
                ],
            ),
            Example(
                title="Monitor GPU Task",
                description="Watch a GPU task and check resource usage.",
                commands=[
                    '# Watch task progress',
                    'parhelia task watch task-abc123',
                    '',
                    '# Check container health (includes GPU status)',
                    'parhelia container health sandbox-xyz789',
                ],
            ),
            Example(
                title="Cost-Efficient GPU Usage",
                description="Optimize GPU costs with proper sizing.",
                commands=[
                    '# Use T4 for light GPU workloads (cheapest)',
                    'parhelia task create "Run unit tests with GPU" --gpu T4',
                    '',
                    '# Check estimated cost before running',
                    'parhelia budget status',
                    '',
                    '# Set budget ceiling before GPU work',
                    'parhelia budget set 20',
                ],
                notes="T4 is ~$0.50/hr, A10G ~$1/hr, A100 ~$3/hr, H100 ~$5/hr",
            ),
        ],
        "checkpoint": [
            Example(
                title="Create a Checkpoint",
                description="Save session state before making changes.",
                commands=[
                    '# Create checkpoint from active session',
                    'parhelia checkpoint create session-abc123',
                    '',
                    '# Create with custom name',
                    'parhelia checkpoint create session-abc123 --name "before-refactor"',
                ],
            ),
            Example(
                title="List and View Checkpoints",
                description="Browse available checkpoints.",
                commands=[
                    '# List all checkpoints',
                    'parhelia checkpoint list',
                    '',
                    '# List with details',
                    'parhelia checkpoint list --verbose',
                ],
            ),
            Example(
                title="Restore from Checkpoint",
                description="Restore a previous session state.",
                commands=[
                    '# Restore checkpoint to new session',
                    'parhelia checkpoint restore chkpt-abc123',
                    '',
                    '# Dry run to see what would be restored',
                    'parhelia checkpoint restore chkpt-abc123 --dry-run',
                ],
            ),
            Example(
                title="Compare Checkpoints",
                description="See what changed between checkpoints.",
                commands=[
                    '# Compare two checkpoints',
                    'parhelia checkpoint diff chkpt-before chkpt-after',
                    '',
                    '# Compare with JSON output for scripting',
                    'parhelia checkpoint diff chkpt-before chkpt-after --json',
                ],
            ),
        ],
        "budget": [
            Example(
                title="Set Up Budget",
                description="Configure spending limits before starting work.",
                commands=[
                    '# Set $50 budget ceiling',
                    'parhelia budget set 50',
                    '',
                    '# Set with custom warning threshold',
                    'parhelia budget set 100 --warning 70',
                ],
            ),
            Example(
                title="Monitor Budget",
                description="Track spending and remaining budget.",
                commands=[
                    '# Check current status',
                    'parhelia budget status',
                    '',
                    '# View spending history',
                    'parhelia budget history',
                    '',
                    '# Export history to JSON',
                    'parhelia budget history --json > budget.json',
                ],
            ),
            Example(
                title="Budget-Aware Task Creation",
                description="Check costs before running expensive tasks.",
                commands=[
                    '# Check budget before GPU task',
                    'parhelia budget status',
                    '',
                    '# Run with dry-run to see estimated cost',
                    'parhelia task create "Heavy computation" --gpu A100 --dry-run',
                    '',
                    '# If OK, run for real',
                    'parhelia task create "Heavy computation" --gpu A100',
                ],
            ),
        ],
        "debug": [
            Example(
                title="Debug a Failed Task",
                description="Investigate why a task failed.",
                commands=[
                    '# Check task status and error',
                    'parhelia task show task-abc123',
                    '',
                    '# View container events',
                    'parhelia container events sandbox-xyz789',
                    '',
                    '# Replay all events for task',
                    'parhelia events replay task-abc123',
                ],
            ),
            Example(
                title="Debug Container Issues",
                description="Troubleshoot container problems.",
                commands=[
                    '# Check container health',
                    'parhelia container health sandbox-xyz789',
                    '',
                    '# View container details',
                    'parhelia container show sandbox-xyz789',
                    '',
                    '# Check reconciler for drift',
                    'parhelia reconciler status',
                ],
            ),
            Example(
                title="Debug Session Attachment",
                description="Troubleshoot SSH attachment issues.",
                commands=[
                    '# Check session state',
                    'parhelia session list',
                    '',
                    '# View session details',
                    'parhelia task show task-abc123',
                    '',
                    '# Force reconciliation',
                    'parhelia reconciler run',
                ],
            ),
        ],
        "interactive": [
            Example(
                title="Start an Interactive Session",
                description="Create a task and attach for interactive work.",
                commands=[
                    '# Create task and wait for session',
                    'parhelia task create "Set up development environment" --sync',
                    '',
                    '# List sessions to find the new one',
                    'parhelia session list',
                    '',
                    '# Attach to session',
                    'parhelia session attach session-abc123',
                ],
            ),
            Example(
                title="Checkpoint During Interactive Work",
                description="Save progress during an interactive session.",
                commands=[
                    '# In a separate terminal while attached:',
                    '',
                    '# Create checkpoint',
                    'parhelia checkpoint create session-abc123 --name "work-in-progress"',
                    '',
                    '# Continue working in the attached session',
                    '',
                    '# Create another checkpoint after more work',
                    'parhelia checkpoint create session-abc123 --name "feature-complete"',
                ],
            ),
            Example(
                title="Resume Work Later",
                description="Restore an interactive session from checkpoint.",
                commands=[
                    '# List available checkpoints',
                    'parhelia checkpoint list',
                    '',
                    '# Restore the checkpoint',
                    'parhelia checkpoint restore chkpt-work-in-progress',
                    '',
                    '# Attach to the restored session',
                    'parhelia session list',
                    'parhelia session attach session-new123',
                ],
            ),
        ],
        "task": [
            Example(
                title="Create and Monitor a Task",
                description="Basic task workflow from creation to completion.",
                commands=[
                    '# Create a task',
                    'parhelia task create "Fix the authentication bug in login.py"',
                    '',
                    '# Watch progress',
                    'parhelia task watch task-abc123',
                    '',
                    '# Check final result',
                    'parhelia task show task-abc123',
                ],
            ),
            Example(
                title="Synchronous Task Execution",
                description="Wait for task completion and see output.",
                commands=[
                    '# Create task and wait for completion',
                    'parhelia task create "Run test suite" --sync',
                    '',
                    '# Output is displayed when complete',
                ],
            ),
            Example(
                title="Retry a Failed Task",
                description="Retry a task that failed.",
                commands=[
                    '# Check why it failed',
                    'parhelia task show task-abc123',
                    '',
                    '# Retry with same parameters',
                    'parhelia task retry task-abc123',
                    '',
                    '# Or create a new task with adjustments',
                    'parhelia task create "Fix bug" --memory 8  # More memory',
                ],
            ),
        ],
        "workflow": [
            Example(
                title="Complete Development Workflow",
                description="End-to-end workflow for a feature implementation.",
                commands=[
                    '# 1. Set budget',
                    'parhelia budget set 25',
                    '',
                    '# 2. Create task',
                    'parhelia task create "Implement user authentication" --gpu none --sync',
                    '',
                    '# 3. Attach to session',
                    'parhelia session list',
                    'parhelia session attach session-abc123',
                    '',
                    '# 4. Work interactively, then checkpoint',
                    'parhelia checkpoint create session-abc123 --name "auth-v1"',
                    '',
                    '# 5. Check budget usage',
                    'parhelia budget status',
                ],
            ),
            Example(
                title="Parallel Task Execution",
                description="Run multiple tasks in parallel.",
                commands=[
                    '# Submit multiple tasks without waiting',
                    'parhelia task create "Fix bug A"',
                    'parhelia task create "Fix bug B"',
                    'parhelia task create "Add feature C"',
                    '',
                    '# Monitor all running tasks',
                    'parhelia task list --status running',
                    '',
                    '# Watch a specific task',
                    'parhelia task watch task-abc123',
                ],
            ),
        ],
    }

    def get_examples(self, topic: str) -> list[Example] | None:
        """Get examples for a topic.

        Args:
            topic: Topic name (gpu, checkpoint, budget, etc.).

        Returns:
            List of examples or None if topic not found.
        """
        return self._EXAMPLES.get(topic.lower())

    def list_topics(self) -> list[str]:
        """List all available example topics.

        Returns:
            List of topic names.
        """
        return list(self._EXAMPLES.keys())

    def format_examples(self, topic: str) -> str | None:
        """Format examples for display.

        Args:
            topic: Topic name.

        Returns:
            Formatted example text or None if topic not found.
        """
        examples = self.get_examples(topic)
        if not examples:
            return None

        lines = [
            f"Examples: {topic.capitalize()}",
            "=" * (11 + len(topic)),
            "",
        ]

        for i, example in enumerate(examples, 1):
            lines.append(f"{i}. {example.title}")
            lines.append("-" * (3 + len(example.title)))
            lines.append(example.description)
            lines.append("")

            for cmd in example.commands:
                lines.append(f"  {cmd}")

            if example.notes:
                lines.append("")
                lines.append(f"  Note: {example.notes}")

            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_smart_prompt(cache_dir: Path | None = None) -> SmartPrompt:
    """Get a SmartPrompt instance with default cache location.

    Args:
        cache_dir: Optional custom cache directory.

    Returns:
        Configured SmartPrompt instance.
    """
    return SmartPrompt(cache_dir=cache_dir)


def get_help_system() -> HelpSystem:
    """Get a HelpSystem instance.

    Returns:
        HelpSystem instance.
    """
    return HelpSystem()


def get_example_system() -> ExampleSystem:
    """Get an ExampleSystem instance.

    Returns:
        ExampleSystem instance.
    """
    return ExampleSystem()
