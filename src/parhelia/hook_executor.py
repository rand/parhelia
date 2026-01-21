"""Hook execution framework for dispatch lifecycle.

Implements [SPEC-10.50] Hook Framework and [SPEC-10.51] Hook Types.

Provides pre-dispatch and post-dispatch hooks for:
- Budget validation before dispatch
- Resource estimation
- Audit logging
- Beads integration (task tracking)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class HookType(str, Enum):
    """Types of dispatch hooks."""

    PRE_DISPATCH = "pre-dispatch"
    POST_DISPATCH = "post-dispatch"


class HookResult(str, Enum):
    """Result of hook execution."""

    ALLOW = "allow"  # Proceed with dispatch
    REJECT = "reject"  # Block dispatch (pre-dispatch only)
    WARN = "warn"  # Proceed but log warning
    ERROR = "error"  # Hook failed to execute


@dataclass
class HookContext:
    """Context passed to hooks.

    Serialized as JSON environment variable PARHELIA_HOOK_CONTEXT.
    """

    hook_type: HookType
    task_id: str
    task_type: str
    prompt: str
    requirements: dict[str, Any]
    budget_remaining_usd: float | None = None
    estimated_cost_usd: float | None = None
    worker_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hook_type": self.hook_type.value,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "requirements": self.requirements,
            "budget_remaining_usd": self.budget_remaining_usd,
            "estimated_cost_usd": self.estimated_cost_usd,
            "worker_id": self.worker_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class HookOutput:
    """Output from a hook execution."""

    result: HookResult
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    @classmethod
    def from_exit_code(cls, code: int, stdout: str, stderr: str, duration_ms: int) -> "HookOutput":
        """Create HookOutput from process exit code and output."""
        if code == 0:
            # Try to parse JSON output
            try:
                data = json.loads(stdout) if stdout.strip() else {}
                return cls(
                    result=HookResult(data.get("result", "allow")),
                    message=data.get("message"),
                    data=data,
                    duration_ms=duration_ms,
                )
            except (json.JSONDecodeError, ValueError):
                return cls(
                    result=HookResult.ALLOW,
                    message=stdout.strip() if stdout else None,
                    duration_ms=duration_ms,
                )
        elif code == 1:
            # Reject
            return cls(
                result=HookResult.REJECT,
                message=stderr.strip() or stdout.strip() or "Hook rejected dispatch",
                duration_ms=duration_ms,
            )
        elif code == 2:
            # Warn but allow
            return cls(
                result=HookResult.WARN,
                message=stderr.strip() or stdout.strip() or "Hook warning",
                duration_ms=duration_ms,
            )
        else:
            # Error
            return cls(
                result=HookResult.ERROR,
                message=f"Hook exited with code {code}: {stderr or stdout}",
                duration_ms=duration_ms,
            )


@dataclass
class HookConfig:
    """Configuration for hook execution."""

    hooks_dir: Path = field(default_factory=lambda: Path.home() / ".parhelia" / "hooks")
    timeout_seconds: int = 30
    enabled: bool = True
    fail_open: bool = True  # If True, dispatch continues if hooks fail

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HookConfig":
        """Create from dictionary."""
        return cls(
            hooks_dir=Path(data.get("hooks_dir", str(Path.home() / ".parhelia" / "hooks"))),
            timeout_seconds=data.get("timeout_seconds", 30),
            enabled=data.get("enabled", True),
            fail_open=data.get("fail_open", True),
        )


class HookExecutor:
    """Executes dispatch lifecycle hooks.

    Hooks are executable scripts in the hooks directory:
    - ~/.parhelia/hooks/pre-dispatch.py (or .sh)
    - ~/.parhelia/hooks/post-dispatch.py (or .sh)

    Hook scripts receive context via PARHELIA_HOOK_CONTEXT env var (JSON).

    Exit codes:
    - 0: Allow (continue dispatch)
    - 1: Reject (block dispatch, pre-dispatch only)
    - 2: Warn (continue but log warning)
    - Other: Error

    Hooks can output JSON to stdout for structured responses:
    {"result": "allow|reject|warn", "message": "...", "data": {...}}
    """

    HOOK_EXTENSIONS = [".py", ".sh", ""]  # Order of preference

    def __init__(self, config: HookConfig | None = None):
        """Initialize hook executor.

        Args:
            config: Hook configuration. Uses defaults if not provided.
        """
        self.config = config or HookConfig()
        self._log_callback: Callable[[str], None] | None = None

    def set_log_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for log messages."""
        self._log_callback = callback

    def _log(self, message: str) -> None:
        """Log a message."""
        if self._log_callback:
            self._log_callback(message)

    def find_hook(self, hook_type: HookType) -> Path | None:
        """Find hook script for given type.

        Args:
            hook_type: The hook type to find.

        Returns:
            Path to hook script, or None if not found.
        """
        if not self.config.hooks_dir.exists():
            return None

        for ext in self.HOOK_EXTENSIONS:
            hook_path = self.config.hooks_dir / f"{hook_type.value}{ext}"
            if hook_path.exists() and hook_path.is_file():
                return hook_path

        return None

    async def execute_hook(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> HookOutput:
        """Execute a hook script.

        Args:
            hook_type: Type of hook to execute.
            context: Context to pass to hook.

        Returns:
            HookOutput with result and message.
        """
        if not self.config.enabled:
            return HookOutput(result=HookResult.ALLOW, message="Hooks disabled")

        hook_path = self.find_hook(hook_type)
        if not hook_path:
            return HookOutput(result=HookResult.ALLOW, message=f"No {hook_type.value} hook found")

        self._log(f"Executing {hook_type.value} hook: {hook_path}")

        # Prepare environment with context
        env = os.environ.copy()
        env["PARHELIA_HOOK_CONTEXT"] = context.to_json()
        env["PARHELIA_HOOK_TYPE"] = hook_type.value
        env["PARHELIA_TASK_ID"] = context.task_id

        start_time = datetime.now()

        try:
            # Determine how to run the hook
            if hook_path.suffix == ".py":
                cmd = ["python3", str(hook_path)]
            else:
                cmd = [str(hook_path)]

            # Run hook with timeout
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return HookOutput(
                    result=HookResult.ERROR if not self.config.fail_open else HookResult.WARN,
                    message=f"Hook timed out after {self.config.timeout_seconds}s",
                    duration_ms=self.config.timeout_seconds * 1000,
                )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return HookOutput.from_exit_code(
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                duration_ms,
            )

        except FileNotFoundError:
            return HookOutput(
                result=HookResult.ERROR if not self.config.fail_open else HookResult.WARN,
                message=f"Hook not executable: {hook_path}",
            )
        except Exception as e:
            return HookOutput(
                result=HookResult.ERROR if not self.config.fail_open else HookResult.WARN,
                message=f"Hook execution failed: {e}",
            )

    async def run_pre_dispatch(self, context: HookContext) -> HookOutput:
        """Run pre-dispatch hook.

        Args:
            context: Hook context with task info.

        Returns:
            HookOutput. If result is REJECT, dispatch should be blocked.
        """
        return await self.execute_hook(HookType.PRE_DISPATCH, context)

    async def run_post_dispatch(self, context: HookContext) -> HookOutput:
        """Run post-dispatch hook.

        Args:
            context: Hook context with task and worker info.

        Returns:
            HookOutput. Post-dispatch hooks cannot block (already dispatched).
        """
        output = await self.execute_hook(HookType.POST_DISPATCH, context)

        # Post-dispatch cannot reject (already dispatched)
        if output.result == HookResult.REJECT:
            output.result = HookResult.WARN
            output.message = f"Post-dispatch hook attempted to reject (ignored): {output.message}"

        return output


def create_hook_context_from_task(
    task: Any,  # Task from orchestrator
    hook_type: HookType,
    budget_remaining_usd: float | None = None,
    estimated_cost_usd: float | None = None,
    worker_id: str | None = None,
) -> HookContext:
    """Create HookContext from a Task object.

    Args:
        task: Task object from orchestrator.
        hook_type: Type of hook being executed.
        budget_remaining_usd: Remaining budget.
        estimated_cost_usd: Estimated cost of this task.
        worker_id: Worker ID (for post-dispatch).

    Returns:
        HookContext ready for hook execution.
    """
    return HookContext(
        hook_type=hook_type,
        task_id=task.id,
        task_type=task.task_type.value,
        prompt=task.prompt,
        requirements={
            "needs_gpu": task.requirements.needs_gpu,
            "gpu_type": task.requirements.gpu_type,
            "min_memory_gb": task.requirements.min_memory_gb,
            "min_cpu": task.requirements.min_cpu,
            "estimated_duration_minutes": task.requirements.estimated_duration_minutes,
        },
        budget_remaining_usd=budget_remaining_usd,
        estimated_cost_usd=estimated_cost_usd,
        worker_id=worker_id,
        session_id=f"ph-{task.id}",
        metadata=task.metadata,
    )
