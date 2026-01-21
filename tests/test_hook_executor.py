"""Tests for hook executor."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from parhelia.hook_executor import (
    HookConfig,
    HookContext,
    HookExecutor,
    HookOutput,
    HookResult,
    HookType,
)


# =============================================================================
# HookContext Tests
# =============================================================================


class TestHookContext:
    """Tests for HookContext class."""

    def test_to_dict(self):
        """HookContext MUST serialize to dict."""
        ctx = HookContext(
            hook_type=HookType.PRE_DISPATCH,
            task_id="task-123",
            task_type="generic",
            prompt="Test prompt",
            requirements={"needs_gpu": False},
            budget_remaining_usd=10.0,
        )

        result = ctx.to_dict()

        assert result["hook_type"] == "pre-dispatch"
        assert result["task_id"] == "task-123"
        assert result["budget_remaining_usd"] == 10.0

    def test_to_json(self):
        """HookContext MUST serialize to JSON."""
        ctx = HookContext(
            hook_type=HookType.POST_DISPATCH,
            task_id="task-456",
            task_type="test_run",
            prompt="Run tests",
            requirements={},
            worker_id="worker-abc",
        )

        result = ctx.to_json()
        parsed = json.loads(result)

        assert parsed["hook_type"] == "post-dispatch"
        assert parsed["worker_id"] == "worker-abc"


# =============================================================================
# HookOutput Tests
# =============================================================================


class TestHookOutput:
    """Tests for HookOutput class."""

    def test_from_exit_code_zero_allow(self):
        """Exit code 0 with no output MUST result in ALLOW."""
        output = HookOutput.from_exit_code(0, "", "", 100)

        assert output.result == HookResult.ALLOW
        assert output.duration_ms == 100

    def test_from_exit_code_zero_json(self):
        """Exit code 0 with JSON output MUST parse result."""
        stdout = '{"result": "warn", "message": "Budget low"}'
        output = HookOutput.from_exit_code(0, stdout, "", 50)

        assert output.result == HookResult.WARN
        assert output.message == "Budget low"

    def test_from_exit_code_one_reject(self):
        """Exit code 1 MUST result in REJECT."""
        output = HookOutput.from_exit_code(1, "", "Rejected: over budget", 75)

        assert output.result == HookResult.REJECT
        assert "over budget" in output.message

    def test_from_exit_code_two_warn(self):
        """Exit code 2 MUST result in WARN."""
        output = HookOutput.from_exit_code(2, "Warning message", "", 25)

        assert output.result == HookResult.WARN
        assert "Warning message" in output.message

    def test_from_exit_code_other_error(self):
        """Other exit codes MUST result in ERROR."""
        output = HookOutput.from_exit_code(127, "", "Command not found", 10)

        assert output.result == HookResult.ERROR
        assert "127" in output.message


# =============================================================================
# HookConfig Tests
# =============================================================================


class TestHookConfig:
    """Tests for HookConfig class."""

    def test_from_dict_defaults(self):
        """HookConfig MUST use defaults for missing keys."""
        config = HookConfig.from_dict({})

        assert config.timeout_seconds == 30
        assert config.enabled is True
        assert config.fail_open is True

    def test_from_dict_custom(self):
        """HookConfig MUST accept custom values."""
        config = HookConfig.from_dict({
            "hooks_dir": "/custom/hooks",
            "timeout_seconds": 60,
            "enabled": False,
        })

        assert str(config.hooks_dir) == "/custom/hooks"
        assert config.timeout_seconds == 60
        assert config.enabled is False


# =============================================================================
# HookExecutor Tests
# =============================================================================


class TestHookExecutor:
    """Tests for HookExecutor class."""

    @pytest.fixture
    def hooks_dir(self, tmp_path):
        """Create temporary hooks directory."""
        hooks_path = tmp_path / "hooks"
        hooks_path.mkdir()
        return hooks_path

    @pytest.fixture
    def executor(self, hooks_dir):
        """Create executor with test hooks directory."""
        config = HookConfig(hooks_dir=hooks_dir)
        return HookExecutor(config)

    @pytest.fixture
    def test_context(self):
        """Create test hook context."""
        return HookContext(
            hook_type=HookType.PRE_DISPATCH,
            task_id="task-test",
            task_type="generic",
            prompt="Test task",
            requirements={"needs_gpu": False},
        )

    def test_find_hook_not_found(self, executor):
        """find_hook MUST return None if hook doesn't exist."""
        result = executor.find_hook(HookType.PRE_DISPATCH)
        assert result is None

    def test_find_hook_py(self, executor, hooks_dir):
        """find_hook MUST find .py hooks."""
        hook_path = hooks_dir / "pre-dispatch.py"
        hook_path.write_text("#!/usr/bin/env python3\nprint('ok')")

        result = executor.find_hook(HookType.PRE_DISPATCH)
        assert result == hook_path

    def test_find_hook_sh(self, executor, hooks_dir):
        """find_hook MUST find .sh hooks."""
        hook_path = hooks_dir / "post-dispatch.sh"
        hook_path.write_text("#!/bin/bash\necho ok")

        result = executor.find_hook(HookType.POST_DISPATCH)
        assert result == hook_path

    @pytest.mark.asyncio
    async def test_execute_hook_disabled(self, hooks_dir, test_context):
        """execute_hook MUST return ALLOW if hooks disabled."""
        config = HookConfig(hooks_dir=hooks_dir, enabled=False)
        executor = HookExecutor(config)

        output = await executor.execute_hook(HookType.PRE_DISPATCH, test_context)

        assert output.result == HookResult.ALLOW
        assert "disabled" in output.message.lower()

    @pytest.mark.asyncio
    async def test_execute_hook_not_found(self, executor, test_context):
        """execute_hook MUST return ALLOW if hook not found."""
        output = await executor.execute_hook(HookType.PRE_DISPATCH, test_context)

        assert output.result == HookResult.ALLOW
        assert "no" in output.message.lower() and "hook" in output.message.lower()

    @pytest.mark.asyncio
    async def test_execute_hook_success(self, executor, hooks_dir, test_context):
        """execute_hook MUST execute hook and return result."""
        hook_path = hooks_dir / "pre-dispatch.py"
        hook_path.write_text("""#!/usr/bin/env python3
import json
print(json.dumps({"result": "allow", "message": "Validation passed"}))
""")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR)

        output = await executor.execute_hook(HookType.PRE_DISPATCH, test_context)

        assert output.result == HookResult.ALLOW
        assert "passed" in output.message.lower()

    @pytest.mark.asyncio
    async def test_execute_hook_reject(self, executor, hooks_dir, test_context):
        """execute_hook MUST handle rejection."""
        hook_path = hooks_dir / "pre-dispatch.py"
        hook_path.write_text("""#!/usr/bin/env python3
import sys
print("Budget exceeded", file=sys.stderr)
sys.exit(1)
""")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR)

        output = await executor.execute_hook(HookType.PRE_DISPATCH, test_context)

        assert output.result == HookResult.REJECT
        assert "exceeded" in output.message.lower()

    @pytest.mark.asyncio
    async def test_execute_hook_receives_context(self, executor, hooks_dir, test_context):
        """execute_hook MUST pass context to hook via environment."""
        hook_path = hooks_dir / "pre-dispatch.py"
        hook_path.write_text("""#!/usr/bin/env python3
import json
import os

ctx = json.loads(os.environ.get("PARHELIA_HOOK_CONTEXT", "{}"))
task_id = ctx.get("task_id", "unknown")
print(json.dumps({"result": "allow", "message": f"Got task {task_id}"}))
""")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR)

        output = await executor.execute_hook(HookType.PRE_DISPATCH, test_context)

        assert output.result == HookResult.ALLOW
        assert "task-test" in output.message

    @pytest.mark.asyncio
    async def test_run_post_dispatch_cannot_reject(self, executor, hooks_dir):
        """run_post_dispatch MUST convert REJECT to WARN."""
        hook_path = hooks_dir / "post-dispatch.py"
        hook_path.write_text("""#!/usr/bin/env python3
import sys
sys.exit(1)  # Try to reject
""")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR)

        context = HookContext(
            hook_type=HookType.POST_DISPATCH,
            task_id="task-post",
            task_type="generic",
            prompt="Test",
            requirements={},
            worker_id="worker-123",
        )

        output = await executor.run_post_dispatch(context)

        # Should be converted to WARN, not REJECT
        assert output.result == HookResult.WARN

    @pytest.mark.asyncio
    async def test_execute_hook_timeout(self, hooks_dir, test_context):
        """execute_hook MUST timeout long-running hooks."""
        config = HookConfig(hooks_dir=hooks_dir, timeout_seconds=1)
        executor = HookExecutor(config)

        hook_path = hooks_dir / "pre-dispatch.py"
        hook_path.write_text("""#!/usr/bin/env python3
import time
time.sleep(10)  # Sleep longer than timeout
""")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR)

        output = await executor.execute_hook(HookType.PRE_DISPATCH, test_context)

        assert output.result in (HookResult.ERROR, HookResult.WARN)
        assert "timed out" in output.message.lower()
