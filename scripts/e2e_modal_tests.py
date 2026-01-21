#!/usr/bin/env python3
"""E2E Modal integration tests for Parhelia.

Validates the full execution pipeline on Modal.

Test levels:
- smoke: Health check, volume verification, Claude binary presence (~30s)
- core: Sandbox creation, simple prompt execution (~2-3 min)
- full: Complete task lifecycle, checkpoints (~10-15 min)

Usage:
    python scripts/e2e_modal_tests.py --level smoke
    python scripts/e2e_modal_tests.py --level core
    python scripts/e2e_modal_tests.py --level full
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    duration_seconds: float
    error: str | None = None


class E2ETestRunner:
    """Runs E2E tests against Modal."""

    def __init__(self, level: str = "core"):
        self.level = level
        self.results: list[TestResult] = []

    def log(self, msg: str) -> None:
        """Print with timestamp."""
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    async def run_test(self, name: str, test_fn) -> TestResult:
        """Run a single test and record result."""
        self.log(f"Running: {name}")
        start = time.time()
        try:
            await test_fn()
            duration = time.time() - start
            self.log(f"  ✅ PASSED ({duration:.1f}s)")
            return TestResult(name=name, passed=True, duration_seconds=duration)
        except Exception as e:
            duration = time.time() - start
            self.log(f"  ❌ FAILED: {e}")
            return TestResult(
                name=name, passed=False, duration_seconds=duration, error=str(e)
            )

    # =========================================================================
    # Smoke Tests
    # =========================================================================

    async def test_health_check(self) -> None:
        """Verify Modal app health check passes."""
        import modal

        # Look up the deployed function
        health_check = modal.Function.from_name("parhelia", "health_check")

        result = health_check.remote()
        assert result["status"] == "ok", f"Health check failed: {result}"
        assert result["volume_mounted"], "Volume not mounted"
        assert result["claude_installed"], f"Claude not installed: {result}"
        assert result["anthropic_key_set"], "Anthropic API key not set"

    async def test_volume_structure(self) -> None:
        """Verify volume directories exist."""
        import modal

        init_volume_structure = modal.Function.from_name("parhelia", "init_volume_structure")

        result = init_volume_structure.remote()
        assert "all_directories" in result, f"Missing directories: {result}"
        expected = ["/vol/parhelia/config/claude", "/vol/parhelia/workspaces"]
        for d in expected:
            assert d in result["all_directories"], f"Missing directory: {d}"

    # =========================================================================
    # Core Tests
    # =========================================================================

    async def test_sandbox_creation(self) -> None:
        """Test sandbox can be created."""
        from parhelia.modal_app import create_claude_sandbox, run_in_sandbox

        sandbox = await create_claude_sandbox("e2e-test-sandbox")
        assert sandbox is not None, "Sandbox creation failed"

        # Verify basic command execution
        output = await run_in_sandbox(sandbox, ["echo", "hello"])
        assert "hello" in output, f"Echo failed: {output}"

    async def test_simple_prompt(self) -> None:
        """Test Claude Code can process a simple prompt."""
        from parhelia.modal_app import create_claude_sandbox, run_in_sandbox

        sandbox = await create_claude_sandbox("e2e-test-prompt")

        # Run Claude with a simple math prompt
        output = await run_in_sandbox(
            sandbox,
            ["/root/.local/bin/claude", "-p", "What is 7*8? Reply with just the number."],
            timeout_seconds=60,
        )
        assert "56" in output, f"Expected 56 in output: {output}"

    async def test_entrypoint_initialization(self) -> None:
        """Test entrypoint runs and creates expected files."""
        from parhelia.modal_app import create_claude_sandbox, run_in_sandbox

        # Create sandbox and run entrypoint manually
        sandbox = await create_claude_sandbox("e2e-entrypoint-test")

        # Run entrypoint
        await run_in_sandbox(
            sandbox,
            ["bash", "-c", "PARHELIA_INTERACTIVE=true /entrypoint.sh"],
            timeout_seconds=30,
        )

        # Check entrypoint created expected files
        output = await run_in_sandbox(sandbox, ["cat", "/tmp/ready"])
        assert "PARHELIA_READY" in output, f"Ready file not found: {output}"

        # Check config linking
        output = await run_in_sandbox(sandbox, ["ls", "-la", "/root/.claude"])
        assert "vol/parhelia" in output or "config" in output, f"Config not linked: {output}"

    # =========================================================================
    # Full Tests
    # =========================================================================

    async def test_cli_submit_sync(self) -> None:
        """Test CLI submit with sync mode."""
        import subprocess

        result = subprocess.run(
            ["uv", "run", "parhelia", "submit", "What is 2+2? Reply with just the number.", "--sync"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "4" in result.stdout, f"Expected 4 in output: {result.stdout}"

    async def test_task_persistence(self) -> None:
        """Test tasks are persisted correctly."""
        from parhelia.persistence import PersistentOrchestrator
        from parhelia.orchestrator import Task, TaskRequirements, TaskType
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        orch = PersistentOrchestrator(db_path=db_path)

        # Create and persist a task
        task = Task(
            id="e2e-persistence-test",
            prompt="Test prompt",
            task_type=TaskType.GENERIC,
            requirements=TaskRequirements(),
        )
        orch.task_store.save(task)

        # Reload and verify
        loaded = orch.task_store.get("e2e-persistence-test")
        assert loaded is not None, "Task not found after save"
        assert loaded.prompt == "Test prompt", "Task prompt mismatch"

    # =========================================================================
    # Test Runner
    # =========================================================================

    async def run_all(self) -> bool:
        """Run all tests for the configured level."""
        self.log(f"Starting E2E tests (level: {self.level})")
        self.log("=" * 50)

        # Smoke tests (always run)
        self.results.append(await self.run_test("Health Check", self.test_health_check))
        self.results.append(
            await self.run_test("Volume Structure", self.test_volume_structure)
        )

        if self.level in ("core", "full"):
            # Core tests
            self.results.append(
                await self.run_test("Sandbox Creation", self.test_sandbox_creation)
            )
            self.results.append(
                await self.run_test("Simple Prompt", self.test_simple_prompt)
            )
            self.results.append(
                await self.run_test(
                    "Entrypoint Initialization", self.test_entrypoint_initialization
                )
            )

        if self.level == "full":
            # Full tests
            self.results.append(
                await self.run_test("CLI Submit Sync", self.test_cli_submit_sync)
            )
            self.results.append(
                await self.run_test("Task Persistence", self.test_task_persistence)
            )

        # Summary
        self.log("=" * 50)
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        total_time = sum(r.duration_seconds for r in self.results)

        self.log(f"Results: {passed}/{total} passed in {total_time:.1f}s")

        if passed < total:
            self.log("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    self.log(f"  - {r.name}: {r.error}")
            return False

        return True


def main():
    parser = argparse.ArgumentParser(description="Run E2E Modal tests")
    parser.add_argument(
        "--level",
        choices=["smoke", "core", "full"],
        default="core",
        help="Test level to run",
    )
    args = parser.parse_args()

    runner = E2ETestRunner(level=args.level)
    success = asyncio.run(runner.run_all())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
