#!/usr/bin/env python3
"""Pre-dispatch hook for Parhelia.

Validates budget and resource availability before dispatching tasks.

Exit codes:
  0 - Allow dispatch
  1 - Reject dispatch (block)
  2 - Warn but allow

Environment:
  PARHELIA_HOOK_CONTEXT - JSON with task details
  PARHELIA_TASK_ID - Task ID being dispatched

Output (JSON to stdout):
  {"result": "allow|reject|warn", "message": "...", "data": {...}}
"""

import json
import os
import sys


def get_context() -> dict:
    """Load hook context from environment."""
    ctx_json = os.environ.get("PARHELIA_HOOK_CONTEXT", "{}")
    return json.loads(ctx_json)


def output(result: str, message: str, **data):
    """Output structured response."""
    print(json.dumps({
        "result": result,
        "message": message,
        **data,
    }))


def validate_budget(ctx: dict) -> tuple[bool, str]:
    """Check if budget allows this task.

    Returns:
        (allowed, message) tuple
    """
    budget_remaining = ctx.get("budget_remaining_usd")
    estimated_cost = ctx.get("estimated_cost_usd")

    if budget_remaining is None:
        return True, "Budget not tracked"

    # HARD STOP: Reject if budget is exhausted (< $0.50)
    if budget_remaining < 0.50:
        return False, f"BUDGET EXHAUSTED: Only ${budget_remaining:.2f} remaining. Run 'parhelia budget set <amount>' to increase."

    if estimated_cost is None:
        # Can't estimate, allow with warning
        return True, "Cost estimate unavailable"

    if estimated_cost > budget_remaining:
        return False, f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${budget_remaining:.2f}"

    # Warn if getting close to budget
    if budget_remaining - estimated_cost < budget_remaining * 0.2:
        return True, f"Warning: This task will use most of remaining budget (${budget_remaining:.2f} remaining)"

    return True, f"Budget OK: ${budget_remaining:.2f} remaining"


def validate_container_limit(ctx: dict) -> tuple[bool, str]:
    """Check if we're under the container limit.

    Returns:
        (allowed, message) tuple
    """
    import subprocess

    max_containers = ctx.get("max_concurrent_containers", 5)

    try:
        # Count active parhelia containers
        result = subprocess.run(
            ["modal", "container", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr

        # Count lines containing 'parhelia'
        active_count = sum(1 for line in output.split('\n') if 'parhelia' in line.lower() and 'ta-' in line)

        if active_count >= max_containers:
            return False, f"CONTAINER LIMIT: {active_count} containers running (max: {max_containers}). Run 'parhelia cleanup' first."

        if active_count >= max_containers - 1:
            return True, f"Warning: {active_count}/{max_containers} containers running"

        return True, f"Containers OK: {active_count}/{max_containers}"

    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Can't check, allow with warning
        return True, "Warning: Could not verify container count"


def validate_resources(ctx: dict) -> tuple[bool, str]:
    """Validate resource requirements.

    Returns:
        (allowed, message) tuple
    """
    requirements = ctx.get("requirements", {})

    # Check for very large memory requests (Modal max is 336GB without GPU)
    min_memory_gb = requirements.get("min_memory_gb", 4)
    if min_memory_gb > 336:
        return False, f"Memory request too large: {min_memory_gb}GB (Modal max is 336GB)"

    # Warn for high memory requests (expensive)
    if min_memory_gb >= 64:
        return True, f"Warning: High memory request ({min_memory_gb}GB) - ~${min_memory_gb * 0.005:.2f}/hr"

    # Warn for expensive GPU types
    gpu_type = requirements.get("gpu_type")
    if gpu_type in ("H100", "A100"):
        return True, f"Warning: {gpu_type} GPU is expensive - ensure this is necessary"

    return True, "Resources OK"


def main():
    ctx = get_context()
    task_id = ctx.get("task_id", "unknown")

    # Run validations (order matters - check limits first)
    container_ok, container_msg = validate_container_limit(ctx)
    budget_ok, budget_msg = validate_budget(ctx)
    resource_ok, resource_msg = validate_resources(ctx)

    # Determine result - reject on any hard failure
    if not container_ok:
        output("reject", container_msg, task_id=task_id)
        sys.exit(1)

    if not budget_ok:
        output("reject", budget_msg, task_id=task_id)
        sys.exit(1)

    if not resource_ok:
        output("reject", resource_msg, task_id=task_id)
        sys.exit(1)

    # Check for warnings
    warnings = []
    if "Warning:" in container_msg:
        warnings.append(container_msg)
    if "Warning:" in budget_msg:
        warnings.append(budget_msg)
    if "Warning:" in resource_msg:
        warnings.append(resource_msg)

    if warnings:
        output("warn", "; ".join(warnings), task_id=task_id)
        sys.exit(2)

    # All good
    output("allow", f"Pre-dispatch validation passed for {task_id}", task_id=task_id)
    sys.exit(0)


if __name__ == "__main__":
    main()
