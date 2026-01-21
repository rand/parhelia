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

    if estimated_cost is None:
        # Can't estimate, allow with warning
        return True, "Cost estimate unavailable"

    if estimated_cost > budget_remaining:
        return False, f"Estimated cost ${estimated_cost:.2f} exceeds remaining budget ${budget_remaining:.2f}"

    # Warn if getting close to budget
    if budget_remaining - estimated_cost < budget_remaining * 0.1:
        return True, f"Warning: This task will use most of remaining budget (${budget_remaining:.2f} remaining)"

    return True, f"Budget OK: ${budget_remaining:.2f} remaining"


def validate_resources(ctx: dict) -> tuple[bool, str]:
    """Validate resource requirements.

    Returns:
        (allowed, message) tuple
    """
    requirements = ctx.get("requirements", {})

    # Check for very large memory requests
    min_memory_gb = requirements.get("min_memory_gb", 4)
    if min_memory_gb > 128:
        return False, f"Memory request too large: {min_memory_gb}GB (max 128GB)"

    # Warn for expensive GPU types
    gpu_type = requirements.get("gpu_type")
    if gpu_type in ("H100", "A100"):
        return True, f"Warning: {gpu_type} GPU is expensive - ensure this is necessary"

    return True, "Resources OK"


def main():
    ctx = get_context()
    task_id = ctx.get("task_id", "unknown")

    # Run validations
    budget_ok, budget_msg = validate_budget(ctx)
    resource_ok, resource_msg = validate_resources(ctx)

    # Determine result
    if not budget_ok:
        output("reject", budget_msg, task_id=task_id)
        sys.exit(1)

    if not resource_ok:
        output("reject", resource_msg, task_id=task_id)
        sys.exit(1)

    # Check for warnings
    warnings = []
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
