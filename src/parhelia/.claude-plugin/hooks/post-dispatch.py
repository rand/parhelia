#!/usr/bin/env python3
"""Post-dispatch hook for Parhelia.

Performs audit logging and optional beads integration after task dispatch.

Exit codes:
  0 - Success
  2 - Warning (logged but doesn't affect dispatch)

Environment:
  PARHELIA_HOOK_CONTEXT - JSON with task and worker details
  PARHELIA_TASK_ID - Task ID that was dispatched

Output (JSON to stdout):
  {"result": "allow|warn", "message": "...", "data": {...}}
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


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


def append_audit_log(ctx: dict) -> bool:
    """Append entry to audit log.

    Returns:
        True if logged successfully
    """
    audit_dir = Path.home() / ".parhelia" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_file = audit_dir / f"dispatch-{datetime.now().strftime('%Y-%m')}.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "dispatch",
        "task_id": ctx.get("task_id"),
        "task_type": ctx.get("task_type"),
        "worker_id": ctx.get("worker_id"),
        "session_id": ctx.get("session_id"),
        "requirements": ctx.get("requirements", {}),
        "estimated_cost_usd": ctx.get("estimated_cost_usd"),
        "prompt_preview": ctx.get("prompt", "")[:100],
    }

    try:
        with open(audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except OSError:
        return False


def try_beads_integration(ctx: dict) -> tuple[bool, str]:
    """Optionally create/update beads issue for task tracking.

    Returns:
        (success, message) tuple
    """
    # Check if beads is available
    try:
        result = subprocess.run(
            ["bd", "--version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return True, "Beads not available, skipping integration"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True, "Beads not installed, skipping integration"

    # Check if .beads directory exists (project uses beads)
    if not Path(".beads").exists():
        return True, "No .beads directory, skipping integration"

    # Check if task is tracked in beads metadata
    metadata = ctx.get("metadata", {})
    beads_id = metadata.get("beads_id")

    if beads_id:
        # Update existing issue to in_progress
        try:
            result = subprocess.run(
                ["bd", "update", beads_id, "--status", "in_progress"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, f"Updated beads issue {beads_id} to in_progress"
        except (subprocess.TimeoutExpired, OSError):
            pass
        return True, f"Beads issue {beads_id} update attempted"

    return True, "No beads_id in task metadata"


def main():
    ctx = get_context()
    task_id = ctx.get("task_id", "unknown")
    worker_id = ctx.get("worker_id", "unknown")

    messages = []

    # Audit logging
    if append_audit_log(ctx):
        messages.append(f"Logged dispatch of {task_id} to {worker_id}")
    else:
        messages.append("Audit logging failed")

    # Beads integration
    beads_ok, beads_msg = try_beads_integration(ctx)
    if beads_msg:
        messages.append(beads_msg)

    # Output result
    output("allow", "; ".join(messages), task_id=task_id, worker_id=worker_id)
    sys.exit(0)


if __name__ == "__main__":
    main()
