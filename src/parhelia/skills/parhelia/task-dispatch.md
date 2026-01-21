---
name: parhelia-task-dispatch
description: Submitting tasks and understanding dispatch modes
category: parhelia
keywords: [dispatch, submit, task, async, sync, modal, execution]
---

# Task Dispatch

**Scope**: Submitting tasks to Modal and understanding dispatch modes
**Lines**: ~300
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Atomic)

## When to Use This Skill

- Submitting tasks for remote execution
- Choosing between sync and async modes
- Understanding task lifecycle and states
- Debugging dispatch failures
- Configuring dispatch options

## Core Concepts

### Task Lifecycle

```
CREATED → PENDING → DISPATCHED → RUNNING → COMPLETED
                         │           │
                         │           └──→ FAILED
                         └──→ DISPATCH_FAILED
```

### Dispatch Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **ASYNC** (default) | Returns immediately with task ID | Long-running tasks, background work |
| **SYNC** | Waits for completion, returns result | Quick tasks, immediate feedback needed |

### Task Types

| Type | Description |
|------|-------------|
| `generic` | General-purpose tasks (default) |
| `code_fix` | Bug fixes and corrections |
| `test` | Running test suites |
| `build` | Building projects |
| `lint` | Linting and formatting |
| `refactor` | Code refactoring |

## Patterns

### Pattern 1: Simple Async Dispatch

**When**: Background task, don't need immediate result

```bash
parhelia submit "Run the full test suite"
```

Output:
```
Task submitted: task-abc12345
Dispatching in async mode...
Worker started: worker-def67890
```

Then check status:
```bash
parhelia task show task-abc12345
```

### Pattern 2: Synchronous Execution

**When**: Need result immediately, task is short

```bash
parhelia submit "Check if the build passes" --sync
```

Blocks until completion, then shows output.

### Pattern 3: GPU Task

**When**: ML training, inference, CUDA workloads

```bash
parhelia submit "Train the model on the dataset" --gpu A10G
```

GPU options: `A10G`, `A100`, `H100`, `T4`

### Pattern 4: Dry Run Testing

**When**: Test dispatch without Modal execution

```bash
parhelia submit "Test task" --dry-run
```

Validates task creation and dispatch logic without cost.

### Pattern 5: Specific Task Type

**When**: Optimize for specific workload

```bash
parhelia submit "Fix the auth bug" --type code_fix
```

Task type influences worker selection and resource allocation.

## Task Requirements

Configure resource requirements:

```bash
parhelia submit "Heavy computation" \
  --gpu A100 \
  --memory 32 \
  --workspace /path/to/project
```

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu` | GPU type | none |
| `--memory` | Min memory (GB) | 4 |
| `--workspace` | Working directory | current |

## Dispatch Flow

```
1. CLI creates Task object
   └─ ID, prompt, type, requirements

2. Task persisted to SQLite
   └─ Survives CLI restart

3. TaskDispatcher evaluates targets
   └─ GPU? → parhelia-gpu
   └─ CPU? → parhelia-cpu

4. Modal Sandbox created
   └─ Image with Claude Code
   └─ Volume mounted
   └─ Secrets injected

5. Entrypoint runs
   └─ Config linked
   └─ Hooks validated
   └─ tmux initialized

6. Claude Code executes prompt
   └─ In tmux session
   └─ With full context
```

## Anti-Patterns

### Anti-Pattern 1: Sync for Long Tasks

**Bad**: Using `--sync` for multi-hour tasks
```bash
parhelia submit "Train for 100 epochs" --sync  # Blocks CLI for hours
```

**Good**: Use async and attach
```bash
parhelia submit "Train for 100 epochs"
parhelia attach task-abc12345
```

### Anti-Pattern 2: Wrong GPU Selection

**Bad**: Using H100 for small inference
```bash
parhelia submit "Run one prediction" --gpu H100  # Expensive overkill
```

**Good**: Match GPU to workload
```bash
parhelia submit "Run one prediction" --gpu T4  # Sufficient and cheaper
```

### Anti-Pattern 3: Missing Workspace

**Bad**: Forgetting workspace context
```bash
cd /other/dir
parhelia submit "Run pytest"  # Wrong directory
```

**Good**: Explicit workspace
```bash
parhelia submit "Run pytest" --workspace /path/to/project
```

## Error Handling

### Dispatch Failed

```
Dispatch failed: No available workers
```

**Solutions**:
1. Check Modal dashboard for region issues
2. Try different region: configure in `parhelia.toml`
3. Reduce resource requirements

### Budget Exceeded

```
Dispatch failed: Task would exceed budget ceiling
```

**Solutions**:
1. Increase budget: `parhelia budget set 50.0`
2. Use cheaper resources (CPU instead of GPU)
3. Break into smaller tasks

### Task Not Found

```
Task not found: task-xyz
```

**Solutions**:
1. Check task ID spelling
2. List tasks: `parhelia list`
3. Task may have been cleaned up

## Related Skills

- `parhelia/gpu-configuration` - GPU selection guidance
- `parhelia/budget-management` - Cost control
- `parhelia/checkpoint-resume` - Session recovery
