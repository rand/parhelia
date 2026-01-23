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

## CRITICAL: What Parhelia Actually Does

**Parhelia dispatches a NEW Claude Code instance to a cloud container.**

It does NOT:
- Run shell commands remotely (use `session attach` for that)
- Access your local filesystem (remote Claude clones from git)
- Access private repos without GITHUB_TOKEN configured on Modal

**For private repos, you MUST push first:**
```bash
git push origin my-branch
parhelia task create "Clone github.com/org/repo, checkout my-branch, run tests"
```

The prompt you provide becomes the initial instruction for a fresh Claude Code session running in Modal.

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
parhelia task create "Run the full test suite"
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
parhelia task create "Check if the build passes" --sync
```

Blocks until completion, then shows output.

### Pattern 3: GPU Task

**When**: ML training, inference, CUDA workloads

```bash
parhelia task create "Train the model on the dataset" --gpu A10G
```

GPU options: `A10G`, `A100`, `H100`, `T4`

### Pattern 4: Dry Run Testing

**When**: Test dispatch without Modal execution

```bash
parhelia task create "Test task" --dry-run
```

Validates task creation and dispatch logic without cost.

### Pattern 5: Specific Task Type

**When**: Optimize for specific workload

```bash
parhelia task create "Fix the auth bug" --type code_fix
```

Task type influences worker selection and resource allocation.

## Task Requirements

Configure resource requirements:

```bash
parhelia task create "Heavy computation" \
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

### Anti-Pattern 1: Expecting Local Filesystem Access

**Bad**: Assuming remote Claude can see your files
```bash
parhelia task create "Run the tests in ./src"  # Remote can't see ./src!
```

**Good**: Be explicit about cloning
```bash
parhelia task create "Clone github.com/me/repo, then run pytest"
```

### Anti-Pattern 2: Dispatching Without Pushing (Private Repos)

**Bad**: Your unpushed changes are invisible to remote
```bash
# Local changes not pushed
parhelia task create "Test my new feature"  # Remote clones old code!
```

**Good**: Push first
```bash
git push origin feature-branch
parhelia task create "Clone repo, checkout feature-branch, run tests"
```

### Anti-Pattern 3: Sync for Long Tasks

**Bad**: Using `--sync` for multi-hour tasks
```bash
parhelia task create "Train for 100 epochs" --sync  # Blocks CLI for hours
```

**Good**: Use async and attach
```bash
parhelia task create "Train for 100 epochs"
parhelia session attach task-abc12345
```

### Anti-Pattern 4: Wrong GPU Selection

**Bad**: Using H100 for small inference
```bash
parhelia task create "Run one prediction" --gpu H100  # Expensive overkill
```

**Good**: Match GPU to workload
```bash
parhelia task create "Run one prediction" --gpu T4  # Sufficient and cheaper
```

### Anti-Pattern 5: Using Parhelia for Quick Local Tasks

**Bad**: Dispatching what should run locally
```bash
parhelia task create "Run cargo check"  # Just run it locally!
```

**Good**: Run quick tasks locally
```bash
cargo check  # Faster, no overhead
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
2. List tasks: `parhelia task list`
3. Task may have been cleaned up

## Related Skills

- `parhelia/gpu-configuration` - GPU selection guidance
- `parhelia/budget-management` - Cost control
- `parhelia/checkpoint-resume` - Session recovery
