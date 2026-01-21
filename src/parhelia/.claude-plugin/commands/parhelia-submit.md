---
name: parhelia-submit
description: Submit a task for remote execution on Modal infrastructure
argument-hint: <prompt> [--gpu TYPE] [--sync] [--dry-run]
---

# Submit Task to Modal

Submit a task for remote execution on Modal.com infrastructure with optional GPU support.

## Usage

```bash
parhelia submit "<prompt>" [options]
```

## Options

| Option | Description |
|--------|-------------|
| `--gpu TYPE` | GPU type: A10G, A100, H100, T4 (default: none) |
| `--memory N` | Minimum memory in GB (default: 4) |
| `--sync` | Wait for completion (synchronous mode) |
| `--dry-run` | Test without actual Modal execution |
| `-t, --type TYPE` | Task type: generic, code_fix, test, build, lint, refactor |
| `-w, --workspace PATH` | Working directory for the task |

## Examples

**Simple task (async)**:
```bash
parhelia submit "Run the test suite and fix any failures"
```

**GPU-enabled task**:
```bash
parhelia submit "Train the model on the dataset" --gpu A10G
```

**Synchronous execution**:
```bash
parhelia submit "Build the project" --sync
```

**Dry run for testing**:
```bash
parhelia submit "Test prompt" --dry-run
```

## Execution Flow

1. **Validate** - Check prompt and requirements
2. **Budget Check** - Verify budget availability
3. **Create Task** - Persist task to orchestrator
4. **Dispatch** - Send to Modal sandbox (unless --no-dispatch)
5. **Return** - Task ID (async) or result (sync)

## Output

**Async mode** (default):
```
Task submitted: task-abc12345
Dispatching in async mode...
Worker started: worker-def67890
```

**Sync mode** (--sync):
```
Task submitted: task-abc12345
Dispatching in sync mode...
Worker started: worker-def67890

Output:
[Task output appears here]
```

## Next Steps

After submitting a task:
- Check status: `parhelia status` or `/parhelia-status`
- Attach to session: `parhelia attach <task-id>` or `/parhelia-attach <task-id>`
- View logs: `parhelia logs <task-id>`

## Arguments

$ARGUMENTS
