---
name: parhelia-submit
description: Submit a task for remote execution on Modal infrastructure
argument-hint: <prompt> [--gpu TYPE] [--sync] [--automated]
---

# Create Task for Remote Execution

Create and dispatch a task to Modal.com infrastructure for remote Claude Code execution.

## Critical: Remote Execution Model

**Remote Claude clones from git** - it cannot see your local files.

Your prompt becomes the initial task for a fresh Claude Code session in the cloud.
You must include clone instructions or the remote Claude won't have access to your code.

## Usage

```bash
# Preferred (canonical command)
parhelia task create "<prompt>" [options]

# Legacy (deprecated but functional)
parhelia submit "<prompt>" [options]
```

## Options

| Option | Description |
|--------|-------------|
| `--gpu TYPE` | GPU type: A10G, A100, H100, T4 (default: none) |
| `--memory N` | Minimum memory in GB (default: 4) |
| `--sync` | Wait for completion (synchronous mode) |
| `--dry-run` | Test without actual Modal execution |
| `--automated` | Skip permission prompts (for CI/agents) |
| `--no-hints` | Suppress pre-flight warnings |
| `-t, --type TYPE` | Task type: generic, code_fix, test, build, lint, refactor |
| `-w, --workspace PATH` | Working directory for the task |

## Agent Workflow

**Before submitting a task:**

1. **Push your changes** - Remote Claude clones from git
   ```bash
   git push origin my-branch
   ```

2. **Include clone instructions** in your prompt:
   ```bash
   parhelia task create "Clone https://github.com/org/repo, checkout my-branch, run cargo test" --sync --automated
   ```

3. **Use --automated** for unattended execution (skips permission prompts)

## Examples

**Agent-driven verification (recommended pattern)**:
```bash
git push origin feature-branch
parhelia task create "Clone github.com/org/repo, checkout feature-branch, run cargo test and report results" --sync --automated
```

**GPU-enabled task**:
```bash
parhelia task create "Clone repo, train the model" --gpu A10G --automated
```

**Dry run for testing**:
```bash
parhelia task create "Test prompt" --dry-run
```

## Execution Flow

1. **Pre-flight checks** - Warn about unpushed commits, missing clone instructions
2. **Create Task** - Persist to orchestrator
3. **Dispatch** - Send to Modal sandbox
4. **Return** - Task ID (async) or result (sync)

## Output

**Async mode** (default):
```
Task created: task-abc12345
Dispatching in async mode...
Worker started: worker-def67890
```

**Sync mode** (--sync):
```
Task created: task-abc12345
Dispatching in sync mode...
Worker started: worker-def67890

Output:
[Claude's response appears here]
```

## Next Steps

After creating a task:
- Check status: `parhelia status` or `/parhelia-status`
- Attach to session: `parhelia session attach <task-id>`
- View task details: `parhelia task show <task-id>`

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| "Run tests" without clone | Include: "Clone github.com/org/repo, run tests" |
| Local changes not visible | Run `git push` before submitting |
| Permission prompts block | Add `--automated` flag |

## Arguments

$ARGUMENTS
