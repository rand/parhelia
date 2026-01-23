---
name: discover-parhelia
description: Remote Claude Code execution on Modal.com - dispatch tasks to cloud containers with GPU support, checkpoint/resume, and budget controls
category: remote-execution
keywords: [parhelia, modal, remote, gpu, cloud, checkpoint, attach, dispatch, sandbox]
---

# Parhelia Remote Execution

**Scope**: Dispatch Claude Code sessions to Modal.com cloud containers
**Last Updated**: 2026-01-22
**Format Version**: 2.0

## Critical Mental Model

**Parhelia runs a SEPARATE Claude Code instance in the cloud.** It does NOT:
- Run arbitrary shell commands remotely
- Give remote Claude access to your local filesystem
- Give remote Claude access to private repos (without GITHUB_TOKEN)
- Execute in your current shell session

Think of it as: "Spawn a fresh Claude Code in a cloud container that will work on a task autonomously."

```
                YOU ARE HERE
                     |
                     v
Local Machine                       Modal.com Cloud
+------------------+               +-------------------------+
| Your terminal    |               | Fresh container         |
| Your filesystem  |   dispatch    | New Claude Code         |
| Private repos    | ------------> | Public repos only*      |
| Your plugins     |               | Synced plugins          |
| Your CLAUDE.md   |               | Synced CLAUDE.md        |
+------------------+               +-------------------------+
                                   * Unless GITHUB_TOKEN set
```

## When to Use Parhelia

**Good use cases:**
- Long-running tasks while you work on other things
- GPU-accelerated ML work (training, inference)
- Parallel execution of multiple independent tasks
- Tasks on public repositories
- Tasks on private repos (after pushing your branch)

**Bad use cases:**
- Running quick local commands (just run them locally)
- Tasks requiring your unpushed local changes
- Tasks on private repos without GITHUB_TOKEN configured
- Interactive debugging (use `--interactive` flag or work locally)

## The Push-First Workflow (Private Repos)

**If you're working on a private repo, you MUST push before dispatching:**

```bash
# 1. Commit and push your current work
git add -A && git commit -m "WIP: current state"
git push origin feature-branch

# 2. THEN dispatch to parhelia with repo/branch context
parhelia task create "Clone github.com/org/repo, checkout feature-branch, run pytest" --sync
```

The remote Claude Code will clone from GitHub - it cannot see your local files.

## Quick Start Commands

```bash
# Simple async task
parhelia task create "Run pytest in the cloned repo"

# Wait for result (--sync)
parhelia task create "Fix the failing tests" --sync

# With GPU
parhelia task create "Train the model" --gpu A10G

# Check what's running
parhelia task list
parhelia session list

# Watch progress
parhelia task watch <task-id>

# Attach interactively (SSH/tmux)
parhelia session attach <session-id>
```

## Full Command Reference

### Task Management
| Command | Purpose |
|---------|---------|
| `parhelia task create "<prompt>"` | Create async task |
| `parhelia task create "<prompt>" --sync` | Create and wait for result |
| `parhelia task create "<prompt>" --gpu A10G` | Request GPU |
| `parhelia task create "<prompt>" --interactive` | Interactive session |
| `parhelia task list` | List all tasks |
| `parhelia task show <id>` | Task details |
| `parhelia task watch <id>` | Real-time monitoring |
| `parhelia task delete <id>` | Delete task |

### Session Management
| Command | Purpose |
|---------|---------|
| `parhelia session list` | List active sessions |
| `parhelia session attach <id>` | SSH into running session |
| `parhelia session kill <id>` | Terminate session |
| `parhelia session recover <id>` | Guided recovery |

### Checkpoint & Resume
| Command | Purpose |
|---------|---------|
| `parhelia checkpoint create <session-id>` | Manual checkpoint |
| `parhelia checkpoint list [<session-id>]` | List checkpoints |
| `parhelia checkpoint restore <checkpoint-id>` | Restore from checkpoint |
| `parhelia resume <session-id>` | Resume from latest |

### Budget & Status
| Command | Purpose |
|---------|---------|
| `parhelia budget show` | Current spending |
| `parhelia budget set <amount>` | Set ceiling (USD) |
| `parhelia status` | System health |
| `parhelia container list` | Running containers |

### Aliases
`t`=task, `s`=session, `cp`=checkpoint, `c`=container, `b`=budget, `e`=events

## DO NOT (Common Mistakes)

### 1. Do NOT expect local filesystem access
```bash
# WRONG - remote Claude can't see your files
parhelia task create "Run the tests in ./src"

# RIGHT - be explicit about cloning
parhelia task create "Clone github.com/me/repo, then run pytest"
```

### 2. Do NOT use for quick local operations
```bash
# WRONG - just run this locally
parhelia task create "Run cargo check"

# RIGHT - run locally
cargo check
```

### 3. Do NOT dispatch without pushing (private repos)
```bash
# WRONG - your changes aren't visible remotely
parhelia task create "Test my new feature"

# RIGHT - push first
git push origin my-branch
parhelia task create "Clone repo, checkout my-branch, run tests"
```

### 4. Do NOT confuse with remote shell execution
```bash
# WRONG - parhelia is not ssh
parhelia task create "ls -la"

# Parhelia runs Claude Code, not arbitrary commands
# Use session attach if you need a shell
parhelia session attach <session-id>
```

### 5. Do NOT forget GITHUB_TOKEN for private repos
```bash
# If you get clone errors on private repos, the remote needs auth
modal secret create github-token GITHUB_TOKEN=ghp_...
```

## Cost Reference

| Resource | Cost/hr | Use Case |
|----------|---------|----------|
| CPU (4 core, 16GB) | ~$0.35 | Tests, builds, coding |
| A10G (24GB VRAM) | ~$1.10 | ML inference |
| A100 (40/80GB VRAM) | ~$2.50 | Model training |
| H100 (80GB VRAM) | ~$4.00 | Maximum performance |

Plus Claude API costs: ~$3/$15 per 1M tokens (Sonnet input/output).

## Setup Requirements

```bash
# 1. Install parhelia
uv tool install parhelia

# 2. Configure Modal
modal token set
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# 3. (Optional) For private repos
modal secret create github-token GITHUB_TOKEN=ghp_...

# 4. Deploy the Modal app
modal deploy src/parhelia/modal_app.py
```

## Troubleshooting

### "Session not found"
```bash
parhelia session list   # Check if it exists
parhelia help E200      # Detailed error help
```

### "Budget exceeded"
```bash
parhelia budget show    # Check current status
parhelia budget set 100 # Raise ceiling if needed
```

### "Can't clone repository"
- Is the repo private? → Configure GITHUB_TOKEN
- Is the URL correct? → Use full github.com/org/repo format

### Task seems stuck
```bash
parhelia task watch <id>      # Monitor progress
parhelia session attach <id>  # Attach interactively
parhelia container health     # Check container state
```

## When to Work Locally Instead

Choose local execution when:
- Changes aren't pushed yet
- Task takes <5 minutes
- You need interactive debugging
- You're iterating rapidly
- Working with local-only files

Choose parhelia when:
- Task will take a long time
- You want to free your laptop
- You need GPU resources
- Running parallel independent tasks
- Changes are pushed and repo is accessible
