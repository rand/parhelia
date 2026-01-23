---
name: parhelia-budget-management
description: Cost tracking, spending limits, and budget alerts
category: parhelia
keywords: [budget, cost, spending, limit, tracking, alert, ceiling]
---

# Budget Management

**Scope**: Tracking and controlling costs for remote execution
**Lines**: ~250
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Atomic)

## When to Use This Skill

- Setting up budget controls
- Understanding cost breakdowns
- Configuring spending alerts
- Debugging budget-related blocks
- Optimizing costs

## Core Concepts

### Cost Components

| Component | Rate | Notes |
|-----------|------|-------|
| **CPU compute** | ~$0.35/hr | Base container (4 CPU, 16GB) |
| **T4 GPU** | ~$0.60/hr | Light ML |
| **A10G GPU** | ~$1.10/hr | Medium ML |
| **A100 GPU** | ~$2.50/hr | Heavy ML |
| **H100 GPU** | ~$4.00/hr | Maximum perf |
| **API tokens (input)** | ~$3/1M | Claude API |
| **API tokens (output)** | ~$15/1M | Claude API |

*Based on [Modal.com pricing](https://modal.com/pricing). Per-second billing.*

### Budget States

| State | Behavior |
|-------|----------|
| **Normal** | Tasks dispatch normally |
| **Warning** | Alert shown, tasks still allowed |
| **Exceeded** | New tasks blocked |

## Patterns

### Pattern 1: Check Budget Status

```bash
parhelia budget show
```

Output:
```
Budget Status
==============================
Ceiling:     $10.00
Used:        $2.50
Remaining:   $7.50
Usage:       25.0%
Tasks:       5
Tokens in:   125,000
Tokens out:  45,000
```

### Pattern 2: Set Budget Ceiling

```bash
parhelia budget set 50.0
```

Sets maximum spend to $50.

### Pattern 3: Reset for New Period

```bash
parhelia budget reset
```

Clears usage tracking (for new billing period).

### Pattern 4: Cost-Aware Task Submission

```bash
# Estimate cost first with dry-run
parhelia task create "Train model" --gpu A10G --dry-run

# Check remaining budget
parhelia budget show

# Submit if budget allows
parhelia task create "Train model" --gpu A10G
```

## Configuration

In `parhelia.toml`:

```toml
[budget]
# Maximum spend before blocking tasks
default_ceiling_usd = 10.0

# Alert when this percentage reached
warning_threshold = 0.8

# Maximum per single task
max_cost_per_task = 5.0

# Daily limit (optional)
max_daily_cost = 100.0
```

## Cost Estimation

### Per-Session Estimates

| Workload | Duration | Resource | Est. Cost |
|----------|----------|----------|-----------|
| Quick test | 10 min | CPU | $0.06 |
| Code review | 1 hr | CPU | $0.35 |
| Build + test | 30 min | CPU | $0.18 |
| ML inference | 1 hr | T4 | $0.60 |
| Model training | 4 hr | A10G | $4.40 |
| Large training | 8 hr | A100 | $20.00 |

### Token Cost Examples

| Conversation | Input | Output | API Cost |
|--------------|-------|--------|----------|
| Short task | 10K | 5K | $0.11 |
| Medium task | 50K | 20K | $0.45 |
| Long session | 200K | 100K | $2.10 |

## Anti-Patterns

### Anti-Pattern 1: No Budget Ceiling

**Bad**: Running without limits
```toml
[budget]
default_ceiling_usd = 999999.0  # No effective limit
```

**Good**: Reasonable ceiling
```toml
[budget]
default_ceiling_usd = 50.0
warning_threshold = 0.8
```

### Anti-Pattern 2: Ignoring Warnings

**Bad**: Continuing when warned
```
⚠ Warning threshold reached (80%)
# User ignores, submits more tasks
# Budget exceeded, tasks blocked
```

**Good**: Respond to warnings
```
⚠ Warning threshold reached (80%)
# Check budget: parhelia budget show
# Either increase ceiling or wait
```

### Anti-Pattern 3: Expensive Resources for Simple Tasks

**Bad**: GPU for non-ML work
```bash
parhelia task create "Run linter" --gpu H100  # $4.00/hr for linting!
```

**Good**: Match resources to task
```bash
parhelia task create "Run linter"  # CPU: $0.35/hr
```

## Budget Exceeded: Recovery

When budget is exceeded:

```
Dispatch failed: Task would exceed budget ceiling ($10.00)
```

**Options**:

1. **Increase ceiling**:
   ```bash
   parhelia budget set 20.0
   ```

2. **Wait for new period**:
   ```bash
   parhelia budget reset  # If starting new billing cycle
   ```

3. **Reduce task cost**:
   ```bash
   # Use CPU instead of GPU
   parhelia task create "Task"
   # Instead of: parhelia task create "Task" --gpu A100
   ```

4. **Break into smaller tasks**:
   ```bash
   # Instead of one long task
   parhelia task create "Part 1 of work"
   # ... complete, then
   parhelia task create "Part 2 of work"
   ```

## Monitoring Tips

### Daily Review

```bash
# Start of day
parhelia budget show

# After each task
parhelia task show <id> | grep cost
```

### Cost Tracking Integration

Tasks record their costs:

```bash
parhelia task show task-abc12345 --json | jq '.result.cost_usd'
```

### Setting Alerts

Configure notifications in `parhelia.toml`:

```toml
[notifications]
budget_warning = true
budget_exceeded = true
channels = ["stdout"]  # or webhook, email
```

## Related Skills

- `parhelia/gpu-configuration` - GPU costs
- `parhelia/task-dispatch` - Task submission
- `parhelia/troubleshooting` - Budget issues
