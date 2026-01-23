---
name: parhelia-budget
description: Budget management - show, set, reset spending limits
argument-hint: <subcommand> [args]
---

# Budget Management

Manage spending limits and track costs for Modal execution.

## Subcommands

### show - Show Budget Status

```bash
parhelia budget show
```

Displays:
- Budget ceiling (USD)
- Current usage and percentage
- Remaining budget
- Task and token counts
- Warning indicators

### set - Set Budget Ceiling

```bash
parhelia budget set <amount>
```

Set the maximum spending limit in USD.

### reset - Reset Budget Tracking

```bash
parhelia budget reset
```

Reset all budget tracking (requires confirmation).

## Budget Protection

Tasks are blocked when:
- Estimated cost would exceed ceiling
- Warning threshold reached (configurable, default 80%)

Pre-dispatch hooks validate budget before task submission.

## Examples

**Check current budget**:
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

**Increase budget**:
```bash
parhelia budget set 50.0
```

**Reset tracking (new billing period)**:
```bash
parhelia budget reset
```

## Cost Estimation

Estimated costs per hour (from [Modal.com](https://modal.com/pricing)):
- **CPU only**: ~$0.35/hr
- **A10G GPU**: ~$1.10/hr
- **A100 GPU**: ~$2.50/hr
- **H100 GPU**: ~$4.00/hr

API token costs are additional.

## Configuration

Budget settings in `parhelia.toml`:

```toml
[budget]
default_ceiling_usd = 10.0
warning_threshold = 0.8
max_cost_per_task = 5.0
max_daily_cost = 100.0
```

## Related Commands

- `/parhelia-submit` - Submit tasks (budget checked)
- `/parhelia-status` - System status including budget

## Arguments

$ARGUMENTS
