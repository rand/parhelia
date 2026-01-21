---
name: parhelia-status
description: Show system status, active sessions, and resource utilization
argument-hint: [--json]
---

# Parhelia System Status

Display system health, active sessions, budget usage, and resource utilization.

## Usage

```bash
parhelia status [options]
```

## Options

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON for programmatic access |

## Output Sections

### Configuration
- Volume name and mount point
- CPU/memory allocation
- Default timeout settings

### Orchestrator
- Pending task count
- Active worker count

### Budget
- Ceiling amount (USD)
- Used amount and percentage
- Remaining budget
- Warning indicators

## Examples

**Human-readable output**:
```bash
parhelia status
```

Output:
```
Parhelia System Status
========================================

Configuration:
  Volume: parhelia-vol
  CPU: 4 cores, 16384MB
  Default timeout: 24h

Orchestrator:
  Pending tasks: 2
  Active workers: 1

Budget:
  Ceiling: $10.00
  Used: $2.50 (25.0%)
  Remaining: $7.50
```

**JSON output for agents**:
```bash
parhelia status --json
```

## Related Commands

- `/parhelia-submit` - Submit new tasks
- `/parhelia-session` - Session management
- `parhelia budget show` - Detailed budget info

## Arguments

$ARGUMENTS
