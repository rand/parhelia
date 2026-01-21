---
name: parhelia-troubleshooting
description: Common issues, diagnostics, and solutions
category: parhelia
keywords: [troubleshooting, debug, error, fix, issue, problem, diagnose]
---

# Troubleshooting

**Scope**: Diagnosing and resolving common Parhelia issues
**Lines**: ~320
**Last Updated**: 2026-01-21
**Format Version**: 1.0 (Atomic)

## When to Use This Skill

- Sessions fail to start
- SSH tunnels won't connect
- Checkpoints not restoring
- Budget unexpectedly exceeded
- Tasks stuck in pending
- Container crashes

## Quick Diagnostics

### Health Check Command

```bash
parhelia status
```

Shows system health including:
- Configuration validity
- Pending/active tasks
- Budget status
- Modal connectivity

### Common First Steps

1. Check status: `parhelia status`
2. Check task: `parhelia task show <id>`
3. Check logs: `parhelia logs <id> --tail 100`
4. Check budget: `parhelia budget show`

## Issue: Session Won't Start

### Symptoms
- Task stuck in PENDING
- Dispatch times out
- "No workers available"

### Diagnostics
```bash
parhelia task show <task-id>
parhelia status
```

### Solutions

**1. Modal API Issues**
```bash
# Verify Modal credentials
modal token show

# Test Modal connectivity
modal run hello.py
```

**2. Region Unavailable**
```toml
# parhelia.toml - try different region
[modal]
region = "us-west"  # instead of us-east
```

**3. Resource Unavailable**
```bash
# GPU might be scarce - try different type
parhelia submit "Task" --gpu A10G  # instead of H100
```

**4. Budget Exceeded**
```bash
parhelia budget show
parhelia budget set 50.0  # increase if needed
```

## Issue: SSH Tunnel Fails

### Symptoms
- "Connection refused"
- "Connection reset"
- Tunnel drops immediately

### Diagnostics
```bash
parhelia task show <id>  # Check state
parhelia attach <id> --verbose
```

### Solutions

**1. Container Still Starting**
```bash
# Wait for RUNNING state
watch parhelia task show <id>
```

**2. Container Crashed**
```bash
parhelia logs <id>  # Check for errors
parhelia session recover <id>  # Recover
```

**3. Network Issues**
```bash
# Test basic connectivity
ping modal.com
curl -I https://modal.com
```

**4. Port Conflicts**
```bash
# Check if port 2222 is in use
lsof -i :2222
# Kill conflicting process if needed
```

## Issue: Checkpoint Won't Restore

### Symptoms
- Resume fails
- "Checkpoint not found"
- "Corrupted checkpoint"

### Diagnostics
```bash
parhelia checkpoint list --session <id>
parhelia checkpoint diff <cp-a> <cp-b>
```

### Solutions

**1. Checkpoint Doesn't Exist**
```bash
# List available checkpoints
parhelia checkpoint list

# Use specific checkpoint
parhelia resume <id> --checkpoint-id <cp-id>
```

**2. Workspace Conflicts**
```bash
# Clean target directory first
rm -rf /target/workspace
parhelia resume <id> --target /target/workspace
```

**3. Corrupted Checkpoint**
```bash
# Try earlier checkpoint
parhelia checkpoint list --session <id>
parhelia resume <id> --checkpoint-id <earlier-cp>
```

## Issue: Task Stuck in Running

### Symptoms
- Task shows RUNNING but no progress
- Can't attach
- No recent logs

### Diagnostics
```bash
parhelia task show <id>
parhelia logs <id> --tail 50
```

### Solutions

**1. Claude Code Hung**
```bash
# Attach and check
parhelia attach <id>
# In tmux, check Claude Code status
ps aux | grep claude
```

**2. Container Frozen**
```bash
# Force kill and recover
parhelia session kill <id>
parhelia session recover <id>
```

**3. Network Partition**
```bash
# Container might be fine but unreachable
# Wait and retry
sleep 60
parhelia attach <id>
```

## Issue: High Unexpected Costs

### Symptoms
- Budget exceeded quickly
- Unexpected charges

### Diagnostics
```bash
parhelia budget show
parhelia list --status completed | head -20
```

### Solutions

**1. Long-Running Sessions**
```bash
# Check for sessions still running
parhelia list --status running

# Kill unnecessary sessions
parhelia session kill <id>
```

**2. GPU When Not Needed**
```bash
# Review recent tasks
parhelia task show <id> | grep gpu

# Use CPU for non-ML tasks
parhelia submit "Task"  # No --gpu
```

**3. Forgotten Sessions**
```bash
# List all sessions
parhelia list

# Set auto-timeout
# parhelia.toml
[modal]
default_timeout_hours = 4  # Shorter timeout
```

## Issue: Hooks Not Working

### Symptoms
- Hooks not executing
- Permission errors
- Hook validation failures

### Diagnostics
```bash
# Check hook validation in logs
parhelia logs <id> | grep -i hook

# Validate hooks manually
python3 -c "from parhelia.hook_validator import HookValidator; v = HookValidator(); print(v.validate())"
```

### Solutions

**1. Hook Not Executable**
```bash
chmod +x ~/.claude/hooks/my_hook.py
```

**2. Missing Shebang**
```python
#!/usr/bin/env python3
# Add this as first line of hook script
```

**3. Hook Path Wrong**
```json
// ~/.claude/settings.json
{
  "hooks": {
    "PreToolUse": [{
      "hooks": ["~/.claude/hooks/my_hook.py"]  // Use full path
    }]
  }
}
```

## Issue: Container Keeps Crashing

### Symptoms
- Multiple restart attempts
- OOM killer messages
- "Container terminated"

### Diagnostics
```bash
parhelia logs <id> --tail 200 | grep -i "error\|killed\|oom"
```

### Solutions

**1. Out of Memory**
```bash
# Increase memory
parhelia submit "Task" --memory 32
```

**2. Disk Full**
```bash
# Clean up checkpoints
parhelia checkpoint list --older-than 7d
# Delete old checkpoints
```

**3. Bad Entrypoint**
```bash
# Check entrypoint logs
parhelia logs <id> | grep entrypoint
```

## Diagnostic Commands Reference

| Command | Purpose |
|---------|---------|
| `parhelia status` | System health |
| `parhelia task show <id>` | Task details |
| `parhelia logs <id>` | Container logs |
| `parhelia budget show` | Cost tracking |
| `parhelia list --status running` | Active sessions |
| `parhelia checkpoint list` | Available checkpoints |
| `parhelia config --json` | Configuration dump |

## Getting Help

If issues persist:

1. Collect diagnostics:
   ```bash
   parhelia status --json > status.json
   parhelia logs <id> > logs.txt
   ```

2. Check Modal status: https://status.modal.com

3. File issue with diagnostics attached

## Related Skills

- `parhelia/checkpoint-resume` - Recovery workflows
- `parhelia/interactive-attach` - SSH issues
- `parhelia/budget-management` - Cost issues
