# Parhelia Agent Friction Report

**Date**: 2026-01-23
**Context**: Claude Code agent attempting to use Parhelia for remote verification of code changes

## Summary

This document captures friction points, confusion, and insights encountered while using Parhelia from an agent's perspective. The goal is to inform improvements to make Parhelia more agent-friendly.

---

## 1. Command Name Confusion

### Issue
The CLI help text creates confusion about command names:

```
Warning: 'parhelia submit' is deprecated. Use 'parhelia task create' instead.
```

But `parhelia task create` doesn't exist:

```bash
$ parhelia task --help
Commands:
  cancel, cleanup, delete, dispatch, list, retry, show, watch
  # No 'create' command
```

### Impact
- Agent tries `parhelia task create` (as instructed by deprecation warning)
- Command fails with "No such command 'create'"
- Agent must discover that `parhelia submit` is the actual working command

### Suggested Fix
Either:
1. Add `parhelia task create` as the actual command
2. Remove the misleading deprecation warning
3. Make `task create` an alias for `submit`

---

## 2. Git Authentication Not Configured

### Issue
The `GITHUB_TOKEN` Modal secret was properly configured and passed to the container as an environment variable, but git wasn't configured to use it for HTTPS authentication.

### Symptom
```
The repository at github.com/rand/paragon either:
1. Doesn't exist at that URL
2. Is a private repository requiring authentication
```

### Root Cause
`entrypoint.sh` didn't configure git credential helper despite `GITHUB_TOKEN` being available.

### Fix Applied
Added to `entrypoint.sh`:
```bash
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
    git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"
fi
```

### Insight
Secrets being *available* as env vars isn't the same as being *usable*. The entrypoint should configure tools to use available credentials.

---

## 3. Mental Model Mismatch

### Issue
Initial assumption was that Parhelia runs commands remotely on the current codebase. Reality: Parhelia dispatches a **new Claude Code instance** that must clone from git.

### Incorrect Mental Model
```
Local Agent → parhelia submit "run tests" → Tests run on my local files remotely
```

### Correct Mental Model
```
Local Agent → parhelia submit "clone X, run tests" → New Claude clones from git, runs tests
```

### Implications
1. **Must push first**: Unpushed changes are invisible to remote Claude
2. **Prompt is instructions**: The prompt becomes the initial task for a fresh Claude session
3. **No shared state**: Remote Claude has no knowledge of local agent's context

### Suggested Improvement
The skills documentation covers this well, but the CLI could reinforce it:
```
$ parhelia submit "run tests"
Warning: Remote Claude cannot see local files. Did you mean to include clone instructions?
Hint: parhelia submit "Clone github.com/org/repo, checkout branch, run tests"
```

---

## 4. Permission Model Blocking Automation

### Issue (Now Fixed)
The `--automated` flag uses `--dangerously-skip-permissions`, which Claude Code blocks when running as root.

### Symptom
```
Claude Code blocks --dangerously-skip-permissions when running as root
```

### Fix Applied
Commit `c3db092`: Run Modal containers as non-root user (`parhelia` user)

### Insight
Security features can conflict with automation. The non-root user approach is the right fix - it maintains security while enabling automation.

---

## 5. Skill Documentation vs CLI Reality

### Issue
The parhelia skills reference commands that don't match the actual CLI:

| Skill Says | Actual Command |
|------------|----------------|
| `parhelia task create "..."` | `parhelia submit "..."` |
| `parhelia session attach task-abc` | Works, but deprecation warning shown |

### Suggested Fix
Align skills documentation with actual CLI commands, or vice versa.

---

## 6. Sync Mode Timeout Ambiguity

### Issue
When using `--sync`, there's no clear indication of:
- Expected wait time
- Whether the task is actually running
- How to gracefully cancel if needed

### Observation
The command just hangs with minimal feedback until completion or timeout.

### Suggested Improvement
```
$ parhelia submit "..." --sync
Task submitted: task-abc123
Waiting for completion... (Ctrl+C to detach, task continues running)
[2m 34s] Status: RUNNING - Claude is working...
[5m 12s] Status: RUNNING - Last activity: "Running cargo check"
```

---

## 7. Workflow for Agents

### Recommended Pattern
Based on experience, here's what works for agent-driven Parhelia usage:

```bash
# 1. Ensure changes are pushed
git push origin my-branch

# 2. Submit with explicit clone instructions
parhelia submit "Clone https://github.com/org/repo, checkout my-branch, run cargo test" --sync --automated

# 3. If sync times out, can still check status
parhelia task show <task-id>
parhelia logs <task-id>
```

### Anti-Patterns Discovered
1. Assuming remote can see local files
2. Using `parhelia task create` (doesn't exist)
3. Forgetting to push before submitting
4. Using `--sync` for long-running tasks without timeout awareness

---

## 8. Positive Observations

### What Works Well
1. **`--automated` flag**: Once non-root was fixed, this works perfectly for CI/agent use
2. **Skill documentation**: The conceptual explanations are excellent (task-dispatch.md especially)
3. **Error messages**: Generally clear about what went wrong
4. **Modal integration**: Seamless container creation and management

### What Agents Need
1. **Predictable commands**: No deprecated commands in help that don't exist
2. **Self-configuring credentials**: If a secret is available, configure tools to use it
3. **Clear feedback loops**: Status updates during sync operations
4. **Explicit mental model reinforcement**: CLI should remind that remote Claude clones from git

---

## Appendix: Session Timeline

| Time | Action | Result |
|------|--------|--------|
| T+0 | Read parhelia skills | Understood conceptual model |
| T+1 | Tried `parhelia task create` | Failed (command doesn't exist) |
| T+2 | Used `parhelia submit` | Worked, but git auth failed |
| T+3 | Investigated entrypoint.sh | Found missing git credential config |
| T+4 | Added git URL rewrite rules | Fixed |
| T+5 | Redeployed Modal app | Success |
| T+6 | Retried with `--sync --automated` | Working |
