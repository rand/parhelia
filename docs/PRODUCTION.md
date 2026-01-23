# Parhelia Production Deployment Guide

This guide covers deploying and operating Parhelia in production environments.

## Prerequisites

- Modal account with active billing
- Anthropic API key with sufficient credits
- GitHub repository for CI/CD (optional but recommended)

## 1. Modal Setup

### 1.1 Create Modal Secrets

Parhelia requires two secrets configured in your Modal dashboard:

```bash
# Create Anthropic API key secret
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...

# Create GitHub token (for repo access in sandboxes)
modal secret create github-token GITHUB_TOKEN=ghp_...
```

### 1.2 Deploy the Application

```bash
# Deploy to Modal
modal deploy src/parhelia/modal_app.py

# Verify deployment
modal app list | grep parhelia
```

### 1.3 Initialize Volume Structure

```bash
# Create required directories on the volume
modal run src/parhelia/modal_app.py::init_volume_structure
```

### 1.4 Verify Health

```bash
# Run health check
modal run src/parhelia/modal_app.py::health_check
```

Expected output:
```json
{
  "status": "ok",
  "volume_mounted": true,
  "claude_installed": true,
  "anthropic_key_set": true,
  "claude_version": "Claude Code v1.x.x"
}
```

## 2. GitHub Actions Setup

### 2.1 Required Secrets

Add these secrets to your GitHub repository (Settings > Secrets and variables > Actions):

| Secret | Description |
|--------|-------------|
| `MODAL_TOKEN_ID` | Your Modal token ID |
| `MODAL_TOKEN_SECRET` | Your Modal token secret |

To get Modal tokens:
```bash
modal token new
```

### 2.2 Workflows

Parhelia includes two GitHub Actions workflows:

1. **Image Rebuild** (`.github/workflows/image-rebuild.yml`)
   - Weekly rebuild to capture Claude Code updates
   - Manual trigger available
   - Creates issue on failure

2. **E2E Tests** (`.github/workflows/e2e-modal.yml`)
   - Daily integration tests
   - Smoke tests on PRs touching Modal code
   - Creates issue on failure

## 3. Cost Management

### 3.1 Compute Costs

| Resource | Cost/sec | Cost/hr | Usage |
|----------|----------|---------|-------|
| CPU sandbox | ~$0.0001 | ~$0.35 | Default for most tasks |
| A10G GPU | ~$0.0003 | ~$1.10 | ML workloads |
| A100 GPU | ~$0.0007 | ~$2.50 | Large model inference |
| H100 GPU | ~$0.0011 | ~$4.00 | Maximum performance |
| Volume storage | - | ~$0.20/GB/month | Checkpoints, configs |

*Pricing from [Modal.com](https://modal.com/pricing). Per-second billing means you only pay for active compute.*

### 3.2 Estimated Monthly Costs

| Usage Pattern | Est. Monthly Cost |
|---------------|-------------------|
| Light (10 tasks/day, CPU) | $3-10 |
| Medium (50 tasks/day, CPU) | $15-50 |
| Heavy (100 tasks/day, mixed) | $50-150 |

*Assumes ~10 min average task duration. GPU tasks cost 3-10x more than CPU.*

### 3.3 Cost Controls

1. **Set Spending Limits** in Modal dashboard (Settings > Spending)
2. **Monitor Usage** via Modal dashboard (Usage tab)
3. **Sandbox Timeouts**: Default 1 hour, configurable per task
4. **E2E Test Costs**: ~$3-15/month for daily runs

### 3.4 Anthropic API Costs

Claude Code usage is billed separately through your Anthropic account:
- Claude Sonnet: ~$3/million input tokens, ~$15/million output tokens
- Monitor via Anthropic Console

## 4. Monitoring & Alerting

### 4.1 Built-in CLI Monitoring

Parhelia provides comprehensive CLI commands for monitoring:

```bash
# System health overview
parhelia status

# Container monitoring
parhelia container list                    # List all containers
parhelia container list --state running    # Filter by state
parhelia container health                  # Health summary
parhelia container watch                   # Real-time updates

# Event monitoring
parhelia events list                       # Recent events
parhelia events list --level error         # Errors only
parhelia events watch                      # Real-time stream

# Reconciler status
parhelia reconciler status                 # Drift detection state
parhelia reconciler run                    # Force reconciliation

# Budget monitoring
parhelia budget show                       # Current spending
parhelia budget history                    # Usage over time
```

### 4.2 Modal Dashboard

- **Health Check**: `modal run src/parhelia/modal_app.py::health_check`
- **E2E Tests**: Daily automated validation
- **GitHub Issues**: Auto-created on CI failures

### 4.3 Recommended Alerts

Set up in Modal dashboard:
1. Spending threshold alerts (e.g., 80% of budget)
2. Error rate alerts (sandbox failures)
3. Function timeout alerts

### 4.4 Log Access

```bash
# View recent function logs
modal app logs parhelia

# View specific function logs
modal app logs parhelia --filter health_check
```

## 5. Security Considerations

### 5.1 API Key Security

- Never commit API keys to git
- Use Modal secrets for all sensitive values
- Rotate keys periodically (recommended: quarterly)
- Use separate API keys for dev/staging/prod

### 5.2 Network Security

- Sandboxes run in isolated Modal infrastructure
- Outbound internet access is allowed by default
- No inbound access except through Modal's proxy

### 5.3 Code Execution

- Claude Code runs arbitrary commands in sandboxes
- Sandboxes are ephemeral (destroyed after timeout)
- Volume data persists across sandbox restarts
- Consider: What code/repos should Claude access?

### 5.4 Access Control

- Modal tokens grant full account access
- GitHub tokens should be scoped to required repos only
- Consider using GitHub App tokens for fine-grained control

## 6. Troubleshooting

### 6.1 Common Issues

#### Sandbox creation fails
```
Error: Function has not been hydrated
```
**Solution**: Ensure the app is deployed: `modal deploy src/parhelia/modal_app.py`

#### Claude Code not found
```
claude_installed: false
```
**Solution**: Rebuild the image: trigger workflow manually or redeploy

#### Volume not mounted
```
volume_mounted: false
```
**Solution**: Check volume name in config matches Modal volume

#### API key not set
```
anthropic_key_set: false
```
**Solution**: Recreate secret: `modal secret create anthropic-api-key ANTHROPIC_API_KEY=...`

### 6.2 Debug Mode

Use CLI commands to inspect system state:
```bash
# Check system health
parhelia status

# View container state
parhelia container list
parhelia container show <container-id>

# Check events for issues
parhelia events list --level error

# View reconciler status
parhelia reconciler status
```

### 6.3 Manual Health Check

```python
import asyncio
from parhelia.modal_app import create_claude_sandbox, run_in_sandbox

async def debug():
    sandbox = await create_claude_sandbox("debug-test")
    output = await run_in_sandbox(sandbox, ["echo", "hello"])
    print(output)

asyncio.run(debug())
```

## 7. Backup & Recovery

### 7.1 Volume Backup

Modal volumes are persistent but not automatically backed up. For critical data:

```bash
# Export checkpoint data (run in sandbox)
tar -czf /vol/parhelia/backup-$(date +%Y%m%d).tar.gz /vol/parhelia/checkpoints

# Download backup
modal volume get parhelia-data /backup-*.tar.gz ./backups/
```

### 7.2 Configuration Backup

Keep `.parhelia/` directory in version control:
```
.parhelia/
├── config.toml      # Local configuration
└── orchestrator.db  # Task history (optional)
```

### 7.3 Recovery Procedures

1. **App Recovery**: Redeploy from git
2. **Volume Recovery**: Restore from backup
3. **Secret Recovery**: Recreate from secure storage

## 8. Scaling Considerations

### 8.1 Concurrent Sandboxes

- Default: 10 concurrent sandboxes
- Configurable via `SandboxManager.max_sandboxes`
- Modal limit: varies by plan

### 8.2 Queue Management

For high-volume workloads:
1. Use `--no-dispatch` to queue without executing
2. Batch dispatch with `dispatch_pending()`
3. Monitor queue depth via task store

### 8.3 GPU Workloads

GPU sandboxes have limited availability:
- A10G: Generally available
- A100: May have queuing
- H100: Limited availability

Consider fallback logic for GPU unavailability.

## 9. Version Management

### 9.1 Image Versioning

Images are tagged on deployment:
- `weekly-YYYYMMDD`: Weekly rebuilds
- `v1.0.0`: Manual version tags

```bash
# Deploy with specific tag
modal deploy src/parhelia/modal_app.py --tag v1.0.0
```

### 9.2 Rollback

```bash
# List available tags
modal app history parhelia

# Rollback (redeploy previous version)
git checkout v1.0.0
modal deploy src/parhelia/modal_app.py --tag rollback-$(date +%Y%m%d)
```

## 10. MCP Integration

Parhelia exposes 24 MCP tools for programmatic integration:

```bash
# Start MCP server
parhelia mcp-server
```

### 10.1 Available Tools

| Category | Tools |
|----------|-------|
| **Tasks** | `parhelia_task_create`, `parhelia_task_list`, `parhelia_task_show`, `parhelia_task_dispatch` |
| **Containers** | `parhelia_containers`, `parhelia_container_show`, `parhelia_container_events` |
| **Events** | `parhelia_events_list`, `parhelia_events_subscribe`, `parhelia_events_stream` |
| **Checkpoints** | `parhelia_checkpoint_create`, `parhelia_checkpoint_list`, `parhelia_checkpoint_restore` |
| **Budget** | `parhelia_budget_status`, `parhelia_budget_estimate` |
| **System** | `parhelia_health`, `parhelia_reconciler_status` |

### 10.2 Claude Code Integration

Add Parhelia to Claude Code's MCP config for AI-assisted task management:

```json
{
  "mcpServers": {
    "parhelia": {
      "command": "parhelia",
      "args": ["mcp-server"]
    }
  }
}
```

## 11. Support

- **Issues**: https://github.com/rand/parhelia/issues
- **Modal Docs**: https://modal.com/docs
- **Anthropic Docs**: https://docs.anthropic.com
