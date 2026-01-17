# SPEC-04: Security Model and Trust Boundaries

**Status**: Draft
**Issue**: ph-kuv
**Author**: Claude + rand
**Date**: 2026-01-16

## Overview

This specification defines security boundaries, authentication, authorization, and secrets management for Parhelia's local-remote execution model.

## Goals

- [SPEC-04.01] Define clear trust boundaries between local and remote environments
- [SPEC-04.02] Secure secrets injection without exposure in logs or checkpoints
- [SPEC-04.03] Implement permission model for remote Claude Code execution
- [SPEC-04.04] Provide audit logging for security-relevant operations
- [SPEC-04.05] Enable secure interactive attachment without credential leakage

## Non-Goals

- Multi-tenant isolation (single-user system for v1)
- Hardware security modules (HSM)
- Compliance certifications (SOC2, HIPAA) - future consideration

---

## Architecture

### Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRUST ZONE 1: LOCAL ENVIRONMENT                          │
│                         (Full Trust - User's Machine)                        │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  ~/.claude/     │  │  SSH Keys       │  │  Local          │             │
│  │  Full Config    │  │  ~/.ssh/        │  │  Secrets        │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Parhelia Orchestrator                            │   │
│  │  - Holds Modal API token                                             │   │
│  │  - Decides what secrets to inject remotely                           │   │
│  │  - Controls task dispatch                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                     [Authenticated TLS Channel]
                     [Modal API + SSH Tunnel]
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRUST ZONE 2: MODAL ENVIRONMENT                          │
│                      (Sandboxed Trust - Ephemeral)                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     gVisor Container Sandbox                         │   │
│  │                                                                       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │   │
│  │  │  Injected       │  │  Claude Code    │  │  Workspace      │      │   │
│  │  │  Secrets Only   │  │  (sandboxed)    │  │  (Volume)       │      │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘      │   │
│  │                                                                       │   │
│  │  Constraints:                                                         │   │
│  │  - No access to local filesystem                                      │   │
│  │  - No access to local network                                         │   │
│  │  - Secrets injected via env vars only                                 │   │
│  │  - Network egress controlled                                          │   │
│  │  - Ephemeral - destroyed on termination                               │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Authentication Flow

```
┌─────────┐     ┌─────────────┐     ┌─────────┐     ┌─────────────┐
│  User   │     │ Orchestrator│     │  Modal  │     │  Container  │
└────┬────┘     └──────┬──────┘     └────┬────┘     └──────┬──────┘
     │                 │                  │                 │
     │  parhelia auth  │                  │                 │
     │────────────────▶│                  │                 │
     │                 │                  │                 │
     │                 │  modal token     │                 │
     │                 │  create          │                 │
     │                 │─────────────────▶│                 │
     │                 │                  │                 │
     │                 │◀─────────────────│                 │
     │                 │  token stored    │                 │
     │◀────────────────│                  │                 │
     │                 │                  │                 │
     │  dispatch task  │                  │                 │
     │────────────────▶│                  │                 │
     │                 │                  │                 │
     │                 │  spawn container │                 │
     │                 │  + inject secrets│                 │
     │                 │─────────────────▶│                 │
     │                 │                  │                 │
     │                 │                  │  create         │
     │                 │                  │────────────────▶│
     │                 │                  │                 │
     │                 │                  │  ANTHROPIC_API  │
     │                 │                  │  _KEY injected  │
     │                 │                  │────────────────▶│
```

---

## Requirements

### [SPEC-04.10] Secrets Classification

Secrets MUST be classified by sensitivity and injection scope:

| Classification | Examples | Local Storage | Remote Injection |
|---------------|----------|---------------|------------------|
| **Critical** | `ANTHROPIC_API_KEY` | Encrypted keychain | Yes, via Modal Secrets |
| **Sensitive** | `GITHUB_TOKEN`, DB passwords | Encrypted keychain | Yes, selective |
| **Project** | API keys for specific projects | Project `.env` | Yes, per-project |
| **Internal** | `MODAL_TOKEN_ID` | Config file | No - orchestrator only |

```python
@dataclass
class Secret:
    name: str
    classification: SecretClassification
    inject_remote: bool = True
    redact_in_logs: bool = True
    include_in_checkpoint: bool = False  # NEVER for secrets

class SecretClassification(Enum):
    CRITICAL = "critical"    # System-wide, highly sensitive
    SENSITIVE = "sensitive"  # Important but replaceable
    PROJECT = "project"      # Project-specific
    INTERNAL = "internal"    # Parhelia internal use only
```

### [SPEC-04.11] Secrets Storage (Local)

Secrets MUST be stored securely on the local machine:

```python
class LocalSecretStore:
    """Store secrets using OS keychain."""

    def __init__(self):
        # Use keyring for cross-platform keychain access
        import keyring
        self.keyring = keyring

    def store_secret(self, name: str, value: str, classification: SecretClassification):
        """Store secret in OS keychain."""
        service_name = f"parhelia-{classification.value}"
        self.keyring.set_password(service_name, name, value)

    def get_secret(self, name: str, classification: SecretClassification) -> str | None:
        """Retrieve secret from OS keychain."""
        service_name = f"parhelia-{classification.value}"
        return self.keyring.get_password(service_name, name)

    def delete_secret(self, name: str, classification: SecretClassification):
        """Remove secret from keychain."""
        service_name = f"parhelia-{classification.value}"
        self.keyring.delete_password(service_name, name)
```

### [SPEC-04.12] Secrets Injection (Remote)

Secrets MUST be injected via Modal's Secrets API:

```python
class RemoteSecretInjector:
    """Inject secrets into Modal containers."""

    async def prepare_secrets_for_task(
        self,
        task: Task,
        project_config: ProjectConfig,
    ) -> list[modal.Secret]:
        """Determine which secrets to inject for a task."""
        secrets = []

        # Always inject Anthropic API key
        secrets.append(modal.Secret.from_name("anthropic-api-key"))

        # Inject project-specific secrets
        if project_config.required_secrets:
            for secret_name in project_config.required_secrets:
                secrets.append(modal.Secret.from_name(secret_name))

        # Inject GitHub token if task involves git operations
        if task.requires_git:
            secrets.append(modal.Secret.from_name("github-token"))

        return secrets

    async def sync_secrets_to_modal(self, secrets: dict[str, str]):
        """Sync local secrets to Modal's secret store."""
        import modal

        for name, value in secrets.items():
            # Create or update Modal secret
            modal.Secret.create_or_update(
                name=name,
                env_dict={name.upper().replace("-", "_"): value}
            )
```

**Security constraints**:
- Secrets are injected as environment variables
- Secrets are NOT written to Volume
- Secrets are NOT included in checkpoints
- Secrets are NOT logged

### [SPEC-04.13] Permission Model for Remote Execution

Remote Claude Code MUST operate under restricted permissions:

```python
@dataclass
class RemotePermissions:
    """Permissions granted to remote Claude Code instance."""

    # Tool permissions
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Read", "Write", "Edit", "Glob", "Grep",
        "Bash", "Task", "TodoWrite",
    ])
    denied_tools: list[str] = field(default_factory=lambda: [
        "WebFetch",  # Controlled via allowlist
    ])

    # Bash restrictions
    bash_allow_network: bool = True
    bash_allow_sudo: bool = False
    bash_blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /",
        "mkfs",
        "dd if=/dev/zero",
        # Add dangerous commands
    ])

    # File access
    allowed_paths: list[str] = field(default_factory=lambda: [
        "/vol/parhelia/workspaces",
        "/vol/parhelia/checkpoints",
        "/tmp",
    ])
    denied_paths: list[str] = field(default_factory=lambda: [
        "/etc/shadow",
        "/etc/passwd",
    ])

    # Network egress
    allowed_domains: list[str] = field(default_factory=lambda: [
        "api.anthropic.com",
        "github.com",
        "*.githubusercontent.com",
        "pypi.org",
        "registry.npmjs.org",
    ])

def build_claude_command(task: Task, permissions: RemotePermissions) -> list[str]:
    """Build Claude Code command with permission restrictions."""
    cmd = [
        "claude",
        "-p", task.prompt,
        "--output-format", "stream-json",
        "--allowedTools", ",".join(permissions.allowed_tools),
    ]

    # In sandboxed Modal environment, we can skip interactive permissions
    if task.trust_level == TrustLevel.AUTOMATED:
        cmd.append("--dangerously-skip-permissions")

    return cmd
```

### [SPEC-04.14] SSH Tunnel Security

Interactive attachment MUST use secure SSH configuration:

```python
class SSHTunnelManager:
    """Manage secure SSH tunnels for interactive attachment."""

    async def setup_ssh_server(self, container_id: str) -> SSHConfig:
        """Configure SSH server in container."""

        # Generate ephemeral host key (new each container)
        await run_command(["ssh-keygen", "-A"])

        # Configure sshd for security
        sshd_config = """
# Parhelia SSH Configuration
Port 2222
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthorizedKeysFile /vol/parhelia/config/authorized_keys
X11Forwarding no
AllowTcpForwarding yes
GatewayPorts no
PermitTunnel no
MaxAuthTries 3
LoginGraceTime 30
ClientAliveInterval 30
ClientAliveCountMax 3
"""
        await write_file("/etc/ssh/sshd_config.d/parhelia.conf", sshd_config)

        return SSHConfig(
            port=2222,
            host_key_fingerprint=await self.get_host_key_fingerprint(),
        )

    async def authorize_user(self, public_key: str):
        """Add user's public key to authorized_keys."""
        authorized_keys_path = "/vol/parhelia/config/authorized_keys"

        # Append public key (idempotent)
        async with aiofiles.open(authorized_keys_path, "a") as f:
            await f.write(f"{public_key}\n")
```

### [SPEC-04.15] Audit Logging

Security-relevant operations MUST be logged:

```python
@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str
    session_id: str | None
    user: str
    action: str
    resource: str
    outcome: str  # "success", "denied", "error"
    details: dict[str, Any]
    source_ip: str | None

class AuditLogger:
    """Log security-relevant events."""

    AUDITED_EVENTS = [
        "auth.login",
        "auth.logout",
        "secret.access",
        "secret.inject",
        "session.create",
        "session.attach",
        "session.detach",
        "permission.denied",
        "bash.dangerous_command",
        "network.egress_blocked",
        "checkpoint.create",
        "checkpoint.restore",
    ]

    async def log(self, event: AuditEvent):
        """Log audit event to secure audit log."""
        # Structured JSON logging
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.event_type,
            "session": event.session_id,
            "user": event.user,
            "action": event.action,
            "resource": event.resource,
            "outcome": event.outcome,
            "details": self._redact_secrets(event.details),
            "source_ip": event.source_ip,
        }

        # Write to append-only audit log
        async with aiofiles.open(
            "/vol/parhelia/audit/audit.jsonl",
            mode="a"
        ) as f:
            await f.write(json.dumps(log_entry) + "\n")

        # Also emit to metrics for monitoring
        if event.outcome == "denied":
            metrics.security_denied_total.inc(
                labels={"event_type": event.event_type}
            )

    def _redact_secrets(self, details: dict) -> dict:
        """Redact any secrets from audit log details."""
        redacted = {}
        for key, value in details.items():
            if any(s in key.lower() for s in ["key", "token", "secret", "password"]):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted
```

### [SPEC-04.16] Checkpoint Security

Checkpoints MUST NOT contain secrets:

```python
class SecureCheckpointManager:
    """Create checkpoints without leaking secrets."""

    REDACT_ENV_VARS = [
        "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "DATABASE_PASSWORD",
        # Pattern-based redaction
    ]

    async def create_secure_checkpoint(self, session: Session) -> Checkpoint:
        """Create checkpoint with secrets redacted."""

        # Capture environment, redacting secrets
        env = {}
        for key, value in session.environment.items():
            if self._is_secret(key):
                env[key] = "[REDACTED]"
            else:
                env[key] = value

        # Capture conversation, redacting any leaked secrets
        conversation = await self._capture_conversation(session)
        conversation = self._redact_conversation(conversation)

        return Checkpoint(
            # ... other fields
            environment=env,
            conversation=conversation,
        )

    def _is_secret(self, key: str) -> bool:
        """Check if environment variable is a secret."""
        key_lower = key.lower()
        secret_patterns = ["key", "token", "secret", "password", "credential"]
        return any(p in key_lower for p in secret_patterns)

    def _redact_conversation(self, conversation: ConversationState) -> ConversationState:
        """Redact any secrets that may have leaked into conversation."""
        # Pattern-based redaction for common secret formats
        patterns = [
            r"sk-[a-zA-Z0-9]{48}",  # Anthropic API key
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub token
            r"AKIA[A-Z0-9]{16}",     # AWS access key
        ]

        for message in conversation.messages:
            for pattern in patterns:
                message.content = re.sub(pattern, "[REDACTED]", message.content)

        return conversation
```

---

## Security Checklist

### Before Deployment

- [ ] Modal API token stored in OS keychain, not plain text
- [ ] Anthropic API key stored as Modal Secret, not in code
- [ ] SSH keys generated fresh, not reused from local
- [ ] Audit logging enabled and writing to secure location
- [ ] Network egress allowlist configured

### Per-Session

- [ ] Secrets injected only for required scope
- [ ] Permissions restricted to task requirements
- [ ] Checkpoint created without secrets
- [ ] Audit events logged for sensitive operations

### Periodic Review

- [ ] Rotate Modal API token quarterly
- [ ] Review audit logs for anomalies
- [ ] Update network egress allowlist
- [ ] Prune old checkpoints

---

## Acceptance Criteria

- [ ] [SPEC-04.AC1] Secrets stored in OS keychain, not plain files
- [ ] [SPEC-04.AC2] Secrets injected via Modal Secrets API only
- [ ] [SPEC-04.AC3] Secrets never appear in logs or checkpoints
- [ ] [SPEC-04.AC4] SSH tunnel uses ephemeral keys
- [ ] [SPEC-04.AC5] Audit log captures all security events
- [ ] [SPEC-04.AC6] Dangerous bash commands blocked
- [ ] [SPEC-04.AC7] Network egress restricted to allowlist

---

## Resolved Questions

### 1. Secret Rotation Without Disrupting Running Sessions

**Problem**: Modal's Secrets API does not support in-place updates—you must delete and recreate secrets.

**Solution**: Versioned secrets with dual-key overlap periods.

```python
# Versioned secret naming
SECRET_NAME = "anthropic-api-key-v{version}"

# Rotation workflow:
# 1. Create new secret: anthropic-api-key-v2
# 2. Deploy new functions referencing v2
# 3. Wait for overlap period (max session duration + buffer)
# 4. Delete old secret: anthropic-api-key-v1

class SecretRefreshManager:
    """Refresh secrets in long-running sessions."""

    REFRESH_INTERVAL = 300  # 5 minutes

    async def refresh_loop(self, session: Session):
        """Periodically check for rotated secrets."""
        while session.state == SessionState.RUNNING:
            await asyncio.sleep(self.REFRESH_INTERVAL)

            # Re-read secret from Modal
            new_key = await self.fetch_current_secret("anthropic-api-key")
            if new_key != session.cached_api_key:
                session.cached_api_key = new_key
                logger.info(f"Refreshed API key for session {session.id}")
```

**Requirements**:
- [SPEC-04.20] Secrets MUST use versioned names: `{name}-v{N}`
- [SPEC-04.21] Dual-key overlap period MUST exceed max session duration (24h + 1h buffer)
- [SPEC-04.22] Long-running sessions (>1 hour) MUST implement secret refresh callbacks
- [SPEC-04.23] Secret rotation events MUST be logged with correlation IDs

### 2. Multi-Tenant Security Changes

**For future multi-tenant support, these isolation boundaries are required:**

| Layer | Single-Tenant (v1) | Multi-Tenant (Future) |
|-------|-------------------|----------------------|
| Compute | gVisor sandbox | gVisor sandbox (no change) |
| Secrets | Flat namespace | Tenant-prefixed: `tenant-{id}-{name}` |
| Volumes | Single Volume | Per-tenant Volume: `vol-{tenant_id}` |
| Network | Allowlist egress | Per-tenant allowlists |
| Audit | Session-level | Tenant ID on every log entry |
| Identity | User tokens | Tenant-scoped OAuth |

**Migration path to multi-tenancy**:

```python
# v1: Single tenant
@app.function(
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    volumes={"/vol/parhelia": volume},
)

# v2: Multi-tenant
@app.function(
    secrets=[modal.Secret.from_name(f"tenant-{tenant_id}-anthropic-api-key")],
    volumes={f"/vol/parhelia/{tenant_id}": get_tenant_volume(tenant_id)},
)
```

**Requirements for multi-tenant**:
- [SPEC-04.30] Tenant ID MUST propagate through all requests
- [SPEC-04.31] Secrets MUST be prefixed with tenant ID
- [SPEC-04.32] Each tenant MUST have dedicated Volume
- [SPEC-04.33] Cross-tenant resource access MUST be a critical security violation
- [SPEC-04.34] All audit logs MUST include tenant_id field

### 3. SOC2 Compliance Controls

**Modal provides SOC 2 Type 2 compliance**, which gives Parhelia inherited controls for:
- Physical security and infrastructure hardening
- gVisor container isolation
- Encryption at rest and in transit
- Network security

**Customer-implemented controls required for SOC2**:

| Control Area | TSC Reference | Parhelia Implementation |
|--------------|---------------|------------------------|
| Access Control | CC6.1-6.8 | RBAC with admin/operator/viewer roles |
| Audit Logging | CC7.2 | Structured logs to immutable storage |
| Security Monitoring | CC7.1, CC7.3 | Real-time alerts, weekly dashboard review |
| Incident Response | CC7.4 | Documented IRP with escalation paths |
| Change Management | CC8 | PR approval, staging tests, deployment workflow |

**Audit log requirements (CC7.2)**:

```python
@dataclass
class SOC2AuditEvent(AuditEvent):
    """Extended audit event for SOC2 compliance."""

    # Required fields for SOC2
    tenant_id: str              # For multi-tenant (use "default" for v1)
    action_type: str            # CREATE, READ, UPDATE, DELETE
    resource_type: str          # session, secret, checkpoint, etc.
    previous_state: str | None  # For UPDATE/DELETE
    new_state: str | None       # For CREATE/UPDATE

    # Immutability
    event_hash: str             # SHA-256 of event content
    previous_hash: str          # Chain to previous event

# Retention: 12 months minimum
# Storage: S3 with Object Lock (WORM)
# Access: Read-only API for tenants
```

**Requirements for SOC2 readiness**:
- [SPEC-04.40] Implement RBAC with minimum roles: admin, operator, viewer
- [SPEC-04.41] Require MFA for all human access
- [SPEC-04.42] Audit logs MUST include: actor, action, resource, timestamp, IP, result
- [SPEC-04.43] Audit log retention MUST be minimum 12 months
- [SPEC-04.44] Audit logs MUST be immutable (append-only storage)
- [SPEC-04.45] Document incident response plan with escalation paths
- [SPEC-04.46] All production changes MUST require PR approval
- [SPEC-04.47] Maintain mapping of Modal's inherited controls vs customer controls

---

## Open Questions

None remaining for v1 scope.

---

## References

### Modal Documentation
- [Modal Secrets Documentation](https://modal.com/docs/guide/secrets)
- [Modal Secrets API Reference](https://modal.com/docs/reference/modal.Secret)
- [Modal gVisor Sandboxing](https://modal.com/docs/reference/sandbox)
- [Modal Security and Privacy](https://modal.com/docs/guide/security)

### Secret Rotation
- [AWS Secrets Manager Rotation Strategies](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotation-strategy.html)
- [HashiCorp Vault Lease Management](https://developer.hashicorp.com/vault/docs/concepts/lease)
- [API Key Rotation Best Practices - GitGuardian](https://blog.gitguardian.com/api-key-rotation-best-practices/)

### Multi-Tenant Security
- [HashiCorp Vault Namespaces](https://developer.hashicorp.com/vault/tutorials/enterprise/namespaces)
- [Multi-Tenant Isolation with Calico](https://www.tigera.io/blog/deep-dive/implementing-tenant-isolation-in-multi-tenant-kubernetes-clusters/)
- [gVisor vs Kata vs Firecracker Comparison](https://dev.to/agentsphere/choosing-a-workspace-for-ai-agents-the-ultimate-showdown-between-gvisor-kata-and-firecracker-b10)

### SOC 2 Compliance
- [SOC 2 Trust Services Criteria](https://secureframe.com/hub/soc-2/trust-services-criteria)
- [SOC 2 CC6 Access Controls](https://www.designcs.net/soc-2-cc6-common-criteria-related-to-logical-and-physical-access/)
- [SOC 2 Incident Response Requirements](https://fractionalciso.com/soc-2-incident-response-whats-required-for-compliance/)
- [SOC 2 Change Management](https://www.thoropass.com/blog/best-practices-for-soc-2-change-management)
- [AWS Shared Responsibility for SOC 2](https://secureframe.com/blog/aws-soc-2-report)

### General Security
- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Claude Code Sandboxing](https://docs.anthropic.com/en/docs/claude-code/sandboxing)
