# SOC2 Compliance Documentation

**Version**: 1.0
**Last Updated**: 2026-01-20
**Spec Reference**: SPEC-04.40-47

## Overview

This document describes Parhelia's SOC2 compliance controls, mapping inherited controls from Modal's SOC 2 Type 2 certification to customer-implemented controls required for full compliance.

---

## Control Inheritance Model

### Modal Inherited Controls (SOC 2 Type 2 Certified)

Modal provides SOC 2 Type 2 compliance for infrastructure-level controls:

| Control Area | TSC Reference | Modal Implementation |
|--------------|---------------|---------------------|
| Physical Security | CC6.4 | Data center physical access controls |
| Infrastructure Hardening | CC6.6 | gVisor container isolation, network segmentation |
| Encryption at Rest | CC6.7 | Volume encryption (AES-256) |
| Encryption in Transit | CC6.7 | TLS 1.3 for all API communications |
| Network Security | CC6.1 | Firewall rules, DDoS protection |
| Availability | A1.1-A1.3 | Multi-AZ deployment, automatic failover |
| Compute Isolation | CC6.6 | gVisor sandboxing for container workloads |

**Reference**: [Modal Security Documentation](https://modal.com/docs/guide/security)

### Customer-Implemented Controls (Parhelia)

| Control Area | TSC Reference | Parhelia Implementation | Status |
|--------------|---------------|------------------------|--------|
| Access Control | CC6.1-6.8 | RBAC with admin/operator/viewer | Implemented |
| Authentication | CC6.1 | API key + MFA for human access | Implemented |
| Audit Logging | CC7.2 | Structured JSON logs, append-only | Implemented |
| Security Monitoring | CC7.1, CC7.3 | Prometheus metrics, Grafana alerts | Implemented |
| Incident Response | CC7.4 | Documented IRP with escalation | Documented |
| Change Management | CC8 | PR approval, CI/CD pipeline | Implemented |
| Secrets Management | CC6.7 | Versioned secrets, rotation policy | Implemented |

---

## Trust Services Criteria Coverage

### CC6: Logical and Physical Access Controls

#### CC6.1: Access Control Policies

**Requirement**: The entity implements logical access security measures to protect against unauthorized access.

**Implementation**:

```python
# parhelia/auth.py - Role definitions
class Role(Enum):
    ADMIN = "admin"       # Full system access
    OPERATOR = "operator" # Session management, no config changes
    VIEWER = "viewer"     # Read-only access to logs and metrics

ROLE_PERMISSIONS = {
    Role.ADMIN: ["*"],
    Role.OPERATOR: [
        "session.create", "session.attach", "session.terminate",
        "checkpoint.create", "checkpoint.restore",
        "logs.read", "metrics.read",
    ],
    Role.VIEWER: [
        "session.list", "checkpoint.list",
        "logs.read", "metrics.read",
    ],
}
```

**Evidence Collection**:
- User access logs: `/vol/parhelia/audit/access.jsonl`
- Role assignments: Configuration in `parhelia.toml`
- Access reviews: Quarterly review documented in `docs/compliance/access-reviews/`

#### CC6.2: User Registration and Authorization

**Requirement**: New users are registered and authorized before access is granted.

**Implementation**:
1. Admin creates user in configuration
2. User receives API token via secure channel
3. First login requires MFA enrollment
4. Access logged in audit trail

**Commands**:
```bash
# Admin creates user
parhelia admin user create --email user@company.com --role operator

# User receives token and enrolls MFA
parhelia auth login --enroll-mfa
```

#### CC6.3: User Access Removal

**Requirement**: Access is removed promptly when no longer required.

**Implementation**:
- Immediate revocation via API token invalidation
- Session termination on access removal
- Audit log entry for all revocations

```bash
# Revoke user access
parhelia admin user revoke --email user@company.com --reason "Offboarding"
```

#### CC6.6: Security for Logical Access

**Requirement**: Logical access to system resources is restricted.

**Implementation**:
- gVisor sandboxing (Modal inherited)
- Restricted file system paths
- Network egress allowlist
- Bash command restrictions

```python
# parhelia/permissions.py
ALLOWED_PATHS = [
    "/vol/parhelia/workspaces",
    "/vol/parhelia/checkpoints",
    "/tmp",
]

BLOCKED_COMMANDS = [
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero",
]

ALLOWED_EGRESS_DOMAINS = [
    "api.anthropic.com",
    "github.com",
    "*.githubusercontent.com",
    "pypi.org",
]
```

#### CC6.7: Encryption

**Requirement**: Data is encrypted at rest and in transit.

**Implementation**:
- At rest: Modal Volume encryption (AES-256) - inherited
- In transit: TLS 1.3 for all communications - inherited
- Secrets: Stored in OS keychain locally, Modal Secrets remotely

---

### CC7: System Operations

#### CC7.1: Security Monitoring

**Requirement**: The entity detects and monitors security events.

**Implementation**:
- Real-time metrics via Prometheus Pushgateway
- Grafana dashboards for visualization
- Alert rules for security events

**Metrics Collected**:
```python
# parhelia/metrics.py
security_denied_total = Counter(
    "parhelia_security_denied_total",
    "Total permission denials",
    ["event_type"]
)

auth_failures_total = Counter(
    "parhelia_auth_failures_total",
    "Authentication failures",
    ["reason"]
)

secrets_accessed_total = Counter(
    "parhelia_secrets_accessed_total",
    "Secret access events",
    ["secret_name", "action"]
)
```

**Alert Rules**:
- Authentication failures > 5 in 5 minutes
- Permission denials > 10 in 1 minute
- Network egress to unknown domain
- Dangerous bash command attempted

#### CC7.2: Audit Logging

**Requirement**: The entity maintains comprehensive audit logs.

**Implementation** [SPEC-04.42-44]:

```python
# parhelia/audit.py
@dataclass
class AuditEvent:
    timestamp: datetime           # [SPEC-04.42] Required
    actor: str                    # [SPEC-04.42] User or system
    action: str                   # [SPEC-04.42] CREATE/READ/UPDATE/DELETE
    resource: str                 # [SPEC-04.42] Target resource
    resource_type: str            # session, checkpoint, secret, etc.
    source_ip: str | None         # [SPEC-04.42] Client IP
    result: str                   # success, denied, error
    details: dict[str, Any]       # Additional context
    event_hash: str               # SHA-256 for integrity
    previous_hash: str            # Chain to previous event
```

**Audit Log Requirements**:
- **Retention**: Minimum 12 months [SPEC-04.43]
- **Immutability**: Append-only storage [SPEC-04.44]
- **Format**: JSONL for easy querying
- **Location**: `/vol/parhelia/audit/audit.jsonl`

**Audited Events**:
| Event Type | Description |
|------------|-------------|
| `auth.login` | User authentication |
| `auth.logout` | User session end |
| `auth.mfa_enroll` | MFA enrollment |
| `auth.failure` | Authentication failure |
| `secret.access` | Secret retrieved |
| `secret.inject` | Secret injected to container |
| `secret.rotate` | Secret rotation |
| `session.create` | New session started |
| `session.attach` | User attached to session |
| `session.detach` | User detached from session |
| `session.terminate` | Session ended |
| `checkpoint.create` | Checkpoint created |
| `checkpoint.restore` | Checkpoint restored |
| `permission.denied` | Access denied |
| `bash.dangerous` | Dangerous command blocked |
| `network.egress_blocked` | Egress blocked |

#### CC7.3: Security Incident Response

**Requirement**: The entity has procedures for responding to security incidents.

**Implementation**: See [Incident Response Plan](#incident-response-plan) below.

#### CC7.4: Security Event Evaluation

**Requirement**: The entity evaluates security events to determine potential incidents.

**Implementation**:
- Automated alert correlation
- Weekly security review of audit logs
- Documented triage procedures

---

### CC8: Change Management

#### CC8.1: Change Control Procedures

**Requirement** [SPEC-04.46]: Changes to infrastructure and applications require approval.

**Implementation**:
1. All changes via pull request
2. Automated CI/CD testing required
3. Approval from code owner required
4. Staging deployment before production
5. Rollback procedures documented

**Change Control Workflow**:
```
Developer → PR → CI Tests → Code Review → Staging → Production
                    ↓           ↓
              (automated)  (human approval)
```

**Evidence**:
- GitHub PR history with approvals
- CI/CD logs in GitHub Actions
- Deployment logs

---

## Incident Response Plan

### Overview [SPEC-04.45]

This plan defines procedures for responding to security incidents affecting Parhelia.

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P0 - Critical | Active breach, data exfiltration | 15 minutes | Credential compromise, data leak |
| P1 - High | Potential breach, degraded security | 1 hour | Suspicious activity, auth bypass |
| P2 - Medium | Security anomaly, policy violation | 4 hours | Failed auth spike, config drift |
| P3 - Low | Informational, best practice deviation | 24 hours | Audit finding, documentation gap |

### Escalation Path

```
L1: On-call Engineer
    ↓ (P1/P0 or unresolved 30 min)
L2: Security Lead
    ↓ (P0 or unresolved 1 hour)
L3: Engineering Director + Legal
    ↓ (data breach confirmed)
L4: Executive Team + External Counsel
```

### Response Procedures

#### Detection
1. Automated alerts trigger PagerDuty
2. On-call acknowledges within SLA
3. Initial triage determines severity

#### Containment
1. Isolate affected systems
2. Revoke compromised credentials
3. Preserve evidence (audit logs, metrics)

#### Eradication
1. Identify root cause
2. Remove malicious access/code
3. Patch vulnerabilities

#### Recovery
1. Restore from known-good state
2. Verify integrity
3. Monitor for recurrence

#### Post-Incident
1. Document timeline and actions
2. Conduct retrospective
3. Update procedures as needed
4. File incident report in `docs/compliance/incidents/`

### Contact Information

| Role | Contact | Escalation |
|------|---------|------------|
| On-Call Engineer | PagerDuty rotation | Slack #security-oncall |
| Security Lead | security@company.com | Direct page |
| Engineering Director | eng-director@company.com | Phone |

---

## Evidence Collection Procedures

### Continuous Collection

| Evidence Type | Source | Retention | Location |
|---------------|--------|-----------|----------|
| Audit Logs | AuditLogger | 12 months | `/vol/parhelia/audit/` |
| Access Logs | Auth system | 12 months | `/vol/parhelia/audit/access.jsonl` |
| Metrics | Prometheus | 90 days | Grafana/Prometheus |
| Change Records | GitHub | Indefinite | GitHub repository |
| Session Logs | Claude Code | 30 days | `/vol/parhelia/logs/` |

### Periodic Collection

| Evidence Type | Frequency | Procedure | Output |
|---------------|-----------|-----------|--------|
| Access Reviews | Quarterly | Review user roles and permissions | `docs/compliance/access-reviews/YYYY-QN.md` |
| Security Scans | Monthly | Run vulnerability scanner | `docs/compliance/scans/YYYY-MM.md` |
| Incident Reviews | Per incident | Post-incident retrospective | `docs/compliance/incidents/YYYY-MM-DD-title.md` |
| Control Testing | Annually | Test all controls | `docs/compliance/control-tests/YYYY.md` |

---

## MFA Requirements [SPEC-04.41]

### Scope

MFA is required for:
- All human access to Parhelia CLI
- Admin access to Modal dashboard (inherited)
- Access to audit logs
- Access to production systems

### Implementation

```python
# parhelia/auth.py
class MFAManager:
    """Manage multi-factor authentication."""

    def enroll(self, user: User) -> TOTPSecret:
        """Enroll user in TOTP-based MFA."""
        secret = pyotp.random_base32()
        # Store encrypted in user profile
        return TOTPSecret(secret=secret, user_id=user.id)

    def verify(self, user: User, code: str) -> bool:
        """Verify TOTP code."""
        totp = pyotp.TOTP(user.mfa_secret)
        return totp.verify(code, valid_window=1)
```

### Supported Methods

| Method | Status | Notes |
|--------|--------|-------|
| TOTP (Authenticator app) | Supported | Primary method |
| Hardware keys (WebAuthn) | Planned | Future enhancement |
| SMS | Not supported | Security concerns |

---

## RBAC Implementation [SPEC-04.40]

### Role Definitions

| Role | Description | Typical Users |
|------|-------------|---------------|
| Admin | Full system access, user management | System administrators |
| Operator | Session management, checkpoint operations | DevOps engineers |
| Viewer | Read-only access to logs and metrics | Auditors, observers |

### Permission Matrix

| Action | Admin | Operator | Viewer |
|--------|-------|----------|--------|
| User management | Yes | No | No |
| System configuration | Yes | No | No |
| Session create | Yes | Yes | No |
| Session attach | Yes | Yes | No |
| Session terminate | Yes | Yes | No |
| Checkpoint create | Yes | Yes | No |
| Checkpoint restore | Yes | Yes | No |
| View logs | Yes | Yes | Yes |
| View metrics | Yes | Yes | Yes |
| Export audit logs | Yes | No | Yes |
| Secrets management | Yes | No | No |

### Configuration

```toml
# parhelia.toml
[auth.roles]
admin = ["admin@company.com"]
operator = ["dev1@company.com", "dev2@company.com"]
viewer = ["auditor@company.com"]
```

---

## Compliance Checklist

### Initial Setup

- [x] RBAC roles defined and implemented
- [x] MFA enrollment required for human access
- [x] Audit logging enabled
- [x] Secrets stored securely
- [x] Network egress allowlist configured
- [x] Incident response plan documented
- [x] Change management workflow established

### Ongoing Operations

- [ ] Quarterly access reviews
- [ ] Monthly security scans
- [ ] Annual control testing
- [ ] Incident retrospectives as needed
- [ ] Audit log retention verified

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-20 | Parhelia Team | Initial release |

---

## References

- [SPEC-04: Security Model](../spec/SPEC-04-security-model.md)
- [Modal SOC 2 Type 2 Report](https://modal.com/security)
- [AICPA Trust Services Criteria](https://www.aicpa.org/resources/landing/trust-services-criteria)
- [SOC 2 Compliance Guide](https://secureframe.com/hub/soc-2)
