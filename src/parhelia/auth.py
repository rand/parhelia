"""Authentication and authorization for Parhelia MCP server.

Implements secure token-based authentication following MCP authorization spec.

Supports:
- Token-based authentication (Bearer tokens)
- Environment variable configuration
- Audit logging of authenticated requests
- Scope-based authorization (future)

Usage:
    # Environment setup
    export PARHELIA_AUTH_TOKENS="token1,token2"  # Comma-separated valid tokens
    export PARHELIA_AUTH_REQUIRED=true           # Require auth (default: false for stdio)

    # In MCP server
    auth = AuthManager()
    if not auth.validate_token(token):
        raise AuthenticationError("Invalid token")
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AuthScope(str, Enum):
    """Authorization scopes for MCP tools."""

    READ = "read"  # Read-only operations (list, show, status)
    WRITE = "write"  # Mutating operations (create, delete, update)
    ADMIN = "admin"  # Administrative operations (cleanup, terminate, config)
    BUDGET = "budget"  # Budget operations (set ceiling, estimate)

    @classmethod
    def all(cls) -> set[AuthScope]:
        """Return all scopes."""
        return {cls.READ, cls.WRITE, cls.ADMIN, cls.BUDGET}


# Tool to scope mapping
TOOL_SCOPES: dict[str, set[AuthScope]] = {
    # Read-only tools
    "parhelia_containers": {AuthScope.READ},
    "parhelia_container_show": {AuthScope.READ},
    "parhelia_container_events": {AuthScope.READ},
    "parhelia_health": {AuthScope.READ},
    "parhelia_reconciler_status": {AuthScope.READ},
    "parhelia_task_list": {AuthScope.READ},
    "parhelia_task_show": {AuthScope.READ},
    "parhelia_session_list": {AuthScope.READ},
    "parhelia_session_attach_info": {AuthScope.READ},
    "parhelia_checkpoint_list": {AuthScope.READ},
    "parhelia_budget_status": {AuthScope.READ},
    "parhelia_budget_estimate": {AuthScope.READ},
    # Write tools
    "parhelia_task_create": {AuthScope.WRITE},
    "parhelia_task_cancel": {AuthScope.WRITE},
    "parhelia_task_retry": {AuthScope.WRITE},
    "parhelia_checkpoint_create": {AuthScope.WRITE},
    "parhelia_checkpoint_restore": {AuthScope.WRITE},
    # Admin tools
    "parhelia_container_terminate": {AuthScope.ADMIN},
    "parhelia_session_kill": {AuthScope.ADMIN},
    # Budget tools (separate scope)
    "parhelia_budget_set": {AuthScope.BUDGET},
}


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class AuthorizationError(Exception):
    """Raised when authorization fails (valid token, insufficient scope)."""

    pass


@dataclass
class AuthToken:
    """Authenticated token with metadata."""

    token_hash: str  # SHA-256 hash of token (never store plaintext)
    scopes: set[AuthScope]
    created_at: datetime
    expires_at: datetime | None = None
    name: str | None = None  # Optional human-readable name
    last_used_at: datetime | None = None
    use_count: int = 0

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def has_scope(self, scope: AuthScope) -> bool:
        """Check if token has the given scope."""
        return scope in self.scopes

    def has_any_scope(self, scopes: set[AuthScope]) -> bool:
        """Check if token has any of the given scopes."""
        return bool(self.scopes & scopes)


@dataclass
class AuditLogEntry:
    """Audit log entry for MCP requests."""

    timestamp: datetime
    tool_name: str
    token_name: str | None  # Name or hash prefix if anonymous
    success: bool
    client_info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_ms: float | None = None


class AuthManager:
    """Manages authentication and authorization for MCP server.

    Token Storage:
    - Tokens from PARHELIA_AUTH_TOKENS env var (comma-separated)
    - Tokens are stored as SHA-256 hashes
    - Never logs or stores plaintext tokens

    Security:
    - Constant-time token comparison to prevent timing attacks
    - Audit logging of all authentication attempts
    - Token expiration support
    """

    def __init__(
        self,
        require_auth: bool | None = None,
        tokens: list[str] | None = None,
        audit_log_path: str | None = None,
    ):
        """Initialize auth manager.

        Args:
            require_auth: Whether auth is required. If None, uses PARHELIA_AUTH_REQUIRED env.
            tokens: List of valid tokens. If None, uses PARHELIA_AUTH_TOKENS env.
            audit_log_path: Path for audit log. If None, uses PARHELIA_AUDIT_LOG env.
        """
        # Determine if auth is required
        if require_auth is None:
            require_auth = os.environ.get("PARHELIA_AUTH_REQUIRED", "").lower() in (
                "true",
                "1",
                "yes",
            )
        self.require_auth = require_auth

        # Load tokens
        self._tokens: dict[str, AuthToken] = {}
        if tokens:
            for token in tokens:
                self._add_token(token)
        else:
            env_tokens = os.environ.get("PARHELIA_AUTH_TOKENS", "")
            if env_tokens:
                for token in env_tokens.split(","):
                    token = token.strip()
                    if token:
                        self._add_token(token)

        # Audit logging
        self.audit_log_path = audit_log_path or os.environ.get("PARHELIA_AUDIT_LOG")
        self._audit_entries: list[AuditLogEntry] = []

    def _hash_token(self, token: str) -> str:
        """Hash a token using SHA-256."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _add_token(
        self,
        token: str,
        scopes: set[AuthScope] | None = None,
        name: str | None = None,
        expires_in: timedelta | None = None,
    ) -> None:
        """Add a token to the valid tokens set.

        Args:
            token: The plaintext token
            scopes: Authorized scopes (default: all)
            name: Human-readable name for audit logs
            expires_in: Token lifetime (None for no expiration)
        """
        token_hash = self._hash_token(token)
        expires_at = datetime.now() + expires_in if expires_in else None

        self._tokens[token_hash] = AuthToken(
            token_hash=token_hash,
            scopes=scopes or AuthScope.all(),
            created_at=datetime.now(),
            expires_at=expires_at,
            name=name,
        )

    def validate_token(self, token: str | None) -> AuthToken | None:
        """Validate a token and return its metadata.

        Uses constant-time comparison to prevent timing attacks.

        Args:
            token: The token to validate

        Returns:
            AuthToken if valid, None if invalid
        """
        if not token:
            return None

        token_hash = self._hash_token(token)

        # Use constant-time comparison
        for stored_hash, auth_token in self._tokens.items():
            if hmac.compare_digest(token_hash, stored_hash):
                if auth_token.is_expired():
                    logger.warning(f"Expired token used: {stored_hash[:8]}...")
                    return None

                # Update usage stats
                auth_token.last_used_at = datetime.now()
                auth_token.use_count += 1

                return auth_token

        return None

    def check_auth(
        self,
        token: str | None,
        required_scopes: set[AuthScope] | None = None,
    ) -> AuthToken:
        """Check authentication and authorization.

        Args:
            token: The token to validate
            required_scopes: Required scopes for this operation

        Returns:
            AuthToken if valid

        Raises:
            AuthenticationError: If token is invalid
            AuthorizationError: If token lacks required scopes
        """
        # If auth not required and no token provided, return anonymous token
        if not self.require_auth and not token:
            return AuthToken(
                token_hash="anonymous",
                scopes=AuthScope.all(),
                created_at=datetime.now(),
                name="anonymous",
            )

        # Validate token
        auth_token = self.validate_token(token)
        if not auth_token:
            raise AuthenticationError("Invalid or expired token")

        # Check scopes
        if required_scopes and not auth_token.has_any_scope(required_scopes):
            raise AuthorizationError(
                f"Token lacks required scopes: {required_scopes}"
            )

        return auth_token

    def check_tool_auth(self, token: str | None, tool_name: str) -> AuthToken:
        """Check if token is authorized for a specific tool.

        Args:
            token: The authentication token
            tool_name: The MCP tool being called

        Returns:
            AuthToken if authorized

        Raises:
            AuthenticationError: If token is invalid
            AuthorizationError: If token lacks required scope for tool
        """
        required_scopes = TOOL_SCOPES.get(tool_name, {AuthScope.READ})
        return self.check_auth(token, required_scopes)

    def log_request(
        self,
        tool_name: str,
        auth_token: AuthToken | None,
        success: bool,
        error: str | None = None,
        duration_ms: float | None = None,
        client_info: dict[str, Any] | None = None,
    ) -> None:
        """Log an MCP request for audit trail.

        Args:
            tool_name: The tool that was called
            auth_token: The authenticated token (or None)
            success: Whether the request succeeded
            error: Error message if failed
            duration_ms: Request duration in milliseconds
            client_info: Additional client information
        """
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            tool_name=tool_name,
            token_name=auth_token.name or (auth_token.token_hash[:8] + "...") if auth_token else "anonymous",
            success=success,
            error=error,
            duration_ms=duration_ms,
            client_info=client_info or {},
        )

        self._audit_entries.append(entry)

        # Keep only last 1000 entries in memory
        if len(self._audit_entries) > 1000:
            self._audit_entries = self._audit_entries[-1000:]

        # Log to file if configured
        if self.audit_log_path:
            try:
                import json

                with open(self.audit_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": entry.timestamp.isoformat(),
                                "tool": entry.tool_name,
                                "token": entry.token_name,
                                "success": entry.success,
                                "error": entry.error,
                                "duration_ms": entry.duration_ms,
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

        # Also log to standard logger
        log_msg = f"MCP {tool_name} by {entry.token_name}: {'OK' if success else 'FAILED'}"
        if error:
            log_msg += f" - {error}"
        if duration_ms:
            log_msg += f" ({duration_ms:.1f}ms)"

        if success:
            logger.info(log_msg)
        else:
            logger.warning(log_msg)

    def get_audit_log(self, limit: int = 100) -> list[AuditLogEntry]:
        """Get recent audit log entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent audit entries
        """
        return self._audit_entries[-limit:]

    def generate_token(
        self,
        scopes: set[AuthScope] | None = None,
        name: str | None = None,
        expires_in: timedelta | None = None,
    ) -> str:
        """Generate a new secure token.

        Args:
            scopes: Authorized scopes (default: all)
            name: Human-readable name
            expires_in: Token lifetime

        Returns:
            The generated token (store securely, shown only once)
        """
        token = secrets.token_urlsafe(32)
        self._add_token(token, scopes, name, expires_in)
        return token

    def revoke_token(self, token: str) -> bool:
        """Revoke a token.

        Args:
            token: The token to revoke

        Returns:
            True if token was revoked, False if not found
        """
        token_hash = self._hash_token(token)
        if token_hash in self._tokens:
            del self._tokens[token_hash]
            logger.info(f"Token revoked: {token_hash[:8]}...")
            return True
        return False

    @property
    def token_count(self) -> int:
        """Number of valid tokens."""
        return len(self._tokens)

    @property
    def is_auth_enabled(self) -> bool:
        """Whether authentication is enabled (tokens configured or required)."""
        return self.require_auth or self.token_count > 0
