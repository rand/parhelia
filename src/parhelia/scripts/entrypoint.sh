#!/bin/bash
# Parhelia container entrypoint script
# Implements [SPEC-01.13] Environment Initialization
#
# This script runs on container startup to:
# 1. Symlink Claude config from volume
# 2. Verify Claude Code installation
# 3. Start MCP servers (if configured)
# 4. Initialize tmux session
# 5. Signal readiness to orchestrator

set -euo pipefail

# Configuration
VOLUME_ROOT="${PARHELIA_VOLUME_ROOT:-/vol/parhelia}"
VOLUME_CONFIG="${VOLUME_ROOT}/config/claude"
WORKSPACE_DIR="${VOLUME_ROOT}/workspaces"
READY_FILE="/tmp/ready"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"

log() {
    echo "[parhelia-entrypoint] $(date -Iseconds) $*"
}

error() {
    echo "[parhelia-entrypoint] ERROR: $*" >&2
}

# =============================================================================
# 1. Configure git credentials for private repo access
# =============================================================================
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    log "Configuring git credential helper with GITHUB_TOKEN..."
    # Use URL rewrite to inject token for all github.com HTTPS URLs
    git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"
    git config --global url."https://x-access-token:${GITHUB_TOKEN}@github.com/".insteadOf "git@github.com:"
    log "Git configured for private repo access"
else
    log "Warning: GITHUB_TOKEN not set, private repos will not be accessible"
fi

# =============================================================================
# 2. Link configuration
# =============================================================================
log "Linking Claude configuration from volume..."

if [[ -d "${VOLUME_CONFIG}" ]]; then
    ln -sfn "${VOLUME_CONFIG}" ~/.claude
    log "Linked ~/.claude -> ${VOLUME_CONFIG}"
else
    log "Warning: Volume config not found at ${VOLUME_CONFIG}, creating empty config"
    mkdir -p ~/.claude
fi

# =============================================================================
# 3. Verify Claude Code installation
# =============================================================================
log "Verifying Claude Code installation..."

if [[ -x "${CLAUDE_BIN}" ]]; then
    CLAUDE_VERSION=$("${CLAUDE_BIN}" --version 2>&1 || echo "unknown")
    log "Claude Code version: ${CLAUDE_VERSION}"
elif command -v claude &> /dev/null; then
    CLAUDE_VERSION=$(claude --version 2>&1 || echo "unknown")
    log "Claude Code version: ${CLAUDE_VERSION}"
else
    error "Claude Code not found at ${CLAUDE_BIN} or in PATH"
    exit 1
fi

# =============================================================================
# 4. Start MCP servers (if launcher exists)
# =============================================================================
if command -v parhelia-mcp-launcher &> /dev/null; then
    log "Starting MCP servers..."
    parhelia-mcp-launcher &
    MCP_PID=$!
    log "MCP launcher started (PID: ${MCP_PID})"
else
    log "MCP launcher not found, skipping MCP server startup"
fi

# =============================================================================
# 5. Initialize tmux session
# =============================================================================
log "Initializing tmux session..."

# Ensure workspace directory exists
mkdir -p "${WORKSPACE_DIR}"

# Kill any existing tmux server (clean slate)
tmux kill-server 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s main -c "${WORKSPACE_DIR}"
log "tmux session 'main' created in ${WORKSPACE_DIR}"

# =============================================================================
# 6. Signal ready
# =============================================================================
log "Signaling readiness..."
echo "PARHELIA_READY" > "${READY_FILE}"
log "Container ready (${READY_FILE} written)"

# =============================================================================
# 7. Idle monitoring and auto-termination
# =============================================================================
IDLE_TIMEOUT_MINUTES="${PARHELIA_IDLE_TIMEOUT:-30}"
IDLE_CHECK_INTERVAL=60  # Check every 60 seconds

get_tmux_activity() {
    # Get last activity time from tmux (format: Unix timestamp)
    tmux display-message -p -t main '#{session_activity}' 2>/dev/null || echo "0"
}

check_idle_and_terminate() {
    local last_activity
    local current_time
    local idle_seconds
    local idle_minutes

    last_activity=$(get_tmux_activity)
    current_time=$(date +%s)
    idle_seconds=$((current_time - last_activity))
    idle_minutes=$((idle_seconds / 60))

    if [[ ${idle_minutes} -ge ${IDLE_TIMEOUT_MINUTES} ]]; then
        log "IDLE TIMEOUT: No activity for ${idle_minutes} minutes (threshold: ${IDLE_TIMEOUT_MINUTES})"
        log "Auto-terminating container to prevent cost overrun"

        # Create a checkpoint marker before exit
        echo "IDLE_TERMINATED at $(date -Iseconds)" > /tmp/termination_reason

        # Clean shutdown
        tmux kill-server 2>/dev/null || true
        exit 0
    fi
}

# Keep container alive with idle monitoring
if [[ "${PARHELIA_INTERACTIVE:-false}" != "true" ]]; then
    log "Running in non-interactive mode with idle monitoring (timeout: ${IDLE_TIMEOUT_MINUTES}min)"

    while true; do
        sleep ${IDLE_CHECK_INTERVAL}
        check_idle_and_terminate
    done
fi
