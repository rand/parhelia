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
# 1. Link configuration
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
# 2. Verify Claude Code installation
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
# 3. Start MCP servers (if launcher exists)
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
# 4. Initialize tmux session
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
# 5. Signal ready
# =============================================================================
log "Signaling readiness..."
echo "PARHELIA_READY" > "${READY_FILE}"
log "Container ready (${READY_FILE} written)"

# Keep container alive (if not running interactively)
if [[ "${PARHELIA_INTERACTIVE:-false}" != "true" ]]; then
    log "Running in non-interactive mode, waiting..."
    # Wait indefinitely (container will be terminated by orchestrator)
    exec tail -f /dev/null
fi
