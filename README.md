# Parhelia

Remote Claude Code execution system using Modal.com.

## Overview

Parhelia enables running Claude Code with full configuration (plugins, skills, CLAUDE.md, MCP servers) in Modal.com containers with:

- tmux-based session management
- Checkpoint/resume for resilience
- Resource capacity broadcasting
- Dynamic parallel dispatch

## Installation

```bash
uv pip install -e ".[dev]"
```

## Usage

See `docs/spec/` for detailed specifications.
