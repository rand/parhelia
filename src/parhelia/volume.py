"""Volume management and pre-warming.

Implements [SPEC-01.15] Cold Start Optimization - Volume Pre-warming.
"""

from __future__ import annotations

import os
from pathlib import Path


def prewarm_volume(volume_path: str) -> None:
    """Pre-warm volume with standard directory structure.

    Implements [SPEC-01.15] - Volume pre-warming for cold start optimization.

    Creates the standard Parhelia volume directory structure if it doesn't exist.
    This is idempotent and safe to call multiple times - existing files are preserved.

    Args:
        volume_path: Root path of the Parhelia volume.

    Directory structure created:
        {volume_path}/
        ├── config/
        │   └── claude/           # ~/.claude contents
        ├── plugins/              # Cloned plugin repos
        ├── skills/               # Cloned skill repos
        ├── checkpoints/          # Session state persistence
        ├── workspaces/           # Cloned project repos
        └── cas/                  # Content-addressable storage
            ├── blobs/
            │   └── sha256/
            ├── trees/
            └── actions/
    """
    root = Path(volume_path)

    # Standard directories
    directories = [
        "config/claude",
        "config/env",
        "plugins",
        "skills",
        "checkpoints",
        "workspaces",
        "cas/blobs/sha256",
        "cas/trees",
        "cas/actions/sha256",
    ]

    for dir_path in directories:
        full_path = root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)


def get_volume_stats(volume_path: str) -> dict:
    """Get statistics about volume usage.

    Args:
        volume_path: Root path of the Parhelia volume.

    Returns:
        Dict with volume statistics.
    """
    root = Path(volume_path)

    def dir_size(path: Path) -> int:
        """Calculate total size of directory."""
        if not path.exists():
            return 0
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
        return total

    def file_count(path: Path) -> int:
        """Count files in directory."""
        if not path.exists():
            return 0
        return sum(1 for _ in path.rglob("*") if _.is_file())

    return {
        "total_size_bytes": dir_size(root),
        "config_size_bytes": dir_size(root / "config"),
        "plugins_size_bytes": dir_size(root / "plugins"),
        "skills_size_bytes": dir_size(root / "skills"),
        "checkpoints_size_bytes": dir_size(root / "checkpoints"),
        "workspaces_size_bytes": dir_size(root / "workspaces"),
        "cas_size_bytes": dir_size(root / "cas"),
        "checkpoint_count": file_count(root / "checkpoints"),
        "workspace_count": len(list((root / "workspaces").iterdir())) if (root / "workspaces").exists() else 0,
    }


def cleanup_old_checkpoints(
    volume_path: str,
    max_age_days: int = 30,
    keep_min: int = 5,
) -> list[str]:
    """Clean up old checkpoint files.

    Implements retention policy for checkpoints to prevent unbounded growth.

    Args:
        volume_path: Root path of the Parhelia volume.
        max_age_days: Delete checkpoints older than this.
        keep_min: Always keep at least this many checkpoints per session.

    Returns:
        List of deleted checkpoint paths.
    """
    import time

    root = Path(volume_path)
    checkpoints_dir = root / "checkpoints"

    if not checkpoints_dir.exists():
        return []

    deleted = []
    now = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    # Group checkpoints by session
    sessions: dict[str, list[Path]] = {}
    for session_dir in checkpoints_dir.iterdir():
        if session_dir.is_dir():
            checkpoints = sorted(
                session_dir.iterdir(),
                key=lambda p: p.stat().st_mtime,
                reverse=True,  # Newest first
            )
            sessions[session_dir.name] = checkpoints

    # Clean up old checkpoints, keeping minimum per session
    for session_id, checkpoints in sessions.items():
        for i, checkpoint in enumerate(checkpoints):
            # Always keep minimum
            if i < keep_min:
                continue

            # Check age
            age = now - checkpoint.stat().st_mtime
            if age > max_age_seconds:
                # Delete old checkpoint
                if checkpoint.is_file():
                    checkpoint.unlink()
                elif checkpoint.is_dir():
                    import shutil
                    shutil.rmtree(checkpoint)
                deleted.append(str(checkpoint))

    return deleted
