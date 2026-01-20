"""Parhelia configuration management.

Loads configuration from parhelia.toml with sensible defaults.

Implements:
- [SPEC-07.20.02] Escalation Policies (approval config)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:
    import tomllib as tomli  # Python 3.11+ stdlib
except ImportError:
    import tomli  # Backport for older Python

if TYPE_CHECKING:
    from parhelia.approval import ApprovalConfig


@dataclass
class ModalConfig:
    """Modal-specific configuration."""

    region: Literal["us-east", "us-west", "eu-west"] = "us-east"
    volume_name: str = "parhelia-vol"
    cpu_count: int = 4
    memory_mb: int = 16384
    default_timeout_hours: int = 24


@dataclass
class PathsConfig:
    """Path configuration."""

    volume_root: str = "/vol/parhelia"
    checkpoint_root: str = "/vol/parhelia/checkpoints"
    audit_root: str = "/vol/parhelia/audit"
    cas_root: str | None = None  # Enable CAS mode if set


@dataclass
class ParheliaConfig:
    """Root configuration for Parhelia."""

    modal: ModalConfig = field(default_factory=ModalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    approval: "ApprovalConfig | None" = None  # Lazy loaded to avoid circular import


def load_config(config_path: Path | None = None) -> ParheliaConfig:
    """Load configuration from parhelia.toml.

    Args:
        config_path: Path to config file. If None, searches current directory
                     and parent directories for parhelia.toml.

    Returns:
        ParheliaConfig with values from file or defaults.
    """
    if config_path is None:
        config_path = _find_config_file()

    if config_path is None or not config_path.exists():
        return ParheliaConfig()

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    return _parse_config(data)


def _find_config_file() -> Path | None:
    """Search for parhelia.toml in current and parent directories."""
    current = Path.cwd()

    for directory in [current, *current.parents]:
        config_file = directory / "parhelia.toml"
        if config_file.exists():
            return config_file

    return None


def _parse_config(data: dict) -> ParheliaConfig:
    """Parse configuration dictionary into ParheliaConfig."""
    modal_data = data.get("modal", {})
    paths_data = data.get("paths", {})

    modal_config = ModalConfig(
        region=modal_data.get("region", "us-east"),
        volume_name=modal_data.get("volume_name", "parhelia-vol"),
        cpu_count=modal_data.get("cpu_count", 4),
        memory_mb=modal_data.get("memory_mb", 16384),
        default_timeout_hours=modal_data.get("default_timeout_hours", 24),
    )

    paths_config = PathsConfig(
        volume_root=paths_data.get("volume_root", "/vol/parhelia"),
        checkpoint_root=paths_data.get("checkpoint_root", "/vol/parhelia/checkpoints"),
        audit_root=paths_data.get("audit_root", "/vol/parhelia/audit"),
        cas_root=paths_data.get("cas_root"),
    )

    # Parse approval config if present [SPEC-07.20.02]
    approval_config = None
    if "approval" in data:
        from parhelia.approval import parse_approval_config

        approval_config = parse_approval_config(data["approval"])

    return ParheliaConfig(
        modal=modal_config,
        paths=paths_config,
        approval=approval_config,
    )
