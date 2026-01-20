"""Parhelia configuration management.

Loads configuration from parhelia.toml with sensible defaults.

Implements:
- [SPEC-07.20.02] Escalation Policies (approval config)
- [SPEC-07.21] Notification Service (notification config)
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
    from parhelia.notification import NotificationConfig


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
class BudgetConfig:
    """Budget configuration [SPEC-05.14]."""

    default_ceiling_usd: float = 10.0  # Default budget ceiling per session
    warning_threshold: float = 0.8  # Warn at 80% usage
    max_cost_per_task: float = 5.0  # Max cost for single task
    max_daily_cost: float = 100.0  # Daily spending limit


@dataclass
class ParheliaConfig:
    """Root configuration for Parhelia."""

    modal: ModalConfig = field(default_factory=ModalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)  # [SPEC-05.14]
    approval: "ApprovalConfig | None" = None  # Lazy loaded to avoid circular import
    notifications: "NotificationConfig | None" = None  # [SPEC-07.21]


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
    budget_data = data.get("budget", {})

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

    budget_config = BudgetConfig(
        default_ceiling_usd=budget_data.get("default_ceiling_usd", 10.0),
        warning_threshold=budget_data.get("warning_threshold", 0.8),
        max_cost_per_task=budget_data.get("max_cost_per_task", 5.0),
        max_daily_cost=budget_data.get("max_daily_cost", 100.0),
    )

    # Parse approval config if present [SPEC-07.20.02]
    approval_config = None
    if "approval" in data:
        from parhelia.approval import parse_approval_config

        approval_config = parse_approval_config(data["approval"])

    # Parse notification config if present [SPEC-07.21]
    notification_config = None
    if "notifications" in data:
        from parhelia.notification import parse_notification_config

        notification_config = parse_notification_config(data["notifications"])

    return ParheliaConfig(
        modal=modal_config,
        paths=paths_config,
        budget=budget_config,
        approval=approval_config,
        notifications=notification_config,
    )
