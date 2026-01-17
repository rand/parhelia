"""Parhelia configuration management.

Loads configuration from parhelia.toml with sensible defaults.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib as tomli  # Python 3.11+ stdlib
except ImportError:
    import tomli  # Backport for older Python


@dataclass
class ModalConfig:
    """Modal-specific configuration."""

    region: Literal["us-east", "us-west", "eu-west"] = "us-east"
    volume_name: str = "parhelia-vol"
    cpu_count: int = 4
    memory_mb: int = 16384
    default_timeout_hours: int = 24


@dataclass
class ParheliaConfig:
    """Root configuration for Parhelia."""

    modal: ModalConfig = field(default_factory=ModalConfig)


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

    modal_config = ModalConfig(
        region=modal_data.get("region", "us-east"),
        volume_name=modal_data.get("volume_name", "parhelia-vol"),
        cpu_count=modal_data.get("cpu_count", 4),
        memory_mb=modal_data.get("memory_mb", 16384),
        default_timeout_hours=modal_data.get("default_timeout_hours", 24),
    )

    return ParheliaConfig(modal=modal_config)
