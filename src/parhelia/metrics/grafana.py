"""Grafana dashboard configuration.

Implements:
- [SPEC-06.15] Grafana Dashboard Configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GrafanaDashboardConfig:
    """Configuration for a Grafana dashboard.

    Implements [SPEC-06.15].
    """

    title: str = "Parhelia Dashboard"
    uid: str = "parhelia-dashboard"
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"
    tags: list[str] = field(default_factory=lambda: ["parhelia"])


class GrafanaPanelBuilder:
    """Builder for Grafana dashboard panels.

    Implements [SPEC-06.15].
    """

    _panel_id = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._panel_id += 1
        return cls._panel_id

    @classmethod
    def _base_panel(
        cls,
        title: str,
        panel_type: str,
        grid_pos: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create base panel structure."""
        return {
            "id": cls._next_id(),
            "title": title,
            "type": panel_type,
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
            "datasource": {"type": "prometheus", "uid": "prometheus"},
        }

    @classmethod
    def gauge(
        cls,
        title: str,
        metric: str,
        unit: str = "none",
        thresholds: list[dict] | None = None,
        grid_pos: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create a gauge panel.

        Args:
            title: Panel title.
            metric: Prometheus metric query.
            unit: Display unit.
            thresholds: Color thresholds.
            grid_pos: Grid position.

        Returns:
            Panel configuration dict.
        """
        panel = cls._base_panel(title, "gauge", grid_pos)
        panel["targets"] = [
            {
                "expr": metric,
                "refId": "A",
            }
        ]
        panel["fieldConfig"] = {
            "defaults": {
                "unit": unit,
                "thresholds": {
                    "mode": "absolute",
                    "steps": thresholds
                    or [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 70},
                        {"color": "red", "value": 90},
                    ],
                },
            }
        }
        return panel

    @classmethod
    def timeseries(
        cls,
        title: str,
        metric: str,
        unit: str = "none",
        legend: str | None = None,
        grid_pos: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create a timeseries panel.

        Args:
            title: Panel title.
            metric: Prometheus metric query.
            unit: Display unit.
            legend: Legend format.
            grid_pos: Grid position.

        Returns:
            Panel configuration dict.
        """
        panel = cls._base_panel(title, "timeseries", grid_pos)
        panel["targets"] = [
            {
                "expr": metric,
                "legendFormat": legend or "{{container_id}}",
                "refId": "A",
            }
        ]
        panel["fieldConfig"] = {
            "defaults": {
                "unit": unit,
            }
        }
        return panel

    @classmethod
    def stat(
        cls,
        title: str,
        metric: str,
        unit: str = "none",
        color_mode: str = "value",
        grid_pos: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create a stat panel.

        Args:
            title: Panel title.
            metric: Prometheus metric query.
            unit: Display unit.
            color_mode: Color mode.
            grid_pos: Grid position.

        Returns:
            Panel configuration dict.
        """
        panel = cls._base_panel(title, "stat", grid_pos)
        panel["targets"] = [
            {
                "expr": metric,
                "refId": "A",
            }
        ]
        panel["options"] = {
            "colorMode": color_mode,
            "graphMode": "area",
            "justifyMode": "auto",
        }
        panel["fieldConfig"] = {
            "defaults": {
                "unit": unit,
            }
        }
        return panel

    @classmethod
    def table(
        cls,
        title: str,
        metrics: list[str],
        grid_pos: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create a table panel.

        Args:
            title: Panel title.
            metrics: List of Prometheus metrics.
            grid_pos: Grid position.

        Returns:
            Panel configuration dict.
        """
        panel = cls._base_panel(title, "table", grid_pos)
        panel["targets"] = [
            {"expr": metric, "refId": chr(65 + i), "format": "table"}
            for i, metric in enumerate(metrics)
        ]
        panel["transformations"] = [
            {"id": "merge", "options": {}},
        ]
        return panel


def generate_overview_dashboard() -> dict[str, Any]:
    """Generate the main Parhelia overview dashboard.

    Implements [SPEC-06.15].

    Returns:
        Dashboard configuration dict.
    """
    # Reset panel ID counter
    GrafanaPanelBuilder._panel_id = 0

    config = GrafanaDashboardConfig(
        title="Parhelia Overview",
        uid="parhelia-overview",
    )

    panels = [
        # Row 1: Key metrics
        GrafanaPanelBuilder.stat(
            title="Active Sessions",
            metric="sum(parhelia_sessions_active)",
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
        ),
        GrafanaPanelBuilder.stat(
            title="Session Capacity",
            metric="sum(parhelia_sessions_capacity)",
            grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
        ),
        GrafanaPanelBuilder.stat(
            title="Total Cost/Hour",
            metric="sum(parhelia_cost_per_hour_usd)",
            unit="currencyUSD",
            grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
        ),
        GrafanaPanelBuilder.stat(
            title="Environments",
            metric="count(parhelia_cpu_total_cores)",
            grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
        ),
        # Row 2: Gauges
        GrafanaPanelBuilder.gauge(
            title="CPU Usage",
            metric="avg(parhelia_cpu_usage_percent)",
            unit="percent",
            grid_pos={"x": 0, "y": 4, "w": 8, "h": 6},
        ),
        GrafanaPanelBuilder.gauge(
            title="Memory Usage",
            metric="avg(parhelia_memory_usage_percent)",
            unit="percent",
            grid_pos={"x": 8, "y": 4, "w": 8, "h": 6},
        ),
        GrafanaPanelBuilder.gauge(
            title="Session Utilization",
            metric="sum(parhelia_sessions_active) / sum(parhelia_sessions_capacity) * 100",
            unit="percent",
            grid_pos={"x": 16, "y": 4, "w": 8, "h": 6},
        ),
        # Row 3: Time series
        GrafanaPanelBuilder.timeseries(
            title="CPU Usage Over Time",
            metric="parhelia_cpu_usage_percent",
            unit="percent",
            grid_pos={"x": 0, "y": 10, "w": 12, "h": 8},
        ),
        GrafanaPanelBuilder.timeseries(
            title="Memory Usage Over Time",
            metric="parhelia_memory_usage_percent",
            unit="percent",
            grid_pos={"x": 12, "y": 10, "w": 12, "h": 8},
        ),
        # Row 4: Sessions and cost
        GrafanaPanelBuilder.timeseries(
            title="Active Sessions Over Time",
            metric="parhelia_sessions_active",
            grid_pos={"x": 0, "y": 18, "w": 12, "h": 8},
        ),
        GrafanaPanelBuilder.timeseries(
            title="Cost Over Time",
            metric="parhelia_cost_per_hour_usd",
            unit="currencyUSD",
            grid_pos={"x": 12, "y": 18, "w": 12, "h": 8},
        ),
    ]

    return {
        "title": config.title,
        "uid": config.uid,
        "tags": config.tags,
        "timezone": "browser",
        "schemaVersion": 38,
        "refresh": config.refresh,
        "time": {
            "from": config.time_from,
            "to": config.time_to,
        },
        "panels": panels,
    }


def generate_container_dashboard() -> dict[str, Any]:
    """Generate container detail dashboard.

    Implements [SPEC-06.15].

    Returns:
        Dashboard configuration dict.
    """
    # Reset panel ID counter
    GrafanaPanelBuilder._panel_id = 0

    config = GrafanaDashboardConfig(
        title="Parhelia Container Details",
        uid="parhelia-containers",
    )

    panels = [
        GrafanaPanelBuilder.table(
            title="Container Status",
            metrics=[
                "parhelia_cpu_usage_percent",
                "parhelia_memory_usage_percent",
                "parhelia_sessions_active",
                "parhelia_cost_per_hour_usd",
            ],
            grid_pos={"x": 0, "y": 0, "w": 24, "h": 8},
        ),
        GrafanaPanelBuilder.timeseries(
            title="Container CPU Usage",
            metric="parhelia_cpu_usage_percent",
            unit="percent",
            grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
        ),
        GrafanaPanelBuilder.timeseries(
            title="Container Memory Usage",
            metric="parhelia_memory_usage_percent",
            unit="percent",
            grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
        ),
    ]

    return {
        "title": config.title,
        "uid": config.uid,
        "tags": config.tags,
        "timezone": "browser",
        "schemaVersion": 38,
        "refresh": config.refresh,
        "time": {
            "from": config.time_from,
            "to": config.time_to,
        },
        "panels": panels,
    }


def generate_datasource_provisioning(
    prometheus_url: str = "http://prometheus:9090",
) -> dict[str, Any]:
    """Generate Grafana datasource provisioning config.

    Implements [SPEC-06.15].

    Args:
        prometheus_url: URL of Prometheus server.

    Returns:
        Datasource provisioning YAML content as dict.
    """
    return {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "uid": "prometheus",
                "access": "proxy",
                "url": prometheus_url,
                "isDefault": True,
                "editable": False,
            }
        ],
    }


def generate_dashboard_provisioning(
    dashboard_path: str = "/var/lib/grafana/dashboards/parhelia",
) -> dict[str, Any]:
    """Generate Grafana dashboard provisioning config.

    Implements [SPEC-06.15].

    Args:
        dashboard_path: Path to dashboard JSON files.

    Returns:
        Dashboard provisioning YAML content as dict.
    """
    return {
        "apiVersion": 1,
        "providers": [
            {
                "name": "Parhelia",
                "orgId": 1,
                "folder": "Parhelia",
                "type": "file",
                "disableDeletion": False,
                "updateIntervalSeconds": 30,
                "options": {
                    "path": dashboard_path,
                },
            }
        ],
    }


def generate_grafana_compose(
    port: int = 3000,
    include_prometheus: bool = False,
    prometheus_port: int = 9090,
) -> str:
    """Generate Docker Compose for Grafana stack.

    Implements [SPEC-06.15].

    Args:
        port: Grafana port.
        include_prometheus: Whether to include Prometheus service.
        prometheus_port: Prometheus port if included.

    Returns:
        Docker Compose YAML content.
    """
    compose = f"""version: '3.8'

services:
  grafana:
    image: grafana/grafana:latest
    container_name: parhelia-grafana
    ports:
      - "{port}:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./provisioning/dashboards:/etc/grafana/provisioning/dashboards
      - ./dashboards:/var/lib/grafana/dashboards/parhelia
    restart: unless-stopped
    networks:
      - parhelia
"""

    if include_prometheus:
        compose += f"""
  prometheus:
    image: prom/prometheus:latest
    container_name: parhelia-prometheus
    ports:
      - "{prometheus_port}:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - parhelia
"""

    compose += """
networks:
  parhelia:
    driver: bridge

volumes:
  grafana-data:
"""

    if include_prometheus:
        compose += "  prometheus-data:\n"

    return compose
