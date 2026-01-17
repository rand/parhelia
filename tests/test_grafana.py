"""Tests for Grafana dashboard configuration - SPEC-06.15."""

from __future__ import annotations

import json

import pytest


class TestGrafanaDashboardConfig:
    """Tests for Grafana dashboard configuration."""

    def test_dashboard_config_creation(self):
        """@trace SPEC-06.15 - Dashboard config MUST be creatable."""
        from parhelia.metrics.grafana import GrafanaDashboardConfig

        config = GrafanaDashboardConfig(
            title="Parhelia Overview",
            uid="parhelia-overview",
        )

        assert config.title == "Parhelia Overview"
        assert config.uid == "parhelia-overview"

    def test_dashboard_config_defaults(self):
        """@trace SPEC-06.15 - Dashboard config SHOULD have sensible defaults."""
        from parhelia.metrics.grafana import GrafanaDashboardConfig

        config = GrafanaDashboardConfig()

        assert config.refresh == "30s"
        assert config.time_from == "now-1h"
        assert config.time_to == "now"


class TestGrafanaPanelBuilder:
    """Tests for Grafana panel building."""

    def test_gauge_panel_creation(self):
        """@trace SPEC-06.15 - Builder MUST create gauge panels."""
        from parhelia.metrics.grafana import GrafanaPanelBuilder

        panel = GrafanaPanelBuilder.gauge(
            title="CPU Usage",
            metric="parhelia_cpu_usage_percent",
            unit="percent",
        )

        assert panel["title"] == "CPU Usage"
        assert panel["type"] == "gauge"
        assert "targets" in panel

    def test_timeseries_panel_creation(self):
        """@trace SPEC-06.15 - Builder MUST create timeseries panels."""
        from parhelia.metrics.grafana import GrafanaPanelBuilder

        panel = GrafanaPanelBuilder.timeseries(
            title="Memory Over Time",
            metric="parhelia_memory_usage_percent",
            unit="percent",
        )

        assert panel["title"] == "Memory Over Time"
        assert panel["type"] == "timeseries"

    def test_stat_panel_creation(self):
        """@trace SPEC-06.15 - Builder MUST create stat panels."""
        from parhelia.metrics.grafana import GrafanaPanelBuilder

        panel = GrafanaPanelBuilder.stat(
            title="Active Sessions",
            metric="parhelia_sessions_active",
        )

        assert panel["title"] == "Active Sessions"
        assert panel["type"] == "stat"

    def test_table_panel_creation(self):
        """@trace SPEC-06.15 - Builder MUST create table panels."""
        from parhelia.metrics.grafana import GrafanaPanelBuilder

        panel = GrafanaPanelBuilder.table(
            title="Container Status",
            metrics=["parhelia_cpu_usage_percent", "parhelia_memory_usage_percent"],
        )

        assert panel["title"] == "Container Status"
        assert panel["type"] == "table"


class TestGrafanaDashboardGenerator:
    """Tests for Grafana dashboard JSON generation."""

    def test_generate_overview_dashboard(self):
        """@trace SPEC-06.15 - Generator MUST create overview dashboard."""
        from parhelia.metrics.grafana import generate_overview_dashboard

        dashboard = generate_overview_dashboard()

        assert dashboard["title"] == "Parhelia Overview"
        assert "panels" in dashboard
        assert len(dashboard["panels"]) > 0

    def test_overview_dashboard_has_cpu_panel(self):
        """@trace SPEC-06.15 - Overview MUST include CPU utilization."""
        from parhelia.metrics.grafana import generate_overview_dashboard

        dashboard = generate_overview_dashboard()
        titles = [p["title"] for p in dashboard["panels"]]

        assert any("CPU" in t for t in titles)

    def test_overview_dashboard_has_memory_panel(self):
        """@trace SPEC-06.15 - Overview MUST include memory utilization."""
        from parhelia.metrics.grafana import generate_overview_dashboard

        dashboard = generate_overview_dashboard()
        titles = [p["title"] for p in dashboard["panels"]]

        assert any("Memory" in t for t in titles)

    def test_overview_dashboard_has_session_panel(self):
        """@trace SPEC-06.15 - Overview MUST include session counts."""
        from parhelia.metrics.grafana import generate_overview_dashboard

        dashboard = generate_overview_dashboard()
        titles = [p["title"] for p in dashboard["panels"]]

        assert any("Session" in t for t in titles)

    def test_overview_dashboard_has_cost_panel(self):
        """@trace SPEC-06.15 - Overview MUST include cost tracking."""
        from parhelia.metrics.grafana import generate_overview_dashboard

        dashboard = generate_overview_dashboard()
        titles = [p["title"] for p in dashboard["panels"]]

        assert any("Cost" in t for t in titles)

    def test_generate_container_dashboard(self):
        """@trace SPEC-06.15 - Generator MUST create container detail dashboard."""
        from parhelia.metrics.grafana import generate_container_dashboard

        dashboard = generate_container_dashboard()

        assert dashboard["title"] == "Parhelia Container Details"
        assert "panels" in dashboard

    def test_dashboard_is_valid_json(self):
        """@trace SPEC-06.15 - Dashboard MUST be valid JSON."""
        from parhelia.metrics.grafana import generate_overview_dashboard

        dashboard = generate_overview_dashboard()

        # Should serialize without error
        json_str = json.dumps(dashboard)
        # Should parse back
        parsed = json.loads(json_str)
        assert parsed["title"] == dashboard["title"]


class TestGrafanaProvisioning:
    """Tests for Grafana provisioning configuration."""

    def test_generate_datasource_provisioning(self):
        """@trace SPEC-06.15 - Generator MUST create datasource provisioning."""
        from parhelia.metrics.grafana import generate_datasource_provisioning

        config = generate_datasource_provisioning(
            prometheus_url="http://localhost:9090",
        )

        assert "datasources" in config
        assert config["datasources"][0]["type"] == "prometheus"

    def test_generate_dashboard_provisioning(self):
        """@trace SPEC-06.15 - Generator MUST create dashboard provisioning."""
        from parhelia.metrics.grafana import generate_dashboard_provisioning

        config = generate_dashboard_provisioning(
            dashboard_path="/var/lib/grafana/dashboards",
        )

        assert "providers" in config
        assert config["providers"][0]["type"] == "file"


class TestGrafanaDockerCompose:
    """Tests for Grafana Docker Compose generation."""

    def test_generate_grafana_compose(self):
        """@trace SPEC-06.15 - Generator MUST create Docker Compose for Grafana."""
        from parhelia.metrics.grafana import generate_grafana_compose

        compose = generate_grafana_compose(port=3000)

        assert "grafana" in compose
        assert "3000" in compose

    def test_grafana_compose_includes_prometheus(self):
        """@trace SPEC-06.15 - Compose SHOULD include Prometheus."""
        from parhelia.metrics.grafana import generate_grafana_compose

        compose = generate_grafana_compose(include_prometheus=True)

        assert "prometheus" in compose
