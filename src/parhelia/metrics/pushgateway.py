"""Prometheus Pushgateway management.

Implements:
- [SPEC-06.13a] Pushgateway Configuration
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiohttp


@dataclass
class PushgatewayConfig:
    """Configuration for Prometheus Pushgateway.

    Implements [SPEC-06.13a].
    """

    port: int = 9091
    public_host: str | None = None  # For remote access
    group_ttl_seconds: int = 60  # How long to keep groups after last push
    persistence_file: str | None = None  # Optional persistence path

    @property
    def local_url(self) -> str:
        """Get local URL for Pushgateway."""
        return f"http://localhost:{self.port}"


class PushgatewayManager:
    """Manage local Pushgateway instance.

    Implements [SPEC-06.13a].
    """

    def __init__(self, config: PushgatewayConfig):
        """Initialize the Pushgateway manager.

        Args:
            config: Pushgateway configuration.
        """
        self.config = config
        self.process: subprocess.Popen | None = None
        self._running = False

    async def start(self) -> None:
        """Start Pushgateway process.

        Implements [SPEC-06.13a].
        """
        await self._spawn_process()
        await self._wait_ready()
        self._running = True

    async def _spawn_process(self) -> None:
        """Spawn the Pushgateway process."""
        cmd = [
            "pushgateway",
            f"--web.listen-address=:{self.config.port}",
        ]

        # Add persistence if configured
        if self.config.persistence_file:
            cmd.append(f"--persistence.file={self.config.persistence_file}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    async def _wait_ready(self, timeout: int = 10) -> None:
        """Wait for Pushgateway to be ready.

        Args:
            timeout: Maximum seconds to wait.

        Raises:
            TimeoutError: If Pushgateway doesn't start in time.
        """
        async with aiohttp.ClientSession() as session:
            for _ in range(timeout * 10):
                try:
                    async with session.get(
                        f"{self.config.local_url}/-/ready"
                    ) as resp:
                        if resp.status == 200:
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.1)

        raise TimeoutError("Pushgateway failed to start")

    async def stop(self) -> None:
        """Stop Pushgateway process."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self._running = False

    def get_url(self) -> str:
        """Get Pushgateway URL for containers.

        Implements [SPEC-06.13a].

        Returns:
            URL that containers should push metrics to.
        """
        # Check environment variable first
        env_url = os.environ.get("PARHELIA_PUSHGATEWAY_URL")
        if env_url:
            return env_url

        # Use public host if configured, otherwise localhost
        host = self.config.public_host or "localhost"
        return f"http://{host}:{self.config.port}"

    def get_metrics_url(self) -> str:
        """Get URL for metrics endpoint.

        Returns:
            URL to fetch metrics from Pushgateway.
        """
        return f"{self.config.local_url}/metrics"

    def get_api_url(self) -> str:
        """Get URL for API endpoint.

        Returns:
            URL for Pushgateway API.
        """
        return f"{self.config.local_url}/api/v1/metrics"

    async def is_healthy(self) -> bool:
        """Check if Pushgateway is healthy.

        Returns:
            True if Pushgateway is responding.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.config.local_url}/-/ready",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def delete_group(self, job: str, grouping_key: dict[str, str]) -> bool:
        """Delete a metric group from Pushgateway.

        Args:
            job: Job name.
            grouping_key: Grouping key labels.

        Returns:
            True if deletion was successful.
        """
        # Build URL path from grouping key
        path_parts = [f"{k}/{v}" for k, v in grouping_key.items()]
        path = "/".join(path_parts)
        url = f"{self.config.local_url}/metrics/job/{job}/{path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url) as resp:
                    return resp.status == 202
        except aiohttp.ClientError:
            return False


def generate_docker_compose(config: PushgatewayConfig) -> str:
    """Generate Docker Compose configuration for Pushgateway.

    Implements [SPEC-06.13a].

    Args:
        config: Pushgateway configuration.

    Returns:
        Docker Compose YAML content.
    """
    persistence_cmd = ""
    if config.persistence_file:
        persistence_cmd = f"--persistence.file={config.persistence_file}"

    compose = f"""version: '3.8'

services:
  pushgateway:
    image: prom/pushgateway:latest
    container_name: parhelia-pushgateway
    ports:
      - "{config.port}:9091"
    command:
      - --web.listen-address=:9091
      - --push.disable-consistency-check
      {f"- {persistence_cmd}" if persistence_cmd else ""}
    restart: unless-stopped
    networks:
      - parhelia

networks:
  parhelia:
    driver: bridge
"""
    return compose


def generate_prometheus_scrape_config(pushgateway_port: int = 9091) -> str:
    """Generate Prometheus scrape config for Pushgateway.

    Args:
        pushgateway_port: Pushgateway port.

    Returns:
        Prometheus scrape config YAML.
    """
    return f"""scrape_configs:
  - job_name: 'pushgateway'
    honor_labels: true
    static_configs:
      - targets: ['localhost:{pushgateway_port}']
"""
