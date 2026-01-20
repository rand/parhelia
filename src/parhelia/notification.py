"""Notification service for alerting users.

Implements:
- [SPEC-07.21.01] Notification Channels
- [SPEC-07.21.02] Notification Priority
- [SPEC-07.21.03] Notification Events
- [SPEC-07.21.04] Notification Content
"""

from __future__ import annotations

import json
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Literal

import aiohttp


class NotificationPriority(Enum):
    """Notification priority levels.

    Implements [SPEC-07.21.02].
    """

    INFO = "info"  # Routine events
    NOTICE = "notice"  # Actionable events
    WARNING = "warning"  # Attention needed
    CRITICAL = "critical"  # Immediate attention


# Event types that trigger notifications
NotificationEventType = Literal[
    "session.started",
    "session.completed",
    "session.error",
    "checkpoint.needs_review",
    "checkpoint.approved",
    "checkpoint.rejected",
    "escalation.triggered",
    "budget.threshold_reached",
    "budget.exceeded",
]


@dataclass
class NotificationEvent:
    """Event that triggers a notification.

    Implements [SPEC-07.21.03], [SPEC-07.21.04].
    """

    event_type: NotificationEventType
    session_id: str
    session_name: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: NotificationPriority = NotificationPriority.INFO
    context: dict[str, Any] = field(default_factory=dict)

    # Standard context fields
    cost_usd: float | None = None
    duration_minutes: int | None = None
    files_changed: int | None = None
    error_message: str | None = None
    checkpoint_id: str | None = None

    def get_title(self) -> str:
        """Get notification title based on event type."""
        titles = {
            "session.started": "Session Started",
            "session.completed": "Session Completed",
            "session.error": "Session Error",
            "checkpoint.needs_review": "Checkpoint Needs Review",
            "checkpoint.approved": "Checkpoint Approved",
            "checkpoint.rejected": "Checkpoint Rejected",
            "escalation.triggered": "Escalation Triggered",
            "budget.threshold_reached": "Budget Threshold Reached",
            "budget.exceeded": "Budget Exceeded",
        }
        return f"Parhelia: {titles.get(self.event_type, 'Notification')}"

    def get_body(self) -> str:
        """Get notification body with context.

        Implements [SPEC-07.21.04].
        """
        lines = [
            f"Session: {self.session_name or self.session_id}",
            f"Event: {self.event_type}",
        ]

        # Add context
        if self.cost_usd is not None:
            lines.append(f"Cost: ${self.cost_usd:.2f}")
        if self.duration_minutes is not None:
            lines.append(f"Duration: {self.duration_minutes}m")
        if self.files_changed is not None:
            lines.append(f"Files changed: {self.files_changed}")
        if self.error_message:
            lines.append(f"Error: {self.error_message[:200]}")
        if self.checkpoint_id:
            lines.append(f"Checkpoint: {self.checkpoint_id}")

        # Add action link
        lines.append("")
        lines.append(f"Review: parhelia session review {self.session_id}")

        return "\n".join(lines)


# Default priority for each event type
DEFAULT_EVENT_PRIORITIES: dict[NotificationEventType, NotificationPriority] = {
    "session.started": NotificationPriority.INFO,
    "session.completed": NotificationPriority.NOTICE,
    "session.error": NotificationPriority.WARNING,
    "checkpoint.needs_review": NotificationPriority.NOTICE,
    "checkpoint.approved": NotificationPriority.INFO,
    "checkpoint.rejected": NotificationPriority.NOTICE,
    "escalation.triggered": NotificationPriority.WARNING,
    "budget.threshold_reached": NotificationPriority.WARNING,
    "budget.exceeded": NotificationPriority.CRITICAL,
}


class NotificationChannel(ABC):
    """Base class for notification channels.

    Implements [SPEC-07.21.01].
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name identifier."""
        ...

    @abstractmethod
    async def send(self, event: NotificationEvent) -> bool:
        """Send notification via this channel.

        Args:
            event: The notification event to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        ...

    def is_configured(self) -> bool:
        """Check if channel is properly configured."""
        return True


@dataclass
class SlackChannelConfig:
    """Configuration for Slack channel."""

    webhook_url: str
    username: str = "Parhelia"
    icon_emoji: str = ":robot_face:"


class SlackChannel(NotificationChannel):
    """Slack webhook notification channel.

    Implements [SPEC-07.21.01].
    """

    def __init__(self, config: SlackChannelConfig):
        """Initialize Slack channel.

        Args:
            config: Slack channel configuration.
        """
        self.config = config

    @property
    def name(self) -> str:
        return "slack"

    def is_configured(self) -> bool:
        return bool(self.config.webhook_url)

    async def send(self, event: NotificationEvent) -> bool:
        """Send notification to Slack.

        Args:
            event: The notification event.

        Returns:
            True if sent successfully.
        """
        if not self.is_configured():
            return False

        # Build Slack message
        color = {
            NotificationPriority.INFO: "#36a64f",
            NotificationPriority.NOTICE: "#2196F3",
            NotificationPriority.WARNING: "#ff9800",
            NotificationPriority.CRITICAL: "#f44336",
        }.get(event.priority, "#808080")

        payload = {
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "title": event.get_title(),
                    "text": event.get_body(),
                    "footer": f"Parhelia | {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    "mrkdwn_in": ["text"],
                }
            ],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 200
        except Exception:
            return False


@dataclass
class DiscordChannelConfig:
    """Configuration for Discord channel."""

    webhook_url: str
    username: str = "Parhelia"
    avatar_url: str | None = None


class DiscordChannel(NotificationChannel):
    """Discord webhook notification channel.

    Implements [SPEC-07.21.01].
    """

    def __init__(self, config: DiscordChannelConfig):
        """Initialize Discord channel.

        Args:
            config: Discord channel configuration.
        """
        self.config = config

    @property
    def name(self) -> str:
        return "discord"

    def is_configured(self) -> bool:
        return bool(self.config.webhook_url)

    async def send(self, event: NotificationEvent) -> bool:
        """Send notification to Discord.

        Args:
            event: The notification event.

        Returns:
            True if sent successfully.
        """
        if not self.is_configured():
            return False

        # Build Discord embed
        color = {
            NotificationPriority.INFO: 0x36A64F,
            NotificationPriority.NOTICE: 0x2196F3,
            NotificationPriority.WARNING: 0xFF9800,
            NotificationPriority.CRITICAL: 0xF44336,
        }.get(event.priority, 0x808080)

        payload: dict[str, Any] = {
            "username": self.config.username,
            "embeds": [
                {
                    "title": event.get_title(),
                    "description": event.get_body(),
                    "color": color,
                    "timestamp": event.timestamp.isoformat(),
                }
            ],
        }

        if self.config.avatar_url:
            payload["avatar_url"] = self.config.avatar_url

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status in (200, 204)
        except Exception:
            return False


@dataclass
class NtfyChannelConfig:
    """Configuration for ntfy.sh channel."""

    server: str = "https://ntfy.sh"
    topic: str = ""
    token: str | None = None  # Optional access token


class NtfyChannel(NotificationChannel):
    """ntfy.sh push notification channel.

    Implements [SPEC-07.21.01].
    """

    def __init__(self, config: NtfyChannelConfig):
        """Initialize ntfy channel.

        Args:
            config: ntfy channel configuration.
        """
        self.config = config

    @property
    def name(self) -> str:
        return "ntfy"

    def is_configured(self) -> bool:
        return bool(self.config.topic)

    async def send(self, event: NotificationEvent) -> bool:
        """Send notification via ntfy.sh.

        Args:
            event: The notification event.

        Returns:
            True if sent successfully.
        """
        if not self.is_configured():
            return False

        # Map priority to ntfy priority (1-5)
        priority = {
            NotificationPriority.INFO: "2",  # low
            NotificationPriority.NOTICE: "3",  # default
            NotificationPriority.WARNING: "4",  # high
            NotificationPriority.CRITICAL: "5",  # urgent
        }.get(event.priority, "3")

        url = f"{self.config.server.rstrip('/')}/{self.config.topic}"

        headers = {
            "Title": event.get_title(),
            "Priority": priority,
            "Tags": event.event_type.replace(".", ","),
        }

        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=event.get_body(),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 200
        except Exception:
            return False


@dataclass
class EmailChannelConfig:
    """Configuration for email channel."""

    smtp_host: str
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    from_address: str = ""
    to_address: str = ""
    use_tls: bool = True


class EmailChannel(NotificationChannel):
    """Email notification channel (SMTP).

    Implements [SPEC-07.21.01].
    """

    def __init__(self, config: EmailChannelConfig):
        """Initialize email channel.

        Args:
            config: Email channel configuration.
        """
        self.config = config

    @property
    def name(self) -> str:
        return "email"

    def is_configured(self) -> bool:
        return bool(
            self.config.smtp_host
            and self.config.from_address
            and self.config.to_address
        )

    async def send(self, event: NotificationEvent) -> bool:
        """Send notification via email.

        Note: This uses synchronous SMTP but is wrapped for async interface.

        Args:
            event: The notification event.

        Returns:
            True if sent successfully.
        """
        if not self.is_configured():
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = event.get_title()
            msg["From"] = self.config.from_address
            msg["To"] = self.config.to_address

            # Plain text body
            text_body = event.get_body()

            # HTML body with basic formatting
            html_body = f"""
            <html>
            <body>
                <h2>{event.get_title()}</h2>
                <pre>{event.get_body()}</pre>
                <hr>
                <p style="color: #666; font-size: 12px;">
                    Sent by Parhelia at {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </body>
            </html>
            """

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.use_tls:
                    server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.sendmail(
                    self.config.from_address, [self.config.to_address], msg.as_string()
                )

            return True
        except Exception:
            return False


@dataclass
class WebhookChannelConfig:
    """Configuration for generic webhook channel."""

    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    include_full_event: bool = True


class WebhookChannel(NotificationChannel):
    """Generic HTTP webhook notification channel.

    Implements [SPEC-07.21.01].
    """

    def __init__(self, config: WebhookChannelConfig):
        """Initialize webhook channel.

        Args:
            config: Webhook channel configuration.
        """
        self.config = config

    @property
    def name(self) -> str:
        return "webhook"

    def is_configured(self) -> bool:
        return bool(self.config.url)

    async def send(self, event: NotificationEvent) -> bool:
        """Send notification via webhook.

        Args:
            event: The notification event.

        Returns:
            True if sent successfully.
        """
        if not self.is_configured():
            return False

        # Build payload
        payload: dict[str, Any] = {
            "title": event.get_title(),
            "body": event.get_body(),
            "priority": event.priority.value,
            "event_type": event.event_type,
            "session_id": event.session_id,
            "timestamp": event.timestamp.isoformat(),
        }

        if self.config.include_full_event:
            payload["context"] = {
                "session_name": event.session_name,
                "cost_usd": event.cost_usd,
                "duration_minutes": event.duration_minutes,
                "files_changed": event.files_changed,
                "error_message": event.error_message,
                "checkpoint_id": event.checkpoint_id,
                **event.context,
            }

        headers = {
            "Content-Type": "application/json",
            **self.config.headers,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    self.config.method,
                    self.config.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return 200 <= response.status < 300
        except Exception:
            return False


@dataclass
class RoutingConfig:
    """Routing rules for notification priorities.

    Implements [SPEC-07.21.02].
    """

    info: list[str] = field(default_factory=list)  # Channel names
    notice: list[str] = field(default_factory=lambda: ["slack"])
    warning: list[str] = field(default_factory=lambda: ["slack", "ntfy"])
    critical: list[str] = field(default_factory=lambda: ["slack", "ntfy", "email"])

    def get_channels(self, priority: NotificationPriority) -> list[str]:
        """Get channel names for a priority level."""
        return {
            NotificationPriority.INFO: self.info,
            NotificationPriority.NOTICE: self.notice,
            NotificationPriority.WARNING: self.warning,
            NotificationPriority.CRITICAL: self.critical,
        }.get(priority, [])


@dataclass
class NotificationConfig:
    """Full notification configuration.

    Implements [SPEC-07.21].
    """

    enabled: bool = True
    default_channel: str = "slack"
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    channels: dict[str, NotificationChannel] = field(default_factory=dict)


class NotificationService:
    """Service for sending notifications.

    Implements [SPEC-07.21].
    """

    def __init__(self, config: NotificationConfig | None = None):
        """Initialize notification service.

        Args:
            config: Notification configuration.
        """
        self.config = config or NotificationConfig()
        self._sent_count = 0
        self._failed_count = 0

    @property
    def channels(self) -> dict[str, NotificationChannel]:
        """Get configured channels."""
        return self.config.channels

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel.

        Args:
            channel: The channel to add.
        """
        self.config.channels[channel.name] = channel

    def remove_channel(self, name: str) -> None:
        """Remove a notification channel.

        Args:
            name: Name of channel to remove.
        """
        self.config.channels.pop(name, None)

    async def notify(
        self,
        event: NotificationEvent,
        channels: list[str] | None = None,
    ) -> dict[str, bool]:
        """Send notification to appropriate channels.

        Args:
            event: The notification event.
            channels: Explicit channel list (overrides routing).

        Returns:
            Dict mapping channel name to success status.
        """
        if not self.config.enabled:
            return {}

        # Determine which channels to use
        if channels is not None:
            target_channels = channels
        else:
            # Use routing config based on priority
            target_channels = self.config.routing.get_channels(event.priority)

        # Send to each channel
        results: dict[str, bool] = {}
        for channel_name in target_channels:
            channel = self.config.channels.get(channel_name)
            if channel and channel.is_configured():
                success = await channel.send(event)
                results[channel_name] = success
                if success:
                    self._sent_count += 1
                else:
                    self._failed_count += 1
            else:
                results[channel_name] = False
                self._failed_count += 1

        return results

    async def notify_session_started(
        self,
        session_id: str,
        session_name: str | None = None,
    ) -> dict[str, bool]:
        """Notify that a session has started.

        Args:
            session_id: The session ID.
            session_name: Optional session name.

        Returns:
            Channel results.
        """
        event = NotificationEvent(
            event_type="session.started",
            session_id=session_id,
            session_name=session_name,
            priority=NotificationPriority.INFO,
        )
        return await self.notify(event)

    async def notify_session_completed(
        self,
        session_id: str,
        session_name: str | None = None,
        cost_usd: float | None = None,
        duration_minutes: int | None = None,
        files_changed: int | None = None,
    ) -> dict[str, bool]:
        """Notify that a session has completed.

        Args:
            session_id: The session ID.
            session_name: Optional session name.
            cost_usd: Session cost.
            duration_minutes: Session duration.
            files_changed: Number of files changed.

        Returns:
            Channel results.
        """
        event = NotificationEvent(
            event_type="session.completed",
            session_id=session_id,
            session_name=session_name,
            priority=NotificationPriority.NOTICE,
            cost_usd=cost_usd,
            duration_minutes=duration_minutes,
            files_changed=files_changed,
        )
        return await self.notify(event)

    async def notify_session_error(
        self,
        session_id: str,
        error_message: str,
        session_name: str | None = None,
        is_critical: bool = False,
    ) -> dict[str, bool]:
        """Notify of a session error.

        Args:
            session_id: The session ID.
            error_message: The error message.
            session_name: Optional session name.
            is_critical: Whether this is a critical error.

        Returns:
            Channel results.
        """
        event = NotificationEvent(
            event_type="session.error",
            session_id=session_id,
            session_name=session_name,
            priority=NotificationPriority.CRITICAL
            if is_critical
            else NotificationPriority.WARNING,
            error_message=error_message,
        )
        return await self.notify(event)

    async def notify_checkpoint_needs_review(
        self,
        session_id: str,
        checkpoint_id: str,
        session_name: str | None = None,
        cost_usd: float | None = None,
        duration_minutes: int | None = None,
        files_changed: int | None = None,
    ) -> dict[str, bool]:
        """Notify that a checkpoint needs review.

        Args:
            session_id: The session ID.
            checkpoint_id: The checkpoint ID.
            session_name: Optional session name.
            cost_usd: Session cost so far.
            duration_minutes: Session duration.
            files_changed: Files changed in checkpoint.

        Returns:
            Channel results.
        """
        event = NotificationEvent(
            event_type="checkpoint.needs_review",
            session_id=session_id,
            session_name=session_name,
            priority=NotificationPriority.NOTICE,
            checkpoint_id=checkpoint_id,
            cost_usd=cost_usd,
            duration_minutes=duration_minutes,
            files_changed=files_changed,
        )
        return await self.notify(event)

    async def notify_budget_exceeded(
        self,
        session_id: str,
        cost_usd: float,
        budget_usd: float,
        session_name: str | None = None,
    ) -> dict[str, bool]:
        """Notify that budget has been exceeded.

        Args:
            session_id: The session ID.
            cost_usd: Current cost.
            budget_usd: Budget limit.
            session_name: Optional session name.

        Returns:
            Channel results.
        """
        event = NotificationEvent(
            event_type="budget.exceeded",
            session_id=session_id,
            session_name=session_name,
            priority=NotificationPriority.CRITICAL,
            cost_usd=cost_usd,
            context={"budget_usd": budget_usd},
        )
        return await self.notify(event)

    def get_stats(self) -> dict[str, int]:
        """Get notification statistics.

        Returns:
            Dict with sent_count and failed_count.
        """
        return {
            "sent_count": self._sent_count,
            "failed_count": self._failed_count,
        }


def parse_notification_config(data: dict) -> NotificationConfig:
    """Parse notification configuration from TOML data.

    Args:
        data: The 'notifications' section from parhelia.toml.

    Returns:
        NotificationConfig parsed from data.
    """
    enabled = data.get("enabled", True)
    default_channel = data.get("default_channel", "slack")

    # Parse routing
    routing_data = data.get("routing", {})
    routing = RoutingConfig(
        info=routing_data.get("info", []),
        notice=routing_data.get("notice", ["slack"]),
        warning=routing_data.get("warning", ["slack", "ntfy"]),
        critical=routing_data.get("critical", ["slack", "ntfy", "email"]),
    )

    # Parse channels
    channels: dict[str, NotificationChannel] = {}
    channels_data = data.get("channels", {})

    # Slack
    if "slack" in channels_data:
        slack_data = channels_data["slack"]
        channels["slack"] = SlackChannel(
            SlackChannelConfig(
                webhook_url=slack_data.get("webhook_url", ""),
                username=slack_data.get("username", "Parhelia"),
                icon_emoji=slack_data.get("icon_emoji", ":robot_face:"),
            )
        )

    # Discord
    if "discord" in channels_data:
        discord_data = channels_data["discord"]
        channels["discord"] = DiscordChannel(
            DiscordChannelConfig(
                webhook_url=discord_data.get("webhook_url", ""),
                username=discord_data.get("username", "Parhelia"),
                avatar_url=discord_data.get("avatar_url"),
            )
        )

    # ntfy
    if "ntfy" in channels_data:
        ntfy_data = channels_data["ntfy"]
        channels["ntfy"] = NtfyChannel(
            NtfyChannelConfig(
                server=ntfy_data.get("server", "https://ntfy.sh"),
                topic=ntfy_data.get("topic", ""),
                token=ntfy_data.get("token"),
            )
        )

    # Email
    if "email" in channels_data:
        email_data = channels_data["email"]
        channels["email"] = EmailChannel(
            EmailChannelConfig(
                smtp_host=email_data.get("smtp_host", ""),
                smtp_port=email_data.get("smtp_port", 587),
                smtp_user=email_data.get("smtp_user"),
                smtp_password=email_data.get("smtp_password"),
                from_address=email_data.get("from_address", ""),
                to_address=email_data.get("to_address", ""),
                use_tls=email_data.get("use_tls", True),
            )
        )

    # Generic webhook
    if "webhook" in channels_data:
        webhook_data = channels_data["webhook"]
        channels["webhook"] = WebhookChannel(
            WebhookChannelConfig(
                url=webhook_data.get("url", ""),
                method=webhook_data.get("method", "POST"),
                headers=webhook_data.get("headers", {}),
                include_full_event=webhook_data.get("include_full_event", True),
            )
        )

    return NotificationConfig(
        enabled=enabled,
        default_channel=default_channel,
        routing=routing,
        channels=channels,
    )
