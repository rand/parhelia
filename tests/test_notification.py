"""Tests for notification service.

@trace SPEC-07.21.01 - Notification Channels
@trace SPEC-07.21.02 - Notification Priority
@trace SPEC-07.21.03 - Notification Events
@trace SPEC-07.21.04 - Notification Content
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class AsyncContextManagerMock:
    """Helper for mocking async context managers like aiohttp.ClientSession."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

from parhelia.notification import (
    DEFAULT_EVENT_PRIORITIES,
    DiscordChannel,
    DiscordChannelConfig,
    EmailChannel,
    EmailChannelConfig,
    NotificationConfig,
    NotificationEvent,
    NotificationPriority,
    NotificationService,
    NtfyChannel,
    NtfyChannelConfig,
    RoutingConfig,
    SlackChannel,
    SlackChannelConfig,
    WebhookChannel,
    WebhookChannelConfig,
    parse_notification_config,
)


class TestNotificationPriority:
    """Tests for notification priority levels."""

    def test_priority_values(self):
        """@trace SPEC-07.21.02 - Priority levels MUST include info, notice, warning, critical."""
        assert NotificationPriority.INFO.value == "info"
        assert NotificationPriority.NOTICE.value == "notice"
        assert NotificationPriority.WARNING.value == "warning"
        assert NotificationPriority.CRITICAL.value == "critical"

    def test_default_event_priorities(self):
        """@trace SPEC-07.21.03 - Events MUST have default priority mappings."""
        assert DEFAULT_EVENT_PRIORITIES["session.started"] == NotificationPriority.INFO
        assert DEFAULT_EVENT_PRIORITIES["session.completed"] == NotificationPriority.NOTICE
        assert DEFAULT_EVENT_PRIORITIES["session.error"] == NotificationPriority.WARNING
        assert DEFAULT_EVENT_PRIORITIES["checkpoint.needs_review"] == NotificationPriority.NOTICE
        assert DEFAULT_EVENT_PRIORITIES["budget.exceeded"] == NotificationPriority.CRITICAL


class TestNotificationEvent:
    """Tests for NotificationEvent."""

    def test_event_creation(self):
        """@trace SPEC-07.21.03 - Events MUST include required fields."""
        event = NotificationEvent(
            event_type="session.completed",
            session_id="test-session-123",
            session_name="Test Session",
            priority=NotificationPriority.NOTICE,
        )

        assert event.event_type == "session.completed"
        assert event.session_id == "test-session-123"
        assert event.session_name == "Test Session"
        assert event.priority == NotificationPriority.NOTICE
        assert event.timestamp is not None

    def test_event_get_title(self):
        """@trace SPEC-07.21.04 - Notifications MUST have descriptive titles."""
        event = NotificationEvent(
            event_type="checkpoint.needs_review",
            session_id="test-session",
        )

        title = event.get_title()
        assert "Parhelia" in title
        assert "Review" in title

    def test_event_get_body(self):
        """@trace SPEC-07.21.04 - Notifications MUST include context."""
        event = NotificationEvent(
            event_type="session.completed",
            session_id="test-session-123",
            session_name="Fix Auth Bug",
            cost_usd=0.45,
            duration_minutes=12,
            files_changed=3,
        )

        body = event.get_body()
        assert "Fix Auth Bug" in body
        assert "$0.45" in body
        assert "12m" in body
        assert "3" in body
        assert "parhelia session review" in body

    def test_event_with_error(self):
        """@trace SPEC-07.21.04 - Error notifications MUST include error message."""
        event = NotificationEvent(
            event_type="session.error",
            session_id="test-session",
            error_message="Connection timeout after 30s",
        )

        body = event.get_body()
        assert "Connection timeout" in body

    def test_event_with_checkpoint(self):
        """@trace SPEC-07.21.04 - Checkpoint notifications MUST include checkpoint ID."""
        event = NotificationEvent(
            event_type="checkpoint.needs_review",
            session_id="test-session",
            checkpoint_id="cp-abc123",
        )

        body = event.get_body()
        assert "cp-abc123" in body


class TestRoutingConfig:
    """Tests for routing configuration."""

    def test_default_routing(self):
        """@trace SPEC-07.21.02 - Default routing MUST follow spec."""
        routing = RoutingConfig()

        assert routing.info == []
        assert "slack" in routing.notice
        assert "slack" in routing.warning
        assert "ntfy" in routing.warning
        assert "slack" in routing.critical
        assert "ntfy" in routing.critical
        assert "email" in routing.critical

    def test_get_channels_by_priority(self):
        """@trace SPEC-07.21.02 - Routing MUST return correct channels for priority."""
        routing = RoutingConfig(
            info=["webhook"],
            notice=["slack"],
            warning=["slack", "ntfy"],
            critical=["slack", "ntfy", "email"],
        )

        assert routing.get_channels(NotificationPriority.INFO) == ["webhook"]
        assert routing.get_channels(NotificationPriority.NOTICE) == ["slack"]
        assert routing.get_channels(NotificationPriority.WARNING) == ["slack", "ntfy"]
        assert routing.get_channels(NotificationPriority.CRITICAL) == ["slack", "ntfy", "email"]


class TestSlackChannel:
    """Tests for Slack channel."""

    def test_slack_config(self):
        """@trace SPEC-07.21.01 - Slack channel MUST support webhook configuration."""
        config = SlackChannelConfig(
            webhook_url="https://hooks.slack.com/services/xxx",
            username="TestBot",
            icon_emoji=":test:",
        )

        channel = SlackChannel(config)
        assert channel.name == "slack"
        assert channel.is_configured() is True

    def test_slack_not_configured(self):
        """@trace SPEC-07.21.01 - Unconfigured channel MUST report not configured."""
        config = SlackChannelConfig(webhook_url="")
        channel = SlackChannel(config)
        assert channel.is_configured() is False

    @pytest.mark.asyncio
    async def test_slack_send_success(self):
        """@trace SPEC-07.21.01 - Slack channel MUST send to webhook."""
        config = SlackChannelConfig(webhook_url="https://hooks.slack.com/test")
        channel = SlackChannel(config)

        event = NotificationEvent(
            event_type="session.completed",
            session_id="test-session",
            priority=NotificationPriority.NOTICE,
        )

        # Create properly nested async context manager mocks
        mock_response = MagicMock(status=200)

        async def mock_post(*args, **kwargs):
            return mock_response

        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=AsyncContextManagerMock(mock_response))

        with patch("parhelia.notification.aiohttp.ClientSession") as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(mock_session_instance)

            result = await channel.send(event)
            assert result is True

    @pytest.mark.asyncio
    async def test_slack_send_failure(self):
        """@trace SPEC-07.21.01 - Slack channel MUST handle failures gracefully."""
        config = SlackChannelConfig(webhook_url="https://hooks.slack.com/test")
        channel = SlackChannel(config)

        event = NotificationEvent(
            event_type="session.error",
            session_id="test-session",
            priority=NotificationPriority.WARNING,
        )

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Network error")

            result = await channel.send(event)
            assert result is False


class TestDiscordChannel:
    """Tests for Discord channel."""

    def test_discord_config(self):
        """@trace SPEC-07.21.01 - Discord channel MUST support webhook configuration."""
        config = DiscordChannelConfig(
            webhook_url="https://discord.com/api/webhooks/xxx",
            username="Parhelia",
        )

        channel = DiscordChannel(config)
        assert channel.name == "discord"
        assert channel.is_configured() is True

    @pytest.mark.asyncio
    async def test_discord_send(self):
        """@trace SPEC-07.21.01 - Discord channel MUST send embeds."""
        config = DiscordChannelConfig(webhook_url="https://discord.com/api/webhooks/xxx")
        channel = DiscordChannel(config)

        event = NotificationEvent(
            event_type="session.completed",
            session_id="test-session",
            priority=NotificationPriority.NOTICE,
        )

        mock_response = MagicMock(status=204)
        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=AsyncContextManagerMock(mock_response))

        with patch("parhelia.notification.aiohttp.ClientSession") as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(mock_session_instance)

            result = await channel.send(event)
            assert result is True


class TestNtfyChannel:
    """Tests for ntfy.sh channel."""

    def test_ntfy_config(self):
        """@trace SPEC-07.21.01 - ntfy channel MUST support server and topic configuration."""
        config = NtfyChannelConfig(
            server="https://ntfy.sh",
            topic="my-topic",
            token="secret-token",
        )

        channel = NtfyChannel(config)
        assert channel.name == "ntfy"
        assert channel.is_configured() is True

    def test_ntfy_not_configured_without_topic(self):
        """@trace SPEC-07.21.01 - ntfy channel MUST require topic."""
        config = NtfyChannelConfig(topic="")
        channel = NtfyChannel(config)
        assert channel.is_configured() is False

    @pytest.mark.asyncio
    async def test_ntfy_send(self):
        """@trace SPEC-07.21.01 - ntfy channel MUST send push notifications."""
        config = NtfyChannelConfig(server="https://ntfy.sh", topic="test-topic")
        channel = NtfyChannel(config)

        event = NotificationEvent(
            event_type="checkpoint.needs_review",
            session_id="test-session",
            priority=NotificationPriority.NOTICE,
        )

        mock_response = MagicMock(status=200)
        mock_session_instance = MagicMock()
        mock_session_instance.post = MagicMock(return_value=AsyncContextManagerMock(mock_response))

        with patch("parhelia.notification.aiohttp.ClientSession") as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(mock_session_instance)

            result = await channel.send(event)
            assert result is True


class TestEmailChannel:
    """Tests for email channel."""

    def test_email_config(self):
        """@trace SPEC-07.21.01 - Email channel MUST support SMTP configuration."""
        config = EmailChannelConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_address="parhelia@example.com",
            to_address="user@example.com",
        )

        channel = EmailChannel(config)
        assert channel.name == "email"
        assert channel.is_configured() is True

    def test_email_not_configured(self):
        """@trace SPEC-07.21.01 - Email channel MUST require host and addresses."""
        config = EmailChannelConfig(smtp_host="", from_address="", to_address="")
        channel = EmailChannel(config)
        assert channel.is_configured() is False


class TestWebhookChannel:
    """Tests for generic webhook channel."""

    def test_webhook_config(self):
        """@trace SPEC-07.21.01 - Webhook channel MUST support URL configuration."""
        config = WebhookChannelConfig(
            url="https://example.com/webhook",
            method="POST",
            headers={"X-Custom": "value"},
        )

        channel = WebhookChannel(config)
        assert channel.name == "webhook"
        assert channel.is_configured() is True

    @pytest.mark.asyncio
    async def test_webhook_send(self):
        """@trace SPEC-07.21.01 - Webhook channel MUST send JSON payload."""
        config = WebhookChannelConfig(url="https://example.com/webhook")
        channel = WebhookChannel(config)

        event = NotificationEvent(
            event_type="session.completed",
            session_id="test-session",
            priority=NotificationPriority.NOTICE,
            cost_usd=1.23,
        )

        mock_response = MagicMock(status=200)
        mock_session_instance = MagicMock()
        mock_session_instance.request = MagicMock(return_value=AsyncContextManagerMock(mock_response))

        with patch("parhelia.notification.aiohttp.ClientSession") as mock_session_class:
            mock_session_class.return_value = AsyncContextManagerMock(mock_session_instance)

            result = await channel.send(event)
            assert result is True


class TestNotificationService:
    """Tests for NotificationService."""

    @pytest.fixture
    def mock_slack_channel(self) -> SlackChannel:
        """Create mock Slack channel."""
        channel = SlackChannel(SlackChannelConfig(webhook_url="https://test.slack.com"))
        return channel

    @pytest.fixture
    def mock_ntfy_channel(self) -> NtfyChannel:
        """Create mock ntfy channel."""
        channel = NtfyChannel(NtfyChannelConfig(topic="test-topic"))
        return channel

    @pytest.fixture
    def service(self, mock_slack_channel, mock_ntfy_channel) -> NotificationService:
        """Create NotificationService with mock channels."""
        config = NotificationConfig(
            enabled=True,
            routing=RoutingConfig(
                info=[],
                notice=["slack"],
                warning=["slack", "ntfy"],
                critical=["slack", "ntfy"],
            ),
            channels={
                "slack": mock_slack_channel,
                "ntfy": mock_ntfy_channel,
            },
        )
        return NotificationService(config)

    def test_add_channel(self):
        """@trace SPEC-07.21.01 - Service MUST allow adding channels."""
        service = NotificationService()
        channel = SlackChannel(SlackChannelConfig(webhook_url="https://test.com"))

        service.add_channel(channel)
        assert "slack" in service.channels

    def test_remove_channel(self, service):
        """@trace SPEC-07.21.01 - Service MUST allow removing channels."""
        service.remove_channel("slack")
        assert "slack" not in service.channels

    @pytest.mark.asyncio
    async def test_notify_routes_by_priority(self, service):
        """@trace SPEC-07.21.02 - Service MUST route by priority."""
        event = NotificationEvent(
            event_type="session.completed",
            session_id="test-session",
            priority=NotificationPriority.WARNING,
        )

        with patch.object(service.config.channels["slack"], "send", new_callable=AsyncMock) as mock_slack:
            with patch.object(service.config.channels["ntfy"], "send", new_callable=AsyncMock) as mock_ntfy:
                mock_slack.return_value = True
                mock_ntfy.return_value = True

                results = await service.notify(event)

                assert results["slack"] is True
                assert results["ntfy"] is True
                mock_slack.assert_called_once()
                mock_ntfy.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_explicit_channels(self, service):
        """@trace SPEC-07.21.02 - Service MUST support explicit channel override."""
        event = NotificationEvent(
            event_type="session.started",
            session_id="test-session",
            priority=NotificationPriority.INFO,
        )

        with patch.object(service.config.channels["slack"], "send", new_callable=AsyncMock) as mock_slack:
            mock_slack.return_value = True

            # Override routing with explicit channel
            results = await service.notify(event, channels=["slack"])

            assert results["slack"] is True
            mock_slack.assert_called_once()

    @pytest.mark.asyncio
    async def test_notify_disabled(self):
        """@trace SPEC-07.21 - Disabled service MUST not send notifications."""
        config = NotificationConfig(enabled=False)
        service = NotificationService(config)

        event = NotificationEvent(
            event_type="budget.exceeded",
            session_id="test-session",
            priority=NotificationPriority.CRITICAL,
        )

        results = await service.notify(event)
        assert results == {}

    @pytest.mark.asyncio
    async def test_notify_session_started(self, service):
        """@trace SPEC-07.21.03 - Service MUST have convenience method for session.started."""
        with patch.object(service, "notify", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = {"slack": True}

            await service.notify_session_started("session-123", "My Session")

            mock_notify.assert_called_once()
            event = mock_notify.call_args[0][0]
            assert event.event_type == "session.started"
            assert event.session_id == "session-123"

    @pytest.mark.asyncio
    async def test_notify_session_completed(self, service):
        """@trace SPEC-07.21.03 - Service MUST have convenience method for session.completed."""
        with patch.object(service, "notify", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = {"slack": True}

            await service.notify_session_completed(
                "session-123",
                session_name="My Session",
                cost_usd=0.50,
                duration_minutes=10,
                files_changed=5,
            )

            mock_notify.assert_called_once()
            event = mock_notify.call_args[0][0]
            assert event.event_type == "session.completed"
            assert event.cost_usd == 0.50
            assert event.duration_minutes == 10
            assert event.files_changed == 5

    @pytest.mark.asyncio
    async def test_notify_session_error(self, service):
        """@trace SPEC-07.21.03 - Service MUST have convenience method for session.error."""
        with patch.object(service, "notify", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = {"slack": True}

            await service.notify_session_error(
                "session-123",
                error_message="Something went wrong",
                is_critical=True,
            )

            mock_notify.assert_called_once()
            event = mock_notify.call_args[0][0]
            assert event.event_type == "session.error"
            assert event.error_message == "Something went wrong"
            assert event.priority == NotificationPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_notify_checkpoint_needs_review(self, service):
        """@trace SPEC-07.21.03 - Service MUST have convenience method for checkpoint.needs_review."""
        with patch.object(service, "notify", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = {"slack": True}

            await service.notify_checkpoint_needs_review(
                "session-123",
                checkpoint_id="cp-abc123",
            )

            mock_notify.assert_called_once()
            event = mock_notify.call_args[0][0]
            assert event.event_type == "checkpoint.needs_review"
            assert event.checkpoint_id == "cp-abc123"

    @pytest.mark.asyncio
    async def test_notify_budget_exceeded(self, service):
        """@trace SPEC-07.21.03 - Service MUST have convenience method for budget.exceeded."""
        with patch.object(service, "notify", new_callable=AsyncMock) as mock_notify:
            mock_notify.return_value = {"slack": True}

            await service.notify_budget_exceeded(
                "session-123",
                cost_usd=15.0,
                budget_usd=10.0,
            )

            mock_notify.assert_called_once()
            event = mock_notify.call_args[0][0]
            assert event.event_type == "budget.exceeded"
            assert event.cost_usd == 15.0
            assert event.priority == NotificationPriority.CRITICAL

    def test_get_stats(self, service):
        """@trace SPEC-07.21 - Service MUST track notification statistics."""
        stats = service.get_stats()
        assert "sent_count" in stats
        assert "failed_count" in stats


class TestParseNotificationConfig:
    """Tests for parsing notification config from TOML."""

    def test_parse_empty_config(self):
        """Parse empty config should use defaults."""
        config = parse_notification_config({})

        assert config.enabled is True
        assert config.default_channel == "slack"
        assert config.channels == {}

    def test_parse_full_config(self):
        """Parse complete notification configuration."""
        data = {
            "enabled": True,
            "default_channel": "ntfy",
            "routing": {
                "info": ["webhook"],
                "notice": ["slack"],
                "warning": ["slack", "ntfy"],
                "critical": ["slack", "ntfy", "email"],
            },
            "channels": {
                "slack": {
                    "webhook_url": "https://hooks.slack.com/xxx",
                    "username": "TestBot",
                },
                "ntfy": {
                    "server": "https://ntfy.sh",
                    "topic": "my-topic",
                },
                "discord": {
                    "webhook_url": "https://discord.com/api/webhooks/xxx",
                },
                "email": {
                    "smtp_host": "smtp.example.com",
                    "smtp_port": 587,
                    "from_address": "parhelia@example.com",
                    "to_address": "user@example.com",
                },
                "webhook": {
                    "url": "https://example.com/hook",
                },
            },
        }

        config = parse_notification_config(data)

        assert config.enabled is True
        assert config.default_channel == "ntfy"
        assert config.routing.info == ["webhook"]
        assert "slack" in config.channels
        assert "ntfy" in config.channels
        assert "discord" in config.channels
        assert "email" in config.channels
        assert "webhook" in config.channels

    def test_parse_slack_channel(self):
        """Parse Slack channel configuration."""
        data = {
            "channels": {
                "slack": {
                    "webhook_url": "https://hooks.slack.com/test",
                    "username": "MyBot",
                    "icon_emoji": ":robot:",
                }
            }
        }

        config = parse_notification_config(data)

        slack = config.channels["slack"]
        assert isinstance(slack, SlackChannel)
        assert slack.config.webhook_url == "https://hooks.slack.com/test"
        assert slack.config.username == "MyBot"
        assert slack.config.icon_emoji == ":robot:"

    def test_parse_ntfy_channel(self):
        """Parse ntfy channel configuration."""
        data = {
            "channels": {
                "ntfy": {
                    "server": "https://custom.ntfy.sh",
                    "topic": "my-alerts",
                    "token": "secret123",
                }
            }
        }

        config = parse_notification_config(data)

        ntfy = config.channels["ntfy"]
        assert isinstance(ntfy, NtfyChannel)
        assert ntfy.config.server == "https://custom.ntfy.sh"
        assert ntfy.config.topic == "my-alerts"
        assert ntfy.config.token == "secret123"

    def test_parse_email_channel(self):
        """Parse email channel configuration."""
        data = {
            "channels": {
                "email": {
                    "smtp_host": "smtp.gmail.com",
                    "smtp_port": 465,
                    "smtp_user": "user@gmail.com",
                    "smtp_password": "app-password",
                    "from_address": "parhelia@gmail.com",
                    "to_address": "alerts@example.com",
                    "use_tls": True,
                }
            }
        }

        config = parse_notification_config(data)

        email = config.channels["email"]
        assert isinstance(email, EmailChannel)
        assert email.config.smtp_host == "smtp.gmail.com"
        assert email.config.smtp_port == 465
        assert email.config.smtp_user == "user@gmail.com"

    def test_parse_webhook_channel(self):
        """Parse webhook channel configuration."""
        data = {
            "channels": {
                "webhook": {
                    "url": "https://api.example.com/notify",
                    "method": "PUT",
                    "headers": {"Authorization": "Bearer xxx"},
                    "include_full_event": False,
                }
            }
        }

        config = parse_notification_config(data)

        webhook = config.channels["webhook"]
        assert isinstance(webhook, WebhookChannel)
        assert webhook.config.url == "https://api.example.com/notify"
        assert webhook.config.method == "PUT"
        assert webhook.config.headers == {"Authorization": "Bearer xxx"}
        assert webhook.config.include_full_event is False
