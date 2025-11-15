"""Unit tests for shared/notifications.py.

Tests toast notification sending with PromptServer integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from triple_ksampler.shared import notifications


class TestSendToastNotification:
    """Tests for send_toast_notification function."""

    def test_send_notification_successfully(self):
        """Test sending notification when PromptServer is available."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_server.instance = mock_instance

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            result = notifications.send_toast_notification(
                message_type="test_message",
                severity="info",
                summary="Test Summary",
                detail="Test Detail",
                life=5000,
            )

        # Assert
        assert result is True
        mock_instance.send_sync.assert_called_once_with(
            "test_message",
            {
                "severity": "info",
                "summary": "Test Summary",
                "detail": "Test Detail",
                "life": 5000,
            },
        )

    def test_send_notification_without_life_parameter(self):
        """Test sending notification without life parameter."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_server.instance = mock_instance

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            result = notifications.send_toast_notification(
                message_type="test_message",
                severity="warn",
                summary="Warning",
                detail="Warning details",
                life=None,
            )

        # Assert
        assert result is True
        # Should not include 'life' in message_data
        mock_instance.send_sync.assert_called_once()
        call_args = mock_instance.send_sync.call_args[0]
        message_data = call_args[1]
        assert "life" not in message_data

    def test_send_notification_prompt_server_none(self):
        """Test that notification returns False when PromptServer is None."""
        # Arrange
        with patch.object(notifications, "PromptServer", None):
            # Act
            result = notifications.send_toast_notification(
                message_type="test_message",
                severity="info",
                summary="Test",
                detail="Test",
            )

        # Assert
        assert result is False

    def test_send_notification_prompt_server_no_instance(self):
        """Test that notification returns False when PromptServer.instance is None."""
        # Arrange
        mock_server = MagicMock()
        mock_server.instance = None

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            result = notifications.send_toast_notification(
                message_type="test_message",
                severity="info",
                summary="Test",
                detail="Test",
            )

        # Assert
        assert result is False

    def test_send_notification_handles_exception(self):
        """Test that notification handles exceptions gracefully."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_instance.send_sync.side_effect = Exception("Network error")
        mock_server.instance = mock_instance

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            result = notifications.send_toast_notification(
                message_type="test_message",
                severity="error",
                summary="Error",
                detail="Error details",
            )

        # Assert
        assert result is False

    def test_send_notification_different_severities(self):
        """Test sending notifications with different severity levels."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_server.instance = mock_instance

        severities = ["info", "warn", "error"]

        with patch.object(notifications, "PromptServer", mock_server):
            for severity in severities:
                mock_instance.reset_mock()

                # Act
                result = notifications.send_toast_notification(
                    message_type="test_message",
                    severity=severity,
                    summary=f"{severity.upper()} Summary",
                    detail="Details",
                )

                # Assert
                assert result is True
                call_data = mock_instance.send_sync.call_args[0][1]
                assert call_data["severity"] == severity


class TestSendDryRunNotification:
    """Tests for send_dry_run_notification convenience function."""

    def test_send_dry_run_notification_success(self):
        """Test sending dry-run notification successfully."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_server.instance = mock_instance

        detail = "Stage 1: 10 steps\nStage 2: 5 steps\nStage 3: 3 steps"

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            result = notifications.send_dry_run_notification(detail)

        # Assert
        assert result is True
        mock_instance.send_sync.assert_called_once_with(
            "triple_ksampler_dry_run",
            {
                "severity": "info",
                "summary": "TripleKSampler: Dry Run Complete",
                "detail": detail,
                "life": notifications.TOAST_LIFE_DRY_RUN,
            },
        )

    def test_send_dry_run_notification_prompt_server_unavailable(self):
        """Test dry-run notification returns False when PromptServer unavailable."""
        # Arrange
        with patch.object(notifications, "PromptServer", None):
            # Act
            result = notifications.send_dry_run_notification("Details")

        # Assert
        assert result is False


class TestSendOverlapWarning:
    """Tests for send_overlap_warning convenience function."""

    def test_send_overlap_warning_success(self):
        """Test sending overlap warning successfully."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_server.instance = mock_instance

        overlap_pct = 25.5

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            result = notifications.send_overlap_warning(overlap_pct)

        # Assert
        assert result is True
        mock_instance.send_sync.assert_called_once()
        call_args = mock_instance.send_sync.call_args[0]
        assert call_args[0] == "triple_ksampler_overlap"

        message_data = call_args[1]
        assert message_data["severity"] == "warn"
        assert message_data["summary"] == "TripleKSampler: Stage overlap"
        assert "25.5%" in message_data["detail"]
        assert message_data["life"] == notifications.TOAST_LIFE_OVERLAP

    def test_send_overlap_warning_formats_percentage(self):
        """Test that overlap percentage is formatted correctly."""
        # Arrange
        mock_server = MagicMock()
        mock_instance = MagicMock()
        mock_server.instance = mock_instance

        with patch.object(notifications, "PromptServer", mock_server):
            # Act
            notifications.send_overlap_warning(33.333333)

        # Assert
        call_data = mock_instance.send_sync.call_args[0][1]
        # Should format to 1 decimal place
        assert "33.3%" in call_data["detail"]

    def test_send_overlap_warning_prompt_server_unavailable(self):
        """Test overlap warning returns False when PromptServer unavailable."""
        # Arrange
        with patch.object(notifications, "PromptServer", None):
            # Act
            result = notifications.send_overlap_warning(25.0)

        # Assert
        assert result is False


class TestConstants:
    """Tests for module constants."""

    def test_toast_life_overlap_value(self):
        """Test TOAST_LIFE_OVERLAP constant value."""
        assert notifications.TOAST_LIFE_OVERLAP == 8000

    def test_toast_life_dry_run_value(self):
        """Test TOAST_LIFE_DRY_RUN constant value."""
        assert notifications.TOAST_LIFE_DRY_RUN == 12000
