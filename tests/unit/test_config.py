"""Unit tests for shared/config.py.

Tests configuration loading, fallback hierarchy, and value extraction.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from triple_ksampler.shared import config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_user_config(self, tmp_path):
        """Test loading user config.toml successfully."""
        # Arrange
        config_dir = tmp_path
        user_config = config_dir / "config.toml"
        user_config.write_text("""
[sampling]
base_quality_threshold = 25

[boundaries]
default_t2v = 0.850
default_i2v = 0.925

[logging]
level = "DEBUG"
""")

        # Act
        result = config.load_config(config_dir)

        # Assert
        assert result["sampling"]["base_quality_threshold"] == 25
        assert result["boundaries"]["default_t2v"] == 0.850
        assert result["boundaries"]["default_i2v"] == 0.925
        assert result["logging"]["level"] == "DEBUG"

    def test_load_config_auto_creates_from_template(self, tmp_path):
        """Test auto-creating config.toml from config.example.toml."""
        # Arrange
        config_dir = tmp_path
        template_config = config_dir / "config.example.toml"
        template_config.write_text("""
[sampling]
base_quality_threshold = 30

[boundaries]
default_t2v = 0.800
default_i2v = 0.900

[logging]
level = "INFO"
""")

        # Act
        result = config.load_config(config_dir)

        # Assert
        user_config = config_dir / "config.toml"
        assert user_config.exists()  # Auto-created
        assert result["sampling"]["base_quality_threshold"] == 30
        assert result["boundaries"]["default_t2v"] == 0.800

    def test_load_config_fallback_to_template(self, tmp_path):
        """Test falling back to config.example.toml when user config doesn't exist."""
        # Arrange
        config_dir = tmp_path
        template_config = config_dir / "config.example.toml"
        template_config.write_text("""
[sampling]
base_quality_threshold = 15

[boundaries]
default_t2v = 0.875
default_i2v = 0.900

[logging]
level = "INFO"
""")

        # Block auto-creation by making directory read-only temporarily
        with patch("shutil.copy2", side_effect=OSError("Permission denied")):
            # Act
            result = config.load_config(config_dir)

        # Assert
        assert result["sampling"]["base_quality_threshold"] == 15

    def test_load_config_fallback_to_defaults(self, tmp_path):
        """Test falling back to hardcoded defaults when no TOML files exist."""
        # Arrange
        config_dir = tmp_path  # Empty directory

        # Act
        result = config.load_config(config_dir)

        # Assert
        assert result["sampling"]["base_quality_threshold"] == config.DEFAULT_BASE_QUALITY_THRESHOLD
        assert result["boundaries"]["default_t2v"] == config.DEFAULT_BOUNDARY_T2V
        assert result["boundaries"]["default_i2v"] == config.DEFAULT_BOUNDARY_I2V
        assert result["logging"]["level"] == config.DEFAULT_LOG_LEVEL

    def test_load_config_handles_invalid_user_toml(self, tmp_path):
        """Test gracefully handling invalid user config.toml."""
        # Arrange
        config_dir = tmp_path
        user_config = config_dir / "config.toml"
        user_config.write_text("invalid toml [[[")  # Malformed TOML

        template_config = config_dir / "config.example.toml"
        template_config.write_text("""
[sampling]
base_quality_threshold = 20
""")

        # Act
        result = config.load_config(config_dir)

        # Assert - Should fall back to template
        assert result["sampling"]["base_quality_threshold"] == 20

    def test_load_config_handles_invalid_template_toml(self, tmp_path):
        """Test gracefully handling invalid config.example.toml."""
        # Arrange
        config_dir = tmp_path
        template_config = config_dir / "config.example.toml"
        template_config.write_text("invalid toml [[[")  # Malformed TOML

        # Act
        result = config.load_config(config_dir)

        # Assert - Should fall back to defaults
        assert result["sampling"]["base_quality_threshold"] == config.DEFAULT_BASE_QUALITY_THRESHOLD

    def test_load_config_without_tomllib(self, tmp_path):
        """Test loading config when TOML parser is not available."""
        # Arrange
        config_dir = tmp_path

        # Mock both tomllib and tomli as unavailable
        with patch.dict("sys.modules", {"tomllib": None, "tomli": None}):
            with patch("builtins.__import__", side_effect=ImportError("No TOML parser")):
                # Act
                result = config.load_config(config_dir)

        # Assert - Should use hardcoded defaults
        assert result["sampling"]["base_quality_threshold"] == config.DEFAULT_BASE_QUALITY_THRESHOLD

    def test_load_config_default_directory(self):
        """Test loading config with default directory (None)."""
        # Act
        result = config.load_config(config_dir=None)

        # Assert - Should successfully load (either from files or defaults)
        assert "sampling" in result
        assert "boundaries" in result
        assert "logging" in result


class TestGetConfigValues:
    """Tests for configuration value extraction functions."""

    def test_get_base_quality_threshold_from_config(self):
        """Test extracting base quality threshold from config dict."""
        # Arrange
        test_config = {"sampling": {"base_quality_threshold": 50}}

        # Act
        result = config.get_base_quality_threshold(test_config)

        # Assert
        assert result == 50

    def test_get_base_quality_threshold_missing_key(self):
        """Test extracting base quality threshold with missing key."""
        # Arrange
        test_config = {}  # Empty config

        # Act
        result = config.get_base_quality_threshold(test_config)

        # Assert
        assert result == config.DEFAULT_BASE_QUALITY_THRESHOLD

    def test_get_boundary_t2v_from_config(self):
        """Test extracting T2V boundary from config dict."""
        # Arrange
        test_config = {"boundaries": {"default_t2v": 0.850}}

        # Act
        result = config.get_boundary_t2v(test_config)

        # Assert
        assert result == 0.850

    def test_get_boundary_t2v_missing_key(self):
        """Test extracting T2V boundary with missing key."""
        # Arrange
        test_config = {}  # Empty config

        # Act
        result = config.get_boundary_t2v(test_config)

        # Assert
        assert result == config.DEFAULT_BOUNDARY_T2V

    def test_get_boundary_i2v_from_config(self):
        """Test extracting I2V boundary from config dict."""
        # Arrange
        test_config = {"boundaries": {"default_i2v": 0.925}}

        # Act
        result = config.get_boundary_i2v(test_config)

        # Assert
        assert result == 0.925

    def test_get_boundary_i2v_missing_key(self):
        """Test extracting I2V boundary with missing key."""
        # Arrange
        test_config = {}  # Empty config

        # Act
        result = config.get_boundary_i2v(test_config)

        # Assert
        assert result == config.DEFAULT_BOUNDARY_I2V

    def test_get_log_level_string_from_config(self):
        """Test extracting log level string from config dict."""
        # Arrange
        test_config = {"logging": {"level": "DEBUG"}}

        # Act
        result = config.get_log_level_string(test_config)

        # Assert
        assert result == "DEBUG"

    def test_get_log_level_string_missing_key(self):
        """Test extracting log level string with missing key."""
        # Arrange
        test_config = {}  # Empty config

        # Act
        result = config.get_log_level_string(test_config)

        # Assert
        assert result == config.DEFAULT_LOG_LEVEL

    def test_get_log_level_debug(self):
        """Test converting DEBUG log level string to constant."""
        # Arrange
        test_config = {"logging": {"level": "DEBUG"}}

        # Act
        result = config.get_log_level(test_config)

        # Assert
        assert result == logging.DEBUG

    def test_get_log_level_info(self):
        """Test converting INFO log level string to constant."""
        # Arrange
        test_config = {"logging": {"level": "INFO"}}

        # Act
        result = config.get_log_level(test_config)

        # Assert
        assert result == logging.INFO

    def test_get_log_level_uppercase_conversion(self):
        """Test that log level string is case-insensitive."""
        # Arrange
        test_config_lower = {"logging": {"level": "debug"}}
        test_config_mixed = {"logging": {"level": "DeBuG"}}

        # Act
        result_lower = config.get_log_level(test_config_lower)
        result_mixed = config.get_log_level(test_config_mixed)

        # Assert
        assert result_lower == logging.DEBUG
        assert result_mixed == logging.DEBUG

    def test_get_log_level_invalid_defaults_to_info(self):
        """Test that invalid log level defaults to INFO."""
        # Arrange
        test_config = {"logging": {"level": "INVALID"}}

        # Act
        result = config.get_log_level(test_config)

        # Assert
        assert result == logging.INFO


class TestGetDefaultConfig:
    """Tests for _get_default_config function."""

    def test_get_default_config_structure(self):
        """Test that default config has correct structure."""
        # Act
        result = config._get_default_config()

        # Assert
        assert "sampling" in result
        assert "boundaries" in result
        assert "logging" in result
        assert "base_quality_threshold" in result["sampling"]
        assert "default_t2v" in result["boundaries"]
        assert "default_i2v" in result["boundaries"]
        assert "level" in result["logging"]

    def test_get_default_config_values(self):
        """Test that default config has expected default values."""
        # Act
        result = config._get_default_config()

        # Assert
        assert result["sampling"]["base_quality_threshold"] == config.DEFAULT_BASE_QUALITY_THRESHOLD
        assert result["boundaries"]["default_t2v"] == config.DEFAULT_BOUNDARY_T2V
        assert result["boundaries"]["default_i2v"] == config.DEFAULT_BOUNDARY_I2V
        assert result["logging"]["level"] == config.DEFAULT_LOG_LEVEL


class TestConstants:
    """Tests for module constants."""

    def test_default_base_quality_threshold(self):
        """Test DEFAULT_BASE_QUALITY_THRESHOLD constant value."""
        assert config.DEFAULT_BASE_QUALITY_THRESHOLD == 20

    def test_default_boundary_t2v(self):
        """Test DEFAULT_BOUNDARY_T2V constant value."""
        assert config.DEFAULT_BOUNDARY_T2V == 0.875

    def test_default_boundary_i2v(self):
        """Test DEFAULT_BOUNDARY_I2V constant value."""
        assert config.DEFAULT_BOUNDARY_I2V == 0.900

    def test_default_log_level(self):
        """Test DEFAULT_LOG_LEVEL constant value."""
        assert config.DEFAULT_LOG_LEVEL == "INFO"
