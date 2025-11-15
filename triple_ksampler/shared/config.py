"""Configuration loading and management for TripleKSampler.

This module handles TOML configuration file loading with a hierarchical fallback system:
1. config.toml (user editable, gitignored)
2. config.example.toml (template tracked in git)
3. Hardcoded defaults

The configuration system is designed to be user-friendly while providing sensible defaults
for all settings. If config.toml doesn't exist but config.example.toml does, the template
will be automatically copied to create an editable user configuration.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

# Default configuration values
DEFAULT_BASE_QUALITY_THRESHOLD = 20
DEFAULT_BOUNDARY_T2V = 0.875
DEFAULT_BOUNDARY_I2V = 0.900
DEFAULT_LOG_LEVEL = "INFO"


def load_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load configuration from TOML files or fallback to defaults.

    Implements a hierarchical fallback system with automatic template copying:
    1. Try loading config.toml (user-editable, gitignored)
    2. If not found, try copying from config.example.toml
    3. Try loading config.example.toml (template in git)
    4. Fallback to hardcoded defaults if all else fails

    Args:
        config_dir: Directory containing config files. If None, uses the
            parent directory of this module (default: None).

    Returns:
        Configuration dict with keys:
            - "sampling": {"base_quality_threshold": int}
            - "boundaries": {"default_t2v": float, "default_i2v": float}
            - "logging": {"level": str}

    Example:
        >>> config = load_config()
        >>> threshold = config["sampling"]["base_quality_threshold"]
        >>> print(threshold)
        20
    """
    # Try to import TOML parser (Python 3.11+ or tomli package)
    tomllib = None
    try:
        # Python 3.11+
        import tomllib as tomllib  # type: ignore
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning(
                "[TripleKSampler] TOML parser not available. "
                "Install 'tomli' for Python < 3.11 to enable config.toml support."
            )
            return _get_default_config()

    # Determine config directory
    if config_dir is None:
        # Default to parent directory of shared/ (triple_ksampler/ root)
        config_dir = Path(__file__).resolve().parent.parent.parent

    user_config_path = config_dir / "config.toml"
    template_config_path = config_dir / "config.example.toml"

    # Auto-create config.toml from template if necessary
    if not user_config_path.exists() and template_config_path.exists():
        try:
            shutil.copy2(template_config_path, user_config_path)
            logging.getLogger(__name__).info("[TripleKSampler] Created config.toml from template")
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "[TripleKSampler] Failed to create config.toml from template: %s", exc
            )

    # Try user config first
    if user_config_path.exists():
        try:
            with user_config_path.open("rb") as f:
                return tomllib.load(f)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[TripleKSampler] Failed to load user config.toml: %s", exc
            )

    # Try template config
    if template_config_path.exists():
        try:
            with template_config_path.open("rb") as f:
                return tomllib.load(f)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[TripleKSampler] Failed to load config.example.toml: %s", exc
            )

    # Final fallback to hardcoded defaults
    return _get_default_config()


def _get_default_config() -> dict[str, Any]:
    """Return hardcoded default configuration.

    This is the final fallback when no TOML files can be loaded.

    Returns:
        Default configuration dict
    """
    return {
        "sampling": {"base_quality_threshold": DEFAULT_BASE_QUALITY_THRESHOLD},
        "boundaries": {
            "default_t2v": DEFAULT_BOUNDARY_T2V,
            "default_i2v": DEFAULT_BOUNDARY_I2V,
        },
        "logging": {"level": DEFAULT_LOG_LEVEL},
    }


def get_base_quality_threshold(config: dict[str, Any]) -> int:
    """Extract base quality threshold from configuration.

    Args:
        config: Configuration dict from load_config()

    Returns:
        Base quality threshold value (default: 20)

    Example:
        >>> config = load_config()
        >>> threshold = get_base_quality_threshold(config)
        >>> print(threshold >= 1)
        True
    """
    return config.get("sampling", {}).get("base_quality_threshold", DEFAULT_BASE_QUALITY_THRESHOLD)


def get_boundary_t2v(config: dict[str, Any]) -> float:
    """Extract T2V boundary value from configuration.

    Args:
        config: Configuration dict from load_config()

    Returns:
        T2V boundary value (default: 0.875)

    Example:
        >>> config = load_config()
        >>> boundary = get_boundary_t2v(config)
        >>> print(0.0 <= boundary <= 1.0)
        True
    """
    return config.get("boundaries", {}).get("default_t2v", DEFAULT_BOUNDARY_T2V)


def get_boundary_i2v(config: dict[str, Any]) -> float:
    """Extract I2V boundary value from configuration.

    Args:
        config: Configuration dict from load_config()

    Returns:
        I2V boundary value (default: 0.900)

    Example:
        >>> config = load_config()
        >>> boundary = get_boundary_i2v(config)
        >>> print(0.0 <= boundary <= 1.0)
        True
    """
    return config.get("boundaries", {}).get("default_i2v", DEFAULT_BOUNDARY_I2V)


def get_log_level_string(config: dict[str, Any]) -> str:
    """Extract log level string from configuration.

    Args:
        config: Configuration dict from load_config()

    Returns:
        Log level string ("DEBUG" or "INFO", default: "INFO")

    Example:
        >>> config = load_config()
        >>> level = get_log_level_string(config)
        >>> print(level in ["DEBUG", "INFO"])
        True
    """
    return config.get("logging", {}).get("level", DEFAULT_LOG_LEVEL)


def get_log_level(config: dict[str, Any]) -> int:
    """Convert log level string to logging module constant.

    Only supports DEBUG and INFO levels. WARNING and ERROR messages
    are always shown regardless of LOG_LEVEL setting.

    Args:
        config: Configuration dict from load_config()

    Returns:
        logging level constant (logging.DEBUG or logging.INFO)

    Example:
        >>> import logging
        >>> config = load_config()
        >>> level = get_log_level(config)
        >>> print(level in [logging.DEBUG, logging.INFO])
        True
    """
    log_level_str = get_log_level_string(config)
    if str(log_level_str).upper() == "DEBUG":
        return logging.DEBUG
    return logging.INFO
