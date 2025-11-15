"""Centralized logging configuration for all TripleKSampler modules.

This module provides a single function to configure the logging hierarchy for the entire
package. Both ksampler_nodes.py and wvsampler_nodes.py call this function to ensure
consistent logging configuration.

Design: DRY principle - configure logging once, use everywhere.
"""

import logging


def configure_package_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Configure the triple_ksampler package logging hierarchy.

    Sets up two loggers:
    1. Package root logger ("triple_ksampler") - All child loggers inherit from this
    2. Separator logger ("triple_ksampler.separator") - For blank line visual separators

    All child modules using logger names starting with "triple_ksampler.*" will
    automatically inherit the configuration.

    Args:
        log_level: Logging level to use (default: logging.INFO)

    Returns:
        logging.Logger: The configured package root logger

    Example:
        >>> from triple_ksampler.shared.logging_config import configure_package_logging
        >>> logger = configure_package_logging(logging.INFO)
        >>> logger.info("Package configured")
        [TripleKSampler] INFO: Package configured
    """
    # Configure package root logger (all "triple_ksampler.*" loggers inherit from this)
    package_logger = logging.getLogger("triple_ksampler")
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[TripleKSampler] %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        package_logger.addHandler(handler)
    package_logger.propagate = False
    package_logger.setLevel(log_level)

    # Configure bare logger for visual separators (clean empty lines)
    bare_logger = logging.getLogger("triple_ksampler.separator")
    if not bare_logger.handlers:
        bare_handler = logging.StreamHandler()
        bare_handler.setFormatter(logging.Formatter(""))  # No formatting for clean output
        bare_logger.addHandler(bare_handler)
    bare_logger.propagate = False
    bare_logger.setLevel(logging.INFO)  # Always show separators

    return package_logger
