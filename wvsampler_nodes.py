"""Triple-stage WanVideo sampler for Wan2.2 split models with Lightning LoRA.

This module implements triple-stage sampling nodes for WanVideo models,
specifically designed for Wan2.2 split models with Lightning LoRA integration
via ComfyUI-WanVideoWrapper.

The sampling process includes base denoising, lightning high-model
processing, and lightning low-model refinement stages using WanVideoSampler.

PRIMARY ENTRY POINT: This file imports all classes from
triple_ksampler.wvsampler package. The actual implementations
are in triple_ksampler/wvsampler/ package.
"""

from __future__ import annotations

from pathlib import Path

# Import shared config for logging setup
from triple_ksampler.shared import config as core_config

# Load configuration at import time (delegate to shared.config)
_CONFIG = core_config.load_config(config_dir=Path(__file__).resolve().parent)

# Extract log level for logging configuration
_LOG_LEVEL = core_config.get_log_level(_CONFIG)

# ============================================================================
# Logging Configuration
# ============================================================================
#
# TripleWVSampler uses the same dual-logger system as TripleKSampler:
# 1. Main logger: Structured messages with [TripleKSampler] prefix
# 2. Bare logger: Clean visual separators without prefixes
#
# The log level is controlled by config.toml [logging] level setting:
# - "DEBUG": Shows all messages including detailed calculations
# - "INFO": Shows essential workflow information only
# - WARNING and ERROR always appear regardless of level
#
# NOTE: Logging is now centralized in triple_ksampler.shared.logging_config
# ============================================================================

# Import centralized logging configuration
from triple_ksampler.shared.logging_config import configure_package_logging

# Configure package-wide logging hierarchy
logger = configure_package_logging(log_level=_LOG_LEVEL)

# ============================================================================
# Import node classes from new module structure
# ============================================================================

# Import node classes from package
# NOTE: Filesystem check for WanVideo availability happens in __init__.py
# This module is only imported if ComfyUI-WanVideoWrapper directory exists
try:
    from triple_ksampler.wvsampler import (
        TripleWVSampler,
        TripleWVSamplerAdvanced,
        TripleWVSamplerAdvancedAlt,
    )

    # Set availability flag (filesystem check done in __init__.py)
    # NO verification call here to avoid import-time load order issues
    WANVIDEO_AVAILABLE = True

except ImportError:
    # WanVideo not available - set flag, no node registration
    WANVIDEO_AVAILABLE = False
    TripleWVSampler = None
    TripleWVSamplerAdvanced = None
    TripleWVSamplerAdvancedAlt = None

# Re-export for backward compatibility
__all__ = [
    "TripleWVSamplerAdvancedAlt",
    "TripleWVSamplerAdvanced",
    "TripleWVSampler",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WANVIDEO_AVAILABLE",  # Export availability flag
]

# ============================================================================
# ComfyUI Node Registration
# ============================================================================

# Conditionally register nodes based on WanVideo availability
if WANVIDEO_AVAILABLE:
    # WanVideo available - register all nodes
    NODE_CLASS_MAPPINGS = {
        "TripleWVSamplerAdvancedAlt": TripleWVSamplerAdvancedAlt,
        "TripleWVSamplerAdvanced": TripleWVSamplerAdvanced,
        "TripleWVSampler": TripleWVSampler,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "TripleWVSamplerAdvancedAlt": "TripleWVSampler (Advanced Alt)",
        "TripleWVSamplerAdvanced": "TripleWVSampler (Advanced)",
        "TripleWVSampler": "TripleWVSampler (Simple)",
    }
else:
    # WanVideo not available - don't register any nodes
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
