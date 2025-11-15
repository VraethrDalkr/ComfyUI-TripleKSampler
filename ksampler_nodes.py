"""Triple-stage KSampler for Wan2.2 split models with Lightning LoRA.

This module implements triple-stage sampling nodes for ComfyUI,
specifically designed for Wan2.2 split models with Lightning LoRA
integration.

The sampling process includes base denoising, lightning high-model
processing, and lightning low-model refinement stages.

PRIMARY ENTRY POINT: This file imports all classes from
triple_ksampler.ksampler package. The actual implementations
are in triple_ksampler/ksampler/ package.
"""

from __future__ import annotations

from pathlib import Path

# Import shared config for logging setup
from triple_ksampler.shared import config as core_config

# Load configuration at import time (delegate to core.config)
_CONFIG = core_config.load_config(config_dir=Path(__file__).resolve().parent)

# Extract log level for logging configuration
_LOG_LEVEL = core_config.get_log_level(_CONFIG)

# ============================================================================
# Logging Configuration
# ============================================================================
#
# TripleKSampler uses a dual-logger system:
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

# Import all node classes from the refactored module structure
from triple_ksampler.ksampler import (
    SwitchStrategyAdvanced,
    SwitchStrategySimple,
    TripleKSampler,
    TripleKSamplerAdvanced,
    TripleKSamplerAdvancedAlt,
    TripleKSamplerBase,
)

# Re-export for backward compatibility
__all__ = [
    "TripleKSamplerBase",
    "TripleKSampler",
    "TripleKSamplerAdvanced",
    "TripleKSamplerAdvancedAlt",
    "SwitchStrategySimple",
    "SwitchStrategyAdvanced",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

# ============================================================================
# ComfyUI Node Registration
# ============================================================================

# Node registration mapping (ComfyUI expects these names)
NODE_CLASS_MAPPINGS = {
    "TripleKSamplerWan22Lightning": TripleKSampler,
    "TripleKSamplerWan22LightningAdvanced": TripleKSamplerAdvanced,
    "TripleKSamplerWan22LightningAdvancedAlt": TripleKSamplerAdvancedAlt,
    "SwitchStrategySimple": SwitchStrategySimple,
    "SwitchStrategyAdvanced": SwitchStrategyAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripleKSamplerWan22Lightning": "TripleKSampler (Simple)",
    "TripleKSamplerWan22LightningAdvanced": "TripleKSampler (Advanced)",
    "TripleKSamplerWan22LightningAdvancedAlt": "TripleKSampler (Advanced Alt)",
    "SwitchStrategySimple": "Switch Strategy (Simple)",
    "SwitchStrategyAdvanced": "Switch Strategy (Advanced)",
}
