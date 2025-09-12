"""
Configuration constants for ComfyUI-TripleKSampler.

This module contains configurable constants that affect the behavior of the
TripleKSampler nodes. These values can be modified to tune the sampling
behavior without changing the core implementation.
"""

# Quality threshold for automatic base step calculation
# This ensures that auto-computed base_steps result in sufficient total sampling
# steps for good quality output. The formula used is:
# base_steps = lightning_start * ceil(MIN_TOTAL_STEPS / lightning_steps)
MIN_TOTAL_STEPS = 20

# Development toggle for latent consistency debugging
# When enabled, performs additional validation checks on latent tensors
# between sampling stages. Useful for development and debugging but may
# impact performance. Should remain False for production use.
ENABLE_CONSISTENCY_CHECK = False

# Default sigma boundaries for different model types
# These values are used when sigma boundary-based model switching is enabled
DEFAULT_BOUNDARY_T2V = 0.875  # Text-to-video models
DEFAULT_BOUNDARY_I2V = 0.900  # Image-to-video models

# Experimental features
# Enable efficient total steps calculation (EXPERIMENTAL)
# When True: total_base_steps = MIN_TOTAL_STEPS (faster, but may cause stage overlap)
# When False: uses current calculation (safe, no overlap, but more steps)
USE_EFFICIENT_TOTAL_STEPS = False

# Logging configuration
LOGGER_PREFIX = "[TripleKSampler]"
LOG_LEVEL = "INFO"  # Can be: DEBUG, INFO, WARNING, ERROR