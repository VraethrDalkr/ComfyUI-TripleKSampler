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

# Default sigma boundaries for different model types
# These values are used when sigma boundary-based model switching is enabled
DEFAULT_BOUNDARY_T2V = 0.875  # Text-to-video models
DEFAULT_BOUNDARY_I2V = 0.900  # Image-to-video models

# Logging configuration
# DEBUG: Shows all messages including input parameters and internal calculations
# INFO: Shows essential workflow information (auto-calculation, stage execution, model switching)
# WARNING and ERROR messages are always shown regardless of LOG_LEVEL setting
LOG_LEVEL = "DEBUG"
