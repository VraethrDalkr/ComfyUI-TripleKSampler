"""
ComfyUI-TripleKSampler: Advanced triple-stage sampling for Wan2.2 split models.

This package provides custom nodes for ComfyUI that implement a sophisticated
triple-stage sampling workflow optimized for Wan2.2 split models with Lightning LoRA.
"""

from .nodes import (
    TripleKSamplerWan22Lightning,
    TripleKSamplerWan22LightningAdvanced,
    SwitchStrategySimple,
    SwitchStrategyAdvanced,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__version__ = "0.8.5"
__author__ = "VraethrDalkr"
__description__ = "Triple-stage KSampler for Wan2.2 split models with Lightning LoRA"

# Export for ComfyUI node registration
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TripleKSamplerWan22Lightning",
    "TripleKSamplerWan22LightningAdvanced",
    "SwitchStrategySimple",
    "SwitchStrategyAdvanced",
]

# ComfyUI will look for these specific variables
WEB_DIRECTORY = "./web"