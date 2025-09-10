"""
ComfyUI-TripleKSampler: Advanced triple-stage sampling for Wan2.2 split models.

This package provides custom nodes for ComfyUI that implement a sophisticated
triple-stage sampling workflow optimized for Wan2.2 split models with Lightning LoRA.
"""

from .triple_ksampler_wan22 import (
    TripleKSamplerWan22Lightning,
    SimpleTripleKSamplerWan22Lightning,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__version__ = "1.0.0"
__author__ = "dduval"
__description__ = "Triple-stage KSampler for Wan2.2 split models with Lightning LoRA"

# Export for ComfyUI node registration
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TripleKSamplerWan22Lightning",
    "SimpleTripleKSamplerWan22Lightning",
]

# ComfyUI will look for these specific variables
WEB_DIRECTORY = "./web"