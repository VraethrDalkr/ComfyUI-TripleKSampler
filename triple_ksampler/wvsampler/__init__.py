"""ComfyUI node implementations for WanVideo triple-stage sampling.

This package contains the node class implementations for TripleWVSampler,
organized by complexity and purpose:

- utils.py: get_wanvideo_components - WanVideo lazy loader
- advanced.py: TripleWVSamplerAdvancedAlt + TripleWVSamplerAdvanced - Full params (static/dynamic UI)
- simple.py: TripleWVSampler - Simplified interface with smart defaults

Base class (TripleKSamplerBase) is in triple_ksampler.base (package root).
All nodes delegate calculation logic to triple_ksampler.shared modules.
"""

from .advanced import TripleWVSamplerAdvanced, TripleWVSamplerAdvancedAlt
from .simple import TripleWVSampler
from .utils import get_wanvideo_components

__all__ = [
    "TripleWVSamplerAdvancedAlt",
    "TripleWVSamplerAdvanced",
    "TripleWVSampler",
    "get_wanvideo_components",
]
