"""ComfyUI node implementations for triple-stage sampling.

This package contains the node class implementations for ComfyUI-TripleKSampler,
organized by complexity and purpose:

- simple.py: TripleKSampler - Simplified interface with smart defaults
- advanced.py: TripleKSamplerAdvanced/Alt - Full control with all parameters

Base class (TripleKSamplerBase) is in triple_ksampler.base (package root).
Strategy utility nodes (SwitchStrategySimple/Advanced) are in triple_ksampler.shared.

All nodes delegate calculation logic to triple_ksampler.shared modules.
"""

from triple_ksampler.base import TripleKSamplerBase
from triple_ksampler.shared.strategy_nodes import SwitchStrategyAdvanced, SwitchStrategySimple

from .advanced import TripleKSamplerAdvanced, TripleKSamplerAdvancedAlt
from .simple import TripleKSampler

__all__ = [
    "TripleKSamplerBase",
    "TripleKSampler",
    "TripleKSamplerAdvanced",
    "TripleKSamplerAdvancedAlt",
    "SwitchStrategySimple",
    "SwitchStrategyAdvanced",
]
