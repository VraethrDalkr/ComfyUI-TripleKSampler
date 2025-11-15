"""Core logic modules for TripleKSampler.

This package contains shared calculation, validation, and utility logic used by
both regular TripleKSampler nodes and WanVideo wrapper nodes.

Also includes strategy utility nodes (SwitchStrategySimple/Advanced) that work
with both ksampler and wvsampler node types.
"""

from .strategy_nodes import SwitchStrategyAdvanced, SwitchStrategySimple

__all__ = [
    "SwitchStrategySimple",
    "SwitchStrategyAdvanced",
]
