"""Strategy utility nodes for external strategy control.

This module contains:
- SwitchStrategySimple: 5 strategies (50% of steps, T2V/I2V boundary + refined variants)
- SwitchStrategyAdvanced: 8 strategies (50%, Manual switch step, T2V/I2V/Manual boundary + refined variants)

These nodes allow users to calculate switching strategy externally and pass results
to the main sampling nodes, enabling workflow-level control and experimentation.
"""

from __future__ import annotations

from typing import Any


class SwitchStrategySimple:
    """Utility node for selecting switch strategies for TripleKSampler (Simple).

    Outputs a strategy value compatible with TripleKSampler (Simple).
    Supports the 5 strategies available in the simple node.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Return INPUT_TYPES for simple strategy selector."""
        return {
            "required": {
                "switch_strategy": (
                    [
                        "50% of steps",
                        "T2V boundary",
                        "I2V boundary",
                        "T2V boundary (refined)",
                        "I2V boundary (refined)",
                    ],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between models. Refined variants auto-tune sigma_shift for perfect boundary alignment at the switch step.",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        [
            "50% of steps",
            "T2V boundary",
            "I2V boundary",
            "T2V boundary (refined)",
            "I2V boundary (refined)",
        ],
    )
    RETURN_NAMES = ("switch_strategy",)
    FUNCTION = "select_strategy"
    CATEGORY = "TripleKSampler/utilities"
    DESCRIPTION = (
        "Strategy selector for TripleKSampler (Simple). "
        "Outputs one of 5 available strategies: 50% of steps, T2V/I2V boundary, and refined variants."
    )

    def select_strategy(self, switch_strategy: str) -> tuple[str]:
        """Return the selected strategy as a string tuple.

        Args:
            switch_strategy: Selected strategy name

        Returns:
            Tuple containing the strategy string
        """
        return (switch_strategy,)


class SwitchStrategyAdvanced:
    """Utility node for selecting switch strategies for TripleKSampler (Advanced).

    Outputs a strategy value compatible with TripleKSampler (Advanced).
    Supports all 8 strategies including manual control options and refined variants.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Return INPUT_TYPES for advanced strategy selector."""
        return {
            "required": {
                "switch_strategy": (
                    [
                        "50% of steps",
                        "Manual switch step",
                        "T2V boundary",
                        "I2V boundary",
                        "Manual boundary",
                        "T2V boundary (refined)",
                        "I2V boundary (refined)",
                        "Manual boundary (refined)",
                    ],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between models. Refined variants auto-tune sigma_shift for perfect boundary alignment at the switch step.",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        [
            "50% of steps",
            "Manual switch step",
            "T2V boundary",
            "I2V boundary",
            "Manual boundary",
            "T2V boundary (refined)",
            "I2V boundary (refined)",
            "Manual boundary (refined)",
        ],
    )
    RETURN_NAMES = ("switch_strategy",)
    FUNCTION = "select_strategy"
    CATEGORY = "TripleKSampler/utilities"
    DESCRIPTION = (
        "Strategy selector for TripleKSampler (Advanced). "
        "Outputs one of 8 available strategies including manual switch step, manual boundary control, and refined variants."
    )

    def select_strategy(self, switch_strategy: str) -> tuple[str]:
        """Return the selected strategy as a string tuple.

        Args:
            switch_strategy: Selected strategy name

        Returns:
            Tuple containing the strategy string
        """
        return (switch_strategy,)
