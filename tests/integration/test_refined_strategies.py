"""Integration tests for sigma shift refinement with refined strategies.

This module tests that refined strategies are properly integrated across all node
variants, focusing on structural validation and strategy availability rather than
complex mocking of sampling behavior (which is covered by existing test_validation.py
and test_integration_focused.py).
"""

from __future__ import annotations

# Check if WVSampler is available (direct filesystem check for test environment)
import os
import sys

import pytest

# Import node classes from package
from triple_ksampler.ksampler import advanced as ksampler_advanced
from triple_ksampler.ksampler import simple as ksampler_simple
from triple_ksampler.shared import strategy_nodes

WVSAMPLER_AVAILABLE = False
try:
    # Check if WanVideoWrapper exists in custom_nodes (3 levels up: integration/ -> tests/ -> project/ -> custom_nodes/)
    test_dir = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.join(test_dir, "../../..", "ComfyUI-WanVideoWrapper")

    # Verify the wrapper exists by checking for key file
    if os.path.exists(custom_nodes_dir) and os.path.exists(
        os.path.join(custom_nodes_dir, "nodes_sampler.py")
    ):
        # Try importing WVSampler nodes
        from triple_ksampler.wvsampler import advanced as wvsampler_advanced
        from triple_ksampler.wvsampler import simple as wvsampler_simple

        WVSAMPLER_AVAILABLE = True
except Exception:
    # Any import or path error means WVSampler not available
    WVSAMPLER_AVAILABLE = False


class TestRefinedStrategyAvailability:
    """Test that refined strategies are available in all node variants."""

    def test_ksampler_advanced_has_refined_strategies(self):
        """Test that TripleKSamplerAdvanced includes all 3 refined strategies."""
        # Arrange
        node = ksampler_advanced.TripleKSamplerAdvanced()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Should include refined strategies
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert "Manual boundary (refined)" in strategy_list
        assert len(strategy_list) == 8  # 5 base + 3 refined

    def test_ksampler_simple_has_refined_strategies(self):
        """Test that TripleKSampler (Simple) includes 2 refined strategies."""
        # Arrange
        node = ksampler_simple.TripleKSampler()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Should include refined strategies
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert len(strategy_list) == 5  # 3 base + 2 refined

    @pytest.mark.skipif(not WVSAMPLER_AVAILABLE, reason="WanVideo wrapper not available")
    def test_wvsampler_advanced_has_refined_strategies(self):
        """Test that TripleWVSamplerAdvanced includes all 3 refined strategies."""
        # Arrange
        node = wvsampler_advanced.TripleWVSamplerAdvanced()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Should include refined strategies
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert "Manual boundary (refined)" in strategy_list
        assert len(strategy_list) == 8  # 5 base + 3 refined

    @pytest.mark.skipif(not WVSAMPLER_AVAILABLE, reason="WanVideo wrapper not available")
    def test_wvsampler_simple_has_refined_strategies(self):
        """Test that TripleWVSampler (Simple) includes 2 refined strategies."""
        # Arrange
        node = wvsampler_simple.TripleWVSampler()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Should include refined strategies
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert len(strategy_list) == 5  # 3 base + 2 refined


class TestStrategyUtilityNodesRefinement:
    """Test that strategy utility nodes support refined strategies."""

    def test_simple_strategy_node_includes_refined_strategies(self):
        """Test that SwitchStrategySimple includes refined strategies."""
        # Arrange
        node = strategy_nodes.SwitchStrategySimple()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert len(strategy_list) == 5

    def test_advanced_strategy_node_includes_refined_strategies(self):
        """Test that SwitchStrategyAdvanced includes refined strategies."""
        # Arrange
        node = strategy_nodes.SwitchStrategyAdvanced()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert "Manual boundary (refined)" in strategy_list
        assert len(strategy_list) == 8


class TestRefinedStrategyTooltips:
    """Test that refined strategies have appropriate tooltips/descriptions."""

    def test_ksampler_advanced_refined_strategy_tooltip(self):
        """Test that Advanced node has informative tooltip for refined strategies."""
        # Arrange
        node = ksampler_advanced.TripleKSamplerAdvanced()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_spec = input_types["required"]["switch_strategy"]
        tooltip = strategy_spec[1].get("tooltip", "")

        # Assert - Tooltip should mention refinement
        assert "refined" in tooltip.lower()
        assert "sigma_shift" in tooltip.lower() or "sigma shift" in tooltip.lower()

    def test_ksampler_simple_refined_strategy_tooltip(self):
        """Test that Simple node has informative tooltip for refined strategies."""
        # Arrange
        node = ksampler_simple.TripleKSampler()
        input_types = node.INPUT_TYPES()

        # Act
        strategy_spec = input_types["required"]["switch_strategy"]
        tooltip = strategy_spec[1].get("tooltip", "")

        # Assert - Tooltip should mention refinement
        assert "refined" in tooltip.lower()
        assert "sigma_shift" in tooltip.lower() or "sigma shift" in tooltip.lower()
