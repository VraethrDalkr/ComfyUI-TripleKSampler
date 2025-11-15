"""Regression tests for sigma shift refinement backward compatibility.

This module tests that the addition of refined strategies maintains full backward
compatibility with existing workflows, configurations, and behavior.
"""

from __future__ import annotations

# Check if WVSampler is available (direct filesystem check for test environment)
import os
import sys

import pytest

# Import node classes from package
from triple_ksampler.ksampler import advanced as ksampler_advanced
from triple_ksampler.ksampler import simple as ksampler_simple
from triple_ksampler.shared import config as tk_config
from triple_ksampler.shared import strategy_nodes

WVSAMPLER_AVAILABLE = False
try:
    # Check if WanVideoWrapper exists in custom_nodes (3 levels up: regression/ -> tests/ -> project/ -> custom_nodes/)
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


class TestStrategyDropdownIndices:
    """Test that strategy dropdown indices are preserved for workflow JSON compatibility."""

    def test_advanced_strategy_list_order_preserved(self):
        """Test that Advanced node strategy list order is preserved."""
        # Arrange
        node = ksampler_advanced.TripleKSamplerAdvanced()
        input_types = node.INPUT_TYPES()

        # Get strategy list
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Verify original strategies are in first 5 positions
        assert strategy_list[0] == "50% of steps"
        assert strategy_list[1] == "Manual switch step"
        assert strategy_list[2] == "T2V boundary"
        assert strategy_list[3] == "I2V boundary"
        assert strategy_list[4] == "Manual boundary"

        # Assert - Verify refined strategies are appended at end
        assert strategy_list[5] == "T2V boundary (refined)"
        assert strategy_list[6] == "I2V boundary (refined)"
        assert strategy_list[7] == "Manual boundary (refined)"

        # Assert - Total count
        assert len(strategy_list) == 8

    def test_simple_strategy_list_order_preserved(self):
        """Test that Simple node strategy list order is preserved."""
        # Arrange
        node = ksampler_simple.TripleKSampler()
        input_types = node.INPUT_TYPES()

        # Get strategy list
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Verify original strategies are in first 3 positions
        assert strategy_list[0] == "50% of steps"
        assert strategy_list[1] == "T2V boundary"
        assert strategy_list[2] == "I2V boundary"

        # Assert - Verify refined strategies are appended at end
        assert strategy_list[3] == "T2V boundary (refined)"
        assert strategy_list[4] == "I2V boundary (refined)"

        # Assert - Total count
        assert len(strategy_list) == 5

    @pytest.mark.skipif(not WVSAMPLER_AVAILABLE, reason="WanVideo wrapper not available")
    def test_wvsampler_advanced_strategy_list_order_preserved(self):
        """Test that WVSampler Advanced strategy list order is preserved."""
        # Arrange
        node = wvsampler_advanced.TripleWVSamplerAdvanced()
        input_types = node.INPUT_TYPES()

        # Get strategy list
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Verify original strategies are in first 5 positions
        assert strategy_list[0] == "50% of steps"
        assert strategy_list[1] == "Manual switch step"
        assert strategy_list[2] == "T2V boundary"
        assert strategy_list[3] == "I2V boundary"
        assert strategy_list[4] == "Manual boundary"

        # Assert - Verify refined strategies are appended at end
        assert strategy_list[5] == "T2V boundary (refined)"
        assert strategy_list[6] == "I2V boundary (refined)"
        assert strategy_list[7] == "Manual boundary (refined)"

        # Assert - Total count
        assert len(strategy_list) == 8

    @pytest.mark.skipif(not WVSAMPLER_AVAILABLE, reason="WanVideo wrapper not available")
    def test_wvsampler_simple_strategy_list_order_preserved(self):
        """Test that WVSampler Simple strategy list order is preserved."""
        # Arrange
        node = wvsampler_simple.TripleWVSampler()
        input_types = node.INPUT_TYPES()

        # Get strategy list
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Verify original strategies are in first 3 positions
        assert strategy_list[0] == "50% of steps"
        assert strategy_list[1] == "T2V boundary"
        assert strategy_list[2] == "I2V boundary"

        # Assert - Verify refined strategies are appended at end
        assert strategy_list[3] == "T2V boundary (refined)"
        assert strategy_list[4] == "I2V boundary (refined)"

        # Assert - Total count
        assert len(strategy_list) == 5


class TestConfigLoadingBackwardCompatibility:
    """Test that config loading works with missing refinement sections."""

    def test_config_loads_without_refinement_section(self):
        """Test that config loads successfully without [sigma_shift_refinement] section."""
        # Arrange - Mock config without refinement section (proper nested structure)
        mock_config = {
            "sampling": {"base_quality_threshold": 55},
            "boundaries": {"default_t2v": 0.875, "default_i2v": 0.900},
            "logging": {"level": "INFO"},
        }

        # Act - Should not raise exception
        base_threshold = tk_config.get_base_quality_threshold(mock_config)

        # Assert
        assert base_threshold == 55

    def test_default_config_has_no_refinement_section(self):
        """Test that default config does not include refinement parameters."""
        # Act
        default_config = tk_config._get_default_config()

        # Assert - Refinement section should not exist
        assert "sigma_shift_refinement" not in default_config

    def test_missing_config_uses_hardcoded_search_interval(self):
        """Test that missing config correctly falls back to hardcoded search interval."""
        # This is implicitly tested by the algorithm working without config
        # The _SEARCH_INTERVAL constant in models.py should be 0.01
        from triple_ksampler.shared import models

        # Assert
        assert hasattr(models, "_SEARCH_INTERVAL")
        assert models._SEARCH_INTERVAL == 0.01


class TestStrategyUtilityNodes:
    """Test that strategy utility nodes preserve backward compatibility."""

    def test_simple_strategy_node_output_compatible(self):
        """Test that SwitchStrategySimple output is compatible with Simple node."""
        # Arrange
        node = strategy_nodes.SwitchStrategySimple()

        # Act - Test non-refined strategy
        result = node.select_strategy("T2V boundary")

        # Assert
        assert result == ("T2V boundary",)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_simple_strategy_node_supports_refined_strategies(self):
        """Test that SwitchStrategySimple includes refined strategies."""
        # Arrange
        node = strategy_nodes.SwitchStrategySimple()
        input_types = node.INPUT_TYPES()

        # Get strategy list
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Should include refined strategies
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert len(strategy_list) == 5

    def test_advanced_strategy_node_output_compatible(self):
        """Test that SwitchStrategyAdvanced output is compatible with Advanced node."""
        # Arrange
        node = strategy_nodes.SwitchStrategyAdvanced()

        # Act - Test non-refined strategy
        result = node.select_strategy("Manual switch step")

        # Assert
        assert result == ("Manual switch step",)
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_advanced_strategy_node_supports_refined_strategies(self):
        """Test that SwitchStrategyAdvanced includes refined strategies."""
        # Arrange
        node = strategy_nodes.SwitchStrategyAdvanced()
        input_types = node.INPUT_TYPES()

        # Get strategy list
        strategy_list = input_types["required"]["switch_strategy"][0]

        # Assert - Should include refined strategies
        assert "T2V boundary (refined)" in strategy_list
        assert "I2V boundary (refined)" in strategy_list
        assert "Manual boundary (refined)" in strategy_list
        assert len(strategy_list) == 8
