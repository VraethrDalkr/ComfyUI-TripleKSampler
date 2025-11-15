"""Unit tests for shared/logging.py.

Tests logging formatters, percentage calculations, and toast message compaction.
"""

import pytest

from triple_ksampler.shared import logging as core_logging


class TestCalculatePercentage:
    """Tests for calculate_percentage function."""

    def test_calculate_percentage_normal(self):
        """Test calculating percentage with normal values."""
        # Arrange & Act
        result = core_logging.calculate_percentage(5, 20)

        # Assert
        assert result == 25.0

    def test_calculate_percentage_whole_number(self):
        """Test calculating percentage that results in whole number."""
        # Arrange & Act
        result = core_logging.calculate_percentage(10, 20)

        # Assert
        assert result == 50.0

    def test_calculate_percentage_divide_by_zero(self):
        """Test calculating percentage with zero denominator."""
        # Arrange & Act
        result = core_logging.calculate_percentage(5, 0)

        # Assert
        assert result == 0.0

    def test_calculate_percentage_exceeds_100(self):
        """Test calculating percentage that exceeds 100%."""
        # Arrange & Act
        result = core_logging.calculate_percentage(15, 10)

        # Assert
        assert result == 100.0  # Clamped to 100

    def test_calculate_percentage_negative_clamped(self):
        """Test that negative percentages are clamped to 0."""
        # Arrange & Act
        result = core_logging.calculate_percentage(-5, 10)

        # Assert
        assert result == 0.0  # Clamped to 0

    def test_calculate_percentage_decimal_result(self):
        """Test calculating percentage with decimal result."""
        # Arrange & Act
        result = core_logging.calculate_percentage(1, 3)

        # Assert
        assert result == 33.3  # Rounded to 1 decimal

    def test_calculate_percentage_zero_numerator(self):
        """Test calculating percentage with zero numerator."""
        # Arrange & Act
        result = core_logging.calculate_percentage(0, 10)

        # Assert
        assert result == 0.0

    def test_calculate_percentage_float_inputs(self):
        """Test calculating percentage with float inputs."""
        # Arrange & Act
        result = core_logging.calculate_percentage(7.5, 15.0)

        # Assert
        assert result == 50.0


class TestFormatStageRange:
    """Tests for format_stage_range function."""

    def test_format_stage_range_normal(self):
        """Test formatting stage range with normal values."""
        # Arrange & Act
        result = core_logging.format_stage_range(0, 5, 20)

        # Assert
        assert result == "steps 0-5 of 20 (denoising 0.0%–25.0%)"

    def test_format_stage_range_full_range(self):
        """Test formatting stage range for full range."""
        # Arrange & Act
        result = core_logging.format_stage_range(0, 20, 20)

        # Assert
        assert result == "steps 0-20 of 20 (denoising 0.0%–100.0%)"

    def test_format_stage_range_middle_section(self):
        """Test formatting stage range for middle section."""
        # Arrange & Act
        result = core_logging.format_stage_range(10, 20, 20)

        # Assert
        assert result == "steps 10-20 of 20 (denoising 50.0%–100.0%)"

    def test_format_stage_range_single_step(self):
        """Test formatting stage range for single step."""
        # Arrange & Act
        result = core_logging.format_stage_range(5, 5, 10)

        # Assert
        assert result == "steps 5-5 of 10 (denoising 50.0%–50.0%)"

    def test_format_stage_range_negative_start_clamped(self):
        """Test that negative start is clamped to 0."""
        # Arrange & Act
        result = core_logging.format_stage_range(-5, 10, 20)

        # Assert
        assert result.startswith("steps 0-")

    def test_format_stage_range_end_before_start(self):
        """Test that end < start is clamped to start."""
        # Arrange & Act
        result = core_logging.format_stage_range(10, 5, 20)

        # Assert
        assert result == "steps 10-10 of 20 (denoising 50.0%–50.0%)"

    def test_format_stage_range_zero_total(self):
        """Test that zero total is clamped to 1."""
        # Arrange & Act
        result = core_logging.format_stage_range(0, 0, 0)

        # Assert
        assert "of 1" in result  # Total clamped to 1


class TestFormatBaseCalculationCompact:
    """Tests for format_base_calculation_compact function."""

    def test_format_base_calculation_simple_math(self):
        """Test formatting auto-calculated message with simple math."""
        # Arrange
        msg = "Auto-calculated base_steps = 10, total_base_steps = 50 (simple math)"

        # Act
        result = core_logging.format_base_calculation_compact(msg)

        # Assert
        assert result == "Base steps: 10, Total: 50 (simple math)"

    def test_format_base_calculation_mathematical_search(self):
        """Test formatting auto-calculated message with mathematical search."""
        # Arrange
        msg = "Auto-calculated base_steps = 8, total_base_steps = 20 (mathematical_search)"

        # Act
        result = core_logging.format_base_calculation_compact(msg)

        # Assert
        assert result == "Base steps: 8, Total: 20 (mathematical_search)"

    def test_format_base_calculation_fallback(self):
        """Test formatting fallback calculation message."""
        # Arrange
        msg = "Auto-calculated base_steps = 15 (fallback - no perfect alignment found)"

        # Act
        result = core_logging.format_base_calculation_compact(msg)

        # Assert
        assert result == "Base steps: 15 (fallback)"

    def test_format_base_calculation_manual(self):
        """Test formatting manual base_steps calculation message."""
        # Arrange
        msg = "Auto-calculated total_base_steps = 17 for manual base_steps = 10"

        # Act
        result = core_logging.format_base_calculation_compact(msg)

        # Assert
        assert result == "Base steps: 10, Total: 17 (manual)"

    def test_format_base_calculation_no_match_returns_original(self):
        """Test that unrecognized messages are returned unchanged."""
        # Arrange
        msg = "Some other message format"

        # Act
        result = core_logging.format_base_calculation_compact(msg)

        # Assert
        assert result == msg


class TestFormatSwitchInfoCompact:
    """Tests for format_switch_info_compact function."""

    def test_format_switch_info_t2v_boundary(self):
        """Test formatting T2V boundary switching message."""
        # Arrange
        msg = "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: T2V boundary → step 7 of 10"

    def test_format_switch_info_i2v_boundary(self):
        """Test formatting I2V boundary switching message."""
        # Arrange
        msg = "Model switching: I2V boundary (boundary = 0.900) → switch at step 8 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: I2V boundary → step 8 of 10"

    def test_format_switch_info_manual_boundary(self):
        """Test formatting manual boundary switching message."""
        # Arrange
        msg = "Model switching: Manual boundary (boundary = 0.85) → switch at step 6 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: Manual boundary → step 6 of 10"

    def test_format_switch_info_fifty_percent(self):
        """Test formatting 50% strategy switching message."""
        # Arrange
        msg = "Model switching: 50% of steps → switch at step 5 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: 50% of steps → step 5 of 10"

    def test_format_switch_info_seventy_five_percent(self):
        """Test formatting 75% strategy switching message."""
        # Arrange
        msg = "Model switching: 75% of steps → switch at step 7 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: 75% of steps → step 7 of 10"

    def test_format_switch_info_manual_switch_step(self):
        """Test formatting manual switch step message."""
        # Arrange
        msg = "Model switching: Manual switch step → switch at step 6 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: Manual switch step → step 6 of 10"

    def test_format_switch_info_fifty_percent_auto(self):
        """Test formatting 50% auto strategy message."""
        # Arrange
        msg = "Model switching: 50% of steps (auto) → switch at step 5 of 10"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == "Switch: 50% of steps (auto) → step 5 of 10"

    def test_format_switch_info_no_match_returns_original(self):
        """Test that unrecognized messages are returned unchanged."""
        # Arrange
        msg = "Some other switch message format"

        # Act
        result = core_logging.format_switch_info_compact(msg)

        # Assert
        assert result == msg
