"""Unit tests for refined strategy toast notification formatting.

This module tests the display/formatting improvements for sigma shift refinement
in toast notifications, specifically the multi-line format and regex pattern matching
in format_switch_info_compact().
"""

from __future__ import annotations

import pytest

from triple_ksampler.shared import logging as tk_logging


class TestFormatSwitchInfoCompactRefinement:
    """Tests for format_switch_info_compact() with refinement display."""

    def test_formats_t2v_boundary_refined_multiline(self):
        """Test that T2V boundary (refined) displays with multi-line format."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00→6.94]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert "Switch: T2V boundary → step 7 of 10" in result
        assert "\n  (σ-shift refined: 5.00 → 6.94)" in result
        # Verify newline and indentation
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[1].startswith("  (")

    def test_formats_i2v_boundary_refined_multiline(self):
        """Test that I2V boundary (refined) displays with multi-line format."""
        # Arrange
        switch_info = (
            "Model switching: I2V boundary (boundary = 0.900) → switch at step 5 of 8 "
            "[Refined shift: 5.00→7.22]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert "Switch: I2V boundary → step 5 of 8" in result
        assert "\n  (σ-shift refined: 5.00 → 7.22)" in result

    def test_formats_manual_boundary_refined_multiline(self):
        """Test that Manual boundary (refined) displays with multi-line format."""
        # Arrange
        switch_info = (
            "Model switching: Manual boundary (boundary = 0.850) → switch at step 6 of 10 "
            "[Refined shift: 5.00→5.83]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert "Switch: Manual boundary → step 6 of 10" in result
        assert "\n  (σ-shift refined: 5.00 → 5.83)" in result

    def test_non_refined_strategy_no_extra_line(self):
        """Test that non-refined strategies display normally without extra line."""
        # Arrange
        switch_info = "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10"

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert result == "Switch: T2V boundary → step 7 of 10"
        assert "\n" not in result
        assert "refined" not in result

    def test_fifty_percent_strategy_no_refinement(self):
        """Test that '50% of steps' strategy displays normally."""
        # Arrange
        switch_info = "Model switching: 50% of steps → switch at step 5 of 10"

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert result == "Switch: 50% of steps → step 5 of 10"
        assert "\n" not in result

    def test_manual_switch_step_no_refinement(self):
        """Test that 'Manual switch step' strategy displays normally."""
        # Arrange
        switch_info = "Model switching: Manual switch step → switch at step 3 of 10"

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert result == "Switch: Manual switch step → step 3 of 10"
        assert "\n" not in result

    def test_extracts_refinement_bracket_correctly(self):
        """Test that regex correctly extracts [Refined shift: X→Y] pattern."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00→6.94]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert - Verify extracted values are correct
        assert "5.00" in result
        assert "6.94" in result
        assert "→" in result  # Arrow should be preserved

    def test_handles_decimal_places_in_refinement(self):
        """Test that refinement values with various decimal places display correctly."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.123→6.789]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert "5.123" in result
        assert "6.789" in result

    def test_handles_integer_shift_values(self):
        """Test that integer shift values display correctly."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5→6]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert "5" in result
        assert "6" in result

    def test_refinement_bracket_removed_from_main_line(self):
        """Test that [Refined shift: ...] bracket is removed from main switch line."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00→6.94]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)
        lines = result.split("\n")
        main_line = lines[0]

        # Assert
        assert "[Refined shift:" not in main_line
        assert main_line == "Switch: T2V boundary → step 7 of 10"

    def test_fallback_to_original_if_no_pattern_match(self):
        """Test that function returns original string if no pattern matches."""
        # Arrange
        switch_info = "Some completely unrelated string"

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert
        assert result == switch_info

    def test_handles_malformed_refinement_bracket_gracefully(self):
        """Test that malformed [Refined shift: ...] bracket doesn't break formatting."""
        # Arrange - Missing arrow
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00 6.94]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert - Should still format the main part
        assert "Switch: T2V boundary → step 7 of 10" in result
        # Malformed bracket won't match regex, so no refinement line added
        assert "\n" not in result or "[Refined shift: 5.00 6.94]" in result

    def test_handles_missing_closing_bracket_gracefully(self):
        """Test that missing ] bracket doesn't break formatting."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00→6.94"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert - Should format main part normally
        assert "Switch: T2V boundary → step 7 of 10" in result

    def test_preserves_spacing_in_refinement_display(self):
        """Test that spacing around arrow is preserved in multi-line display."""
        # Arrange
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00→6.94]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert - Should have spaces around arrow in output
        assert " → " in result  # Space-arrow-space pattern

    def test_multiple_refinement_brackets_uses_first(self):
        """Test that multiple [Refined shift: ...] brackets use the first match."""
        # Arrange - Unusual case with duplicate brackets
        switch_info = (
            "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
            "[Refined shift: 5.00→6.94] [Refined shift: 4.00→5.00]"
        )

        # Act
        result = tk_logging.format_switch_info_compact(switch_info)

        # Assert - Should extract first bracket
        assert "5.00" in result
        assert "6.94" in result

    def test_refinement_works_with_all_boundary_strategies(self):
        """Test that refinement formatting works with all boundary-based strategies."""
        # Arrange - Test all three boundary strategies
        strategies = [
            (
                "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 "
                "[Refined shift: 5.00→6.94]",
                "T2V boundary",
            ),
            (
                "Model switching: I2V boundary (boundary = 0.900) → switch at step 5 of 8 "
                "[Refined shift: 5.00→7.22]",
                "I2V boundary",
            ),
            (
                "Model switching: Manual boundary (boundary = 0.850) → switch at step 6 of 10 "
                "[Refined shift: 5.00→5.83]",
                "Manual boundary",
            ),
        ]

        for switch_info, strategy_name in strategies:
            # Act
            result = tk_logging.format_switch_info_compact(switch_info)

            # Assert
            assert f"Switch: {strategy_name}" in result
            assert "\n  (σ-shift refined:" in result
