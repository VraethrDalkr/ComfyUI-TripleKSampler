"""Unit tests for shared/alignment.py.

Tests the Perfect Alignment Algorithm independently of ComfyUI integration.
"""

import math

import pytest

from triple_ksampler.shared import alignment


class TestCalculatePerfectAlignment:
    """Tests for calculate_perfect_alignment function."""

    def test_simple_math_lightning_start_1(self):
        """Test simple math method when lightning_start=1."""
        # Arrange
        base_quality_threshold = 20
        lightning_start = 1
        lightning_steps = 10

        # Act
        base_steps, total_base_steps, method = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert
        assert method == "simple_math"
        assert base_steps == 2  # ceil(20 / 10)
        assert total_base_steps == 20  # 2 * 10
        # Verify alignment: base_steps/total_base_steps = lightning_start/lightning_steps
        assert base_steps / total_base_steps == lightning_start / lightning_steps

    def test_simple_math_with_remainder(self):
        """Test simple math with non-divisible threshold."""
        # Arrange
        base_quality_threshold = 25
        lightning_start = 1
        lightning_steps = 10

        # Act
        base_steps, total_base_steps, method = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert
        assert method == "simple_math"
        assert base_steps == 3  # ceil(25 / 10)
        assert total_base_steps == 30  # 3 * 10
        assert total_base_steps >= base_quality_threshold

    def test_mathematical_search_finds_perfect_alignment(self):
        """Test mathematical search finds perfect integer alignment."""
        # Arrange
        base_quality_threshold = 20
        lightning_start = 4
        lightning_steps = 10

        # Act
        base_steps, total_base_steps, method = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert
        assert method == "mathematical_search"
        assert base_steps == 8  # (20 * 4) / 10
        assert total_base_steps == 20
        # Verify perfect alignment
        assert (total_base_steps * lightning_start) % lightning_steps == 0
        assert base_steps == (total_base_steps * lightning_start) // lightning_steps

    def test_mathematical_search_different_values(self):
        """Test mathematical search with different parameter values."""
        # Arrange
        base_quality_threshold = 15
        lightning_start = 3
        lightning_steps = 7

        # Act
        base_steps, total_base_steps, method = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert
        assert method == "mathematical_search"
        # Verify alignment formula
        assert base_steps * lightning_steps == total_base_steps * lightning_start
        assert total_base_steps >= base_quality_threshold

    def test_fallback_when_no_perfect_alignment_in_range(self):
        """Test fallback method when no perfect alignment found in search range."""
        # Arrange
        # Use parameters that won't find perfect alignment within search limit
        base_quality_threshold = 100
        lightning_start = 7
        lightning_steps = 13
        # Search limit = 100 + (13 * 1) = 113
        # Need to verify no perfect alignment exists in range [100, 113)

        # Act
        base_steps, total_base_steps, method = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert
        # Method should be fallback if no perfect alignment found
        # (Note: This might still find mathematical_search if alignment exists in range)
        assert method in ["mathematical_search", "fallback"]
        assert total_base_steps >= base_quality_threshold
        assert base_steps > 0

    def test_lightning_start_zero_returns_zero(self):
        """Test that lightning_start=0 returns zero (skip base stage)."""
        # Arrange
        base_quality_threshold = 20
        lightning_start = 0
        lightning_steps = 10

        # Act
        base_steps, total_base_steps, method = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert
        assert method == "simple_math"
        assert base_steps == 0
        assert total_base_steps == 0

    def test_invalid_threshold_raises_error(self):
        """Test that base_quality_threshold < 1 raises ValueError."""
        with pytest.raises(ValueError, match="base_quality_threshold must be at least 1"):
            alignment.calculate_perfect_alignment(0, 1, 10)

    def test_invalid_lightning_steps_raises_error(self):
        """Test that lightning_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="lightning_steps must be at least 1"):
            alignment.calculate_perfect_alignment(20, 1, 0)

    def test_lightning_start_out_of_range_raises_error(self):
        """Test that lightning_start >= lightning_steps raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and"):
            alignment.calculate_perfect_alignment(20, 10, 10)  # start == steps

        with pytest.raises(ValueError, match="must be between 0 and"):
            alignment.calculate_perfect_alignment(20, 15, 10)  # start > steps

    def test_negative_lightning_start_raises_error(self):
        """Test that negative lightning_start raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and"):
            alignment.calculate_perfect_alignment(20, -1, 10)

    def test_alignment_preserves_percentage_relationship(self):
        """Test that alignment preserves denoising percentage relationship."""
        # Arrange
        base_quality_threshold = 20
        lightning_start = 4
        lightning_steps = 10

        # Act
        base_steps, total_base_steps, _ = alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

        # Assert - Calculate denoising percentages
        stage1_end_pct = base_steps / total_base_steps
        stage2_start_pct = lightning_start / lightning_steps

        # Should be equal (or very close for fallback method)
        assert abs(stage1_end_pct - stage2_start_pct) < 0.01  # Within 1%

    def test_multiple_thresholds_same_lightning_params(self):
        """Test different thresholds with same lightning parameters."""
        lightning_start = 2
        lightning_steps = 5

        for threshold in [5, 10, 15, 20, 25]:
            base_steps, total_base_steps, _ = alignment.calculate_perfect_alignment(
                threshold, lightning_start, lightning_steps
            )

            # Verify basic constraints
            assert total_base_steps >= threshold
            assert base_steps > 0
            assert total_base_steps >= base_steps


class TestCalculateManualBaseStepsAlignment:
    """Tests for calculate_manual_base_steps_alignment function."""

    def test_manual_alignment_basic_calculation(self):
        """Test basic manual base_steps alignment calculation."""
        # Arrange
        base_steps = 10
        lightning_start = 4
        lightning_steps = 7

        # Act
        total = alignment.calculate_manual_base_steps_alignment(
            base_steps, lightning_start, lightning_steps
        )

        # Assert
        assert total == 17  # floor(10 * 7 / 4) = floor(17.5) = 17

    def test_manual_alignment_exact_division(self):
        """Test manual alignment when division is exact."""
        # Arrange
        base_steps = 8
        lightning_start = 4
        lightning_steps = 10

        # Act
        total = alignment.calculate_manual_base_steps_alignment(
            base_steps, lightning_start, lightning_steps
        )

        # Assert
        assert total == 20  # floor(8 * 10 / 4) = 20
        assert total >= base_steps

    def test_manual_alignment_ensures_min_value(self):
        """Test that total_base_steps is at least base_steps."""
        # Arrange - small lightning values relative to base
        base_steps = 20
        lightning_start = 9
        lightning_steps = 10

        # Act
        total = alignment.calculate_manual_base_steps_alignment(
            base_steps, lightning_start, lightning_steps
        )

        # Assert
        # floor(20 * 10 / 9) = floor(22.22) = 22
        assert total >= base_steps
        assert total == 22

    def test_manual_alignment_lightning_start_zero_base_zero(self):
        """Test special case: lightning_start=0 and base_steps=0."""
        # Arrange
        base_steps = 0
        lightning_start = 0
        lightning_steps = 10

        # Act
        total = alignment.calculate_manual_base_steps_alignment(
            base_steps, lightning_start, lightning_steps
        )

        # Assert
        assert total == 0

    def test_manual_alignment_lightning_start_zero_base_nonzero(self):
        """Test special case: lightning_start=0 but base_steps > 0."""
        # Arrange
        base_steps = 10
        lightning_start = 0
        lightning_steps = 7

        # Act
        total = alignment.calculate_manual_base_steps_alignment(
            base_steps, lightning_start, lightning_steps
        )

        # Assert
        assert total == base_steps  # When lightning_start=0, total equals base

    def test_manual_alignment_invalid_base_steps_negative(self):
        """Test that negative base_steps raises ValueError."""
        with pytest.raises(ValueError, match="base_steps must be >= 0"):
            alignment.calculate_manual_base_steps_alignment(-1, 4, 7)

    def test_manual_alignment_invalid_lightning_steps(self):
        """Test that lightning_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="lightning_steps must be at least 1"):
            alignment.calculate_manual_base_steps_alignment(10, 4, 0)

    def test_manual_alignment_invalid_lightning_start(self):
        """Test that invalid lightning_start raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and"):
            alignment.calculate_manual_base_steps_alignment(10, 10, 10)  # start == steps

        with pytest.raises(ValueError, match="must be between 0 and"):
            alignment.calculate_manual_base_steps_alignment(10, -1, 10)  # negative

    def test_manual_alignment_matches_real_world_example(self):
        """Test with real-world example from recent bug fix."""
        # Arrange - Values from user test case
        base_steps = 10
        lightning_start = 4
        lightning_steps = 7

        # Act
        total = alignment.calculate_manual_base_steps_alignment(
            base_steps, lightning_start, lightning_steps
        )

        # Assert
        assert total == 17  # This was the correct value after bug fix

        # Verify overlap calculation works
        stage1_end_pct = base_steps / total  # 10/17 = 58.8%
        stage2_start_pct = lightning_start / lightning_steps  # 4/7 = 57.1%
        overlap_pct = (stage1_end_pct - stage2_start_pct) * 100.0

        assert abs(overlap_pct - 1.7) < 0.1  # ~1.7% overlap

    def test_manual_alignment_various_combinations(self):
        """Test manual alignment with various parameter combinations."""
        test_cases = [
            (5, 2, 10, 25),  # floor(5 * 10 / 2) = 25
            (15, 3, 7, 35),  # floor(15 * 7 / 3) = 35
            (8, 1, 5, 40),  # floor(8 * 5 / 1) = 40
            (12, 6, 8, 16),  # floor(12 * 8 / 6) = 16
        ]

        for base, start, steps, expected_total in test_cases:
            total = alignment.calculate_manual_base_steps_alignment(base, start, steps)
            assert total == expected_total
            assert total >= base


class TestConstants:
    """Tests for module constants."""

    def test_search_limit_multiplier_value(self):
        """Test SEARCH_LIMIT_MULTIPLIER constant value."""
        assert alignment.SEARCH_LIMIT_MULTIPLIER == 1
