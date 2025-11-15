"""Unit tests for sigma shift refinement algorithm.

Tests calculate_perfect_shift_for_step and related helper functions.
"""

from unittest.mock import MagicMock, patch

import pytest

from triple_ksampler.shared import models


class TestCalculatePerfectShiftForStep:
    """Tests for calculate_perfect_shift_for_step refinement algorithm."""

    def test_finds_perfect_shift_increasing_direction(self, mock_model_factory):
        """Test algorithm finds optimal shift when target is higher."""
        # Arrange
        mock_model = mock_model_factory()

        # Simulate sigma schedule where target is at higher shift
        # Initial: 5.0 → sigma=0.8
        # Target: 0.875 (need to increase shift to reach it)
        # Optimal: 6.0 → sigma=0.875
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            # Simulate: higher shift → higher sigma
            return 0.8 + (shift - 5.0) * 0.075

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=5.0,
                is_wanvideo=False,
            )

        # Assert
        assert refined_shift == pytest.approx(6.0, abs=0.02)
        assert "increasing" in message.lower() or "converged" in message.lower()
        assert "iterations" in message.lower()

    def test_finds_perfect_shift_decreasing_direction(self, mock_model_factory):
        """Test algorithm finds optimal shift when target is lower."""
        # Arrange
        mock_model = mock_model_factory()

        # Simulate sigma schedule where target is at lower shift
        # Initial: 7.0 → sigma=0.95
        # Target: 0.875 (need to decrease shift to reach it)
        # Optimal: 6.0 → sigma=0.875
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            return 0.95 - (7.0 - shift) * 0.075

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=7.0,
                is_wanvideo=False,
            )

        # Assert
        assert refined_shift == pytest.approx(6.0, abs=0.02)
        assert "decreasing" in message.lower() or "converged" in message.lower()

    def test_local_optimum_returns_initial_shift(self, mock_model_factory):
        """Test algorithm recognizes when initial shift is already optimal."""
        # Arrange
        mock_model = mock_model_factory()

        # Simulate: initial shift is already at local optimum
        # Moving in either direction increases error
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            # Parabola centered at 5.0 → minimum error at shift=5.0
            return 0.875 + abs(shift - 5.0) * 0.01

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=5.0,
                is_wanvideo=False,
            )

        # Assert
        assert refined_shift == 5.0
        assert "optimum" in message.lower()

    def test_handles_boundary_reached_upper_limit(self, mock_model_factory):
        """Test algorithm respects upper boundary (100.0) clamp."""
        # Arrange
        mock_model = mock_model_factory()

        # Simulate: sigma increases with shift, optimal is beyond 100.0
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            # Linear: higher shift = higher sigma, target requires shift > 100
            return 0.5 + (shift * 0.01)

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=2.0,  # Impossibly high target
                initial_shift=99.0,
                is_wanvideo=False,
            )

        # Assert - Should not exceed 100.0
        assert refined_shift <= 100.0

    def test_handles_boundary_reached_lower_limit(self, mock_model_factory):
        """Test algorithm respects lower boundary (0.0) clamp."""
        # Arrange
        mock_model = mock_model_factory()

        # Simulate: sigma decreases with shift, optimal is below 0.0
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            # Inverse: lower shift = lower sigma, target requires shift < 0
            return 1.0 - (shift * 0.1)

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.0,  # Very low target
                initial_shift=1.0,
                is_wanvideo=False,
            )

        # Assert - Should not go below 0.0
        assert refined_shift >= 0.0

    def test_overshoot_detection_returns_previous_value(self, mock_model_factory):
        """Test algorithm detects overshoot and returns previous best value."""
        # Arrange
        mock_model = mock_model_factory()

        # Simulate: sigma gets closer, then overshoots
        # Target: 0.875
        # Shift 5.0 → 0.85 (diff 0.025)
        # Shift 5.01 → 0.87 (diff 0.005) ← closer
        # Shift 5.02 → 0.88 (diff 0.005) ← same distance
        # Shift 5.03 → 0.89 (diff 0.015) ← overshoot!
        call_count = [0]

        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            call_count[0] += 1
            if shift <= 5.0:
                return 0.85
            elif shift <= 5.01:
                return 0.87
            elif shift <= 5.02:
                return 0.88
            else:
                return 0.89  # Overshoot

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=5.0,
                is_wanvideo=False,
            )

        # Assert - Should return the step before overshoot
        assert refined_shift == pytest.approx(5.02, abs=0.01)
        assert "converged" in message.lower()

    def test_returns_tuple_of_shift_and_message(self, mock_model_factory):
        """Test function returns tuple of (float, str)."""
        # Arrange
        mock_model = mock_model_factory()

        with patch("triple_ksampler.shared.models._get_sigma_at_step_comfy", return_value=0.875):
            # Act
            result = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=5.0,
                is_wanvideo=False,
            )

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        refined_shift, message = result
        assert isinstance(refined_shift, float)
        assert isinstance(message, str)

    def test_works_with_comfyui_mode(self, mock_model_factory):
        """Test algorithm works correctly with ComfyUI samplers."""
        # Arrange
        mock_model = mock_model_factory()

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", return_value=0.875
        ) as mock_comfy:
            with patch("triple_ksampler.shared.models._get_sigma_at_step_wanvideo") as mock_wv:
                # Act
                models.calculate_perfect_shift_for_step(
                    model=mock_model,
                    scheduler="normal",
                    total_steps=10,
                    target_step=5,
                    target_sigma=0.875,
                    initial_shift=5.0,
                    is_wanvideo=False,  # ComfyUI mode
                )

        # Assert - Should use ComfyUI getter
        assert mock_comfy.called
        assert not mock_wv.called

    def test_works_with_wanvideo_mode(self, mock_model_factory):
        """Test algorithm works correctly with WanVideo schedulers."""
        # Arrange
        mock_model = mock_model_factory()

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_wanvideo", return_value=0.875
        ) as mock_wv:
            with patch("triple_ksampler.shared.models._get_sigma_at_step_comfy") as mock_comfy:
                # Act
                models.calculate_perfect_shift_for_step(
                    model=mock_model,
                    scheduler="normal",
                    total_steps=10,
                    target_step=5,
                    target_sigma=0.875,
                    initial_shift=5.0,
                    is_wanvideo=True,  # WanVideo mode
                )

        # Assert - Should use WanVideo getter
        assert mock_wv.called
        assert not mock_comfy.called

    def test_handles_zero_initial_shift(self, mock_model_factory):
        """Test algorithm handles initial_shift=0.0 correctly."""
        # Arrange
        mock_model = mock_model_factory()

        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            return 0.5 + shift * 0.05

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=0.0,
                is_wanvideo=False,
            )

        # Assert
        assert refined_shift >= 0.0
        assert isinstance(message, str)

    def test_handles_maximum_initial_shift(self, mock_model_factory):
        """Test algorithm handles initial_shift=100.0 correctly."""
        # Arrange
        mock_model = mock_model_factory()

        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            return 1.0 - (100.0 - shift) * 0.001

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=100.0,
                is_wanvideo=False,
            )

        # Assert
        assert refined_shift <= 100.0
        assert isinstance(message, str)

    def test_handles_very_close_target_sigma(self, mock_model_factory):
        """Test algorithm with initial shift already very close to target."""
        # Arrange
        mock_model = mock_model_factory()

        # Initial shift gives sigma=0.8749 (only 0.0001 off from target 0.875)
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            return 0.8749 + (shift - 5.0) * 0.001

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=5.0,
                is_wanvideo=False,
            )

        # Assert - Should converge quickly
        assert abs(refined_shift - 5.0) < 0.5
        assert isinstance(message, str)

    def test_handles_far_away_target_sigma(self, mock_model_factory):
        """Test algorithm with initial shift very far from target."""
        # Arrange
        mock_model = mock_model_factory()

        # Initial shift gives sigma=0.5 (0.375 off from target 0.875)
        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            return 0.5 + (shift - 1.0) * 0.05

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            refined_shift, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=1.0,
                is_wanvideo=False,
            )

        # Assert - Should still converge
        assert isinstance(refined_shift, float)
        assert 0.0 <= refined_shift <= 100.0
        assert "iterations" in message.lower()

    def test_message_contains_iteration_count(self, mock_model_factory):
        """Test that returned message contains iteration count."""
        # Arrange
        mock_model = mock_model_factory()

        call_count = [0]

        def mock_get_sigma(model, scheduler, steps, target_step, shift):
            call_count[0] += 1
            # Converge after a few iterations
            if call_count[0] > 10:
                return 0.875 - 0.001  # Close enough to trigger overshoot
            return 0.85 + call_count[0] * 0.001

        with patch(
            "triple_ksampler.shared.models._get_sigma_at_step_comfy", side_effect=mock_get_sigma
        ):
            # Act
            _, message = models.calculate_perfect_shift_for_step(
                model=mock_model,
                scheduler="normal",
                total_steps=10,
                target_step=5,
                target_sigma=0.875,
                initial_shift=5.0,
                is_wanvideo=False,
            )

        # Assert
        assert "iteration" in message.lower()
        # Should contain a number
        assert any(char.isdigit() for char in message)
