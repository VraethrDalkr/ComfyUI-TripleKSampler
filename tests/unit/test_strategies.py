"""Unit tests for shared/strategies.py.

Tests strategy calculation logic independently of ComfyUI integration.
"""

import math
from unittest.mock import MagicMock, patch

import pytest

from triple_ksampler.shared import strategies


class TestCalculateSwitchStepAndStrategy:
    """Tests for calculate_switch_step_and_strategy function."""

    def test_manual_switch_step_explicit_value(self, mock_model_factory):
        """Test manual switch step with explicit value."""
        # Arrange
        switch_strategy = "Manual switch step"
        switch_step = 5
        lightning_steps = 10
        mock_model = mock_model_factory()

        # Act
        step, effective_strategy, info = strategies.calculate_switch_step_and_strategy(
            switch_strategy, switch_step, 0.875, lightning_steps, mock_model, "normal"
        )

        # Assert
        assert step == 5
        assert effective_strategy == "Manual switch step"
        assert "switch at step 5 of 10" in info

    def test_manual_switch_step_auto_calculation(self, mock_model_factory):
        """Test manual switch step with auto-calculation (-1)."""
        # Arrange
        switch_strategy = "Manual switch step"
        switch_step = -1  # Auto-calculate
        lightning_steps = 10
        mock_model = mock_model_factory()

        # Act
        step, effective_strategy, info = strategies.calculate_switch_step_and_strategy(
            switch_strategy, switch_step, 0.875, lightning_steps, mock_model, "normal"
        )

        # Assert
        assert step == 5  # 10 // 2
        assert effective_strategy == "50% of steps (auto)"
        assert "switch at step 5 of 10" in info

    def test_manual_switch_step_odd_steps(self, mock_model_factory):
        """Test manual switch step auto-calculation with odd number of steps."""
        # Arrange
        switch_strategy = "Manual switch step"
        switch_step = -1
        lightning_steps = 11
        mock_model = mock_model_factory()

        # Act
        step, _, _ = strategies.calculate_switch_step_and_strategy(
            switch_strategy, switch_step, 0.875, lightning_steps, mock_model, "normal"
        )

        # Assert
        assert step == 5  # 11 // 2 = 5 (integer division)

    @patch("triple_ksampler.shared.strategies.compute_boundary_switching_step")
    def test_t2v_boundary_strategy(self, mock_compute, mock_model_factory):
        """Test T2V boundary strategy uses correct boundary value."""
        # Arrange
        mock_compute.return_value = 7
        mock_model = mock_model_factory()
        mock_sampling = MagicMock()
        mock_model.get_model_object.return_value = mock_sampling

        # Act
        step, effective_strategy, info = strategies.calculate_switch_step_and_strategy(
            "T2V boundary", -1, 0.999, 10, mock_model, "normal"
        )

        # Assert
        assert step == 7
        assert effective_strategy == "T2V boundary"
        mock_compute.assert_called_once_with(
            mock_sampling, "normal", 10, strategies.DEFAULT_BOUNDARY_T2V
        )
        assert "boundary = 0.875" in info  # DEFAULT_BOUNDARY_T2V

    @patch("triple_ksampler.shared.strategies.compute_boundary_switching_step")
    def test_i2v_boundary_strategy(self, mock_compute, mock_model_factory):
        """Test I2V boundary strategy uses correct boundary value."""
        # Arrange
        mock_compute.return_value = 8
        mock_model = mock_model_factory()
        mock_sampling = MagicMock()
        mock_model.get_model_object.return_value = mock_sampling

        # Act
        step, effective_strategy, info = strategies.calculate_switch_step_and_strategy(
            "I2V boundary", -1, 0.999, 10, mock_model, "normal"
        )

        # Assert
        assert step == 8
        assert effective_strategy == "I2V boundary"
        mock_compute.assert_called_once_with(
            mock_sampling, "normal", 10, strategies.DEFAULT_BOUNDARY_I2V
        )
        assert "boundary = 0.9" in info  # DEFAULT_BOUNDARY_I2V

    @patch("triple_ksampler.shared.strategies.compute_boundary_switching_step")
    def test_manual_boundary_strategy(self, mock_compute, mock_model_factory):
        """Test manual boundary strategy uses user-specified boundary."""
        # Arrange
        mock_compute.return_value = 6
        mock_model = mock_model_factory()
        mock_sampling = MagicMock()
        mock_model.get_model_object.return_value = mock_sampling
        custom_boundary = 0.850

        # Act
        step, effective_strategy, info = strategies.calculate_switch_step_and_strategy(
            "Manual boundary", -1, custom_boundary, 10, mock_model, "normal"
        )

        # Assert
        assert step == 6
        assert effective_strategy == "Manual boundary"
        mock_compute.assert_called_once_with(mock_sampling, "normal", 10, custom_boundary)
        assert "boundary = 0.85" in info

    def test_default_strategy_fallback(self, mock_model_factory):
        """Test default strategy falls back to 50% of steps."""
        # Arrange
        mock_model = mock_model_factory()

        # Act
        step, effective_strategy, info = strategies.calculate_switch_step_and_strategy(
            "50% of steps", -1, 0.875, 10, mock_model, "normal"
        )

        # Assert
        assert step == 5  # math.ceil(10 / 2)
        assert effective_strategy == "50% of steps"
        assert "switch at step 5 of 10" in info

    def test_invalid_lightning_steps_raises_error(self, mock_model_factory):
        """Test that lightning_steps < 1 raises ValueError."""
        # Arrange
        mock_model = mock_model_factory()

        # Act & Assert
        with pytest.raises(ValueError, match="lightning_steps must be at least 1"):
            strategies.calculate_switch_step_and_strategy(
                "Manual switch step", -1, 0.875, 0, mock_model, "normal"
            )

    def test_custom_boundary_values(self, mock_model_factory):
        """Test custom T2V and I2V boundary values."""
        # Arrange
        mock_model = mock_model_factory()
        custom_t2v = 0.850

        with patch(
            "triple_ksampler.shared.strategies.compute_boundary_switching_step"
        ) as mock_compute:
            mock_sampling = MagicMock()
            mock_model.get_model_object.return_value = mock_sampling
            mock_compute.return_value = 7

            # Act - T2V with custom boundary
            _, _, info = strategies.calculate_switch_step_and_strategy(
                "T2V boundary",
                -1,
                0.999,
                10,
                mock_model,
                "normal",
                boundary_t2v=custom_t2v,
            )

            # Assert
            mock_compute.assert_called_with(mock_sampling, "normal", 10, custom_t2v)
            assert "boundary = 0.85" in info

    def test_switch_step_final_is_int(self, mock_model_factory):
        """Test that returned switch_step is always an integer."""
        # Arrange
        mock_model = mock_model_factory()

        # Act
        step, _, _ = strategies.calculate_switch_step_and_strategy(
            "Manual switch step", 5, 0.875, 10, mock_model, "normal"
        )

        # Assert
        assert isinstance(step, int)


class TestComputeBoundarySwitchingStep:
    """Tests for compute_boundary_switching_step function."""

    @patch("triple_ksampler.shared.strategies.comfy")
    def test_boundary_switching_basic(self, mock_comfy, mock_model_factory):
        """Test basic boundary switching calculation."""
        # Arrange
        mock_sampling = MagicMock()
        mock_sampling.timestep = lambda sigma_val: sigma_val * 1000.0

        # Create mock sigmas (decreasing values representing noise levels)
        mock_sigmas = [
            MagicMock(item=lambda: 1.0),  # High noise
            MagicMock(item=lambda: 0.95),
            MagicMock(item=lambda: 0.90),
            MagicMock(item=lambda: 0.85),  # Should switch here (< 0.875)
            MagicMock(item=lambda: 0.80),
            MagicMock(item=lambda: 0.70),  # Low noise
        ]
        mock_comfy.samplers.calculate_sigmas.return_value = mock_sigmas

        # Act
        result = strategies.compute_boundary_switching_step(mock_sampling, "normal", 5, 0.875)

        # Assert
        assert result == 3  # First step where timestep < 0.875
        mock_comfy.samplers.calculate_sigmas.assert_called_once_with(mock_sampling, "normal", 5)

    @patch("triple_ksampler.shared.strategies.comfy")
    def test_boundary_never_crossed_returns_last_step(self, mock_comfy, mock_model_factory):
        """Test that if boundary is never crossed, returns steps-1."""
        # Arrange
        mock_sampling = MagicMock()
        mock_sampling.timestep = lambda sigma_val: sigma_val * 1000.0

        # All timesteps above boundary
        mock_sigmas = [
            MagicMock(item=lambda: 1.0),
            MagicMock(item=lambda: 0.95),
            MagicMock(item=lambda: 0.92),
            MagicMock(item=lambda: 0.90),
        ]
        mock_comfy.samplers.calculate_sigmas.return_value = mock_sigmas

        # Act
        result = strategies.compute_boundary_switching_step(mock_sampling, "normal", 3, 0.875)

        # Assert
        assert result == 2  # steps - 1 = 3 - 1

    @patch("triple_ksampler.shared.strategies.comfy")
    def test_handles_scalar_sigmas(self, mock_comfy, mock_model_factory):
        """Test that function handles scalar sigmas without .item() method."""
        # Arrange
        mock_sampling = MagicMock()
        mock_sampling.timestep = lambda sigma_val: sigma_val * 1000.0

        # Scalar sigmas (no .item() method)
        mock_comfy.samplers.calculate_sigmas.return_value = [1.0, 0.9, 0.85, 0.8, 0.7]

        # Act
        result = strategies.compute_boundary_switching_step(mock_sampling, "normal", 4, 0.875)

        # Assert
        assert isinstance(result, int)
        assert 0 <= result < 4

    def test_invalid_steps_raises_error(self, mock_model_factory):
        """Test that steps < 1 raises ValueError."""
        # Arrange
        mock_sampling = MagicMock()

        # Act & Assert
        with pytest.raises(ValueError, match="steps must be at least 1"):
            strategies.compute_boundary_switching_step(mock_sampling, "normal", 0, 0.875)

    def test_invalid_boundary_low_raises_error(self, mock_model_factory):
        """Test that boundary < 0.0 raises ValueError."""
        # Arrange
        mock_sampling = MagicMock()

        # Act & Assert
        with pytest.raises(ValueError, match="boundary must be between 0.0 and 1.0"):
            strategies.compute_boundary_switching_step(mock_sampling, "normal", 10, -0.1)

    def test_invalid_boundary_high_raises_error(self, mock_model_factory):
        """Test that boundary > 1.0 raises ValueError."""
        # Arrange
        mock_sampling = MagicMock()

        # Act & Assert
        with pytest.raises(ValueError, match="boundary must be between 0.0 and 1.0"):
            strategies.compute_boundary_switching_step(mock_sampling, "normal", 10, 1.5)

    def test_comfy_not_available_raises_import_error(self, mock_model_factory):
        """Test that missing comfy module raises ImportError."""
        # Arrange
        with patch("triple_ksampler.shared.strategies.comfy", None):
            mock_sampling = MagicMock()

            # Act & Assert
            with pytest.raises(ImportError, match="comfy.samplers module is not available"):
                strategies.compute_boundary_switching_step(mock_sampling, "normal", 10, 0.875)


class TestValidateStrategy:
    """Tests for validate_strategy function."""

    def test_valid_strategies_do_not_raise(self, mock_model_factory):
        """Test that all valid strategies pass validation."""
        valid_strategies = [
            "Manual switch step",
            "T2V boundary",
            "I2V boundary",
            "Manual boundary",
            "50% of steps",
        ]

        for strategy in valid_strategies:
            # Should not raise
            strategies.validate_strategy(strategy)

    def test_invalid_strategy_raises_value_error(self, mock_model_factory):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            strategies.validate_strategy("invalid_strategy")

    def test_case_sensitive_validation(self, mock_model_factory):
        """Test that validation is case-sensitive."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            strategies.validate_strategy("manual switch step")  # lowercase

    def test_error_message_lists_valid_strategies(self, mock_model_factory):
        """Test that error message includes list of valid strategies."""
        with pytest.raises(ValueError, match="Valid strategies:.*Manual switch step"):
            strategies.validate_strategy("bad")


class TestConstants:
    """Tests for module constants."""

    def test_default_boundary_t2v_value(self, mock_model_factory):
        """Test DEFAULT_BOUNDARY_T2V constant value."""
        assert strategies.DEFAULT_BOUNDARY_T2V == 0.875

    def test_default_boundary_i2v_value(self, mock_model_factory):
        """Test DEFAULT_BOUNDARY_I2V constant value."""
        assert strategies.DEFAULT_BOUNDARY_I2V == 0.900
