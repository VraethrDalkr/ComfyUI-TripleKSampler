"""
Unit tests for TripleKSampler calculation methods.

Tests the mathematical and calculation methods that are critical to the
sampling algorithm but were previously untested.
"""

import math
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import from conftest (classes are loaded there)
from conftest import COMFYUI_AVAILABLE, TripleKSamplerAdvanced, TripleKSamplerBase


@pytest.mark.unit
@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestCalculatePerfectAlignment:
    """Test the critical _calculate_perfect_alignment method."""

    def test_simple_case_lightning_start_1(self):
        """Test simple case with lightning_start=1."""
        # Simple case: lightning_start=1, should use direct calculation
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=20, lightning_start=1, lightning_steps=8
        )

        # Should use simple math: ceil(20/8) = 3
        assert base_steps == 3
        assert total_base_steps == 24  # 3 * 8
        assert method == "simple_math"

    def test_simple_case_different_values(self):
        """Test simple case with different parameter values."""
        # Test with different values
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=15, lightning_start=1, lightning_steps=6
        )

        # ceil(15/6) = 3
        assert base_steps == 3
        assert total_base_steps == 18  # 3 * 6
        assert method == "simple_math"

    def test_complex_case_mathematical_search(self):
        """Test complex case that finds perfect alignment."""
        # Complex case: lightning_start=2, should find mathematical solution
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=20, lightning_start=2, lightning_steps=8
        )

        # Should find a solution where (total * lightning_start) % lightning_steps == 0
        assert method == "mathematical_search"
        assert total_base_steps >= 20  # Must meet quality threshold
        assert (total_base_steps * 2) % 8 == 0  # Perfect alignment condition
        assert base_steps == (total_base_steps * 2) // 8

    def test_complex_case_lightning_start_3(self):
        """Test complex case with lightning_start=3."""
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=20, lightning_start=3, lightning_steps=8
        )

        # Should find mathematical solution
        assert method == "mathematical_search"
        assert total_base_steps >= 20
        assert (total_base_steps * 3) % 8 == 0
        assert base_steps == (total_base_steps * 3) // 8

    def test_fallback_case(self):
        """Test fallback when no perfect alignment is found."""
        # Create a case where perfect alignment is unlikely in the search range
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=100, lightning_start=7, lightning_steps=11
        )

        # May use fallback if no perfect alignment found in search range
        assert method in ["mathematical_search", "fallback"]
        assert total_base_steps >= 100  # Must meet quality threshold
        assert base_steps > 0

    def test_edge_case_lightning_start_0(self):
        """Test edge case with lightning_start=0."""
        # Edge case: lightning_start=0 (lightning-only mode, skip base stage)
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=20, lightning_start=0, lightning_steps=8
        )

        # Special case returns immediately with simple_math
        assert method == "simple_math"
        assert base_steps == 0
        assert total_base_steps == 0

    def test_large_values(self):
        """Test with larger parameter values."""
        base_steps, total_base_steps, method = TripleKSamplerBase._calculate_perfect_alignment(
            base_quality_threshold=50, lightning_start=1, lightning_steps=16
        )

        # Simple case with larger values
        assert method == "simple_math"
        assert base_steps == math.ceil(50 / 16)  # ceil(3.125) = 4
        assert total_base_steps == base_steps * 16


@pytest.mark.unit
@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestCalculatePercentage:
    """Test the _calculate_percentage helper method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.base_instance = TripleKSamplerBase()

    def test_normal_percentages(self):
        """Test normal percentage calculations."""
        # Test various normal cases
        assert self.base_instance._calculate_percentage(25, 100) == 25.0
        assert self.base_instance._calculate_percentage(50, 100) == 50.0
        assert self.base_instance._calculate_percentage(75, 100) == 75.0
        assert self.base_instance._calculate_percentage(100, 100) == 100.0

    def test_fractional_percentages(self):
        """Test fractional percentages with rounding."""
        # Test rounding to 1 decimal place
        assert self.base_instance._calculate_percentage(1, 3) == 33.3
        assert self.base_instance._calculate_percentage(2, 3) == 66.7
        assert self.base_instance._calculate_percentage(1, 6) == 16.7
        assert self.base_instance._calculate_percentage(1, 7) == 14.3

    def test_division_by_zero(self):
        """Test division by zero handling."""
        # Should return 0.0 for division by zero
        assert self.base_instance._calculate_percentage(10, 0) == 0.0
        assert self.base_instance._calculate_percentage(0, 0) == 0.0

    def test_boundary_clamping(self):
        """Test clamping to 0-100 range."""
        # Test values > 100%
        assert self.base_instance._calculate_percentage(150, 100) == 100.0
        assert self.base_instance._calculate_percentage(200, 100) == 100.0

        # Test negative values (clamped to 0)
        assert self.base_instance._calculate_percentage(-10, 100) == 0.0

    def test_zero_numerator(self):
        """Test with zero numerator."""
        assert self.base_instance._calculate_percentage(0, 100) == 0.0
        assert self.base_instance._calculate_percentage(0, 50) == 0.0

    def test_float_inputs(self):
        """Test with float inputs."""
        assert self.base_instance._calculate_percentage(12.5, 50.0) == 25.0
        assert self.base_instance._calculate_percentage(33.333, 100.0) == 33.3


@pytest.mark.unit
@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestFormatStageRange:
    """Test the _format_stage_range method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.base_instance = TripleKSamplerBase()

    def test_normal_stage_ranges(self):
        """Test normal stage range formatting."""
        # Test typical stage ranges
        result = self.base_instance._format_stage_range(0, 5, 20)
        assert "steps 0-5 of 20" in result
        assert "0.0%–25.0%" in result

        result = self.base_instance._format_stage_range(5, 10, 20)
        assert "steps 5-10 of 20" in result
        assert "25.0%–50.0%" in result

    def test_single_step_stage(self):
        """Test single step stage formatting."""
        result = self.base_instance._format_stage_range(3, 4, 20)
        assert "steps 3-4 of 20" in result
        assert "15.0%–20.0%" in result

    def test_full_range(self):
        """Test full range formatting."""
        result = self.base_instance._format_stage_range(0, 20, 20)
        assert "steps 0-20 of 20" in result
        assert "0.0%–100.0%" in result

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with step 0
        result = self.base_instance._format_stage_range(0, 1, 10)
        assert "steps 0-1 of 10" in result
        assert "0.0%–10.0%" in result

        # Test larger ranges
        result = self.base_instance._format_stage_range(10, 20, 50)
        assert "steps 10-20 of 50" in result
        assert "20.0%–40.0%" in result


@pytest.mark.unit
@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestComputeBoundarySwitchingStep:
    """Test the _compute_boundary_switching_step method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerAdvanced()

        # Mock sampling object
        self.mock_sampling = MagicMock()
        self.mock_sampling.timestep.return_value = 1000.0  # Default return value

    @patch("comfy.samplers.calculate_sigmas")
    def test_t2v_boundary(self, mock_calculate_sigmas):
        """Test T2V boundary (0.875) switching step calculation."""
        import torch

        # Mock sigmas that represent a typical denoising schedule
        # Higher sigmas (more noise) at start, lower at end
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock timestep conversion to simulate boundary crossing
        def mock_timestep(sigma):
            # Simulate timestep conversion where boundary 0.875 occurs around step 3-4
            sigma_val = float(sigma.item()) if hasattr(sigma, "item") else float(sigma)
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary (0.85 < 0.875)

        self.mock_sampling.timestep.side_effect = mock_timestep

        switching_step = self.advanced_node._compute_boundary_switching_step(
            self.mock_sampling, "simple", 8, 0.875
        )

        # Should find the step where timestep first goes below 0.875
        assert isinstance(switching_step, int)
        assert 0 <= switching_step < 8

    @patch("comfy.samplers.calculate_sigmas")
    def test_i2v_boundary(self, mock_calculate_sigmas):
        """Test I2V boundary (0.900) switching step calculation."""
        import torch

        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock higher boundary (0.900)
        def mock_timestep(sigma):
            sigma_val = float(sigma.item()) if hasattr(sigma, "item") else float(sigma)
            if sigma_val >= 1.25:
                return 950.0  # Above boundary
            else:
                return 880.0  # Below boundary (0.88 < 0.900)

        self.mock_sampling.timestep.side_effect = mock_timestep

        switching_step = self.advanced_node._compute_boundary_switching_step(
            self.mock_sampling, "simple", 8, 0.900
        )

        assert isinstance(switching_step, int)
        assert 0 <= switching_step < 8

    @patch("comfy.samplers.calculate_sigmas")
    def test_custom_boundary(self, mock_calculate_sigmas):
        """Test custom boundary value."""
        import torch

        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Test custom boundary of 0.5
        def mock_timestep(sigma):
            sigma_val = float(sigma.item()) if hasattr(sigma, "item") else float(sigma)
            if sigma_val >= 1.25:
                return 600.0  # Above 0.5
            else:
                return 400.0  # Below 0.5

        self.mock_sampling.timestep.side_effect = mock_timestep

        switching_step = self.advanced_node._compute_boundary_switching_step(
            self.mock_sampling, "simple", 4, 0.5
        )

        assert isinstance(switching_step, int)
        assert 0 <= switching_step < 4

    @patch("comfy.samplers.calculate_sigmas")
    def test_boundary_never_crossed(self, mock_calculate_sigmas):
        """Test when boundary is never crossed."""
        import torch

        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25])
        mock_calculate_sigmas.return_value = mock_sigmas

        # All timesteps above boundary
        self.mock_sampling.timestep.return_value = 950.0

        switching_step = self.advanced_node._compute_boundary_switching_step(
            self.mock_sampling, "simple", 4, 0.875
        )

        # Should return last valid step (steps-1)
        assert switching_step == 3

    @patch("comfy.samplers.calculate_sigmas")
    def test_boundary_at_zero(self, mock_calculate_sigmas):
        """Test edge case with boundary=0."""
        import torch

        mock_sigmas = torch.tensor([10.0, 5.0])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Even very low timesteps should be above 0
        self.mock_sampling.timestep.return_value = 100.0

        switching_step = self.advanced_node._compute_boundary_switching_step(
            self.mock_sampling, "simple", 2, 0.0
        )

        # Should switch at first step where timestep < 0 (which should be step 1)
        assert switching_step == 1

    @patch("comfy.samplers.calculate_sigmas")
    def test_boundary_at_one(self, mock_calculate_sigmas):
        """Test edge case with boundary=1.0."""
        import torch

        mock_sigmas = torch.tensor([10.0, 5.0])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Timesteps should be < 1.0
        self.mock_sampling.timestep.return_value = 500.0

        switching_step = self.advanced_node._compute_boundary_switching_step(
            self.mock_sampling, "simple", 2, 1.0
        )

        # Should find switching step since 0.5 < 1.0
        assert switching_step == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
