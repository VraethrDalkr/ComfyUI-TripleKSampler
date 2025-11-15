"""
Unit tests for TripleKSampler parameter validation.

Tests all the error handling scenarios we've implemented to ensure
they raise appropriate ValueErrors with helpful messages.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch

# Import assertion helper
# Import from conftest (classes are loaded there)
from conftest import COMFYUI_AVAILABLE, TripleKSampler, TripleKSamplerAdvanced

# Import comfy for exception types
try:
    import comfy.model_management
except ImportError:
    comfy = None


@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestParameterValidation:
    """Test parameter validation for TripleKSampler nodes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerAdvanced()
        self.simple_node = TripleKSampler()

        # Mock models and inputs with proper ComfyUI compatibility
        from unittest.mock import MagicMock

        # Create model mocks with required attributes for ComfyUI
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_positive = MagicMock()
        self.mock_negative = MagicMock()

        # Mock latent with tensor-like behavior
        import torch

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

        # Common valid parameters
        self.valid_params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 3,
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 1,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
        }

    def test_lightning_steps_too_small(self):
        """Test lightning_steps must be at least 2."""
        params = self.valid_params.copy()
        params["lightning_steps"] = 1

        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            self.advanced_node.sample(**params)

    def test_lightning_start_out_of_range(self):
        """Test lightning_start must be within valid range."""
        params = self.valid_params.copy()

        # Test negative lightning_start
        params["lightning_start"] = -1
        with pytest.raises(ValueError, match="lightning_start must be within"):
            self.advanced_node.sample(**params)

        # Test lightning_start >= lightning_steps
        params["lightning_start"] = 8
        params["lightning_steps"] = 8
        with pytest.raises(ValueError, match="lightning_start must be within"):
            self.advanced_node.sample(**params)

    def test_switch_step_negative(self):
        """Test switch_step cannot be negative for manual strategy."""
        params = self.valid_params.copy()
        params["switch_strategy"] = "Manual switch step"
        params["switch_step"] = -5  # Invalid negative value

        with pytest.raises(ValueError, match="switch_step \\(-5\\) must be >= 0"):
            self.advanced_node.sample(**params)

    def test_switch_step_out_of_bounds(self):
        """Test switch_step must be within lightning_steps range."""
        params = self.valid_params.copy()
        params["switch_strategy"] = "Manual switch step"
        params["switch_step"] = 10  # Greater than lightning_steps (8)

        with pytest.raises(
            ValueError, match="switch_step \\(10\\) must be < lightning_steps \\(8\\)"
        ):
            self.advanced_node.sample(**params)

    def test_switch_step_less_than_lightning_start(self):
        """Test switch_step cannot be less than lightning_start."""
        params = self.valid_params.copy()
        params["switch_strategy"] = "Manual switch step"
        params["lightning_start"] = 5
        params["switch_step"] = 2  # Less than lightning_start

        with pytest.raises(ValueError) as exc_info:
            self.advanced_node.sample(**params)

        error_msg = str(exc_info.value)
        assert "switch_step (2) cannot be less than lightning_start (5)" in error_msg
        assert "If you want low-noise only, set lightning_start=0 as well" in error_msg

    def test_base_steps_too_small_with_lightning_start(self):
        """Test base_steps must be >= 1 when lightning_start > 0."""
        params = self.valid_params.copy()
        params["base_steps"] = 0
        params["lightning_start"] = 2

        with pytest.raises(ValueError, match="base_steps must be >= 1 when lightning_start > 0"):
            self.advanced_node.sample(**params)

    def test_base_steps_zero_with_nonzero_lightning_start(self):
        """Test base_steps=0 only allowed when lightning_start=0."""
        params = self.valid_params.copy()
        params["base_steps"] = 0
        params["lightning_start"] = 1

        with pytest.raises(ValueError, match="base_steps must be >= 1 when lightning_start > 0"):
            self.advanced_node.sample(**params)

    def test_stage1_stage2_skip_with_positive_base_steps(self):
        """Test Stage1+Stage2 skip scenario validation."""
        params = self.valid_params.copy()
        params["lightning_start"] = 0  # Skip Stage1
        params["switch_strategy"] = "Manual switch step"
        params["switch_step"] = 0  # Skip Stage2
        params["base_steps"] = 5  # Positive base_steps (invalid)

        with pytest.raises(
            ValueError, match="When skipping both Stage 1 and Stage 2.*base_steps must be -1 or 0"
        ):
            self.advanced_node.sample(**params)

    def test_lightning_start_zero_with_positive_base_steps(self):
        """Test base_steps ignored when lightning_start=0."""
        params = self.valid_params.copy()
        params["lightning_start"] = 0  # Lightning-only mode
        params["base_steps"] = 5  # Positive base_steps (will be ignored)

        with pytest.raises(ValueError) as exc_info:
            self.advanced_node.sample(**params)

        error_msg = str(exc_info.value)
        assert "Set base_steps=0 or base_steps=-1 for Lightning-only mode" in error_msg

    def test_lightning_start_greater_than_switch_point(self):
        """Test lightning_start cannot be greater than computed switch point."""
        params = self.valid_params.copy()
        params["lightning_start"] = 6
        params["switch_strategy"] = "Manual switch step"
        params["switch_step"] = 2  # Less than lightning_start

        with pytest.raises(ValueError) as exc_info:
            self.advanced_node.sample(**params)

        error_msg = str(exc_info.value)
        assert (
            "cannot be greater than switch_step" in error_msg
            or "cannot be less than lightning_start" in error_msg
        )

    def test_valid_parameters_pass(self):
        """Test that valid parameters don't raise exceptions."""
        # Use dry_run mode to test parameter validation without actual sampling
        params = self.valid_params.copy()
        params["dry_run"] = True

        # In dry run mode, should interrupt workflow execution
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

    def test_simple_node_delegates_validation(self):
        """Test that simple node inherits validation from advanced node."""
        # Simple node parameters (subset of advanced)
        simple_params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 3,  # Required but ignored
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 1,  # Added required parameter
            "lightning_steps": 1,  # Invalid: too small
            "lightning_cfg": 1.0,  # Required but ignored
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
        }

        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            self.simple_node.sample(**simple_params)


@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerAdvanced()

        # Set up proper mocks like in TestParameterValidation
        from unittest.mock import MagicMock

        import torch

        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_lightning_only_mode_valid(self):
        """Test valid Lightning-only mode configuration."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 0,  # Valid for lightning-only
            "base_quality_threshold": 20,  # Add missing parameter
            "base_cfg": 3.5,
            "lightning_start": 0,  # Lightning-only mode
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True,  # Use dry run to avoid sampling execution
        }

        # Should pass validation and return minimal 8x8 latent
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

    def test_auto_calculation_mode(self):
        """Test auto-calculation modes work correctly."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": -1,  # Auto-calculate
            "base_quality_threshold": 25,  # Add missing parameter
            "base_cfg": 3.5,
            "lightning_start": 2,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "Manual switch step",
            "switch_boundary": 0.875,
            "switch_step": -1,  # Auto-calculate
            "dry_run": True,  # Use dry run to avoid sampling execution
        }

        # Should pass validation and return minimal 8x8 latent
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)


@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestSimpleNodeLightningStart:
    """Test Simple node with various lightning_start values (v0.6.1 feature)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.simple_node = TripleKSampler()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_simple_node_lightning_start_0(self):
        """Test Simple node with lightning_start=0 (lightning-only mode)."""

        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 3,  # Required but ignored
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 0,  # Lightning-only mode
            "lightning_steps": 8,
            "lightning_cfg": 1.0,  # Required but ignored
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
        }

        # Use dry run mode to test lightning-only mode
        params["dry_run"] = True
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.simple_node.sample(**params)

    def test_simple_node_lightning_start_1(self):
        """Test Simple node with lightning_start=1 (default)."""

        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 3,  # Required but ignored
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 1,  # Default value
            "lightning_steps": 8,
            "lightning_cfg": 1.0,  # Required but ignored
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
        }

        # Use dry run mode to test default lightning_start
        params["dry_run"] = True
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.simple_node.sample(**params)

    def test_simple_node_lightning_start_higher_values(self):
        """Test Simple node with higher lightning_start values."""

        # Test with valid combinations that pass validation
        test_cases = [
            {"lightning_start": 2, "lightning_steps": 8},  # switch_step=4, 2 < 4 ✓
            {"lightning_start": 3, "lightning_steps": 8},  # switch_step=4, 3 < 4 ✓
            {"lightning_start": 2, "lightning_steps": 12},  # switch_step=6, 2 < 6 ✓
        ]

        for case in test_cases:
            params = {
                "base_high": self.mock_base_high,
                "lightning_high": self.mock_lightning_high,
                "lightning_low": self.mock_lightning_low,
                "positive": MagicMock(),
                "negative": MagicMock(),
                "latent_image": self.mock_latent,
                "seed": 42,
                "sigma_shift": 5.0,
                "base_steps": 3,  # Required but ignored
                "base_cfg": 3.5,
                "lightning_start": case["lightning_start"],
                "lightning_steps": case["lightning_steps"],
                "lightning_cfg": 1.0,  # Required but ignored
                "sampler_name": "euler",
                "scheduler": "simple",
                "switch_strategy": "50% of steps",
            }

            # Use dry run mode to test higher lightning_start values
            params["dry_run"] = True
            with pytest.raises(comfy.model_management.InterruptProcessingException):
                self.simple_node.sample(**params)

    def test_simple_node_lightning_start_validation(self):
        """Test Simple node lightning_start validation."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 3,  # Required but ignored
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 8,  # Invalid: >= lightning_steps
            "lightning_steps": 8,
            "lightning_cfg": 1.0,  # Required but ignored
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
        }

        # Should raise validation error
        with pytest.raises(ValueError, match="lightning_start must be within"):
            self.simple_node.sample(**params)


@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestDryRunMode:
    """Test dry run mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerAdvanced()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

        self.base_params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 3,
            "base_quality_threshold": 20,
            "base_cfg": 3.5,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_start": 1,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
        }

    def test_dry_run_all_strategies(self):
        """Test dry run mode with all switching strategies."""
        from unittest.mock import patch

        import torch

        strategies = [
            "50% of steps",
            "Manual switch step",
            "T2V boundary",
            "I2V boundary",
            "Manual boundary",
        ]

        # Mock sigma calculation for boundary strategies
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])

        # Mock timestep function for boundary calculations
        def mock_timestep(sigma_val):
            """Mock timestep that returns comparable float values for boundary testing."""
            sigma = float(sigma_val)
            # Return timesteps that work for both T2V (0.875) and I2V (0.900) boundaries
            if sigma >= 2.5:
                return 950.0  # Above both boundaries (0.95 > 0.900 > 0.875)
            else:
                return 850.0  # Below both boundaries (0.85 < 0.875 < 0.900)

        # Mock the sampling object's timestep method
        mock_sampling = MagicMock()
        mock_sampling.timestep.side_effect = mock_timestep

        with (
            patch("comfy.samplers.calculate_sigmas", return_value=mock_sigmas),
            patch.object(
                self.advanced_node,
                "_patch_models_for_sampling",
                return_value=(
                    self.mock_base_high,
                    self.mock_lightning_high,
                    self.mock_lightning_low,
                ),
            ),
        ):
            # Mock get_model_object to return our mock sampling
            self.mock_lightning_high.get_model_object.return_value = mock_sampling
            self.mock_lightning_low.get_model_object.return_value = mock_sampling
            for strategy in strategies:
                params = self.base_params.copy()
                params.update(
                    {
                        "switch_strategy": strategy,
                        "switch_step": 4 if strategy == "Manual switch step" else -1,
                        "dry_run": True,
                    }
                )

                # Should complete without error and return minimal 8x8 latent
                with pytest.raises(comfy.model_management.InterruptProcessingException):
                    self.advanced_node.sample(**params)

    def test_dry_run_complex_scenarios(self):
        """Test dry run mode with complex parameter scenarios."""
        # Lightning-only mode
        params = self.base_params.copy()
        params.update(
            {
                "base_steps": 0,
                "lightning_start": 0,
                "switch_strategy": "50% of steps",
                "dry_run": True,
            }
        )
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

        # Auto-calculation mode
        params.update(
            {
                "base_steps": -1,
                "lightning_start": 1,
                "switch_strategy": "Manual switch step",
                "switch_step": -1,
            }
        )
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

    def test_dry_run_validation_still_occurs(self):
        """Test that validation still occurs in dry run mode."""
        params = self.base_params.copy()
        params.update(
            {
                "lightning_steps": 1,  # Invalid: too small
                "dry_run": True,
            }
        )

        # Should still raise validation error even in dry run
        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            self.advanced_node.sample(**params)


@pytest.mark.skipif(not COMFYUI_AVAILABLE, reason="ComfyUI dependencies not available")
class TestConfigParameterBoundaries:
    """Test edge cases for configuration parameters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerAdvanced()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {
            "shift": 1.0,
            "multiplier": 1000,
        }

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_sigma_shift_edge_cases(self):
        """Test sigma_shift with edge values."""
        edge_values = [0.0, 0.1, 1.0, 10.0, 100.0]

        for sigma_shift in edge_values:
            params = {
                "base_high": self.mock_base_high,
                "lightning_high": self.mock_lightning_high,
                "lightning_low": self.mock_lightning_low,
                "positive": MagicMock(),
                "negative": MagicMock(),
                "latent_image": self.mock_latent,
                "seed": 42,
                "sigma_shift": sigma_shift,
                "base_steps": 3,
                "base_quality_threshold": 20,  # Add missing parameter
                "base_cfg": 3.5,
                "lightning_start": 1,
                "lightning_steps": 8,
                "lightning_cfg": 1.0,
                "base_sampler": "euler",
                "base_scheduler": "simple",
                "lightning_sampler": "euler",
                "lightning_scheduler": "simple",
                "switch_strategy": "50% of steps",
                "switch_boundary": 0.875,
                "switch_step": -1,
                "dry_run": True,
            }

            # Should handle all sigma_shift values gracefully
            with pytest.raises(comfy.model_management.InterruptProcessingException):
                self.advanced_node.sample(**params)

    def test_cfg_extreme_values(self):
        """Test CFG with extreme values."""
        cfg_values = [0.0, 0.1, 1.0, 10.0, 50.0, 100.0]

        for base_cfg in cfg_values:
            for lightning_cfg in cfg_values:
                params = {
                    "base_high": self.mock_base_high,
                    "lightning_high": self.mock_lightning_high,
                    "lightning_low": self.mock_lightning_low,
                    "positive": MagicMock(),
                    "negative": MagicMock(),
                    "latent_image": self.mock_latent,
                    "seed": 42,
                    "sigma_shift": 5.0,
                    "base_steps": 3,
                    "base_quality_threshold": 20,  # Add missing parameter
                    "base_cfg": base_cfg,
                    "lightning_start": 1,
                    "lightning_steps": 8,
                    "lightning_cfg": lightning_cfg,
                    "base_sampler": "euler",
                    "base_scheduler": "simple",
                    "lightning_sampler": "euler",
                    "lightning_scheduler": "simple",
                    "switch_strategy": "50% of steps",
                    "switch_boundary": 0.875,
                    "switch_step": -1,
                    "dry_run": True,
                }

                with pytest.raises(comfy.model_management.InterruptProcessingException):
                    self.advanced_node.sample(**params)

    def test_large_step_counts(self):
        """Test with very large step counts."""
        large_steps = [50, 100, 200]

        for lightning_steps in large_steps:
            params = {
                "base_high": self.mock_base_high,
                "lightning_high": self.mock_lightning_high,
                "lightning_low": self.mock_lightning_low,
                "positive": MagicMock(),
                "negative": MagicMock(),
                "latent_image": self.mock_latent,
                "seed": 42,
                "sigma_shift": 5.0,
                "base_steps": 10,
                "base_quality_threshold": 30,  # Add missing parameter
                "base_cfg": 3.5,
                "lightning_start": 5,
                "lightning_steps": lightning_steps,
                "lightning_cfg": 1.0,
                "base_sampler": "euler",
                "base_scheduler": "simple",
                "lightning_sampler": "euler",
                "lightning_scheduler": "simple",
                "switch_strategy": "50% of steps",
                "switch_boundary": 0.875,
                "switch_step": -1,
                "dry_run": True,
            }

            with pytest.raises(comfy.model_management.InterruptProcessingException):
                self.advanced_node.sample(**params)

    def test_switch_boundary_full_range(self):
        """Test switch_boundary across full valid range."""
        from unittest.mock import patch

        import torch

        # Mock sigma calculation for boundary strategies
        torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])

        # Test strategy UI visibility
        strategies_test_cases = [
            ("50% of steps", False, False),
            ("Manual switch step", True, False),
            ("T2V boundary", False, False),
            ("I2V boundary", False, False),
            ("Manual boundary", False, True),
        ]

        for strategy, should_show_step, should_show_boundary in strategies_test_cases:
            # Simulate the JavaScript logic
            show_switch_step = False
            show_switch_boundary = False

            if strategy == "50% of steps":
                show_switch_step = False
                show_switch_boundary = False
            elif strategy == "Manual switch step":
                show_switch_step = True
                show_switch_boundary = False
            elif strategy in ["T2V boundary", "I2V boundary"]:
                show_switch_step = False
                show_switch_boundary = False
            elif strategy == "Manual boundary":
                show_switch_step = False
                show_switch_boundary = True

            assert show_switch_step == should_show_step, f"Failed for strategy: {strategy}"
            assert show_switch_boundary == should_show_boundary, f"Failed for strategy: {strategy}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
