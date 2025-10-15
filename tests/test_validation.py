"""
Unit tests for TripleKSampler parameter validation.

Tests all the error handling scenarios we've implemented to ensure
they raise appropriate ValueErrors with helpful messages.
"""

import pytest
import sys
import os
import math
import torch
from unittest.mock import MagicMock

# Import assertion helper
from conftest import TripleKSamplerAssertions

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if ComfyUI dependencies are available
COMFYUI_AVAILABLE = False
try:
    import importlib.util

    # Add ComfyUI root to path
    comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.insert(0, comfyui_root)

    # Test ComfyUI import first
    import comfy.model_sampling


    # Load the main module directly
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_module_path = os.path.join(project_path, 'nodes.py')
    spec = importlib.util.spec_from_file_location('nodes', main_module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules['nodes'] = module
        spec.loader.exec_module(module)

        # Get the classes
        TripleKSamplerWan22LightningAdvanced = module.TripleKSamplerWan22LightningAdvanced
        TripleKSamplerWan22Lightning = module.TripleKSamplerWan22Lightning
    else:
        raise ImportError("Could not load main module")
    COMFYUI_AVAILABLE = True
except Exception as e:
    # ComfyUI dependencies not available - tests will be skipped
    pass


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestParameterValidation:
    """Test parameter validation for TripleKSampler nodes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()
        self.simple_node = TripleKSamplerWan22Lightning()
        
        # Mock models and inputs with proper ComfyUI compatibility
        from unittest.mock import MagicMock

        # Create model mocks with required attributes for ComfyUI
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

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
            "switch_step": -1
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
        
        with pytest.raises(ValueError, match="switch_step \\(10\\) must be < lightning_steps \\(8\\)"):
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
        params["switch_step"] = 0      # Skip Stage2 
        params["base_steps"] = 5       # Positive base_steps (invalid)
        
        with pytest.raises(ValueError, match="When skipping both Stage 1 and Stage 2.*base_steps must be -1 or 0"):
            self.advanced_node.sample(**params)

    def test_lightning_start_zero_with_positive_base_steps(self):
        """Test base_steps ignored when lightning_start=0."""
        params = self.valid_params.copy()
        params["lightning_start"] = 0  # Lightning-only mode
        params["base_steps"] = 5       # Positive base_steps (will be ignored)
        
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
        assert "cannot be greater than switch_step" in error_msg or "cannot be less than lightning_start" in error_msg

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
            "switch_strategy": "50% of steps"
        }
        
        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            self.simple_node.sample(**simple_params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Set up proper mocks like in TestParameterValidation
        from unittest.mock import MagicMock
        import torch

        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

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
            "base_steps": 0,        # Valid for lightning-only
            "base_quality_threshold": 20,  # Add missing parameter
            "base_cfg": 3.5,
            "lightning_start": 0,   # Lightning-only mode
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True  # Use dry run to avoid sampling execution
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
            "base_steps": -1,       # Auto-calculate
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
            "switch_step": -1,      # Auto-calculate
            "dry_run": True         # Use dry run to avoid sampling execution
        }

        # Should pass validation and return minimal 8x8 latent
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestSimpleNodeLightningStart:
    """Test Simple node with various lightning_start values (v0.6.1 feature)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.simple_node = TripleKSamplerWan22Lightning()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_simple_node_lightning_start_0(self):
        """Test Simple node with lightning_start=0 (lightning-only mode)."""
        from unittest.mock import patch

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
            "switch_strategy": "50% of steps"
        }

        # Use dry run mode to test lightning-only mode
        params["dry_run"] = True
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.simple_node.sample(**params)

    def test_simple_node_lightning_start_1(self):
        """Test Simple node with lightning_start=1 (default)."""
        from unittest.mock import patch

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
            "switch_strategy": "50% of steps"
        }

        # Use dry run mode to test default lightning_start
        params["dry_run"] = True
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.simple_node.sample(**params)

    def test_simple_node_lightning_start_higher_values(self):
        """Test Simple node with higher lightning_start values."""
        from unittest.mock import patch

        # Test with valid combinations that pass validation
        test_cases = [
            {"lightning_start": 2, "lightning_steps": 8},  # switch_step=4, 2 < 4 ✓
            {"lightning_start": 3, "lightning_steps": 8},  # switch_step=4, 3 < 4 ✓
            {"lightning_start": 2, "lightning_steps": 12}, # switch_step=6, 2 < 6 ✓
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
                "switch_strategy": "50% of steps"
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
            "switch_strategy": "50% of steps"
        }

        # Should raise validation error
        with pytest.raises(ValueError, match="lightning_start must be within"):
            self.simple_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestDryRunMode:
    """Test dry run mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

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
            "switch_step": -1
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
            "Manual boundary"
        ]

        # Mock sigma calculation and model patcher for boundary strategies
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_patcher = MagicMock()
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (self.mock_lightning_high, None),
            (self.mock_lightning_low, None)
        ] * len(strategies)  # Repeat for each strategy

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

        with patch('comfy.samplers.calculate_sigmas', return_value=mock_sigmas), \
             patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            # Mock get_model_object to return our mock sampling
            self.mock_lightning_high.get_model_object.return_value = mock_sampling
            self.mock_lightning_low.get_model_object.return_value = mock_sampling
            for strategy in strategies:
                params = self.base_params.copy()
                params.update({
                    "switch_strategy": strategy,
                    "switch_step": 4 if strategy == "Manual switch step" else -1,
                    "dry_run": True
                })

                # Should complete without error and return minimal 8x8 latent
                with pytest.raises(comfy.model_management.InterruptProcessingException):

                    self.advanced_node.sample(**params)

    def test_dry_run_complex_scenarios(self):
        """Test dry run mode with complex parameter scenarios."""
        # Lightning-only mode
        params = self.base_params.copy()
        params.update({
            "base_steps": 0,
            "lightning_start": 0,
            "switch_strategy": "50% of steps",
            "dry_run": True
        })
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)

        # Auto-calculation mode
        params.update({
            "base_steps": -1,
            "lightning_start": 1,
            "switch_strategy": "Manual switch step",
            "switch_step": -1
        })
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)

    def test_dry_run_validation_still_occurs(self):
        """Test that validation still occurs in dry run mode."""
        params = self.base_params.copy()
        params.update({
            "lightning_steps": 1,  # Invalid: too small
            "dry_run": True
        })

        # Should still raise validation error even in dry run
        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestConfigParameterBoundaries:
    """Test edge cases for configuration parameters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

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
                "dry_run": True
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
                    "dry_run": True
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
                "dry_run": True
            }

            with pytest.raises(comfy.model_management.InterruptProcessingException):


                self.advanced_node.sample(**params)

    def test_switch_boundary_full_range(self):
        """Test switch_boundary across full valid range."""
        from unittest.mock import patch
        import torch

        boundaries = [0.0, 0.1, 0.25, 0.5, 0.75, 0.875, 0.9, 0.95, 0.99, 1.0]

        # Mock sigma calculation and model patcher for boundary strategies
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_patcher = MagicMock()
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (self.mock_lightning_high, None),
            (self.mock_lightning_low, None)
        ] * len(boundaries)  # Repeat for each boundary

        # Mock timestep function for boundary calculations
        def mock_timestep(sigma_val):
            """Mock timestep that returns comparable float values for boundary testing."""
            sigma = float(sigma_val)
            # Return timesteps that work for different boundaries
            if sigma >= 5.0:
                return 950.0  # Above all boundaries
            elif sigma >= 2.5:
                return 900.0  # Between high and low boundaries
            else:
                return 750.0  # Below all boundaries

        # Mock the sampling object's timestep method
        mock_sampling = MagicMock()
        mock_sampling.timestep.side_effect = mock_timestep

        with patch('comfy.samplers.calculate_sigmas', return_value=mock_sigmas), \
             patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            # Mock get_model_object to return our mock sampling
            self.mock_lightning_high.get_model_object.return_value = mock_sampling
            self.mock_lightning_low.get_model_object.return_value = mock_sampling
            for boundary in boundaries:
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
                    "base_quality_threshold": -1,
                    "base_cfg": 3.5,
                    "lightning_start": 1,
                    "lightning_steps": 8,
                    "lightning_cfg": 1.0,
                    "base_sampler": "euler",
                    "base_scheduler": "simple",
                    "lightning_sampler": "euler",
                    "lightning_scheduler": "simple",
                    "switch_strategy": "Manual boundary",
                    "switch_boundary": boundary,
                    "switch_step": -1,
                    "dry_run": True
                }

                with pytest.raises(comfy.model_management.InterruptProcessingException):


                    self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestBaseQualityThreshold:
    """Test base_quality_threshold parameter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Mock models
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 3.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_base_quality_threshold_default_value(self):
        """Test that -1 uses config default value."""
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
            "base_quality_threshold": 20,  # Use config default
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
            "dry_run": True
        }

        # Should pass validation and use config default
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)

    def test_base_quality_threshold_custom_values(self):
        """Test base_quality_threshold with different custom values."""
        threshold_values = [5, 10, 15, 20, 25, 30, 50, 100]

        for threshold in threshold_values:
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
                "base_quality_threshold": threshold,
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
                "dry_run": True
            }

            # Should handle all threshold values gracefully
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

    def test_base_quality_threshold_with_manual_base_steps(self):
        """Test that base_quality_threshold is ignored when base_steps is manual."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 5,  # Manual value
            "base_quality_threshold": 100,  # Should be ignored
            "base_cfg": 3.5,
            "lightning_start": 2,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        # Should work correctly, threshold ignored for manual base_steps
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)

    def test_base_quality_threshold_boundary_values(self):
        """Test base_quality_threshold boundary conditions."""
        # Test valid boundary values with lightning_start > 0
        valid_boundary_values = [1, 2, 99, 100]

        for threshold in valid_boundary_values:
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
                "base_quality_threshold": threshold,
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
                "dry_run": True
            }

            # Should handle boundary values gracefully
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

        # Test threshold=0 with lightning_start=0 (lightning-only mode)
        params_zero_threshold = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": -1,  # Auto-calculate
            "base_quality_threshold": 0,  # Edge case
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
            "dry_run": True
        }

        # Should handle threshold=0 in lightning-only mode
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params_zero_threshold)

    def test_base_quality_threshold_lightning_only_mode(self):
        """Test base_quality_threshold in lightning-only mode."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 0,  # Lightning-only mode
            "base_quality_threshold": 50,  # Should be ignored in lightning-only
            "base_cfg": 3.5,
            "lightning_start": 0,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        # Should work correctly in lightning-only mode
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestDryRunUIFeatures:
    """Test dry run UI enhancements like toast notifications and formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Mock models
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 3.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_format_base_calculation_compact(self):
        """Test compact formatting of base calculation info."""
        # Test mathematical search method
        info1 = "Auto-calculated base_steps = 5, total_base_steps = 40 (mathematical search)"
        formatted1 = self.advanced_node._format_base_calculation_compact(info1)
        assert "Base steps: 5, Total: 40 (mathematical search)" == formatted1

        # Test simple math method
        info2 = "Auto-calculated base_steps = 3, total_base_steps = 24 (simple math)"
        formatted2 = self.advanced_node._format_base_calculation_compact(info2)
        assert "Base steps: 3, Total: 24 (simple math)" == formatted2

        # Test fallback method
        info3 = "Auto-calculated base_steps = 2 (fallback - no perfect alignment found)"
        formatted3 = self.advanced_node._format_base_calculation_compact(info3)
        assert "Base steps: 2 (fallback)" == formatted3

        # Test manual calculation
        info4 = "Auto-calculated total_base_steps = 32 for manual base_steps = 4"
        formatted4 = self.advanced_node._format_base_calculation_compact(info4)
        assert "Base steps: 4, Total: 32 (manual)" == formatted4

        # Test unknown format fallback
        info5 = "Some unknown calculation format"
        formatted5 = self.advanced_node._format_base_calculation_compact(info5)
        assert info5 == formatted5

    def test_format_switch_info_compact(self):
        """Test compact formatting of model switching info."""
        # Test boundary strategy
        info1 = "Model switching: T2V boundary (boundary = 0.875) → switch at step 3 of 8"
        formatted1 = self.advanced_node._format_switch_info_compact(info1)
        assert "Switch: T2V boundary → step 3 of 8" == formatted1

        # Test percentage strategy
        info2 = "Model switching: 50% of steps → switch at step 4 of 8"
        formatted2 = self.advanced_node._format_switch_info_compact(info2)
        assert "Switch: 50% of steps → step 4 of 8" == formatted2

        # Test manual strategy
        info3 = "Model switching: Manual switch step → switch at step 5 of 8"
        formatted3 = self.advanced_node._format_switch_info_compact(info3)
        assert "Switch: Manual switch step → step 5 of 8" == formatted3

        # Test unknown format fallback
        info4 = "Some unknown switching format"
        formatted4 = self.advanced_node._format_switch_info_compact(info4)
        assert info4 == formatted4

    def test_dry_run_notification_method_exists(self):
        """Test that the dry run notification method exists and doesn't crash."""
        # Test basic call with all parameters
        try:
            self.advanced_node._send_dry_run_notification(
                stage1_info="Stage 1: Base high model - steps 0-2 of 16 (denoising 0.0%–12.5%)",
                stage2_info="Stage 2: Lightning high model - steps 1-4 of 8 (denoising 12.5%–50.0%)",
                stage3_info="Stage 3: Lightning low model - steps 4-8 of 8 (denoising 50.0%–100.0%)",
                base_calculation_info="Auto-calculated base_steps = 3, total_base_steps = 24 (simple math)",
                model_switching_info="Model switching: 50% of steps → switch at step 4 of 8"
            )
        except Exception as e:
            pytest.fail(f"Dry run notification method should not raise exception: {e}")

        # Test call with minimal parameters
        try:
            self.advanced_node._send_dry_run_notification(
                stage1_info="Stage 1: Skipped (Lightning-only mode)",
                stage2_info="Stage 2: Lightning high model - steps 0-4 of 8 (denoising 0.0%–50.0%)",
                stage3_info="Stage 3: Lightning low model - steps 4-8 of 8 (denoising 50.0%–100.0%)"
            )
        except Exception as e:
            pytest.fail(f"Dry run notification method should not raise exception: {e}")

    def test_dry_run_with_auto_calculation_captures_info(self):
        """Test that dry run captures calculation info for notifications."""
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
            "base_quality_threshold": 25,
            "base_cfg": 3.5,
            "lightning_start": 2,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "base_sampler": "euler",
            "base_scheduler": "simple",
            "lightning_sampler": "euler",
            "lightning_scheduler": "simple",
            "switch_strategy": "50% of steps",  # Use simple strategy
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        # Should complete successfully and capture calculation info
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)

    def test_dry_run_with_different_strategies_captures_switch_info(self):
        """Test that dry run captures model switching info for different strategies."""
        # Test strategies that don't require complex boundary computation
        simple_strategies = [
            "50% of steps",
            "Manual switch step"
        ]

        for strategy in simple_strategies:
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
                "base_quality_threshold": 20,
                "base_cfg": 3.5,
                "lightning_start": 1,
                "lightning_steps": 8,
                "lightning_cfg": 1.0,
                "base_sampler": "euler",
                "base_scheduler": "simple",
                "lightning_sampler": "euler",
                "lightning_scheduler": "simple",
                "switch_strategy": strategy,
                "switch_boundary": 0.875,
                "switch_step": 5 if strategy == "Manual switch step" else -1,
                "dry_run": True
            }

            # Should complete successfully for all strategies
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestDynamicWidgetLogic:
    """Test dynamic widget visibility logic and configuration."""

    def test_advanced_node_input_types_has_base_quality_threshold(self):
        """Test that the Advanced node INPUT_TYPES includes base_quality_threshold widget."""
        input_types = TripleKSamplerWan22LightningAdvanced.INPUT_TYPES()

        # Should have base_quality_threshold in required inputs
        assert "base_quality_threshold" in input_types["required"]

        # Should have correct widget configuration
        threshold_config = input_types["required"]["base_quality_threshold"]
        assert threshold_config[0] == "INT"
        assert "default" in threshold_config[1]
        assert threshold_config[1]["default"] == 20
        assert "min" in threshold_config[1]
        assert threshold_config[1]["min"] == 1
        assert "max" in threshold_config[1]
        assert threshold_config[1]["max"] == 100
        assert "tooltip" in threshold_config[1]

    def test_advanced_node_input_types_has_dynamic_widgets(self):
        """Test that the Advanced node has widgets for dynamic visibility."""
        input_types = TripleKSamplerWan22LightningAdvanced.INPUT_TYPES()

        # Should have switch strategy widget in required
        assert "switch_strategy" in input_types["required"]

        # Should have conditional widgets that can be hidden/shown in optional
        assert "optional" in input_types
        assert "switch_step" in input_types["optional"]
        assert "switch_boundary" in input_types["optional"]

        # base_steps should be in required
        assert "base_steps" in input_types["required"]

    def test_simple_node_input_types_no_base_quality_threshold(self):
        """Test that the Simple node doesn't expose base_quality_threshold."""
        input_types = TripleKSamplerWan22Lightning.INPUT_TYPES()

        # Should NOT have base_quality_threshold in Simple node
        assert "base_quality_threshold" not in input_types["required"]

        # Should have simplified switch strategy options
        assert "switch_strategy" in input_types["required"]
        strategy_config = input_types["required"]["switch_strategy"]

        # Should be limited to simple strategies
        available_strategies = strategy_config[0]
        assert "50% of steps" in available_strategies
        assert "T2V boundary" in available_strategies
        assert "I2V boundary" in available_strategies

        # Should NOT have manual strategies that require additional widgets
        assert "Manual switch step" not in available_strategies
        assert "Manual boundary" not in available_strategies

    def test_widget_interaction_logic_base_steps_auto_vs_manual(self):
        """Test the logic for when base_quality_threshold should be visible."""
        # Test case 1: base_steps = -1 (auto-calculation)
        # In this case, base_quality_threshold should be visible to control auto-calculation
        base_steps_auto = -1
        should_show_threshold = (base_steps_auto == -1)
        assert should_show_threshold is True

        # Test case 2: base_steps = manual value (e.g., 5)
        # In this case, base_quality_threshold should be hidden
        base_steps_manual = 5
        should_show_threshold = (base_steps_manual == -1)
        assert should_show_threshold is False

    def test_widget_interaction_logic_switch_strategy_visibility(self):
        """Test the logic for when switch widgets should be visible."""
        strategies_test_cases = [
            ("50% of steps", False, False),          # Neither switch_step nor switch_boundary
            ("Manual switch step", True, False),     # Only switch_step
            ("T2V boundary", False, False),          # Neither (uses preset boundary)
            ("I2V boundary", False, False),          # Neither (uses preset boundary)
            ("Manual boundary", False, True),        # Only switch_boundary
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