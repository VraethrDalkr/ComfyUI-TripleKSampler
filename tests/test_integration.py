"""
Integration tests for TripleKSampler end-to-end functionality.

Tests complete workflow simulations, model patching, and stage execution flow.
"""

import pytest
import sys
import os
import torch
from unittest.mock import MagicMock, patch, call

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
    import comfy.samplers

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


@pytest.mark.integration
@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestWorkflowSimulations:
    """Test complete workflow simulations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()
        self.simple_node = TripleKSamplerWan22Lightning()

        # Create comprehensive model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        self.mock_base_high.clone.return_value = MagicMock()

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        self.mock_lightning_high.clone.return_value = MagicMock()

        # Add proper mock timestep functionality for boundary calculations
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            # Return timesteps that work for both T2V (0.875) and I2V (0.900) boundaries
            if sigma_val >= 2.5:
                return 950.0  # Above both boundaries (0.95 > 0.900 > 0.875)
            else:
                return 850.0  # Below both boundaries (0.85 < 0.875 < 0.900)

        mock_sampling_instance.timestep.side_effect = mock_timestep
        self.mock_lightning_high.get_model_object.return_value = mock_sampling_instance

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        self.mock_lightning_low.clone.return_value = MagicMock()

        self.mock_positive = MagicMock()
        self.mock_negative = MagicMock()
        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_t2v_workflow_simulation(self):
        """Test complete T2V workflow simulation."""
        from unittest.mock import patch
        import torch

        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 3.0,
            "base_steps": 5,
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 1,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "T2V boundary",  # Typical T2V workflow
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True  # Safe integration testing
        }

        # Mock the sigma calculation and model patcher for boundary testing
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_patcher = MagicMock()
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (self.mock_lightning_high, None),
            (self.mock_lightning_low, None)
        ]

        with patch('comfy.samplers.calculate_sigmas', return_value=mock_sigmas), \
             patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

        # Note: In dry run mode, models are not actually cloned, so we don't verify clone calls

    def test_i2v_workflow_simulation(self):
        """Test complete I2V workflow simulation."""
        from unittest.mock import patch
        import torch

        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 123,
            "sigma_shift": 5.0,
            "base_steps": 3,
            "base_quality_threshold": -1,
            "base_cfg": 4.0,
            "lightning_start": 2,
            "lightning_steps": 12,
            "lightning_cfg": 1.2,
            "sampler_name": "dpmpp_2m",
            "scheduler": "karras",
            "switch_strategy": "I2V boundary",  # Typical I2V workflow
            "switch_boundary": 0.900,
            "switch_step": -1,
            "dry_run": True
        }

        # Mock the sigma calculation and model patcher for boundary testing
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_patcher = MagicMock()
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (self.mock_lightning_high, None),
            (self.mock_lightning_low, None)
        ]

        with patch('comfy.samplers.calculate_sigmas', return_value=mock_sigmas), \
             patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

    def test_lightning_only_workflow(self):
        """Test lightning-only workflow (no base model)."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 456,
            "sigma_shift": 2.0,
            "base_steps": 0,        # No base steps
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 0,   # Lightning-only mode
            "lightning_steps": 6,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        with pytest.raises(comfy.model_management.InterruptProcessingException):


            self.advanced_node.sample(**params)

    def test_simple_node_workflow(self):
        """Test simple node complete workflow."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": self.mock_positive,
            "negative": self.mock_negative,
            "latent_image": self.mock_latent,
            "seed": 789,
            "sigma_shift": 4.0,
            "base_steps": 3,  # Required but ignored
            "base_quality_threshold": -1,
            "base_cfg": 3.0,
            "lightning_start": 1,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,  # Required but ignored
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps"
        }

        # Use dry run mode instead of mocking core sampling logic
        params["dry_run"] = True
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.simple_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestModelPatching:
    """Test model patching and cloning behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Create detailed model mocks
        self.original_base = MagicMock()
        self.original_base.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.cloned_base = MagicMock()
        self.original_base.clone.return_value = self.cloned_base

        self.original_lightning_high = MagicMock()
        self.original_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.cloned_lightning_high = MagicMock()
        self.original_lightning_high.clone.return_value = self.cloned_lightning_high

        self.original_lightning_low = MagicMock()
        self.original_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.cloned_lightning_low = MagicMock()
        self.original_lightning_low.clone.return_value = self.cloned_lightning_low

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_model_cloning_occurs(self):
        """Test that all models are cloned before patching."""
        params = {
            "base_high": self.original_base,
            "lightning_high": self.original_lightning_high,
            "lightning_low": self.original_lightning_low,
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
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        with pytest.raises(comfy.model_management.InterruptProcessingException):


            self.advanced_node.sample(**params)

        # Verify all models were cloned
        self.original_base.clone.assert_called_once()
        self.original_lightning_high.clone.assert_called_once()
        self.original_lightning_low.clone.assert_called_once()

    def test_sigma_shift_application(self):
        """Test that sigma shift is applied to cloned models."""
        params = {
            "base_high": self.original_base,
            "lightning_high": self.original_lightning_high,
            "lightning_low": self.original_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 7.5,  # Custom sigma shift
            "base_steps": 3,
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 1,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": False
        }

        # Use dry run mode and mock model patcher to test sigma shift application
        params["dry_run"] = True

        with patch.object(self.advanced_node, '_get_model_patcher') as mock_get_patcher:
            # Mock the patcher to return the expected models
            mock_patcher = MagicMock()
            mock_patcher.patch.side_effect = [
                (self.cloned_base, None),
                (self.cloned_lightning_high, None),
                (self.cloned_lightning_low, None)
            ]
            mock_get_patcher.return_value = mock_patcher

            with pytest.raises(comfy.model_management.InterruptProcessingException):


                self.advanced_node.sample(**params)

            # Verify model patcher was called for sigma shift application
            mock_get_patcher.assert_called_once()
            # Verify patching was called for each model
            assert mock_patcher.patch.call_count == 3

    def test_no_original_model_mutation(self):
        """Test that original models are not mutated."""
        # Track original model state
        original_base_calls = self.original_base.method_calls.copy()
        original_lightning_high_calls = self.original_lightning_high.method_calls.copy()
        original_lightning_low_calls = self.original_lightning_low.method_calls.copy()

        params = {
            "base_high": self.original_base,
            "lightning_high": self.original_lightning_high,
            "lightning_low": self.original_lightning_low,
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
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        # dry_run now raises InterruptProcessingException, but we can still check mocks after
        import comfy.model_management
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

        # Verify only clone() was called on original models
        new_base_calls = [call for call in self.original_base.method_calls if call not in original_base_calls]
        new_lightning_high_calls = [call for call in self.original_lightning_high.method_calls if call not in original_lightning_high_calls]
        new_lightning_low_calls = [call for call in self.original_lightning_low.method_calls if call not in original_lightning_low_calls]

        # Should only have clone() calls
        assert len(new_base_calls) == 1
        assert new_base_calls[0][0] == 'clone'
        assert len(new_lightning_high_calls) == 1
        assert new_lightning_high_calls[0][0] == 'clone'
        assert len(new_lightning_low_calls) == 1
        assert new_lightning_low_calls[0][0] == 'clone'


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestStageExecutionFlow:
    """Test stage execution flow and progression."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Create model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        self.mock_base_high.clone.return_value = MagicMock()

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        self.mock_lightning_high.clone.return_value = MagicMock()

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        self.mock_lightning_low.clone.return_value = MagicMock()

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_stage_progression_normal_workflow(self):
        """Test normal three-stage progression."""
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
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        # dry_run now raises InterruptProcessingException
        import comfy.model_management
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

    def test_stage_skip_scenarios(self):
        """Test stage skipping scenarios."""
        # Test Stage 1 skip (lightning_start=0)
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 0,
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 0,  # Skip Stage 1
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        # dry_run now raises InterruptProcessingException
        import comfy.model_management
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

    def test_noise_addition_logic(self):
        """Test noise addition in different scenarios."""
        # Test Stage 1 skip - Stage 2 should add noise
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 0,
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 0,  # Skip Stage 1
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "Manual switch step",
            "switch_boundary": 0.875,
            "switch_step": 4,
            "dry_run": True
        }

        # dry_run now raises InterruptProcessingException
        import comfy.model_management
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)

    def test_auto_calculation_integration(self):
        """Test integration with auto-calculation features."""
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
            "base_quality_threshold": 20,
            "base_cfg": 3.5,
            "lightning_start": 2,
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "Manual switch step",
            "switch_boundary": 0.875,
            "switch_step": -1,      # Auto-calculate
            "dry_run": True
        }

        # dry_run now raises InterruptProcessingException
        import comfy.model_management
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestEndToEndValidation:
    """Test end-to-end validation and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()
        self.simple_node = TripleKSamplerWan22Lightning()

        # Create model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_comprehensive_parameter_validation(self):
        """Test comprehensive parameter validation in integration context."""
        # Test various invalid parameter combinations
        invalid_configs = [
            # lightning_steps too small
            {"lightning_steps": 1, "expected_error": "lightning_steps must be at least 2"},
            # lightning_start out of range
            {"lightning_start": 10, "lightning_steps": 8, "expected_error": "lightning_start must be within"},
            # Inconsistent base_steps with lightning_start
            {"base_steps": 0, "lightning_start": 1, "expected_error": "base_steps must be >= 1 when lightning_start > 0"},
        ]

        base_params = {
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
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1
        }

        for invalid_config in invalid_configs:
            expected_error = invalid_config.pop("expected_error")
            test_params = base_params.copy()
            test_params.update(invalid_config)

            with pytest.raises(ValueError, match=expected_error):
                self.advanced_node.sample(**test_params)

    def test_robustness_with_edge_cases(self):
        """Test robustness with edge case parameters."""
        edge_case_configs = [
            # Minimal valid configuration
            {"base_steps": 1, "lightning_start": 1, "lightning_steps": 2},
            # Large step counts
            {"base_steps": 50, "lightning_start": 10, "lightning_steps": 100},
            # Lightning-only with minimal steps
            {"base_steps": 0, "lightning_start": 0, "lightning_steps": 2},
        ]

        base_params = {
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
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,
            "dry_run": True
        }

        for edge_config in edge_case_configs:
            test_params = base_params.copy()
            test_params.update(edge_config)

            # Should handle edge cases gracefully
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**test_params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])