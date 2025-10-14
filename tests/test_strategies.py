"""
Unit tests for TripleKSampler switching strategies.

Tests each switching strategy comprehensively to ensure they work correctly
with different parameter combinations and edge cases.
"""

import pytest
import sys
import os
import torch
from unittest.mock import MagicMock, patch

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


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestFiftyPercentStrategy:
    """Test the '50% of steps' strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()
        self.simple_node = TripleKSamplerWan22Lightning()

        # Create proper model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        # Mock get_model_object to return a sampling instance with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            # Default behavior for boundary tests
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary

        mock_sampling_instance.timestep.side_effect = mock_timestep
        self.mock_lightning_high.get_model_object.return_value = mock_sampling_instance

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_fifty_percent_even_steps(self):
        """Test 50% strategy with even number of steps."""
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
            "base_quality_threshold": 20,
            "base_cfg": 3.5,
            "lightning_start": 1,
            "lightning_steps": 8,  # Even number
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "50% of steps",
            "switch_boundary": 0.875,
            "switch_step": -1,  # Auto-calculate
            "dry_run": True  # Safe testing
        }

        # Should not raise exceptions
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)

    def test_fifty_percent_odd_steps(self):
        """Test 50% strategy with odd number of steps."""
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": -1,
            "base_quality_threshold": 20,
            "base_cfg": 3.5,
            "lightning_start": 1,
            "lightning_steps": 7,  # Odd number
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

    def test_fifty_percent_simple_node(self):
        """Test 50% strategy in simple node."""
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
            "switch_strategy": "50% of steps"
        }

        # Simple node should handle this strategy using dry run mode
        params["dry_run"] = True
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            self.simple_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestBoundaryStrategies:
    """Test T2V and I2V boundary strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Create model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        # Mock get_model_object to return a sampling instance with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            # Default behavior for boundary tests
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary

        mock_sampling_instance.timestep.side_effect = mock_timestep
        self.mock_lightning_high.get_model_object.return_value = mock_sampling_instance

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    @patch('comfy.samplers.calculate_sigmas')
    def test_t2v_boundary_strategy(self, mock_calculate_sigmas):
        """Test T2V boundary strategy uses correct boundary value."""
        # Mock sigma calculation
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock the model patcher to return models with proper get_model_object
        mock_patcher = MagicMock()
        mock_patched_lightning_high = MagicMock()

        # Mock sampling object with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            if sigma_val >= 2.5:
                return 950.0  # Above boundary (0.95 > 0.875)
            else:
                return 850.0  # Below boundary (0.85 < 0.875)

        mock_sampling_instance.timestep.side_effect = mock_timestep
        mock_patched_lightning_high.get_model_object.return_value = mock_sampling_instance

        # Mock patcher.patch to return our controlled objects
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (mock_patched_lightning_high, None),
            (self.mock_lightning_low, None)
        ]

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
            "switch_strategy": "T2V boundary",  # Should use 0.875 boundary
            "switch_boundary": 0.999,  # Should be ignored
            "switch_step": -1,
            "dry_run": True
        }

        # Mock the _get_model_patcher method
        with patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

    @patch('comfy.samplers.calculate_sigmas')
    def test_i2v_boundary_strategy(self, mock_calculate_sigmas):
        """Test I2V boundary strategy uses correct boundary value."""
        # Mock sigma calculation
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock the model patcher to return models with proper get_model_object
        mock_patcher = MagicMock()
        mock_patched_lightning_high = MagicMock()

        # Mock sampling object with timestep method for I2V boundary (0.900)
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            if sigma_val >= 1.25:
                return 950.0  # Above boundary (0.95 > 0.900)
            else:
                return 880.0  # Below boundary (0.88 < 0.900)

        mock_sampling_instance.timestep.side_effect = mock_timestep
        mock_patched_lightning_high.get_model_object.return_value = mock_sampling_instance

        # Mock patcher.patch to return our controlled objects
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (mock_patched_lightning_high, None),
            (self.mock_lightning_low, None)
        ]

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
            "switch_strategy": "I2V boundary",  # Should use 0.900 boundary
            "switch_boundary": 0.999,  # Should be ignored
            "switch_step": -1,
            "dry_run": True
        }

        # Mock the _get_model_patcher method
        with patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

    @patch('comfy.samplers.calculate_sigmas')
    def test_manual_boundary_strategy(self, mock_calculate_sigmas):
        """Test manual boundary strategy uses provided boundary value."""
        # Mock sigma calculation
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock the model patcher to return models with proper get_model_object
        mock_patcher = MagicMock()
        mock_patched_lightning_high = MagicMock()

        # Mock sampling object with timestep method for custom boundary (0.5)
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            if sigma_val >= 1.25:
                return 600.0  # Above boundary (0.6 > 0.5)
            else:
                return 400.0  # Below boundary (0.4 < 0.5)

        mock_sampling_instance.timestep.side_effect = mock_timestep
        mock_patched_lightning_high.get_model_object.return_value = mock_sampling_instance

        # Mock patcher.patch to return our controlled objects
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (mock_patched_lightning_high, None),
            (self.mock_lightning_low, None)
        ]
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
            "switch_strategy": "Manual boundary",
            "switch_boundary": 0.5,  # Custom boundary
            "switch_step": -1,
            "dry_run": True
        }

        # Mock the _get_model_patcher method
        with patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

    @patch('comfy.samplers.calculate_sigmas')
    def test_boundary_edge_values(self, mock_calculate_sigmas):
        """Test boundary strategies with edge values."""
        # Mock sigma calculation
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock the model patcher to return models with proper get_model_object
        mock_patcher = MagicMock()
        mock_patched_lightning_high = MagicMock()

        # Mock sampling object with timestep method for boundary never crossed
        mock_sampling_instance = MagicMock()
        mock_sampling_instance.timestep.return_value = 950.0  # Always above boundary
        mock_patched_lightning_high.get_model_object.return_value = mock_sampling_instance

        # Mock patcher.patch to return our controlled objects
        # Need 6 return values for 2 test runs (3 models each)
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None),
            (mock_patched_lightning_high, None),
            (self.mock_lightning_low, None),
            (self.mock_base_high, None),
            (mock_patched_lightning_high, None),
            (self.mock_lightning_low, None)
        ]
        # Test with boundary = 0.0
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
            "switch_strategy": "Manual boundary",
            "switch_boundary": 0.0,
            "switch_step": -1,
            "dry_run": True
        }

        # Mock the _get_model_patcher method
        with patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)

            # Test with boundary = 1.0
            params["switch_boundary"] = 1.0
            with pytest.raises(comfy.model_management.InterruptProcessingException):

                self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestManualSwitchStepStrategy:
    """Test manual switch step strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()

        # Create model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        # Mock get_model_object to return a sampling instance with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            # Default behavior for boundary tests
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary

        mock_sampling_instance.timestep.side_effect = mock_timestep
        self.mock_lightning_high.get_model_object.return_value = mock_sampling_instance

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    def test_manual_switch_step_valid(self):
        """Test manual switch step with valid step."""
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
            "switch_strategy": "Manual switch step",
            "switch_boundary": 0.875,  # Should be ignored
            "switch_step": 4,  # Manual step
            "dry_run": True
        }

        with pytest.raises(comfy.model_management.InterruptProcessingException):


            self.advanced_node.sample(**params)

    def test_manual_switch_step_auto_calculate(self):
        """Test manual switch step with auto-calculation (step=-1)."""
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
            "switch_strategy": "Manual switch step",
            "switch_boundary": 0.875,
            "switch_step": -1,  # Auto-calculate
            "dry_run": True
        }

        with pytest.raises(comfy.model_management.InterruptProcessingException):


            self.advanced_node.sample(**params)

    def test_manual_switch_step_edge_cases(self):
        """Test manual switch step with edge cases."""
        # Test switch at step 0 (immediate switch)
        params = {
            "base_high": self.mock_base_high,
            "lightning_high": self.mock_lightning_high,
            "lightning_low": self.mock_lightning_low,
            "positive": MagicMock(),
            "negative": MagicMock(),
            "latent_image": self.mock_latent,
            "seed": 42,
            "sigma_shift": 5.0,
            "base_steps": 0,  # No base steps for lightning-only
            "base_quality_threshold": -1,
            "base_cfg": 3.5,
            "lightning_start": 0,  # Lightning-only mode
            "lightning_steps": 8,
            "lightning_cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "switch_strategy": "Manual switch step",
            "switch_boundary": 0.875,
            "switch_step": 0,
            "dry_run": True
        }

        with pytest.raises(comfy.model_management.InterruptProcessingException):


            self.advanced_node.sample(**params)

        # Test switch at last step
        params.update({
            "lightning_start": 1,
            "base_steps": 3,
            "switch_step": 7  # Last valid step for 8 steps
        })
        with pytest.raises(comfy.model_management.InterruptProcessingException):

            self.advanced_node.sample(**params)


@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestStrategyIntegration:
    """Test strategy integration and interactions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.advanced_node = TripleKSamplerWan22LightningAdvanced()
        self.simple_node = TripleKSamplerWan22Lightning()

        # Create model mocks
        self.mock_base_high = MagicMock()
        self.mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_lightning_high = MagicMock()
        self.mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        # Mock get_model_object to return a sampling instance with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            # Default behavior for boundary tests
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary

        mock_sampling_instance.timestep.side_effect = mock_timestep
        self.mock_lightning_high.get_model_object.return_value = mock_sampling_instance

        self.mock_lightning_low = MagicMock()
        self.mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        self.mock_latent = {"samples": torch.randn(1, 4, 64, 64)}

    @patch('comfy.samplers.calculate_sigmas')
    def test_simple_node_strategies(self, mock_calculate_sigmas):
        """Test that simple node supports all its strategies."""
        # Mock sigma calculation
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock the model patcher to return models with proper get_model_object
        mock_patcher = MagicMock()
        mock_patched_lightning_high = MagicMock()

        # Mock sampling object with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary

        mock_sampling_instance.timestep.side_effect = mock_timestep
        mock_patched_lightning_high.get_model_object.return_value = mock_sampling_instance

        # Mock patcher.patch to return our controlled objects (3 strategies * 3 models each = 9)
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None),
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None),
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None)
        ]
        strategies = ["50% of steps", "T2V boundary", "I2V boundary"]

        for strategy in strategies:
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
                "lightning_start": 1,
                "lightning_steps": 8,
                "lightning_cfg": 1.0,  # Required but ignored
                "sampler_name": "euler",
                "scheduler": "simple",
                "switch_strategy": strategy
            }

            # Use dry run mode for strategy testing
            params["dry_run"] = True

            with patch.object(self.simple_node, '_get_model_patcher', return_value=mock_patcher):
                with pytest.raises(comfy.model_management.InterruptProcessingException):
                    self.simple_node.sample(**params)

    @patch('comfy.samplers.calculate_sigmas')
    def test_advanced_node_strategies(self, mock_calculate_sigmas):
        """Test that advanced node supports all its strategies."""
        # Mock sigma calculation
        mock_sigmas = torch.tensor([10.0, 5.0, 2.5, 1.25, 0.625, 0.312, 0.156, 0.078])
        mock_calculate_sigmas.return_value = mock_sigmas

        # Mock the model patcher to return models with proper get_model_object
        mock_patcher = MagicMock()
        mock_patched_lightning_high = MagicMock()

        # Mock sampling object with timestep method
        mock_sampling_instance = MagicMock()
        def mock_timestep(sigma):
            sigma_val = float(sigma) if not hasattr(sigma, 'item') else float(sigma)
            if sigma_val >= 2.5:
                return 950.0  # Above boundary
            else:
                return 850.0  # Below boundary

        mock_sampling_instance.timestep.side_effect = mock_timestep
        mock_patched_lightning_high.get_model_object.return_value = mock_sampling_instance

        # Mock patcher.patch to return our controlled objects (5 strategies * 3 models each = 15)
        mock_patcher.patch.side_effect = [
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None),
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None),
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None),
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None),
            (self.mock_base_high, None), (mock_patched_lightning_high, None), (self.mock_lightning_low, None)
        ]
        strategies = [
            "50% of steps",
            "Manual switch step",
            "T2V boundary",
            "I2V boundary",
            "Manual boundary"
        ]

        for strategy in strategies:
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
                "switch_strategy": strategy,
                "switch_boundary": 0.875,
                "switch_step": 4 if strategy == "Manual switch step" else -1,
                "dry_run": True
            }

            # Mock the _get_model_patcher method
            with patch.object(self.advanced_node, '_get_model_patcher', return_value=mock_patcher):
                with pytest.raises(comfy.model_management.InterruptProcessingException):

                    self.advanced_node.sample(**params)

    def test_strategy_with_different_step_counts(self):
        """Test strategies with various lightning step counts."""
        step_counts = [2, 4, 6, 8, 12, 16]

        for steps in step_counts:
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
                "lightning_steps": steps,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])