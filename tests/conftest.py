"""
Pytest configuration and shared fixtures for TripleKSampler tests.

Provides reusable fixtures and test utilities to reduce code duplication
and improve test maintainability.
"""

import pytest
import sys
import os
import torch
from unittest.mock import MagicMock

# Check if ComfyUI dependencies are available
COMFYUI_AVAILABLE = False
try:
    import importlib.util

    # Add ComfyUI root to path FIRST (before project path)
    comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    sys.path.insert(0, comfyui_root)

    # Test ComfyUI import first
    import comfy.model_sampling
    import comfy.samplers

    # CRITICAL: Import ComfyUI's nodes module BEFORE loading our custom nodes.py
    # This ensures nodes.KSamplerAdvanced is available when our code imports it
    import nodes as comfyui_nodes

    # CRITICAL: Lock the nodes module in sys.modules to prevent our nodes.py from overwriting it
    # This is necessary because when we exec_module our nodes.py, it will do "import nodes"
    # and Python needs to find ComfyUI's nodes, not our file
    assert 'nodes' in sys.modules, "ComfyUI nodes module must be in sys.modules"
    assert hasattr(comfyui_nodes, 'KSamplerAdvanced'), "ComfyUI nodes must have KSamplerAdvanced"

    # Load the main module directly with a unique name
    # DO NOT add project path to sys.path - load directly via spec
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_module_path = os.path.join(project_path, 'nodes.py')
    spec = importlib.util.spec_from_file_location('triple_ksampler_nodes', main_module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules['triple_ksampler_nodes'] = module

        # CRITICAL: Pre-inject ComfyUI's nodes into the module's __dict__ BEFORE exec
        # This prevents the "import nodes" statement from importing the wrong module
        # The import statement will see nodes already in globals and use it
        module.__dict__['nodes'] = comfyui_nodes

        spec.loader.exec_module(module)

        # Verify that the module got the correct nodes reference
        assert hasattr(module.nodes, 'KSamplerAdvanced'), \
            f"Module's nodes reference is wrong: {module.nodes}"

        # Get the classes - make them available globally
        TripleKSamplerWan22Base = module.TripleKSamplerWan22Base
        TripleKSamplerWan22LightningAdvanced = module.TripleKSamplerWan22LightningAdvanced
        TripleKSamplerWan22Lightning = module.TripleKSamplerWan22Lightning
    else:
        raise ImportError("Could not load main module")
    COMFYUI_AVAILABLE = True
except Exception as e:
    # ComfyUI dependencies not available - tests will be skipped
    pass


# Export availability for other test modules
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def comfyui_available():
    """Fixture to check if ComfyUI dependencies are available."""
    return COMFYUI_AVAILABLE


@pytest.fixture
def mock_model_factory():
    """Factory fixture for creating consistent model mocks."""
    def _create_mock_model(name="test_model"):
        mock_model = MagicMock()
        mock_model.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}
        mock_model.clone.return_value = MagicMock()
        return mock_model

    return _create_mock_model


@pytest.fixture
def mock_models(mock_model_factory):
    """Fixture providing standard set of mock models."""
    return {
        'base_high': mock_model_factory("base_high"),
        'lightning_high': mock_model_factory("lightning_high"),
        'lightning_low': mock_model_factory("lightning_low")
    }


@pytest.fixture
def mock_conditioning():
    """Fixture providing mock conditioning objects."""
    return {
        'positive': MagicMock(),
        'negative': MagicMock()
    }


@pytest.fixture
def mock_latent():
    """Fixture providing mock latent tensor."""
    return {"samples": torch.randn(1, 4, 64, 64)}


@pytest.fixture
def standard_params(mock_models, mock_conditioning, mock_latent):
    """Fixture providing standard parameter set for testing."""
    return {
        "base_high": mock_models['base_high'],
        "lightning_high": mock_models['lightning_high'],
        "lightning_low": mock_models['lightning_low'],
        "positive": mock_conditioning['positive'],
        "negative": mock_conditioning['negative'],
        "latent_image": mock_latent,
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
        "dry_run": False,
        "switch_boundary": 0.875,
        "switch_step": -1
    }


@pytest.fixture
def advanced_node():
    """Fixture providing TripleKSamplerWan22LightningAdvanced instance."""
    if not COMFYUI_AVAILABLE:
        pytest.skip("ComfyUI dependencies not available")
    return TripleKSamplerWan22LightningAdvanced()


@pytest.fixture
def simple_node():
    """Fixture providing TripleKSamplerWan22Lightning instance."""
    if not COMFYUI_AVAILABLE:
        pytest.skip("ComfyUI dependencies not available")
    return TripleKSamplerWan22Lightning()


@pytest.fixture
def base_node():
    """Fixture providing TripleKSamplerWan22Base instance."""
    if not COMFYUI_AVAILABLE:
        pytest.skip("ComfyUI dependencies not available")
    return TripleKSamplerWan22Base()


# Custom assertion helpers
class TripleKSamplerAssertions:
    """Custom assertion helpers for TripleKSampler tests."""

    @staticmethod
    def assert_valid_percentage(value, tolerance=0.1):
        """Assert that a value is a valid percentage (0-100) within tolerance."""
        assert isinstance(value, (int, float)), f"Expected numeric value, got {type(value)}"
        assert 0.0 <= value <= 100.0, f"Percentage {value} not in range [0, 100]"

    @staticmethod
    def assert_stage_range_format(stage_range_str):
        """Assert that a stage range string has the expected format."""
        assert isinstance(stage_range_str, str), "Stage range must be a string"
        assert "steps" in stage_range_str, "Stage range must contain 'steps'"
        assert "%" in stage_range_str, "Stage range must contain percentage"
        assert "of" in stage_range_str, "Stage range must contain 'of'"

    @staticmethod
    def assert_models_cloned(mock_models):
        """Assert that all provided mock models were cloned."""
        for name, model in mock_models.items():
            model.clone.assert_called_once(), f"Model {name} was not cloned"

    @staticmethod
    def assert_no_original_mutation(mock_models, allowed_calls=None):
        """Assert that original models were not mutated beyond allowed calls."""
        if allowed_calls is None:
            allowed_calls = ['clone']

        for name, model in mock_models.items():
            actual_calls = [call[0] for call in model.method_calls]
            unexpected_calls = [call for call in actual_calls if call not in allowed_calls]
            assert not unexpected_calls, f"Model {name} had unexpected calls: {unexpected_calls}"

    @staticmethod
    def assert_minimal_latent_result(result, original_latent):
        """Assert that result contains a minimal 8x8 latent with correct properties.

        Args:
            result: The tuple result from the sampling function
            original_latent: The original latent dict to match properties from
        """
        import torch

        # Result should be a tuple with one element
        assert isinstance(result, tuple), f"Expected tuple result, got {type(result)}"
        assert len(result) == 1, f"Expected tuple of length 1, got {len(result)}"

        # Extract the latent dict
        latent_dict = result[0]
        assert isinstance(latent_dict, dict), f"Expected dict in result, got {type(latent_dict)}"
        assert "samples" in latent_dict, "Result latent must contain 'samples' key"

        # Get tensors
        result_samples = latent_dict["samples"]
        original_samples = original_latent["samples"]

        # Validate tensor properties
        assert isinstance(result_samples, torch.Tensor), "Result samples must be a tensor"
        assert isinstance(original_samples, torch.Tensor), "Original samples must be a tensor"

        # Check shape: should be (1, channels, 1, 8, 8) where channels matches original
        expected_shape = (1, original_samples.shape[1], 1, 8, 8)
        assert result_samples.shape == expected_shape, \
            f"Expected minimal latent shape {expected_shape}, got {result_samples.shape}"

        # Check device and dtype match original
        assert result_samples.device == original_samples.device, \
            f"Device mismatch: expected {original_samples.device}, got {result_samples.device}"
        assert result_samples.dtype == original_samples.dtype, \
            f"Dtype mismatch: expected {original_samples.dtype}, got {result_samples.dtype}"

        # Verify it's actually zeros (as created by torch.zeros)
        assert torch.all(result_samples == 0), "Minimal latent should be all zeros"


@pytest.fixture
def tks_assert():
    """Fixture providing TripleKSampler-specific assertions."""
    return TripleKSamplerAssertions()


# Parametrized test data
@pytest.fixture(params=[
    {"lightning_steps": 4, "expected_midpoint": 2},
    {"lightning_steps": 6, "expected_midpoint": 3},
    {"lightning_steps": 8, "expected_midpoint": 4},
    {"lightning_steps": 10, "expected_midpoint": 5},
])
def midpoint_test_data(request):
    """Parametrized fixture for midpoint calculation tests."""
    return request.param


@pytest.fixture(params=[
    {"base_quality_threshold": 20, "lightning_start": 1, "lightning_steps": 8},
    {"base_quality_threshold": 25, "lightning_start": 2, "lightning_steps": 6},
    {"base_quality_threshold": 30, "lightning_start": 1, "lightning_steps": 12},
])
def alignment_test_data(request):
    """Parametrized fixture for perfect alignment calculation tests."""
    return request.param


@pytest.fixture(params=[
    "50% of steps",
    "T2V boundary",
    "I2V boundary"
])
def simple_strategies(request):
    """Parametrized fixture for Simple node strategies."""
    return request.param


@pytest.fixture(params=[
    "50% of steps",
    "Manual switch step",
    "T2V boundary",
    "I2V boundary",
    "Manual boundary"
])
def advanced_strategies(request):
    """Parametrized fixture for Advanced node strategies."""
    return request.param


# Test configuration helpers - exported as fixtures for better IDE support
@pytest.fixture
def param_helpers():
    """Fixture providing test parameter helper functions."""
    class ParameterHelpers:
        @staticmethod
        def create_dry_run_params(base_params):
            """
            Helper to create dry run parameters from base parameters.
            Note: dry_run is now a regular parameter, not a node attribute.
            """
            params = base_params.copy()
            params["dry_run"] = True
            return params

        @staticmethod
        def create_lightning_only_params(base_params):
            """Helper to create lightning-only parameters from base parameters."""
            params = base_params.copy()
            params.update({
                "base_steps": 0,
                "lightning_start": 0
            })
            return params

        @staticmethod
        def create_manual_strategy_params(base_params, switch_step=4):
            """Helper to create manual strategy parameters from base parameters."""
            params = base_params.copy()
            params.update({
                "switch_strategy": "Manual switch step",
                "switch_step": switch_step
            })
            return params

        @staticmethod
        def create_simple_node_params(custom_params=None):
            """Helper to create complete parameter set for Simple node testing."""
            # Base parameters that Simple node requires
            base_params = {
                "base_steps": 3,  # Ignored in Simple node
                "base_cfg": 3.5,
                "lightning_start": 1,
                "lightning_steps": 8,
                "lightning_cfg": 1.0,  # Ignored in Simple node
                "sampler_name": "euler",
                "scheduler": "simple",
                "switch_strategy": "50% of steps",
                "dry_run": False,  # Ignored in Simple node
                "switch_boundary": 0.875,  # Ignored in Simple node
                "switch_step": -1,  # Ignored in Simple node
                "base_quality_threshold": -1,  # Ignored in Simple node
            }

            # Apply any custom overrides
            if custom_params:
                base_params.update(custom_params)

            return base_params

    return ParameterHelpers()


# Also export functions directly for convenience (maintains backward compatibility)
def create_dry_run_params(base_params):
    """
    Helper to create dry run parameters from base parameters.
    Note: dry_run is now a regular parameter, not a node attribute.
    """
    params = base_params.copy()
    params["dry_run"] = True
    return params


def create_lightning_only_params(base_params):
    """Helper to create lightning-only parameters from base parameters."""
    params = base_params.copy()
    params.update({
        "base_steps": 0,
        "lightning_start": 0
    })
    return params


def create_manual_strategy_params(base_params, switch_step=4):
    """Helper to create manual strategy parameters from base parameters."""
    params = base_params.copy()
    params.update({
        "switch_strategy": "Manual switch step",
        "switch_step": switch_step
    })
    return params


def create_simple_node_params(custom_params=None):
    """Helper to create complete parameter set for Simple node testing."""
    # Base parameters that Simple node requires
    base_params = {
        "base_steps": 3,  # Ignored in Simple node
        "base_cfg": 3.5,
        "lightning_start": 1,
        "lightning_steps": 8,
        "lightning_cfg": 1.0,  # Ignored in Simple node
        "sampler_name": "euler",
        "scheduler": "simple",
        "switch_strategy": "50% of steps",
        "dry_run": False,  # Ignored in Simple node
        "switch_boundary": 0.875,  # Ignored in Simple node
        "switch_step": -1,  # Ignored in Simple node
        "base_quality_threshold": -1,  # Ignored in Simple node
    }

    # Apply any custom overrides
    if custom_params:
        base_params.update(custom_params)

    return base_params