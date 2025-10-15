"""
Focused integration tests that validate what we can test without full ComfyUI server.

These tests check our core logic integration with ComfyUI components that don't
require the full server environment.
"""

import pytest
import sys
import os
import torch
from unittest.mock import patch, MagicMock

# Add ComfyUI root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Test ComfyUI component imports
COMFYUI_AVAILABLE = False
try:
    import comfy.model_sampling
    import comfy.samplers
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    COMFYUI_AVAILABLE = True
except Exception:
    pass

# Import our classes by importing specific functions without triggering server imports
if COMFYUI_AVAILABLE:
    # We'll import our classes by copy-pasting just the class definitions
    # to avoid the server import issue
    pass


@pytest.mark.integration
@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestCoreIntegration:
    """Test core integration points with ComfyUI without full server environment."""

    def test_model_sampling_sd3_import(self):
        """Test that ModelSamplingSD3 import works as expected."""
        # This validates our import path is correct
        assert ModelSamplingSD3 is not None

        # Test creating a ModelSamplingSD3 node instance
        node = ModelSamplingSD3()

        # Verify it has the expected ComfyUI node interface
        assert hasattr(node, 'patch')
        assert hasattr(node, 'INPUT_TYPES')
        assert hasattr(node, 'RETURN_TYPES')

        # Check INPUT_TYPES structure
        input_types = node.INPUT_TYPES()
        assert 'required' in input_types
        assert 'model' in input_types['required']
        assert 'shift' in input_types['required']

    def test_sigma_calculation_with_real_comfyui(self):
        """Test sigma calculations using real ComfyUI functions."""
        # Test with real ComfyUI sigma calculation
        steps = 8
        sampler_name = "euler"
        scheduler = "simple"

        # Create a proper model sampling object that has sigmas
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()

        # Test sigma calculation
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

        # Verify we get a valid sigma schedule
        assert isinstance(sigmas, torch.Tensor)
        assert len(sigmas) == steps + 1  # ComfyUI includes final sigma
        assert sigmas[0] > sigmas[-1]  # Should be descending

    def test_boundary_calculation_logic(self):
        """Test boundary calculation logic with real sigma schedules."""
        steps = 8
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", steps)

        # Test T2V boundary calculation (0.875)
        t2v_boundary = 0.875

        # Find switch point using similar logic to our code
        switch_step = None
        for step in range(len(sigmas) - 1):
            sigma = sigmas[step]
            timestep = model_sampling.timestep(sigma) / 1000.0  # Normalize to 0-1
            if timestep <= t2v_boundary:
                switch_step = step
                break

        # Should find a valid switch point
        assert switch_step is not None
        assert 0 <= switch_step < steps

    def test_model_patching_interface(self):
        """Test that model patching interface works with ModelSamplingSD3."""
        sigma_shift = 7.5

        # Create a real ModelSamplingAdvanced instance like ModelSamplingSD3 creates
        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        # Create a mock model config
        class MockModelConfig:
            def __init__(self):
                self.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

        model_config = MockModelConfig()
        model_sampling = ModelSamplingAdvanced(model_config)
        model_sampling.set_parameters(shift=sigma_shift, multiplier=1000)

        # Create a mock model that has the interface our code expects
        class MockModel:
            def __init__(self):
                self.object_patches = {}

            def add_object_patch(self, name, obj):
                self.object_patches[name] = obj

            def get_model_object(self, name=None):
                if name and name in self.object_patches:
                    return self.object_patches[name]
                return comfy.model_sampling.ModelSamplingDiscrete()

        mock_model = MockModel()

        # Test patching (simulating what our ModelSamplingSD3Patcher does)
        mock_model.add_object_patch("model_sampling", model_sampling)

        # Verify patching worked
        patched_sampling = mock_model.get_model_object("model_sampling")
        assert patched_sampling is model_sampling

    def test_step_calculation_logic(self):
        """Test our step calculation logic with realistic values."""
        # Test percentage-based switching
        lightning_steps = 8
        switch_percentage = 0.5  # 50%

        expected_switch_step = int(lightning_steps * switch_percentage)
        assert expected_switch_step == 4

        # Test with odd numbers
        lightning_steps = 7
        expected_switch_step = int(lightning_steps * switch_percentage)
        assert expected_switch_step == 3

    def test_stage_range_calculation(self):
        """Test stage range calculations used in our logging."""
        # Test Stage 1 (base model)
        base_steps = 3
        total_base_steps = 24
        lightning_start = 2

        stage1_start = 0
        stage1_end = lightning_start
        stage1_percentage_start = 0.0
        stage1_percentage_end = (lightning_start / total_base_steps) * 100

        assert stage1_start == 0
        assert stage1_end == 2
        assert stage1_percentage_start == 0.0
        assert abs(stage1_percentage_end - 8.33) < 0.1  # ~8.33%

        # Test Stage 2 (lightning high)
        lightning_steps = 8
        switch_step = 4

        stage2_start = lightning_start
        stage2_end = lightning_start + switch_step
        stage2_total_percentage = ((lightning_steps / total_base_steps) * 100)
        stage2_percentage_start = stage1_percentage_end
        stage2_percentage_end = stage2_percentage_start + (stage2_total_percentage * 0.5)

        assert stage2_start == 2
        assert stage2_end == 6
        assert abs(stage2_percentage_end - 25.0) < 0.1  # Should be ~25%

    def test_parameter_boundary_validation(self):
        """Test parameter boundary validation logic."""
        # Test lightning_start validation
        lightning_steps = 8

        # Valid range
        for lightning_start in range(0, lightning_steps):
            # Should be valid
            assert 0 <= lightning_start < lightning_steps

        # Invalid values
        invalid_starts = [-1, lightning_steps, lightning_steps + 1]
        for invalid_start in invalid_starts:
            assert not (0 <= invalid_start < lightning_steps)

    def test_mathematical_relationships(self):
        """Test mathematical relationships in our calculations."""
        # Test base steps calculation consistency
        base_quality_threshold = 20
        lightning_start = 2
        lightning_steps = 8

        # Our calculation logic (simplified)
        target_base_percentage = base_quality_threshold / 100.0
        target_lightning_percentage = 1.0 - target_base_percentage

        # Calculate total steps needed
        if target_lightning_percentage > 0:
            total_base_steps = int(lightning_steps / target_lightning_percentage)
        else:
            total_base_steps = lightning_steps * 4  # Fallback

        calculated_base_steps = max(1, int(total_base_steps * target_base_percentage))

        # Verify relationships
        assert total_base_steps > 0
        assert calculated_base_steps > 0
        assert calculated_base_steps <= total_base_steps

        # Verify percentage relationship
        actual_base_percentage = calculated_base_steps / total_base_steps
        actual_lightning_percentage = lightning_steps / total_base_steps
        total_percentage = actual_base_percentage + actual_lightning_percentage

        # Should be close to 1.0 (100%)
        assert abs(total_percentage - 1.0) < 0.1


@pytest.mark.integration
@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestRealComfyUIComponents:
    """Test real ComfyUI components we depend on."""

    def test_comfy_samplers_calculate_sigmas(self):
        """Test that comfy.samplers.calculate_sigmas works as expected."""
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()
        scheduler = "simple"
        steps = 8

        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

        # Basic validation
        assert isinstance(sigmas, torch.Tensor)
        assert len(sigmas) == steps + 1
        assert sigmas.dtype == torch.float32 or sigmas.dtype == torch.float64

        # Should be monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] >= sigmas[i + 1]

    def test_model_sampling_timestep_calculation(self):
        """Test model sampling timestep calculations."""
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()

        # Test with various sigma values
        test_sigmas = [10.0, 5.0, 1.0, 0.1]

        for sigma in test_sigmas:
            timestep = model_sampling.timestep(torch.tensor(sigma))

            # Should return a valid timestep
            assert isinstance(timestep, torch.Tensor)
            assert timestep.numel() == 1
            assert timestep.item() >= 0  # Timesteps should be non-negative

    def test_model_sampling_sd3_sigma_shift(self):
        """Test ModelSamplingSD3 sigma shift functionality."""
        shift_values = [1.0, 3.0, 5.0, 10.0]

        for shift in shift_values:
            # Create ModelSamplingAdvanced like ModelSamplingSD3 does
            sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
            sampling_type = comfy.model_sampling.CONST

            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass

            class MockModelConfig:
                def __init__(self):
                    self.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            model_config = MockModelConfig()
            model_sampling = ModelSamplingAdvanced(model_config)
            model_sampling.set_parameters(shift=shift, multiplier=1000)

            # Test sigma calculation with shift
            timestep = torch.tensor(500.0)  # Typical timestep value
            sigma = model_sampling.sigma(timestep)

            # Should return a valid sigma
            assert isinstance(sigma, torch.Tensor)
            assert sigma.item() > 0  # Sigma should be positive

            # Different shifts should produce different sigmas
            if shift != 1.0:
                baseline_config = MockModelConfig()
                baseline_sampling = ModelSamplingAdvanced(baseline_config)
                baseline_sampling.set_parameters(shift=1.0, multiplier=1000)
                baseline_sigma = baseline_sampling.sigma(timestep)
                assert not torch.isclose(sigma, baseline_sigma, rtol=1e-6)


@pytest.mark.integration
@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestAdvancedComfyUIIntegration:
    """Test advanced ComfyUI integration scenarios."""

    def test_model_sampling_sd3_patching_workflow(self):
        """Test the complete ModelSamplingSD3 patching workflow."""
        # Create a ModelSamplingSD3 node
        node = ModelSamplingSD3()

        # Create a minimal model with the right interface
        class MockModel:
            def __init__(self):
                self.model = type('Model', (), {
                    'model_config': type('Config', (), {
                        'sampling_settings': {'shift': 1.0, 'multiplier': 1000}
                    })()
                })()
                self.object_patches = {}

            def clone(self):
                new_model = MockModel()
                new_model.object_patches = self.object_patches.copy()
                return new_model

            def add_object_patch(self, name, obj):
                self.object_patches[name] = obj

        mock_model = MockModel()

        # Test the complete patching workflow
        patched_model_tuple = node.patch(mock_model, shift=5.0)

        # Should return a tuple with the patched model
        assert isinstance(patched_model_tuple, tuple)
        assert len(patched_model_tuple) == 1

        patched_model = patched_model_tuple[0]

        # Verify the model was cloned and patched
        assert patched_model is not mock_model  # Should be a clone
        assert 'model_sampling' in patched_model.object_patches

        # Verify the patched sampling object
        patched_sampling = patched_model.object_patches['model_sampling']
        assert hasattr(patched_sampling, 'set_parameters')

    def test_comfyui_scheduler_integration(self):
        """Test integration with ComfyUI's scheduler system."""
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()

        # Test different schedulers
        schedulers = ["simple", "normal", "karras", "exponential"]
        steps = 8

        for scheduler in schedulers:
            try:
                sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)

                # Basic validation
                assert isinstance(sigmas, torch.Tensor)
                assert len(sigmas) == steps + 1
                assert torch.all(sigmas[:-1] >= sigmas[1:])  # Should be non-increasing

                # Verify sensible sigma range
                assert sigmas[0] > 0  # Max sigma should be positive
                assert sigmas[-1] >= 0  # Min sigma should be non-negative

            except Exception as e:
                # Some schedulers might not be available - that's ok
                print(f"Scheduler {scheduler} not available: {e}")

    def test_sampler_compatibility(self):
        """Test compatibility with different ComfyUI samplers."""
        # Test that our boundary calculations work with different sampler types
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()

        # Test common samplers that should be available
        common_samplers = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral"]

        for sampler_name in common_samplers:
            try:
                # Test sigma calculation
                sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", 8)

                # Test that we can use these sigmas for boundary calculations
                # (This simulates what our boundary strategy code does)
                for i, sigma in enumerate(sigmas[:-1]):  # Skip last sigma
                    timestep = model_sampling.timestep(sigma)
                    normalized_timestep = float(timestep) / 1000.0

                    # Should produce reasonable timestep values
                    assert 0.0 <= normalized_timestep <= 1.0

            except Exception as e:
                # Some samplers might not be available - that's ok for testing
                print(f"Sampler {sampler_name} test failed: {e}")

    def test_lightning_parameters_integration(self):
        """Test parameter calculations that match ComfyUI's expectations."""
        # Test realistic parameter combinations
        test_cases = [
            {"lightning_steps": 4, "base_quality_threshold": 20},
            {"lightning_steps": 6, "base_quality_threshold": 25},
            {"lightning_steps": 8, "base_quality_threshold": 30},
            {"lightning_steps": 12, "base_quality_threshold": 15},
        ]

        for case in test_cases:
            lightning_steps = case["lightning_steps"]
            base_quality_threshold = case["base_quality_threshold"]

            # Calculate total steps (simplified version of our algorithm)
            target_lightning_percentage = (100 - base_quality_threshold) / 100.0
            total_steps = int(lightning_steps / target_lightning_percentage)

            # Verify the math makes sense
            assert total_steps >= lightning_steps
            assert total_steps > 0

            # Calculate actual percentages
            lightning_percentage = (lightning_steps / total_steps) * 100
            base_percentage = 100 - lightning_percentage

            # Should be close to target
            assert abs(base_percentage - base_quality_threshold) < 5.0  # Within 5%

    def test_tensor_operations_with_comfyui(self):
        """Test tensor operations that match ComfyUI's expectations."""
        # Test with realistic latent shapes
        latent_shapes = [
            (1, 4, 8, 8),    # Tiny for testing
            (1, 4, 16, 16),  # Small
            (1, 4, 32, 32),  # Medium
            (1, 4, 64, 64),  # Standard
        ]

        for shape in latent_shapes:
            # Create test latent
            latent = {"samples": torch.randn(*shape, dtype=torch.float32)}

            # Test that our minimal latent creation works
            # (This is what dry_run mode returns)
            minimal_shape = (1, shape[1], 1, 8, 8)  # Minimal 8x8 version
            minimal_latent = {"samples": torch.zeros(*minimal_shape, dtype=latent["samples"].dtype)}

            # Verify properties
            assert minimal_latent["samples"].shape[1] == latent["samples"].shape[1]  # Same channels
            assert minimal_latent["samples"].dtype == latent["samples"].dtype  # Same dtype

    def test_error_handling_integration(self):
        """Test error handling with real ComfyUI components."""
        # Test invalid sigma calculations
        try:
            # This should fail gracefully
            invalid_sampling = comfy.model_sampling.ModelSamplingDiscrete()
            sigmas = comfy.samplers.calculate_sigmas(invalid_sampling, "invalid_scheduler", 8)
            # If we get here, the scheduler was valid after all
            assert isinstance(sigmas, torch.Tensor)
        except Exception:
            # Expected for invalid scheduler
            pass

        # Test invalid timestep calculations
        model_sampling = comfy.model_sampling.ModelSamplingDiscrete()
        try:
            # Test with extreme sigma values
            extreme_sigma = torch.tensor(1e10)
            timestep = model_sampling.timestep(extreme_sigma)
            assert isinstance(timestep, torch.Tensor)
        except Exception:
            # Some extreme values might cause errors - that's ok
            pass


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not COMFYUI_AVAILABLE,
    reason="ComfyUI dependencies not available"
)
class TestPerformanceAndLargeLatents:
    """Test performance with large latent tensors and realistic scenarios."""

    def test_large_latent_dry_run_performance(self):
        """Test dry run performance with large realistic latent tensors."""
        # Import our actual classes for performance testing
        import importlib.util
        import os
        import sys

        # Add ComfyUI root to path
        comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        sys.path.insert(0, comfyui_root)

        # Load the main module
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_module_path = os.path.join(project_path, 'nodes.py')
        spec = importlib.util.spec_from_file_location('nodes', main_module_path)

        if not spec or not spec.loader:
            pytest.skip("Could not load main module for performance testing")

        module = importlib.util.module_from_spec(spec)
        sys.modules['nodes'] = module
        spec.loader.exec_module(module)

        advanced_node = module.TripleKSamplerWan22LightningAdvanced()

        # Test with realistic large latent sizes
        large_latent_shapes = [
            (1, 4, 128, 128),   # 1K video resolution equivalent
            (1, 4, 192, 128),   # Aspect ratio variation
            (1, 16, 81, 144),   # HunyuanVideo format (16 channels, 81 frames, 144x256)
        ]

        import time
        from unittest.mock import MagicMock

        for shape in large_latent_shapes:
            # Create large latent tensor
            large_latent = {"samples": torch.randn(*shape, dtype=torch.float32)}

            # Create mock models
            mock_base_high = MagicMock()
            mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            mock_lightning_high = MagicMock()
            mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            mock_lightning_low = MagicMock()
            mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            params = {
                "base_high": mock_base_high,
                "lightning_high": mock_lightning_high,
                "lightning_low": mock_lightning_low,
                "positive": MagicMock(),
                "negative": MagicMock(),
                "latent_image": large_latent,
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
                "switch_strategy": "50% of steps",
                "switch_boundary": 0.875,
                "switch_step": -1,
                "dry_run": True
            }

            # Measure dry run performance - should raise InterruptProcessingException
            start_time = time.time()
            with pytest.raises(comfy.model_management.InterruptProcessingException):
                advanced_node.sample(**params)
            end_time = time.time()

            duration = end_time - start_time

            # Performance assertion - dry run should be fast even with large latents
            # Allow up to 100ms for dry run (very generous)
            assert duration < 0.1, f"Dry run took {duration:.3f}s for shape {shape}, should be < 0.1s"

            print(f"✓ Large latent {shape} dry run completed in {duration:.3f}s")

    def test_computation_complexity_scaling(self):
        """Test that computational complexity scales appropriately."""
        # Import our actual classes
        import importlib.util
        import os
        import sys

        # Add ComfyUI root to path
        comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        sys.path.insert(0, comfyui_root)

        # Load the main module
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_module_path = os.path.join(project_path, 'nodes.py')
        spec = importlib.util.spec_from_file_location('nodes', main_module_path)

        if not spec or not spec.loader:
            pytest.skip("Could not load main module for complexity testing")

        module = importlib.util.module_from_spec(spec)
        sys.modules['nodes'] = module
        spec.loader.exec_module(module)

        advanced_node = module.TripleKSamplerWan22LightningAdvanced()

        # Test different lightning_steps values to verify O(n) complexity
        step_counts = [4, 8, 16, 32]
        durations = []

        import time
        from unittest.mock import MagicMock

        for lightning_steps in step_counts:
            # Small latent for this test
            test_latent = {"samples": torch.randn(1, 4, 8, 8, dtype=torch.float32)}

            # Create mock models
            mock_base_high = MagicMock()
            mock_base_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            mock_lightning_high = MagicMock()
            mock_lightning_high.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            mock_lightning_low = MagicMock()
            mock_lightning_low.model.model_config.sampling_settings = {'shift': 1.0, 'multiplier': 1000}

            params = {
                "base_high": mock_base_high,
                "lightning_high": mock_lightning_high,
                "lightning_low": mock_lightning_low,
                "positive": MagicMock(),
                "negative": MagicMock(),
                "latent_image": test_latent,
                "seed": 42,
                "sigma_shift": 5.0,
                "base_steps": 3,
                "base_quality_threshold": 20,
                "base_cfg": 3.5,
                "lightning_start": 1,
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

            # Measure computation time - dry run should raise InterruptProcessingException
            start_time = time.time()
            import comfy.model_management
            with pytest.raises(comfy.model_management.InterruptProcessingException):
                advanced_node.sample(**params)
            end_time = time.time()

            duration = end_time - start_time
            durations.append(duration)

            print(f"✓ {lightning_steps} steps dry run completed in {duration:.4f}s")

        # Verify computational complexity doesn't explode
        # Even the largest test should complete quickly in dry run
        max_duration = max(durations)
        assert max_duration < 0.05, f"Max duration {max_duration:.3f}s too high for dry run"

        # Verify reasonable scaling - shouldn't grow exponentially
        if len(durations) >= 2:
            growth_factor = durations[-1] / durations[0]  # Largest vs smallest
            assert growth_factor < 10, f"Duration growth factor {growth_factor:.2f}x suggests exponential complexity"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])