"""Test INPUT_TYPES() fallback behavior when WanVideoWrapper registration fails.

This test simulates issue #11 where:
- WanVideoWrapper directory exists (filesystem check passes)
- But WanVideoSampler not in NODE_CLASS_MAPPINGS (registration failed or load order issue)
- INPUT_TYPES() should use fallback instead of crashing
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestInputTypesFallback:
    """Test INPUT_TYPES() fallback when NODE_CLASS_MAPPINGS missing WanVideoSampler."""

    def test_simple_input_types_fallback_no_crash(self):
        """Verify TripleWVSampler.INPUT_TYPES() doesn't crash when WanVideoSampler missing.

        Simulates issue #11: WanVideoWrapper directory exists but registration failed.
        INPUT_TYPES() should use fallback values instead of raising ImportError.
        """
        # Import after mocking to ensure we get the fallback path
        from triple_ksampler.wvsampler.simple import TripleWVSampler

        # Mock NODE_CLASS_MAPPINGS without WanVideoSampler (simulates load order issue)
        mock_mappings = {}  # Empty - WanVideoSampler not registered yet

        # Patch nodes.NODE_CLASS_MAPPINGS (imported inside INPUT_TYPES method)
        with patch("nodes.NODE_CLASS_MAPPINGS", mock_mappings):
            # This should NOT crash - should use fallback
            try:
                result = TripleWVSampler.INPUT_TYPES()
            except ImportError as e:
                pytest.fail(f"INPUT_TYPES() crashed with ImportError: {e}")
            except Exception as e:
                pytest.fail(f"INPUT_TYPES() crashed with unexpected error: {e}")

            # Verify result is valid INPUT_TYPES structure
            assert isinstance(result, dict)
            assert "required" in result
            assert "optional" in result

            # Verify fallback scheduler_list was used (should have all 19 schedulers)
            # Simple node uses single "scheduler" param, not "lightning_scheduler"
            assert "scheduler" in result["required"]
            scheduler_spec = result["required"]["scheduler"]
            assert isinstance(scheduler_spec, tuple)
            scheduler_list = scheduler_spec[0]

            # Verify we got the corrected fallback list (19 schedulers)
            assert len(scheduler_list) >= 19
            assert "unipc" in scheduler_list
            assert "dpm++" in scheduler_list
            assert "euler" in scheduler_list
            assert "multitalk" in scheduler_list

    def test_advanced_input_types_fallback_no_crash(self):
        """Verify TripleWVSamplerAdvanced.INPUT_TYPES() doesn't crash when WanVideoSampler missing."""
        from triple_ksampler.wvsampler.advanced import TripleWVSamplerAdvanced

        # Mock NODE_CLASS_MAPPINGS without WanVideoSampler
        mock_mappings = {}

        with patch("nodes.NODE_CLASS_MAPPINGS", mock_mappings):
            # This should NOT crash - should use fallback
            try:
                result = TripleWVSamplerAdvanced.INPUT_TYPES()
            except ImportError as e:
                pytest.fail(f"INPUT_TYPES() crashed with ImportError: {e}")
            except Exception as e:
                pytest.fail(f"INPUT_TYPES() crashed with unexpected error: {e}")

            # Verify result is valid
            assert isinstance(result, dict)
            assert "required" in result
            assert "optional" in result

    def test_fallback_uses_correct_image_embeds_type(self):
        """Verify fallback uses correct WANVIDIMAGE_EMBEDS type (not IMAGE_EMBEDS)."""
        from triple_ksampler.wvsampler.simple import TripleWVSampler

        mock_mappings = {}

        with patch("nodes.NODE_CLASS_MAPPINGS", mock_mappings):
            result = TripleWVSampler.INPUT_TYPES()

            # Check image_embeds has correct type name
            assert "image_embeds" in result["required"]
            image_embeds_spec = result["required"]["image_embeds"]
            assert isinstance(image_embeds_spec, tuple)
            assert image_embeds_spec[0] == "WANVIDIMAGE_EMBEDS"

    def test_fallback_uses_correct_batched_cfg_default(self):
        """Verify fallback uses correct batched_cfg default (False, not True)."""
        from triple_ksampler.wvsampler.simple import TripleWVSampler

        mock_mappings = {}

        with patch("nodes.NODE_CLASS_MAPPINGS", mock_mappings):
            result = TripleWVSampler.INPUT_TYPES()

            # batched_cfg should be in optional or required
            batched_cfg_spec = result.get("optional", {}).get(
                "batched_cfg", result.get("required", {}).get("batched_cfg")
            )
            assert batched_cfg_spec is not None

            # Check default is False
            if isinstance(batched_cfg_spec, tuple) and len(batched_cfg_spec) == 2:
                assert batched_cfg_spec[1].get("default") is False

    def test_fallback_uses_correct_rope_function_list(self):
        """Verify fallback uses correct rope_function list (all 3 options)."""
        from triple_ksampler.wvsampler.simple import TripleWVSampler

        mock_mappings = {}

        with patch("nodes.NODE_CLASS_MAPPINGS", mock_mappings):
            result = TripleWVSampler.INPUT_TYPES()

            # rope_function should be in optional or required
            rope_function_spec = result.get("optional", {}).get(
                "rope_function", result.get("required", {}).get("rope_function")
            )
            assert rope_function_spec is not None

            # Check it has all 3 options
            if isinstance(rope_function_spec, tuple):
                rope_list = rope_function_spec[0]
                assert "default" in rope_list
                assert "comfy" in rope_list
                assert "comfy_chunked" in rope_list

    def test_real_wanvideo_sampler_available_no_fallback(self):
        """Verify INPUT_TYPES() uses real WanVideoSampler when available (not fallback)."""
        from triple_ksampler.wvsampler.simple import TripleWVSampler

        # Mock NODE_CLASS_MAPPINGS WITH WanVideoSampler
        mock_wanvideo = MagicMock()
        mock_wanvideo.INPUT_TYPES.return_value = {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "shift": ("FLOAT", {"default": 5.0}),
                "scheduler": (
                    ["real_scheduler_1", "real_scheduler_2"],
                    {"default": "real_scheduler_1"},
                ),
                "force_offload": ("BOOLEAN", {"default": True}),
                "riflex_freq_index": ("INT", {"default": 0}),
            },
            "optional": {
                "batched_cfg": ("BOOLEAN", {"default": False}),
                "rope_function": (["default", "comfy", "comfy_chunked"], {"default": "comfy"}),
            },
        }

        mock_mappings = {"WanVideoSampler": mock_wanvideo}

        with patch("nodes.NODE_CLASS_MAPPINGS", mock_mappings):
            result = TripleWVSampler.INPUT_TYPES()

            # Verify it used REAL scheduler list, not fallback
            # Simple node uses "scheduler", not "lightning_scheduler"
            scheduler_spec = result["required"]["scheduler"]
            scheduler_list = scheduler_spec[0]

            # Should have real schedulers, not fallback schedulers
            assert "real_scheduler_1" in scheduler_list
            assert "real_scheduler_2" in scheduler_list
            # Should NOT have fallback-only schedulers like "multitalk"
            # (unless the real one also has them, which it might)
