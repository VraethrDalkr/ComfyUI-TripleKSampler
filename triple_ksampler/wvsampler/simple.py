"""Simplified node with smart defaults and common strategies.

This module contains TripleWVSampler, which provides a streamlined interface with:
- 5 strategies (50%, T2V/I2V boundary + refined variants)
- Auto-calculated defaults for base_steps
- Fixed lightning_cfg=1.0 for Stages 2&3
- Simplified parameter set for ease of use

Recommended for users who want WanVideo triple-stage sampling without advanced configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

# Import shared modules for config
from triple_ksampler.shared import config as core_config

# Import advanced class to inherit from
from .advanced import TripleWVSamplerAdvanced

# Set up logger
logger = logging.getLogger("triple_ksampler.wvsampler.simple")

# Load configuration
_CONFIG = core_config.load_config(config_dir=Path(__file__).resolve().parent.parent.parent)
_DEFAULT_BASE_QUALITY_THRESHOLD = core_config.get_base_quality_threshold(_CONFIG)


class TripleWVSampler(TripleWVSamplerAdvanced):
    """Simplified triple-stage sampler for WanVideo models with smart defaults.

    Inherits from TripleWVSamplerAdvanced but exposes only essential parameters
    with automatic configuration for ease of use. Delegates to parent implementation
    with hardcoded values for advanced parameters.

    Features:
        - Only 3 strategies (50%, T2V boundary, I2V boundary)
        - Single cfg parameter (for Stage 1), Stages 2&3 use fixed 1.0
        - Auto-calculated base_steps (no manual control)
        - Simplified interface for common use cases
    """

    DESCRIPTION = (
        "Triple-stage sampler for WanVideo models with Lightning LoRA. "
        "Simplified interface with auto-calculated parameters."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """Define simplified input types with only essential parameters.

        Uses NODE_CLASS_MAPPINGS runtime lookup to avoid import order issues.
        Provides fallback if WanVideoSampler not yet registered.

        Returns:
            Dict mapping input names to their type specifications
        """
        # Get WanVideoSampler's INPUT_TYPES from NODE_CLASS_MAPPINGS (load order independent)
        from nodes import NODE_CLASS_MAPPINGS

        if "WanVideoSampler" in NODE_CLASS_MAPPINGS:
            # WanVideoWrapper loaded - get actual INPUT_TYPES structure
            original_inputs = NODE_CLASS_MAPPINGS["WanVideoSampler"].INPUT_TYPES()
            # Extract scheduler list from original inputs
            scheduler_spec = original_inputs.get("required", {}).get("scheduler", [])
            scheduler_list = (
                scheduler_spec[0] if isinstance(scheduler_spec, tuple) else scheduler_spec
            )
        else:
            # Fallback: WanVideoWrapper not loaded yet (or broken)
            # Prevents startup crashes when WanVideoWrapper directory exists but registration failed
            # Nodes built with fallback are non-functional (raise RuntimeError at execution)
            # but prevent ComfyUI startup crash during load order race conditions
            scheduler_list = [
                "unipc",
                "unipc/beta",
                "dpm++",
                "dpm++/beta",
                "dpm++_sde",
                "dpm++_sde/beta",
                "euler",
                "euler/beta",
                "longcat_distill_euler",
                "deis",
                "lcm",
                "lcm/beta",
                "res_multistep",
                "flowmatch_causvid",
                "flowmatch_distill",
                "flowmatch_pusa",
                "multitalk",
                "sa_ode_stable",
                "rcm",
            ]
            original_inputs = {
                "required": {
                    "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                    "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "scheduler": (scheduler_list, {"default": "unipc"}),
                    "force_offload": ("BOOLEAN", {"default": True}),
                    "riflex_freq_index": ("INT", {"default": 0}),
                    "batched_cfg": ("BOOLEAN", {"default": False}),
                    "rope_function": (["default", "comfy", "comfy_chunked"],),
                },
                "optional": {},
            }

        # Build simplified required section with exact parameter order
        required = {}

        # 1. Triple-stage model inputs
        required["base_high"] = ("WANVIDEOMODEL", {"tooltip": "Base high-noise model for Stage 1."})
        required["lightning_high"] = (
            "WANVIDEOMODEL",
            {"tooltip": "Lightning high-noise model for Stage 2."},
        )
        required["lightning_low"] = (
            "WANVIDEOMODEL",
            {"tooltip": "Lightning low-noise model for Stage 3."},
        )

        # 2. image_embeds (from WanVideo)
        if "image_embeds" in original_inputs["required"]:
            required["image_embeds"] = original_inputs["required"]["image_embeds"]

        # 3. seed (from WanVideo) - add control_after_generate and tooltip for consistency with TripleKSampler
        if "seed" in original_inputs["required"]:
            seed_spec = original_inputs["required"]["seed"]
            if isinstance(seed_spec, tuple) and len(seed_spec) == 2:
                seed_spec = (
                    seed_spec[0],
                    {
                        **seed_spec[1],
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                )
            required["seed"] = seed_spec

        # 4. sigma_shift (renamed from "shift") - add tooltip for consistency
        if "shift" in original_inputs["required"]:
            shift_spec = original_inputs["required"]["shift"]
            if isinstance(shift_spec, tuple) and len(shift_spec) == 2:
                shift_spec = (
                    shift_spec[0],
                    {
                        **shift_spec[1],
                        "tooltip": "Sigma adjustment applied to all models for WanVideo sampling.",
                    },
                )
            required["sigma_shift"] = shift_spec
        else:
            # Match TripleKSampler specs: max=100.0
            required["sigma_shift"] = (
                "FLOAT",
                {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Sigma adjustment applied to all models for WanVideo sampling.",
                },
            )

        # 5. base_cfg
        # Match TripleKSampler specs: default=3.5, max=100.0, step=0.1
        required["base_cfg"] = (
            "FLOAT",
            {
                "default": 3.5,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "tooltip": "CFG scale for Stage 1 (Stages 2&3 use fixed 1.0).",
            },
        )

        # 6. lightning_start
        required["lightning_start"] = (
            "INT",
            {
                "default": 1,
                "min": 0,
                "max": 99,
                "tooltip": "Starting step within lightning schedule. Set to 0 to skip Stage 1 entirely.",
            },
        )

        # 7. lightning_steps
        required["lightning_steps"] = (
            "INT",
            {
                "default": 8,
                "min": 2,
                "max": 100,
                "tooltip": "Total steps for lightning stages.",
            },
        )

        # 8. scheduler (single scheduler for all stages)
        if "scheduler" in original_inputs["required"]:
            required["scheduler"] = original_inputs["required"]["scheduler"]
        else:
            required["scheduler"] = (
                scheduler_list if scheduler_list else ["unipc", "euler", "dpm++"],
                {"default": "unipc", "tooltip": "Scheduler for all stages."},
            )

        # 9. switch_strategy (5 strategies: 3 basic + 2 refined)
        required["switch_strategy"] = (
            [
                "50% of steps",
                "T2V boundary",
                "I2V boundary",
                "T2V boundary (refined)",
                "I2V boundary (refined)",
            ],
            {
                "default": "50% of steps",
                "tooltip": "Strategy for switching between models. Refined variants auto-tune sigma_shift for perfect boundary alignment at the switch step.",
            },
        )

        # 10. force_offload (from WanVideo, applies to all stages)
        if "force_offload" in original_inputs["required"]:
            required["force_offload"] = original_inputs["required"]["force_offload"]
        else:
            required["force_offload"] = ("BOOLEAN", {"default": True})

        # 11. riflex_freq_index (from WanVideo)
        if "riflex_freq_index" in original_inputs["required"]:
            required["riflex_freq_index"] = original_inputs["required"]["riflex_freq_index"]

        # 12. batched_cfg and rope_function (from WanVideo required/optional)
        # These params are always visible according to user requirements
        if "batched_cfg" in original_inputs["required"]:
            required["batched_cfg"] = original_inputs["required"]["batched_cfg"]
        elif "batched_cfg" in original_inputs.get("optional", {}):
            required["batched_cfg"] = original_inputs["optional"]["batched_cfg"]

        if "rope_function" in original_inputs["required"]:
            required["rope_function"] = original_inputs["required"]["rope_function"]
        elif "rope_function" in original_inputs.get("optional", {}):
            required["rope_function"] = original_inputs["optional"]["rope_function"]

        # Preserve ALL remaining optional parameters from WanVideo
        # Skip internal parameters used for three-stage orchestration
        _INTERNAL_PARAMS = {
            "samples",
            "denoise_strength",
            "start_step",
            "end_step",
            "add_noise_to_samples",
            "batched_cfg",  # Already handled above
            "rope_function",  # Already handled above
        }
        optional = {}
        for param_name, param_spec in original_inputs.get("optional", {}).items():
            if param_name not in _INTERNAL_PARAMS:
                optional[param_name] = param_spec

        return {"required": required, "optional": optional}

    def sample(
        self,
        # Triple-stage models
        base_high: Any,
        lightning_high: Any,
        lightning_low: Any,
        # WanVideoSampler required parameters
        image_embeds: dict[str, Any],
        seed: int,
        sigma_shift: float,
        base_cfg: float,
        lightning_start: int,
        lightning_steps: int,
        scheduler: str,
        switch_strategy: str,
        force_offload: bool,
        riflex_freq_index: int,
        batched_cfg: bool,
        rope_function: str,
        # All other WanVideo optional parameters (dynamically inherited)
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute triple-stage WanVideo sampling with simplified defaults.

        Delegates to parent TripleWVSamplerAdvancedAlt.sample() with hardcoded values:
        - base_quality_threshold: config default
        - base_steps: -1 (auto-calculated)
        - base_scheduler: user-provided scheduler (for Stage 1)
        - lightning_cfg: 1.0 (hardcoded for Stages 2 & 3)
        - lightning_scheduler: user-provided scheduler (same for Stages 2 & 3)
        - switch_step: -1 (auto at 50% for Manual switch step)
        - switch_boundary: 0.875 (default for Manual boundary)
        - dry_run: False (no dry run in simple node)

        Args:
            base_high: Base high-noise model for Stage 1
            lightning_high: Lightning high-noise model for Stage 2
            lightning_low: Lightning low-noise model for Stage 3
            image_embeds: Image embeddings for conditioning
            seed: Random seed
            sigma_shift: Sigma shift applied to all models
            base_cfg: CFG scale for Stage 1 (Stages 2&3 use 1.0)
            lightning_start: Starting step within lightning schedule
            lightning_steps: Total steps for lightning stages
            scheduler: Scheduler type (applied to all stages for simplicity)
            switch_strategy: Strategy for switching (3 options: 50%, T2V, I2V)
            force_offload: Moves models to offload device after each stage
            riflex_freq_index: RIFLEX frequency index
            batched_cfg: Batch cond and uncond for faster sampling
            rope_function: RoPE function implementation to use
            **kwargs: All other WanVideo optional parameters

        Returns:
            Tuple of (samples_latent_dict, denoised_samples_latent_dict)
        """
        # Delegate to parent (AdvancedAlt) with hardcoded advanced parameters
        return super().sample(
            base_high=base_high,
            lightning_high=lightning_high,
            lightning_low=lightning_low,
            image_embeds=image_embeds,
            seed=seed,
            sigma_shift=sigma_shift,
            base_quality_threshold=_DEFAULT_BASE_QUALITY_THRESHOLD,  # Auto from config
            base_steps=-1,  # Auto-calculated
            base_cfg=base_cfg,  # User-provided for Stage 1
            base_scheduler=scheduler,  # User-provided scheduler for Stage 1
            lightning_start=lightning_start,
            lightning_steps=lightning_steps,
            lightning_cfg=1.0,  # Hardcoded for Stages 2 & 3
            lightning_scheduler=scheduler,  # Same scheduler for Stages 2 & 3
            switch_strategy=switch_strategy,
            switch_step=-1,  # Auto at 50% for Manual switch step
            switch_boundary=0.875,  # Default for Manual boundary
            dry_run=False,  # No dry run in simple node
            force_offload=force_offload,
            riflex_freq_index=riflex_freq_index,
            batched_cfg=batched_cfg,
            rope_function=rope_function,
            **kwargs,
        )
