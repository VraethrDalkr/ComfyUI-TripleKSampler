"""Advanced node variants for WanVideo triple-stage sampling.

This module contains:
- TripleWVSamplerAdvancedAlt: Full implementation with static UI (all parameters visible)
- TripleWVSamplerAdvanced: Dynamic UI variant (JavaScript-controlled parameter visibility)

Both nodes expose all 8 switching strategies and provide fine-grained control over
stage boundaries, sigma shifts, and Lightning model configuration for WanVideo models.

Wraps WanVideoSampler.process() for triple-stage orchestration while reusing
TripleKSamplerBase's perfect alignment algorithm and strategy calculation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

# Import base class from package root
from triple_ksampler.base import TripleKSamplerBase

# Import shared logic modules
from triple_ksampler.shared import alignment as core_alignment
from triple_ksampler.shared import config as core_config
from triple_ksampler.shared import models as core_models
from triple_ksampler.shared import notifications as core_notifications

# Import utility functions
from .utils import get_wanvideo_components

# Load configuration
_CONFIG = core_config.load_config(config_dir=Path(__file__).resolve().parent.parent.parent)
_DEFAULT_BASE_QUALITY_THRESHOLD = core_config.get_base_quality_threshold(_CONFIG)
_BOUNDARY_T2V = core_config.get_boundary_t2v(_CONFIG)
_BOUNDARY_I2V = core_config.get_boundary_i2v(_CONFIG)
_LOG_LEVEL = core_config.get_log_level(_CONFIG)

# Algorithm constants
STAGE3_SEED_OFFSET = 1  # Ensure Stage 3 uses different noise pattern

# Get loggers (inherit configuration from wvsampler_nodes.py entry point)
logger = logging.getLogger("triple_ksampler.wvsampler.advanced")
bare_logger = logging.getLogger("triple_ksampler.separator")

# Import ComfyUI components for dry_run and error handling
try:
    import comfy.model_management
    from server import PromptServer
except Exception:
    PromptServer = None  # Graceful fallback if PromptServer unavailable


class TripleWVSamplerAdvancedAlt(TripleKSamplerBase):
    """Advanced triple-stage sampler for WanVideo models with all parameters exposed (static UI).

    Orchestrates 3 sequential WanVideoSampler.process() calls with full manual control:
        - Stage 1: Base high-noise model (base denoising)
        - Stage 2: Lightning high-noise model (high-frequency refinement)
        - Stage 3: Lightning low-noise model (low-frequency refinement)

    Features:
        - All 5 switching strategies (50%, Manual step, T2V/I2V boundary, Manual boundary)
        - Separate CFG controls (base_cfg for Stage 1, lightning_cfg for Stages 2&3)
        - Manual base_steps control (-1 for auto-calculation)
        - Dry run mode for configuration validation
        - Full logging matching TripleKSampler format
    """

    # ComfyUI node metadata
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("samples", "denoised_samples")
    FUNCTION = "sample"
    CATEGORY = "TripleKSampler/wanvideo"
    DESCRIPTION = (
        "Advanced triple-stage sampler for WanVideo models with Lightning LoRA. "
        "Static UI variant with all parameters always visible."
    )

    def __init__(self):
        """Initialize TripleWVSamplerAdvancedAlt with WanVideoSampler instance."""
        # Lazy load the WanVideoSampler class and scheduler components
        WanVideoSamplerClass, _, _ = get_wanvideo_components()
        self.wanvideo_sampler = WanVideoSamplerClass()

    def _compute_wanvideo_boundary_switching_step(
        self,
        model: Any,
        scheduler: str,
        steps: int,
        shift: float,
        boundary: float,
    ) -> int:
        """Compute switch step for WanVideo using its native scheduler system.

        This overrides the base class method to use WanVideo's custom flow-matching
        schedulers instead of ComfyUI's standard diffusion samplers.

        IMPORTANT: WanVideo's schedulers store sigmas as normalized timesteps
        (timesteps/1000, range 0-1) rather than noise levels. This means we can
        use them directly as boundary comparison values without any conversion.

        WanVideo uses composite scheduler names where "/beta" suffix means
        "use beta_sigmas". For example:
        - "euler" → FlowMatchEulerDiscreteScheduler(use_beta_sigmas=False)
        - "euler/beta" → FlowMatchEulerDiscreteScheduler(use_beta_sigmas=True)
        - "unipc" → FlowUniPCMultistepScheduler with default sigmas
        - "unipc/beta" → FlowUniPCMultistepScheduler with beta_sigmas

        Args:
            model: WanVideo model (for getting transformer_dim)
            scheduler: WanVideo scheduler name (e.g., "euler", "euler/beta", "unipc")
            steps: Number of lightning steps
            shift: Sigma shift value
            boundary: Boundary value between 0 and 1 (normalized timestep)

        Returns:
            int: Step index where sigma crosses the boundary
        """
        import torch

        # Get WanVideo's scheduler function
        _, _, get_scheduler = get_wanvideo_components()

        if get_scheduler is None:
            logger.warning(
                "WanVideo get_scheduler not available. Falling back to 50% switch point."
            )
            return math.ceil(steps / 2)

        # Get transformer dimension from model (needed for some WanVideo schedulers)
        try:
            transformer_dim = (
                model.model.diffusion_model.dim if hasattr(model.model, "diffusion_model") else 5120
            )
        except Exception:
            transformer_dim = 5120  # Default for Wan 14B models (most common)

        # Use WanVideo's native scheduler to get sigmas
        device = torch.device("cpu")  # Use CPU for calculation
        try:
            sample_scheduler, timesteps, _, _ = get_scheduler(
                scheduler,
                steps,
                start_step=0,
                end_step=-1,
                shift=shift,
                device=device,
                transformer_dim=transformer_dim,
            )
            sigmas = sample_scheduler.sigmas
        except Exception as e:
            logger.warning(
                f"Failed to get WanVideo scheduler '{scheduler}': {e}. "
                f"Falling back to 50% switch point. Note: WanVideo uses composite names "
                f"like 'euler/beta' where '/beta' means use_beta_sigmas=True."
            )
            return math.ceil(steps / 2)

        # WanVideo's sigmas are already normalized timesteps (timesteps/1000, so 0-1 range)
        # No conversion needed - use them directly as timesteps
        timesteps_normalized = []
        for sigma in sigmas[:-1]:  # Skip last sigma (always 0.0)
            try:
                timestep = float(sigma.item())
            except Exception:
                timestep = float(sigma)
            timesteps_normalized.append(timestep)

        # Format all sigmas with 4 decimal places for debug output
        sigmas_formatted = [f"{t:.4f}" for t in timesteps_normalized]
        logger.debug(f"WanVideo sigmas: [{', '.join(sigmas_formatted)}]")

        # Find the first step where timestep drops below the boundary
        # WanVideo timesteps go from high (1.0) to low (0.0), same direction as diffusion
        switching_step = steps
        for i, timestep in enumerate(timesteps_normalized):
            if timestep < float(boundary):
                switching_step = i
                break

        # Ensure switching step is within valid range
        if switching_step >= steps:
            switching_step = steps - 1

        return int(switching_step)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """Define input types with explicit parameter order for AdvancedAlt (static UI).

        All parameters are in the "required" section for static UI.
        Only called if WanVideoWrapper is available (conditional registration in __init__.py).

        Returns:
            Dict mapping input names to their type specifications
        """
        # Get WanVideoSampler's INPUT_TYPES and scheduler list dynamically
        # No fallback needed - node only registered if WanVideo available
        WanVideoSamplerClass, scheduler_list, _ = get_wanvideo_components()
        original_inputs = WanVideoSamplerClass.INPUT_TYPES()

        # Build required section with exact parameter order
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

        # 2. image_embeds (required by WanVideo)
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

        # 5. Base parameters
        required["base_quality_threshold"] = (
            "INT",
            {
                "default": _DEFAULT_BASE_QUALITY_THRESHOLD,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": f"Minimum total steps for base_steps auto-calculation (config default: {_DEFAULT_BASE_QUALITY_THRESHOLD}). Only applies when base_steps=-1.",
            },
        )
        required["base_steps"] = (
            "INT",
            {
                "default": -1,
                "min": -1,
                "max": 100,
                "tooltip": "Stage 1 steps for base high-noise model. Use -1 for auto-calculation based on quality threshold.",
            },
        )
        # Match TripleKSampler specs: default=3.5, max=100.0, step=0.1
        required["base_cfg"] = (
            "FLOAT",
            {
                "default": 3.5,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "tooltip": "CFG scale for Stage 1 (base model).",
            },
        )
        required["base_scheduler"] = (
            scheduler_list if scheduler_list else ["unipc", "euler", "dpm++"],
            {"default": "unipc", "tooltip": "Scheduler for Stage 1 (base model)."},
        )

        # 6. Lightning parameters
        required["lightning_start"] = (
            "INT",
            {
                "default": 1,
                "min": 0,
                "max": 99,
                "tooltip": "Starting step within lightning schedule. Set to 0 to skip Stage 1 entirely.",
            },
        )
        required["lightning_steps"] = (
            "INT",
            {
                "default": 8,
                "min": 2,
                "max": 100,
                "tooltip": "Total steps for lightning stages.",
            },
        )
        required["lightning_cfg"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "tooltip": "CFG scale for Stage 2 and Stage 3 (lightning models).",
            },
        )
        required["lightning_scheduler"] = (
            scheduler_list if scheduler_list else ["unipc", "euler", "dpm++"],
            {
                "default": "unipc",
                "tooltip": "Scheduler for Stage 2 and Stage 3 (lightning models).",
            },
        )

        # 7. Switching parameters
        required["switch_strategy"] = (
            [
                "50% of steps",
                "Manual switch step",
                "T2V boundary",
                "I2V boundary",
                "Manual boundary",
                "T2V boundary (refined)",
                "I2V boundary (refined)",
                "Manual boundary (refined)",
            ],
            {
                "default": "50% of steps",
                "tooltip": "Strategy for switching between models. Refined variants auto-tune sigma_shift for perfect boundary alignment at the switch step.",
            },
        )
        required["switch_step"] = (
            "INT",
            {
                "default": -1,
                "min": -1,
                "max": 99,
                "tooltip": "Manual step to switch models. Only used when switch_strategy is 'Manual switch step'. Use -1 for auto-calculation at 50% of lightning steps.",
            },
        )
        required["switch_boundary"] = (
            "FLOAT",
            {
                "default": 0.875,
                "min": 0.0,
                "max": 1.0,
                "step": 0.001,
                "tooltip": "Sigma boundary for switching. Only used when switch_strategy is 'Manual boundary'.",
            },
        )

        # 8. force_offload (from WanVideo, applies to all stages)
        if "force_offload" in original_inputs["required"]:
            required["force_offload"] = original_inputs["required"]["force_offload"]
        else:
            required["force_offload"] = ("BOOLEAN", {"default": True})

        # 9. riflex_freq_index (from WanVideo)
        if "riflex_freq_index" in original_inputs["required"]:
            required["riflex_freq_index"] = original_inputs["required"]["riflex_freq_index"]

        # 10. batched_cfg and rope_function (from WanVideo required/optional)
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

        # Add dry_run at the end (appears at bottom after all WanVideo optional params)
        optional["dry_run"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": "Enable dry run mode to test stage calculations without actual sampling.",
            },
        )

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
        base_quality_threshold: int,
        base_steps: int,
        base_cfg: float,
        base_scheduler: str,
        lightning_start: int,
        lightning_steps: int,
        lightning_cfg: float,
        lightning_scheduler: str,
        # TripleKSampler switching parameters
        switch_strategy: str,
        switch_step: int,
        switch_boundary: float,
        dry_run: bool,
        # WanVideo parameters
        force_offload: bool,
        riflex_freq_index: int,
        batched_cfg: bool,
        rope_function: str,
        # All other WanVideo optional parameters
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute triple-stage WanVideo sampling with full manual control.

        Args:
            base_high: Base high-noise model for Stage 1
            lightning_high: Lightning high-noise model for Stage 2
            lightning_low: Lightning low-noise model for Stage 3
            image_embeds: Image embeddings for conditioning
            seed: Random seed
            sigma_shift: Sigma shift applied to all models
            base_quality_threshold: Minimum total steps for base_steps auto-calculation
            base_steps: Stage 1 steps (-1 for auto-calculation)
            base_cfg: CFG scale for Stage 1
            base_scheduler: Scheduler for Stage 1 (base model)
            lightning_start: Starting step within lightning schedule
            lightning_steps: Total steps for lightning stages
            lightning_cfg: CFG scale for Stages 2 and 3
            lightning_scheduler: Scheduler for Stages 2 and 3 (lightning models)
            switch_strategy: Strategy for switching between lightning high and low models
            switch_step: Manual switch step (for "Manual switch step" strategy)
            switch_boundary: Manual boundary (for "Manual boundary" strategy)
            dry_run: If True, skip sampling and return configuration info only
            force_offload: Moves models to offload device after sampling (all stages)
            riflex_freq_index: RIFLEX frequency index
            batched_cfg: Batch cond and uncond for faster sampling
            rope_function: RoPE function implementation to use
            **kwargs: All other WanVideo optional parameters

        Returns:
            Tuple of (samples_latent_dict, denoised_samples_latent_dict)
        """
        # === DEBUG Section: Input Parameters ===
        if str(_LOG_LEVEL).upper() == "DEBUG":
            bare_logger.info("")
        logger.debug("=== TripleWVSampler Node - Input Parameters ===")
        logger.debug("Sampling: seed=%d, sigma_shift=%.3f", seed, sigma_shift)
        logger.debug(
            "Base stage: base_quality_threshold=%d, base_steps=%d, base_cfg=%.1f",
            base_quality_threshold,
            base_steps,
            base_cfg,
        )
        logger.debug("Base scheduler: %s", base_scheduler)
        logger.debug(
            "Lightning: lightning_start=%d, lightning_steps=%d, lightning_cfg=%.1f",
            lightning_start,
            lightning_steps,
            lightning_cfg,
        )
        logger.debug("Lightning scheduler: %s", lightning_scheduler)
        logger.debug("Strategy: %s", switch_strategy)
        if switch_strategy == "Manual boundary":
            logger.debug("  switch_boundary=%.3f", switch_boundary)
        elif switch_strategy == "Manual switch step":
            logger.debug("  switch_step=%d", switch_step)
        logger.debug(
            "WanVideo: force_offload=%s, riflex_freq_index=%d", force_offload, riflex_freq_index
        )
        logger.debug("dry_run=%s", dry_run)

        # Early validation (before auto-calculation resolves base_steps)
        self._validate_basic_parameters(
            lightning_steps, lightning_start, switch_strategy, switch_step
        )

        bare_logger.info("")  # separator before calculations

        # === Step 1: Calculate or use manual base_steps ===
        if base_steps == -1:
            # Auto-calculate using perfect alignment
            base_steps_calc, total_base_steps, alignment_method = self._calculate_perfect_alignment(
                base_quality_threshold, lightning_start, lightning_steps
            )

            # Format alignment method for logging
            if alignment_method == "mathematical_search":
                base_calculation_info = (
                    "Auto-calculated base_steps = %d, total_base_steps = %d (mathematical search)"
                    % (base_steps_calc, total_base_steps)
                )
            elif alignment_method == "simple_math":
                base_calculation_info = (
                    "Auto-calculated base_steps = %d, total_base_steps = %d (simple math)"
                    % (base_steps_calc, total_base_steps)
                )
            else:
                base_calculation_info = (
                    "Auto-calculated base_steps = %d (fallback - no perfect alignment found)"
                    % base_steps_calc
                )
        else:
            # Manual base_steps - calculate optimal total_base_steps for alignment checks
            base_steps_calc = base_steps
            total_base_steps = core_alignment.calculate_manual_base_steps_alignment(
                base_steps, lightning_start, lightning_steps
            )
            base_calculation_info = (
                "Auto-calculated total_base_steps = %d for manual base_steps = %d"
                % (
                    total_base_steps,
                    base_steps_calc,
                )
            )

        logger.info(base_calculation_info)

        # === Stage overlap check (only for manual base_steps) ===
        if (
            base_steps != -1
            and lightning_start > 0
            and base_steps_calc > 0
            and total_base_steps > 0
        ):
            # Calculate denoising percentages
            stage1_end_pct = (base_steps_calc / total_base_steps) * 100.0
            stage2_start_pct = (lightning_start / lightning_steps) * 100.0

            if stage1_end_pct > stage2_start_pct:
                overlap_pct = stage1_end_pct - stage2_start_pct
                logger.warning(
                    "Stage 1 and 2 overlap (%.1f%%) detected! For perfect alignment, use base_steps=-1 or adjust lightning params.",
                    overlap_pct,
                )
                # Send overlap warning toast notification
                core_notifications.send_overlap_warning(overlap_pct)

        # === Step 2: Calculate switch point based on strategy ===
        # Strip "(refined)" suffix to get base strategy for matching
        base_strategy = self._get_base_strategy(switch_strategy)

        if base_strategy == "50% of steps":
            switch_step_calculated = math.ceil(lightning_steps / 2)
            model_switching_info = "Model switching: 50%% of steps → switch at step %d of %d" % (
                switch_step_calculated,
                lightning_steps,
            )
        elif base_strategy == "Manual switch step":
            if switch_step == -1:
                # Auto-calculate at 50%
                switch_step_calculated = math.ceil(lightning_steps / 2)
                model_switching_info = (
                    "Model switching: Manual switch step (auto at 50%%) → switch at step %d of %d"
                    % (switch_step_calculated, lightning_steps)
                )
            else:
                switch_step_calculated = switch_step
                model_switching_info = (
                    "Model switching: Manual switch step → switch at step %d of %d"
                    % (
                        switch_step_calculated,
                        lightning_steps,
                    )
                )
        elif base_strategy == "T2V boundary":
            # Use T2V boundary with WanVideo's native scheduler system
            switch_step_calculated = self._compute_wanvideo_boundary_switching_step(
                lightning_high, lightning_scheduler, lightning_steps, sigma_shift, _BOUNDARY_T2V
            )
            model_switching_info = (
                "Model switching: T2V boundary (boundary = %.3f) → switch at step %d of %d"
                % (_BOUNDARY_T2V, switch_step_calculated, lightning_steps)
            )
        elif base_strategy == "I2V boundary":
            # Use I2V boundary with WanVideo's native scheduler system
            switch_step_calculated = self._compute_wanvideo_boundary_switching_step(
                lightning_high, lightning_scheduler, lightning_steps, sigma_shift, _BOUNDARY_I2V
            )
            model_switching_info = (
                "Model switching: I2V boundary (boundary = %.3f) → switch at step %d of %d"
                % (_BOUNDARY_I2V, switch_step_calculated, lightning_steps)
            )
        elif base_strategy == "Manual boundary":
            # Use manual boundary with WanVideo's native scheduler system
            switch_step_calculated = self._compute_wanvideo_boundary_switching_step(
                lightning_high, lightning_scheduler, lightning_steps, sigma_shift, switch_boundary
            )
            model_switching_info = (
                "Model switching: Manual boundary (boundary = %.3f) → switch at step %d of %d"
                % (switch_boundary, switch_step_calculated, lightning_steps)
            )
        else:
            # Default fallback
            switch_step_calculated = math.ceil(lightning_steps / 2)
            model_switching_info = (
                "Model switching: 50%% of steps (fallback) → switch at step %d of %d"
                % (switch_step_calculated, lightning_steps)
            )

        logger.info(model_switching_info)

        # === Sigma shift refinement phase (for refined strategy variants) ===
        sigma_shift_final = sigma_shift

        if self._is_refined_strategy(switch_strategy):
            try:
                target_boundary = self._get_target_boundary_from_strategy(
                    switch_strategy, switch_boundary
                )

                refined_shift, refine_msg = core_models.calculate_perfect_shift_for_step(
                    model=lightning_high,
                    scheduler=lightning_scheduler,
                    total_steps=lightning_steps,
                    target_step=switch_step_calculated,
                    target_sigma=target_boundary,
                    initial_shift=sigma_shift,
                    is_wanvideo=True,  # WVSampler uses WanVideo schedulers
                )

                logger.info(
                    "Refined sigma_shift: %.2f → %.2f for perfect boundary alignment at step %d",
                    sigma_shift,
                    refined_shift,
                    switch_step_calculated,
                )
                logger.debug("Refinement converged: %s", refine_msg)

                sigma_shift_final = refined_shift

                # Log updated sigmas after refinement for debugging
                try:
                    import torch

                    # Get WanVideo scheduler function
                    _, _, get_scheduler = get_wanvideo_components()

                    # Get transformer dimension from model
                    try:
                        transformer_dim = (
                            lightning_high.model.diffusion_model.dim
                            if hasattr(lightning_high.model, "diffusion_model")
                            else 5120
                        )
                    except Exception:
                        transformer_dim = 5120  # Default for Wan 14B models (most common)

                    # Recalculate sigmas with refined shift to show updated schedule
                    device = torch.device("cpu")
                    sample_scheduler_refined, _, _, _ = get_scheduler(
                        lightning_scheduler,
                        lightning_steps,
                        start_step=0,
                        end_step=-1,
                        shift=refined_shift,
                        device=device,
                        transformer_dim=transformer_dim,
                    )
                    sigmas_refined = sample_scheduler_refined.sigmas
                    timesteps_refined = []
                    for sigma in sigmas_refined[:-1]:  # Skip last sigma (always 0.0)
                        try:
                            timestep = float(sigma.item())
                        except Exception:
                            timestep = float(sigma)
                        timesteps_refined.append(timestep)

                    sigmas_formatted_refined = [f"{t:.4f}" for t in timesteps_refined]
                    logger.debug(
                        f"WanVideo sigmas (after refinement): [{', '.join(sigmas_formatted_refined)}]"
                    )
                except Exception as e:
                    logger.debug(f"Could not recalculate sigmas after refinement: {e}")

                # Append refinement info to model switching info for dry-run display
                model_switching_info += f" [Refined shift: {sigma_shift:.2f}→{refined_shift:.2f}]"

            except Exception as e:
                logger.warning(
                    "Sigma shift refinement failed: %s. Using initial shift %.2f", e, sigma_shift
                )
                sigma_shift_final = sigma_shift

        # === Validate parameters after all calculations ===
        # Validate resolved parameters after auto-calculation
        self._validate_resolved_parameters(lightning_start, base_steps_calc)

        # Validate special modes (lightning-only, skip configurations)
        self._validate_special_modes(
            lightning_start, lightning_steps, base_steps_calc, switch_strategy, switch_step
        )

        # === Dry run mode: Return configuration info without sampling ===
        if dry_run:
            # Determine skip conditions (matching TripleKSampler logic)
            skip_stage1 = lightning_start == 0 or base_steps_calc == 0
            skip_stage2 = lightning_start == switch_step_calculated

            # Calculate stage info strings with skip detection
            if skip_stage1:
                stage1_info = "Skipped (Lightning-only mode)"
            else:
                stage1_info = self._format_stage_range(0, base_steps_calc, total_base_steps)

            if skip_stage2:
                stage2_info = "Skipped (lightning_start equals switch point)"
            else:
                stage2_info = self._format_stage_range(
                    lightning_start, switch_step_calculated, lightning_steps
                )

            stage3_start = max(lightning_start, switch_step_calculated)
            stage3_info = self._format_stage_range(stage3_start, lightning_steps, lightning_steps)

            # Log stage configuration (matching TripleKSampler format)
            bare_logger.info("")
            logger.info("Stage 1: Base high model - %s", stage1_info)

            bare_logger.info("")
            logger.info("Stage 2: Lightning high model - %s", stage2_info)

            bare_logger.info("")
            logger.info("Stage 3: Lightning low model - %s", stage3_info)
            bare_logger.info("")

            # Send toast notification with dry run summary
            summary_lines = []

            # Add calculation insights if available (hide base_calculation_info in lightning-only mode)
            show_base_calc = not skip_stage1 and base_calculation_info
            if show_base_calc or model_switching_info:
                summary_lines.append("Calculations:")

                if show_base_calc:
                    # Transform verbose base calculation info into compact format
                    formatted_base = self._format_base_calculation_compact(base_calculation_info)
                    summary_lines.append(f"• {formatted_base}")

                if model_switching_info:
                    # Transform verbose model switching info into compact format
                    formatted_switch = self._format_switch_info_compact(model_switching_info)
                    summary_lines.append(f"• {formatted_switch}")

                summary_lines.append("")

            # Always add stage configuration
            summary_lines.extend(
                [
                    "Stage Configuration:",
                    f"• {stage1_info}",
                    f"• {stage2_info}",
                    f"• {stage3_info}",
                ]
            )

            detail_text = "\n".join(summary_lines)

            # Send dry-run completion notification
            core_notifications.send_dry_run_notification(detail_text)

            logger.info("[DRY RUN] Complete - interrupting workflow execution (expected behavior)")
            bare_logger.info("")

            raise comfy.model_management.InterruptProcessingException()

        # === Step 3: Prepare shared parameters for all stages (without scheduler/force_offload - passed per-stage) ===
        shared_params = {
            "image_embeds": image_embeds,
            "shift": sigma_shift_final,  # Use refined shift if refinement occurred
            "seed": seed,
            "riflex_freq_index": riflex_freq_index,
            "batched_cfg": batched_cfg,
            "rope_function": rope_function,
            **kwargs,  # All other optional WanVideo params
        }

        # === Step 4: Execute Stage 1 - Base denoising ===
        if lightning_start == 0 or base_steps_calc == 0:
            logger.info("Lightning-only mode: base_steps not applicable (Stage 1 skipped)")
            bare_logger.info("")  # separator before skipped stage log
            logger.info("Stage 1: Skipped (Lightning-only mode)")
            stage1_output = None  # Will trigger add_noise in Stage 2
        else:
            bare_logger.info("")  # separator before stage execution
            stage1_info = self._format_stage_range(0, base_steps_calc, total_base_steps)
            logger.info("Stage 1: Base high model - %s", stage1_info)

            try:
                stage1_result = self.wanvideo_sampler.process(
                    model=base_high,
                    steps=total_base_steps,
                    cfg=base_cfg,
                    scheduler=base_scheduler,  # Use base scheduler
                    force_offload=force_offload,
                    start_step=0,
                    end_step=base_steps_calc,
                    add_noise_to_samples=True,  # Stage 1 always adds noise
                    **shared_params,
                )
                stage1_output, _ = stage1_result
            except comfy.model_management.InterruptProcessingException:
                # User cancelled - re-raise without wrapping to allow graceful cancellation
                raise
            except Exception as exc:
                exc_msg = str(exc).strip()
                if exc_msg:
                    raise RuntimeError(
                        f"Stage 1: sampling failed - {type(exc).__name__}: {exc_msg}"
                    ) from exc
                else:
                    raise RuntimeError(f"Stage 1: sampling failed - {type(exc).__name__}") from exc

        # === Step 5: Execute Stage 2 - Lightning high model ===
        skip_stage2 = lightning_start == switch_step_calculated

        if skip_stage2:
            bare_logger.info("")  # separator before skipped stage log
            logger.info("Stage 2: Skipped (lightning_start equals switch point)")
            stage2_output = stage1_output
        else:
            bare_logger.info("")  # separator before stage execution
            stage2_info = self._format_stage_range(
                lightning_start, switch_step_calculated, lightning_steps
            )
            logger.info("Stage 2: Lightning high model - %s", stage2_info)

            stage2_add_noise = stage1_output is None  # Add noise if Stage 1 was skipped
            try:
                stage2_result = self.wanvideo_sampler.process(
                    model=lightning_high,
                    steps=lightning_steps,
                    cfg=lightning_cfg,
                    scheduler=lightning_scheduler,  # Use lightning scheduler
                    force_offload=force_offload,
                    samples=stage1_output,  # Chain from Stage 1
                    start_step=lightning_start,
                    end_step=switch_step_calculated,
                    add_noise_to_samples=stage2_add_noise,
                    **shared_params,
                )
                stage2_output, _ = stage2_result
            except comfy.model_management.InterruptProcessingException:
                # User cancelled - re-raise without wrapping to allow graceful cancellation
                raise
            except Exception as exc:
                exc_msg = str(exc).strip()
                if exc_msg:
                    raise RuntimeError(
                        f"Stage 2: sampling failed - {type(exc).__name__}: {exc_msg}"
                    ) from exc
                else:
                    raise RuntimeError(f"Stage 2: sampling failed - {type(exc).__name__}") from exc

        # === Step 6: Execute Stage 3 - Lightning low model ===
        bare_logger.info("")  # separator before stage execution
        stage3_start = max(lightning_start, switch_step_calculated)
        stage3_info = self._format_stage_range(stage3_start, lightning_steps, lightning_steps)
        logger.info("Stage 3: Lightning low model - %s", stage3_info)

        # Use incremented seed for Stage 3 to ensure different noise pattern
        stage3_params = shared_params.copy()
        stage3_params["seed"] = seed + STAGE3_SEED_OFFSET

        stage3_add_noise = stage2_output is None  # Add noise if both Stage 1 and 2 were skipped
        try:
            stage3_result = self.wanvideo_sampler.process(
                model=lightning_low,
                steps=lightning_steps,
                cfg=lightning_cfg,
                scheduler=lightning_scheduler,  # Use lightning scheduler
                force_offload=force_offload,
                samples=stage2_output,  # Chain from Stage 2
                start_step=stage3_start,
                end_step=-1,  # -1 means full sampling to the end
                add_noise_to_samples=stage3_add_noise,
                **stage3_params,
            )
            final_samples, final_denoised = stage3_result
        except comfy.model_management.InterruptProcessingException:
            # User cancelled - re-raise without wrapping to allow graceful cancellation
            raise
        except Exception as exc:
            exc_msg = str(exc).strip()
            if exc_msg:
                raise RuntimeError(
                    f"Stage 3: sampling failed - {type(exc).__name__}: {exc_msg}"
                ) from exc
            else:
                raise RuntimeError(f"Stage 3: sampling failed - {type(exc).__name__}") from exc

        bare_logger.info("")  # final separator after all sampling completes

        return (final_samples, final_denoised)


class TripleWVSamplerAdvanced(TripleWVSamplerAdvancedAlt):
    """Advanced triple-stage sampler for WanVideo models with dynamic UI.

    Inherits all functionality from TripleWVSamplerAdvancedAlt with identical
    INPUT_TYPES. Dynamic parameter visibility (base_quality_threshold, switch_step,
    switch_boundary, dry_run) is handled by JavaScript extension, not by moving
    parameters to optional section.

    Features:
        - All 5 switching strategies
        - Separate CFG controls
        - Manual base_steps control
        - Context-aware parameter visibility (dynamic UI via JavaScript)
        - Dry run mode (hidden widget with "Run Dry Run" button)
    """

    DESCRIPTION = (
        "Advanced triple-stage sampler for WanVideo models with Lightning LoRA. "
        "Dynamic UI variant with context-aware parameter visibility."
    )

    # INPUT_TYPES inherited directly from TripleWVSamplerAdvancedAlt
    # JavaScript (web/triple_ksampler_ui.js) handles dynamic visibility:
    # - base_quality_threshold: shown only when base_steps=-1
    # - switch_step: shown only for "Manual switch step" strategy
    # - switch_boundary: shown only for "Manual boundary" strategy
    # - dry_run: hidden, replaced with "Run Dry Run" button
    # - batched_cfg, rope_function: always visible

    # sample() method inherited from TripleWVSamplerAdvancedAlt
    # _get_fallback_input_types() inherited from TripleWVSamplerAdvancedAlt
