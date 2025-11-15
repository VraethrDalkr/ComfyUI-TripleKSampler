"""Advanced node variants with full parameter control.

This module contains:
- TripleKSamplerAdvancedAlt: Full implementation with static UI (all parameters visible)
- TripleKSamplerAdvanced: Dynamic UI variant (parameters in optional section)

Both nodes expose all 8 switching strategies and provide fine-grained control over
stage boundaries, sigma shifts, and Lightning model configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.samplers
import torch

# Import base class
from triple_ksampler.base import TripleKSamplerBase

# Import shared logic modules
from triple_ksampler.shared import alignment as core_alignment
from triple_ksampler.shared import config as core_config
from triple_ksampler.shared import models as core_models
from triple_ksampler.shared import notifications as core_notifications

# Load configuration at import time (delegate to core.config)
_CONFIG = core_config.load_config(config_dir=Path(__file__).resolve().parent.parent.parent)

# Extract configuration values using core.config
_BASE_QUALITY_THRESHOLD = core_config.get_base_quality_threshold(_CONFIG)
_LOG_LEVEL = core_config.get_log_level(_CONFIG)

# Get logger (inherits configuration from ksampler_nodes.py entry point)
logger = logging.getLogger("triple_ksampler.ksampler.advanced")
bare_logger = logging.getLogger("triple_ksampler.separator")

# Algorithm constants
STAGE3_SEED_OFFSET = 1  # Ensure Stage 3 uses different noise pattern


class TripleKSamplerAdvancedAlt(TripleKSamplerBase):
    """Advanced triple-stage node with all parameters exposed (static UI)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Return ComfyUI INPUT_TYPES mapping with all parameters in required section."""
        base_inputs = cls._get_base_input_types()

        return {
            "required": {
                # Models
                "base_high": base_inputs["base_high"],
                "lightning_high": base_inputs["lightning_high"],
                "lightning_low": base_inputs["lightning_low"],
                # Conditioning
                "positive": base_inputs["positive"],
                "negative": base_inputs["negative"],
                "latent_image": base_inputs["latent_image"],
                # Base parameters
                "seed": base_inputs["seed"],
                "sigma_shift": base_inputs["sigma_shift"],
                "base_quality_threshold": (
                    "INT",
                    {
                        "default": core_config.DEFAULT_BASE_QUALITY_THRESHOLD,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": f"Minimum total steps for base_steps auto-calculation (config default: {_BASE_QUALITY_THRESHOLD}). Only applies when base_steps=-1.",
                    },
                ),
                "base_steps": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Stage 1 steps for base high-noise model. Use -1 for auto-calculation based on quality threshold.",
                    },
                ),
                "base_cfg": base_inputs["base_cfg"],
                "base_sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler for Stage 1 (base model)."},
                ),
                "base_scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler for Stage 1 (base model)."},
                ),
                # Lightning parameters
                "lightning_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99,
                        "tooltip": "Starting step within lightning schedule. Set to 0 to skip Stage 1 entirely.",
                    },
                ),
                "lightning_steps": base_inputs["lightning_steps"],
                "lightning_cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for Stage 2 and Stage 3. In regular node, automatically set to 1.0.",
                    },
                ),
                "lightning_sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler for Stage 2 and Stage 3 (lightning models)."},
                ),
                "lightning_scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler for Stage 2 and Stage 3 (lightning models)."},
                ),
                # Switching parameters
                "switch_strategy": (
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
                ),
                "switch_step": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 99,
                        "tooltip": "Manual step to switch models. Only used when switch_strategy is 'Manual switch step'. Use -1 for auto-calculation at 50% of lightning steps.",
                    },
                ),
                "switch_boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sigma boundary for switching. Only used when switch_strategy is 'Manual boundary'.",
                    },
                ),
                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable dry run mode to test stage calculations without actual sampling.",
                    },
                ),
            },
        }

    DESCRIPTION = (
        "Advanced triple-stage cascade sampler with all parameters exposed for "
        "Wan2.2 split models with Lightning LoRA. Static UI variant with all parameters always visible."
    )

    def sample(
        self,
        base_high: Any,
        lightning_high: Any,
        lightning_low: Any,
        positive: Any,
        negative: Any,
        latent_image: dict[str, torch.Tensor],
        seed: int,
        sigma_shift: float,
        base_steps: int,
        base_quality_threshold: int,
        base_cfg: float,
        base_sampler: str,
        base_scheduler: str,
        lightning_start: int,
        lightning_steps: int,
        lightning_cfg: float,
        lightning_sampler: str,
        lightning_scheduler: str,
        switch_strategy: str,
        switch_boundary: float = 0.875,
        switch_step: int = -1,
        dry_run: bool = False,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        """Perform the triple-stage sampling run and return final latent tuple."""
        if str(_LOG_LEVEL).upper() == "DEBUG":
            bare_logger.info("")
        logger.debug("=== TripleKSampler Node - Input Parameters ===")
        logger.debug("Sampling: seed=%d, sigma_shift=%.3f", seed, sigma_shift)
        logger.debug(
            "Base stage: base_quality_threshold=%d, base_steps=%d, base_cfg=%.1f",
            base_quality_threshold,
            base_steps,
            base_cfg,
        )
        logger.debug("Base sampler: %s, scheduler: %s", base_sampler, base_scheduler)
        logger.debug(
            "Lightning: lightning_start=%d, lightning_steps=%d, lightning_cfg=%.1f",
            lightning_start,
            lightning_steps,
            lightning_cfg,
        )
        logger.debug("Lightning sampler: %s, scheduler: %s", lightning_sampler, lightning_scheduler)
        logger.debug("Strategy: %s", switch_strategy)
        if switch_strategy == "Manual boundary":
            logger.debug("  switch_boundary=%.3f", switch_boundary)
        elif switch_strategy == "Manual switch step":
            logger.debug("  switch_step=%d", switch_step)
        logger.debug("dry_run=%s", dry_run)

        # Variables to capture calculation info for dry run notification
        base_calculation_info = ""
        model_switching_info = ""

        # Early validation (before auto-calculation resolves base_steps)
        self._validate_basic_parameters(
            lightning_steps, lightning_start, switch_strategy, switch_step
        )

        bare_logger.info("")  # separator before calculation logs

        # Use the provided base quality threshold
        effective_threshold = base_quality_threshold

        # Calculate base_steps and total_base_steps
        optimal_total_base_steps = None
        if base_steps == -1:
            base_steps, optimal_total_base_steps, method = self._calculate_perfect_alignment(
                effective_threshold, lightning_start, lightning_steps
            )
            if lightning_start > 0:
                if method == "mathematical_search":
                    base_calculation_info = f"Auto-calculated base_steps = {base_steps}, total_base_steps = {optimal_total_base_steps} (mathematical search)"
                    logger.info(base_calculation_info)
                elif method == "simple_math":
                    base_calculation_info = f"Auto-calculated base_steps = {base_steps}, total_base_steps = {optimal_total_base_steps} (simple math)"
                    logger.info(base_calculation_info)
                else:
                    base_calculation_info = f"Auto-calculated base_steps = {base_steps} (fallback - no perfect alignment found)"
                    logger.info(base_calculation_info)
        else:
            # manual base_steps -> compute total_base_steps for alignment checks
            optimal_total_base_steps = core_alignment.calculate_manual_base_steps_alignment(
                base_steps, lightning_start, lightning_steps
            )
            base_calculation_info = f"Auto-calculated total_base_steps = {optimal_total_base_steps} for manual base_steps = {base_steps}"
            logger.info(base_calculation_info)
            # === Stage overlap check ===
            if lightning_start > 0 and base_steps > 0 and optimal_total_base_steps > 0:
                stage1_end_pct = base_steps / optimal_total_base_steps
                stage2_start_pct = lightning_start / lightning_steps
                if stage1_end_pct > stage2_start_pct:
                    overlap_pct = (stage1_end_pct - stage2_start_pct) * 100.0
                    logger.warning(
                        "Stage 1 and 2 overlap (%.1f%%) detected! For perfect alignment, use base_steps=-1 or adjust lightning params.",
                        overlap_pct,
                    )
                    # Send toast notification (delegate to core.notifications)
                    core_notifications.send_overlap_warning(overlap_pct)

        # Validate resolved parameters after auto-calculation
        self._validate_resolved_parameters(lightning_start, base_steps)

        # Validate special modes (lightning-only, skip configurations)
        self._validate_special_modes(
            lightning_start, lightning_steps, base_steps, switch_strategy, switch_step
        )

        # Patch all models with initial sigma shift for discovery (non-mutating)
        patched_base_high, patched_lightning_high, patched_lightning_low = (
            self._patch_models_for_sampling(base_high, lightning_high, lightning_low, sigma_shift)
        )

        # Calculate switch step using initial sigma shift (discovery phase)
        switch_step_discovered, _, model_switching_info = self._calculate_switch_step_and_strategy(
            switch_strategy,
            switch_step,
            switch_boundary,
            lightning_steps,
            patched_lightning_high,
            lightning_scheduler,
        )

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
                    target_step=switch_step_discovered,
                    target_sigma=target_boundary,
                    initial_shift=sigma_shift,
                    is_wanvideo=False,  # KSampler uses ComfyUI samplers
                )

                logger.info(
                    "Refined sigma_shift: %.2f → %.2f for perfect boundary alignment at step %d",
                    sigma_shift,
                    refined_shift,
                    switch_step_discovered,
                )
                logger.debug("Refinement converged: %s", refine_msg)

                sigma_shift_final = refined_shift

                # Append refinement info to model switching info for dry-run display
                model_switching_info += f" [Refined shift: {sigma_shift:.2f}→{refined_shift:.2f}]"

            except Exception as e:
                logger.warning(
                    "Sigma shift refinement failed: %s. Using initial shift %.2f", e, sigma_shift
                )
                sigma_shift_final = sigma_shift

        # Re-patch models with final shift if refinement occurred
        if sigma_shift_final != sigma_shift:
            patched_base_high, patched_lightning_high, patched_lightning_low = (
                self._patch_models_for_sampling(
                    base_high, lightning_high, lightning_low, sigma_shift_final
                )
            )

        # Use discovered switch_step for final execution
        switch_step_final = switch_step_discovered

        # Stage execution logic flags
        skip_stage1 = lightning_start == 0
        skip_stage2 = False
        stage2_skip_reason = ""

        stage1_add_noise = True
        stage2_add_noise = skip_stage1
        stage3_add_noise = False

        # Validate switch step against lightning_start
        if lightning_start > switch_step_final:
            raise ValueError("lightning_start cannot be greater than switch_step.")

        # Log the model switching strategy
        logger.info(model_switching_info)

        # Check if Stage 2 should be skipped
        if lightning_start == switch_step_final:
            skip_stage2 = True
            stage2_skip_reason = "lightning_start equals switch point"

        stage3_add_noise = skip_stage1 and skip_stage2

        # Stage 1: Base denoising
        if skip_stage1:
            logger.info("Lightning-only mode: base_steps not applicable (Stage 1 skipped)")
            bare_logger.info("")  # separator before skipped stage log
            logger.info("Stage 1: Skipped (Lightning-only mode)")
            stage1_info = "Skipped (Lightning-only mode)"
            stage1_output = latent_image
        else:
            if optimal_total_base_steps is not None:
                total_base_steps = optimal_total_base_steps
            else:
                _, total_base_steps, _ = self._calculate_perfect_alignment(
                    effective_threshold, lightning_start, lightning_steps
                )

            stage1_info = self._format_stage_range(0, base_steps, total_base_steps)
            stage1_result = self._run_sampling_stage(
                model=patched_base_high,
                positive=positive,
                negative=negative,
                latent=latent_image,
                seed=seed,
                steps=total_base_steps,
                cfg=base_cfg,
                sampler_name=base_sampler,
                scheduler=base_scheduler,
                start_at_step=0,
                end_at_step=base_steps,
                add_noise=stage1_add_noise,
                return_with_leftover_noise=True,
                dry_run=dry_run,
                stage_name="Stage 1",
                stage_info=stage1_info,
            )
            stage1_output = stage1_result[0]

        # Stage 2: Lightning high
        if skip_stage2:
            bare_logger.info("")  # separator before skipped stage log
            logger.info("Stage 2: Skipped (%s)", stage2_skip_reason)
            stage2_info = f"Skipped ({stage2_skip_reason})"
            stage2_output = stage1_output
        else:
            stage2_info = self._format_stage_range(
                lightning_start, switch_step_final, lightning_steps
            )
            stage2_result = self._run_sampling_stage(
                model=patched_lightning_high,
                positive=positive,
                negative=negative,
                latent=stage1_output,
                seed=seed,
                steps=lightning_steps,
                cfg=lightning_cfg,
                sampler_name=lightning_sampler,
                scheduler=lightning_scheduler,
                start_at_step=lightning_start,
                end_at_step=switch_step_final,
                add_noise=stage2_add_noise,
                return_with_leftover_noise=True,
                dry_run=dry_run,
                stage_name="Stage 2",
                stage_info=stage2_info,
            )
            stage2_output = stage2_result[0]

        # Stage 3: Lightning low
        stage3_start = max(lightning_start, switch_step_final)
        stage3_info = self._format_stage_range(stage3_start, lightning_steps, lightning_steps)
        stage3_result = self._run_sampling_stage(
            model=patched_lightning_low,
            positive=positive,
            negative=negative,
            latent=stage2_output,
            seed=seed + STAGE3_SEED_OFFSET,
            steps=lightning_steps,
            cfg=lightning_cfg,  # Use lightning_cfg parameter for advanced node flexibility
            sampler_name=lightning_sampler,
            scheduler=lightning_scheduler,
            start_at_step=stage3_start,
            end_at_step=lightning_steps,
            add_noise=stage3_add_noise,
            return_with_leftover_noise=False,
            dry_run=dry_run,
            stage_name="Stage 3",
            stage_info=stage3_info,
        )

        bare_logger.info("")  # final separator after all sampling completes
        if dry_run:
            # Send toast notification with dry run summary
            self._send_dry_run_notification(
                stage1_info=stage1_info,
                stage2_info=stage2_info,
                stage3_info=stage3_info,
                base_calculation_info=base_calculation_info,
                model_switching_info=model_switching_info,
            )

            logger.info("[DRY RUN] Complete - interrupting workflow execution (expected behavior)")
            bare_logger.info("")

            raise comfy.model_management.InterruptProcessingException()

        # Always return tuple as expected by RETURN_TYPES
        return stage3_result


class TripleKSamplerAdvanced(TripleKSamplerAdvancedAlt):
    """Advanced triple-stage node with dynamic UI (context-aware parameter visibility)."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Return ComfyUI INPUT_TYPES mapping for the advanced node."""
        base_inputs = cls._get_base_input_types()

        return {
            "required": {
                # Models
                "base_high": base_inputs["base_high"],
                "lightning_high": base_inputs["lightning_high"],
                "lightning_low": base_inputs["lightning_low"],
                # Conditioning
                "positive": base_inputs["positive"],
                "negative": base_inputs["negative"],
                "latent_image": base_inputs["latent_image"],
                # Base parameters
                "seed": base_inputs["seed"],
                "sigma_shift": base_inputs["sigma_shift"],
                "base_quality_threshold": (
                    "INT",
                    {
                        "default": core_config.DEFAULT_BASE_QUALITY_THRESHOLD,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": f"Minimum total steps for base_steps auto-calculation (config default: {_BASE_QUALITY_THRESHOLD}). Only applies when base_steps=-1.",
                    },
                ),
                "base_steps": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Stage 1 steps for base high-noise model. Use -1 for auto-calculation based on quality threshold.",
                    },
                ),
                "base_cfg": base_inputs["base_cfg"],
                "base_sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler for Stage 1 (base model)."},
                ),
                "base_scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler for Stage 1 (base model)."},
                ),
                # Lightning parameters
                "lightning_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99,
                        "tooltip": "Starting step within lightning schedule. Set to 0 to skip Stage 1 entirely.",
                    },
                ),
                "lightning_steps": base_inputs["lightning_steps"],
                "lightning_cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for Stage 2 and Stage 3. In regular node, automatically set to 1.0.",
                    },
                ),
                "lightning_sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler for Stage 2 and Stage 3 (lightning models)."},
                ),
                "lightning_scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler for Stage 2 and Stage 3 (lightning models)."},
                ),
                # Switching parameters
                "switch_strategy": (
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
                ),
            },
            "optional": {
                "switch_step": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 99,
                        "tooltip": "Manual step to switch models. Only used when switch_strategy is 'Manual switch step'. Use -1 for auto-calculation at 50% of lightning steps. Ignored for all other strategies.",
                    },
                ),
                "switch_boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sigma boundary for switching. Only used when switch_strategy is 'Manual boundary'. Ignored for all other strategies.",
                    },
                ),
                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable dry run mode to test stage calculations without actual sampling.",
                    },
                ),
            },
        }

    DESCRIPTION = (
        "Advanced triple-stage cascade sampler with dynamic UI for "
        "Wan2.2 split models with Lightning LoRA. Context-aware parameter visibility."
    )

    # sample() method inherited from TripleKSamplerAdvancedAlt
