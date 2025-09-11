"""
Triple-stage KSampler for Wan2.2 split models with Lightning LoRA.

This module implements sophisticated triple-stage sampling nodes for ComfyUI,
specifically designed for Wan2.2 split models with Lightning LoRA integration.
The sampling process includes base denoising, lightning high-model processing,
and lightning low-model refinement stages.

Classes:
    TripleKSamplerWan22Lightning: Main triple-stage sampler with auto-computed parameters
    TripleKSamplerWan22LightningAdvanced: Advanced variant with full parameter control
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import comfy.model_sampling
import comfy.samplers
import nodes
import torch
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

from .constants import MIN_TOTAL_STEPS, ENABLE_CONSISTENCY_CHECK, LOGGER_PREFIX, DEFAULT_BOUNDARY_T2V, DEFAULT_BOUNDARY_I2V

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"{LOGGER_PREFIX} %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(logging.INFO)


class TripleKSamplerWan22LightningAdvanced:
    """
    Advanced Triple-stage KSampler node for Wan2.2 split models with Lightning LoRA.

    This advanced node provides complete parameter control for sophisticated
    three-stage sampling process:
    1. Base denoising with high-noise model
    2. Lightning high-model processing
    3. Lightning low-model refinement

    The node clones and patches models with sigma shift for optimal sampling
    without mutating the original models. It supports both step-based and
    sigma boundary-based model switching strategies with full configurability.

    Attributes:
        RETURN_TYPES: Output types for ComfyUI
        RETURN_NAMES: Output names for ComfyUI
        FUNCTION: Entry point method name
        CATEGORY: ComfyUI category for node organization
        DESCRIPTION: Node description for ComfyUI
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define input specification for ComfyUI.

        Returns:
            Dict containing required input parameters with their types,
            defaults, ranges, and tooltips.
        """
        return {
            "required": {
                "high_model": (
                    "MODEL",
                    {"tooltip": "High-noise model for base denoising."}
                ),
                "high_model_lx2v": (
                    "MODEL",
                    {"tooltip": "Lightning high model (LightX2V)."}
                ),
                "low_model_lx2v": (
                    "MODEL",
                    {"tooltip": "Lightning low model (LightX2V)."}
                ),
                "positive": (
                    "CONDITIONING",
                    {"tooltip": "Positive prompt conditioning."}
                ),
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Negative prompt conditioning."}
                ),
                "latent_image": (
                    "LATENT",
                    {"tooltip": "Latent image to denoise."}
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "tooltip": "Random seed for noise generation."
                    }
                ),
                "sigma_shift": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Sigma shift applied to model sampling (default 5.0)."
                    }
                ),
                "base_steps": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Number of base (Stage 1) steps. Use -1 for auto-calculation based on lightning_start."
                    }
                ),
                "base_cfg": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for Stage 1."
                    }
                ),
                "lightning_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99,
                        "tooltip": "Start step inside lightning schedule (0 to skip Stage 1)."
                    }
                ),
                "lightning_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 100,
                        "tooltip": "Total steps for lightning stages."
                    }
                ),
                "lightning_cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for Stage 2 and Stage 3 (lightning stages)."
                    }
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler to use."}
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler for noise."}
                ),
                "switch_strategy": (
                    ["50% of steps", "Manual switch step", "T2V boundary", "I2V boundary", "Manual boundary"],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between lightning high and low models."
                    }
                ),
            },
            "optional": {
                "switch_step": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 99,
                        "tooltip": "Manual step to switch from high-noise to low-noise model. Use -1 for auto-calculation."
                    }
                ),
                "switch_boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": (
                            "Boundary for sigma-based model switching. "
                            "Recommended 0.875 for T2V, 0.900 for I2V."
                        )
                    }
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "TripleKSampler/sampling"
    DESCRIPTION = (
        "Advanced triple-stage cascade sampler with full parameter control for "
        "Wan2.2 split models with Lightning LoRA. Supports both step-based and "
        "sigma boundary-based model switching."
    )

    def _get_model_patcher(self) -> ModelSamplingSD3:
        """
        Create ModelSamplingSD3 instance for patching models.

        Returns:
            ModelSamplingSD3 instance for model patching.
        """
        return ModelSamplingSD3()

    def _canonicalize_shift(self, value: float) -> float:
        """
        Convert shift value to canonical Python float.

        Args:
            value: Shift value input (may be tensor or other numeric type).

        Returns:
            Canonical float representation.
        """
        return float(value)

    def _calculate_percentage(self, numerator: float, denominator: float) -> float:
        """
        Calculate percentage with single digit precision for logging.

        Args:
            numerator: Numerator value.
            denominator: Denominator value.

        Returns:
            Float percentage between 0.0 and 100.0 with one decimal place.
        """
        if denominator == 0:
            return 0.0
        percentage = (float(numerator) / float(denominator)) * 100.0
        return round(max(0.0, min(100.0, percentage)), 1)

    def _format_stage_range(self, start: int, end: int, total: int) -> str:
        """
        Format human-readable stage range with denoising percentages.

        Args:
            start: Starting step (inclusive).
            end: Ending step (exclusive).
            total: Total steps in schedule.

        Returns:
            Formatted string like "steps 0-4 of 24 (denoising 0%-16%)".
        """
        start_safe = int(max(0, start))
        end_safe = int(max(start_safe, end))
        total_safe = int(max(1, total))
        
        pct_start = self._calculate_percentage(start_safe, total_safe)
        pct_end = self._calculate_percentage(end_safe, total_safe)
        
        return f"steps {start_safe}-{end_safe} of {total_safe} (denoising {pct_start:.1f}%–{pct_end:.1f}%)"

    def _compute_boundary_switching_step(
        self,
        sampling: Any,
        scheduler: str,
        steps: int,
        boundary: float
    ) -> int:
        """
        Compute model switching step based on sigma boundary.

        Args:
            sampling: Model sampling object.
            scheduler: Scheduler name.
            steps: Number of lightning steps.
            boundary: Timestep boundary (0-1).

        Returns:
            Switching step index in range [0, steps-1].
        """
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
        timesteps: List[float] = []
        
        # Convert tensor sigmas to timesteps
        for sigma in sigmas:
            timestep = sampling.timestep(float(sigma.item())) / 1000.0
            timesteps.append(timestep)
        
        switching_step = steps
        # Start at index 1 to match previous behavior
        for i, timestep in enumerate(timesteps[1:], start=1):
            if timestep < float(boundary):
                switching_step = i
                break
        
        # Ensure switching step is within valid range
        if switching_step >= steps:
            switching_step = steps - 1
            
        return int(switching_step)

    def _run_sampling_stage(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        latent: Dict[str, torch.Tensor],
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        start_at_step: int,
        end_at_step: int,
        add_noise: bool,
        return_with_leftover_noise: bool,
        stage_name: str = "Sampler",
        stage_info: str = None
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Execute a single sampling stage using KSamplerAdvanced.

        Args:
            model: Model to use for sampling.
            positive: Positive conditioning.
            negative: Negative conditioning.
            latent: Input latent dictionary.
            seed: Random seed.
            steps: Total steps in schedule.
            cfg: CFG scale.
            sampler_name: Sampler algorithm name.
            scheduler: Noise scheduler name.
            start_at_step: Starting step (inclusive).
            end_at_step: Ending step (exclusive).
            add_noise: Whether to add initial noise.
            return_with_leftover_noise: Whether to return with remaining noise.
            stage_name: Stage identifier for logging.
            stage_info: Optional stage information to log right before sampling.

        Returns:
            Tuple containing the resulting latent dictionary.

        Raises:
            RuntimeError: If sampling fails.
        """
        if start_at_step >= end_at_step:
            logger.info("%s: start_at_step >= end_at_step, skipping.", stage_name)
            return (latent,)

        # Log stage info right before sampling to appear above progress bar
        if stage_info:
            stage_type = stage_name.replace("Stage 1", "Base denoising").replace("Stage 2", "Lightning high model").replace("Stage 3", "Lightning low model")
            logger.info("%s: %s - %s", stage_name, stage_type, stage_info)

        advanced_sampler = nodes.KSamplerAdvanced()
        add_noise_mode = "enable" if add_noise else "disable"
        return_noise_mode = "enable" if return_with_leftover_noise else "disable"
        
        try:
            result = advanced_sampler.sample(
                model=model,
                add_noise=add_noise_mode,
                noise_seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent,
                start_at_step=start_at_step,
                end_at_step=end_at_step,
                return_with_leftover_noise=return_noise_mode,
                denoise=1.0
            )
        except Exception as exc:
            raise RuntimeError(f"{stage_name}: sampling failed.") from exc
        
        # Add visual separator after sampling stage for better log readability
        if stage_info:
            logger.info("")
            
        return result

    def sample(
        self,
        high_model: Any,
        high_model_lx2v: Any,
        low_model_lx2v: Any,
        positive: Any,
        negative: Any,
        latent_image: Dict[str, torch.Tensor],
        seed: int,
        sigma_shift: float,
        base_steps: int,
        base_cfg: float,
        lightning_start: int,
        lightning_steps: int,
        lightning_cfg: float,
        sampler_name: str,
        scheduler: str,
        switch_strategy: str,
        switch_boundary: float = 0.875,
        switch_step: int = -1
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Execute triple-stage cascade sampling with comprehensive logging.

        Args:
            high_model: High-noise model for base denoising.
            high_model_lx2v: Lightning high model.
            low_model_lx2v: Lightning low model.
            positive: Positive prompt conditioning.
            negative: Negative prompt conditioning.
            latent_image: Input latent image.
            seed: Random seed.
            sigma_shift: Sigma shift value.
            base_steps: Number of base denoising steps (-1 for auto-calculation).
            base_cfg: CFG scale for base stage.
            lightning_start: Starting step in lightning schedule.
            lightning_steps: Total lightning steps.
            lightning_cfg: CFG scale for lightning stages (Stage 2 and 3).
            sampler_name: Sampler algorithm.
            scheduler: Noise scheduler.
            switch_strategy: Strategy for lightning model switching.
            switch_boundary: Sigma boundary for manual boundary strategy (optional).
            switch_step: Manual switch step for manual switch step strategy (optional).

        Returns:
            Tuple containing final latent dictionary.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Validate parameters
        if lightning_steps < 2:
            raise ValueError("lightning_steps must be at least 2.")
        if not (0 <= lightning_start < lightning_steps):
            raise ValueError("lightning_start must be within [0, lightning_steps-1].")
        
        # Auto-calculate base_steps if requested
        if base_steps == -1:
            multiplier = math.ceil(MIN_TOTAL_STEPS / lightning_steps)
            base_steps = lightning_start * multiplier
            logger.info("Auto-calculated base_steps = %d", base_steps)
        
        # Validate base_steps after potential auto-calculation
        if lightning_start > 0 and base_steps < 1:
            raise ValueError("base_steps must be >= 1 when lightning_start > 0.")

        # Clone and patch models with sigma shift
        patcher = self._get_model_patcher()
        shift_value = self._canonicalize_shift(sigma_shift)
        
        patched_high = patcher.patch(high_model, shift_value)[0]
        patched_high_lx2v = patcher.patch(high_model_lx2v, shift_value)[0]
        patched_low_lx2v = patcher.patch(low_model_lx2v, shift_value)[0]

        # Determine model switching strategy based on dropdown selection
        if switch_strategy == "Manual switch step":
            lightning_midpoint = switch_step if switch_step != -1 else lightning_steps // 2
            logger.info("Using manual switch step = %d", lightning_midpoint)
        elif switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
            # Select appropriate boundary value
            if switch_strategy == "T2V boundary":
                boundary_value = DEFAULT_BOUNDARY_T2V
            elif switch_strategy == "I2V boundary":
                boundary_value = DEFAULT_BOUNDARY_I2V
            else:  # Manual boundary
                boundary_value = switch_boundary
            
            sampling = patched_high_lx2v.get_model_object("model_sampling")
            lightning_midpoint = self._compute_boundary_switching_step(
                sampling, scheduler, lightning_steps, boundary_value
            )
        else:  # "50% of steps" strategy
            lightning_midpoint = math.ceil(lightning_steps / 2)
            if lightning_midpoint >= lightning_steps:
                lightning_midpoint = lightning_steps - 1

        lightning_midpoint_int = int(lightning_midpoint)

        # Determine stage execution logic
        skip_stage1 = (lightning_start == 0)
        skip_stage2 = False
        stage2_skip_reason = ""

        if lightning_start > lightning_midpoint_int:
            logger.info("Model switching: Strategy bypassed (lightning_start > switch point).")
            skip_stage2 = True
            stage2_skip_reason = "lightning_start beyond switch point"
        else:
            # Log switching strategy
            if switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
                boundary_value = DEFAULT_BOUNDARY_T2V if switch_strategy == "T2V boundary" else (
                    DEFAULT_BOUNDARY_I2V if switch_strategy == "I2V boundary" else switch_boundary
                )
                logger.info(
                    "Model switching: %s (boundary = %s) → switch at step %d of %d",
                    switch_strategy, boundary_value, lightning_midpoint_int, lightning_steps
                )
            else:
                logger.info(
                    "Model switching: %s → switch at step %d of %d",
                    switch_strategy, lightning_midpoint_int, lightning_steps
                )

            if lightning_start == lightning_midpoint_int:
                skip_stage2 = True
                stage2_skip_reason = "lightning_start equals switch point"

        # Stage 1: Base Denoising
        if skip_stage1:
            logger.info("Stage 1: Skipped (Lightning-only mode).")
            latent_after_stage1 = latent_image
            add_noise_for_stage2 = True
        else:
            total_base_steps = math.floor(base_steps * lightning_steps / max(1, lightning_start))
            total_base_steps = max(total_base_steps, base_steps)
            stage1_info = self._format_stage_range(0, base_steps, total_base_steps)
            
            latent_stage1 = self._run_sampling_stage(
                model=patched_high,
                positive=positive,
                negative=negative,
                latent=latent_image,
                seed=seed,
                steps=total_base_steps,
                cfg=base_cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=0,
                end_at_step=base_steps,
                add_noise=True,
                return_with_leftover_noise=True,
                stage_name="Stage 1",
                stage_info=stage1_info
            )
            latent_after_stage1 = latent_stage1[0]
            add_noise_for_stage2 = False

        # Stage 2: Lightning High Model
        if skip_stage2:
            logger.info("Stage 2: Skipped (%s).", stage2_skip_reason)
            latent_for_stage3 = latent_after_stage1
        else:
            stage2_info = self._format_stage_range(
                lightning_start, lightning_midpoint_int, lightning_steps
            )
            
            latent_stage2 = self._run_sampling_stage(
                model=patched_high_lx2v,
                positive=positive,
                negative=negative,
                latent=latent_after_stage1,
                seed=seed,
                steps=lightning_steps,
                cfg=lightning_cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=lightning_start,
                end_at_step=lightning_midpoint_int,
                add_noise=add_noise_for_stage2,
                return_with_leftover_noise=True,
                stage_name="Stage 2",
                stage_info=stage2_info
            )
            latent_for_stage3 = latent_stage2[0]

        # Stage 3: Lightning Low Model
        stage3_start = max(lightning_start, lightning_midpoint_int)
        stage3_info = self._format_stage_range(stage3_start, lightning_steps, lightning_steps)
        
        latent_final = self._run_sampling_stage(
            model=patched_low_lx2v,
            positive=positive,
            negative=negative,
            latent=latent_for_stage3,
            seed=seed + 1,  # Offset seed for final stage
            steps=lightning_steps,
            cfg=1.0,
            sampler_name=sampler_name,
            scheduler=scheduler,
            start_at_step=stage3_start,
            end_at_step=lightning_steps,
            add_noise=False,
            return_with_leftover_noise=False,
            stage_name="Stage 3",
            stage_info=stage3_info
        )

        return latent_final


class TripleKSamplerWan22Lightning:
    """
    Main Triple-stage KSampler node for Wan2.2 split models with Lightning LoRA.

    This node provides an optimized interface to the triple-stage sampling
    process with auto-computed parameters. Uses fixed lightning_start=1 and
    auto-computed base_steps to ensure base_steps * lightning_steps >= _MIN_TOTAL_STEPS
    for optimal quality.

    This streamlined interface exposes the most essential parameters while
    maintaining the full functionality of the advanced sampling algorithm.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define input specification for the simplified node.

        Returns:
            Dict containing required input parameters with reduced complexity.
        """
        return {
            "required": {
                "high_model": (
                    "MODEL",
                    {"tooltip": "High-noise model for base denoising."}
                ),
                "high_model_lx2v": (
                    "MODEL",
                    {"tooltip": "Lightning high model (LightX2V)."}
                ),
                "low_model_lx2v": (
                    "MODEL",
                    {"tooltip": "Lightning low model (LightX2V)."}
                ),
                "positive": (
                    "CONDITIONING",
                    {"tooltip": "Positive prompt conditioning."}
                ),
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Negative prompt conditioning."}
                ),
                "latent_image": (
                    "LATENT",
                    {"tooltip": "Latent image to denoise."}
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "tooltip": "Random seed for noise generation."
                    }
                ),
                "sigma_shift": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Sigma shift applied to model sampling."
                    }
                ),
                "base_cfg": (
                    "FLOAT",
                    {
                        "default": 3.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for Stage 1."
                    }
                ),
                "lightning_steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 100,
                        "tooltip": "Total steps for lightning stages."
                    }
                ),
                "lightning_cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "CFG scale for lightning stages (Stage 2 and 3)."
                    }
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler to use."}
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler for noise."}
                ),
                "switch_strategy": (
                    ["50% of steps", "T2V boundary", "I2V boundary"],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between lightning high and low models."
                    }
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "TripleKSampler/sampling"
    DESCRIPTION = (
        "Triple-stage sampler for Wan2.2 split models with Lightning LoRA. "
        "Optimized interface with auto-computed parameters for ease of use."
    )

    def _compute_base_steps(self, lightning_steps: int) -> int:
        """
        Compute base_steps to ensure base_steps * lightning_steps >= _MIN_TOTAL_STEPS.

        This ensures sufficient total sampling steps for quality results.

        Args:
            lightning_steps: Number of lightning steps.

        Returns:
            Computed base_steps value (minimum 1).
        """
        if lightning_steps <= 0:
            return 1
        required = math.ceil(MIN_TOTAL_STEPS / float(lightning_steps))
        return max(1, int(required))

    def _calculate_percentage(self, numerator: float, denominator: float) -> float:
        """
        Calculate percentage with single digit precision for logging.

        Args:
            numerator: Numerator value.
            denominator: Denominator value.

        Returns:
            Float percentage between 0.0 and 100.0 with one decimal place.
        """
        if denominator == 0:
            return 0.0
        percentage = (float(numerator) / float(denominator)) * 100.0
        return round(max(0.0, min(100.0, percentage)), 1)

    def sample(
        self,
        high_model: Any,
        high_model_lx2v: Any,
        low_model_lx2v: Any,
        positive: Any,
        negative: Any,
        latent_image: Dict[str, torch.Tensor],
        seed: int,
        sigma_shift: float,
        base_cfg: float,
        lightning_steps: int,
        lightning_cfg: float,
        sampler_name: str,
        scheduler: str,
        switch_strategy: str
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Execute simplified triple-stage sampling pipeline.

        Args:
            high_model: High-noise model for base denoising.
            high_model_lx2v: Lightning high model.
            low_model_lx2v: Lightning low model.
            positive: Positive prompt conditioning.
            negative: Negative prompt conditioning.
            latent_image: Input latent image.
            seed: Random seed.
            sigma_shift: Sigma shift value.
            base_cfg: CFG scale for base stage.
            lightning_steps: Total lightning steps.
            lightning_cfg: CFG scale for lightning stages.
            sampler_name: Sampler algorithm.
            scheduler: Noise scheduler.
            switch_strategy: Strategy for lightning model switching.

        Returns:
            Tuple containing final latent dictionary.
        """
        # Fixed parameters for simplified interface
        lightning_start = 1
        base_steps = self._compute_base_steps(lightning_steps)

        # Log auto-computed parameters for user feedback
        if lightning_start > 0:
            total_base_steps = math.floor(base_steps * lightning_steps / max(1, lightning_start))
            total_base_steps = max(total_base_steps, base_steps)
            pct_end = self._calculate_percentage(base_steps, total_base_steps)
            logger.info(
                "Simple node: base_steps = %d (auto-computed)."
                "Stage 1 will denoise approx. 0%%–%d%%",
                base_steps, pct_end
            )

        # Delegate to advanced implementation
        runner = TripleKSamplerWan22LightningAdvanced()
        return runner.sample(
            high_model=high_model,
            high_model_lx2v=high_model_lx2v,
            low_model_lx2v=low_model_lx2v,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            sigma_shift=sigma_shift,
            base_steps=base_steps,
            base_cfg=base_cfg,
            lightning_start=lightning_start,
            lightning_steps=lightning_steps,
            lightning_cfg=lightning_cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            switch_strategy=switch_strategy,
            switch_boundary=0.875,  # Default value for simple node
            switch_step=-1      # Auto-calculate for simple node
        )


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TripleKSamplerWan22Lightning": TripleKSamplerWan22Lightning,
    "TripleKSamplerWan22LightningAdvanced": TripleKSamplerWan22LightningAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripleKSamplerWan22Lightning": "TripleKSampler (Wan2.2-Lightning)",
    "TripleKSamplerWan22LightningAdvanced": "TripleKSampler Advanced (Wan2.2-Lightning)"
}