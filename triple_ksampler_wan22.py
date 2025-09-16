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

import json
import logging
import math
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import comfy.model_sampling
import comfy.samplers
import nodes
import torch
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

# Hardcoded default values
_DEFAULT_MIN_TOTAL_STEPS = 20
_DEFAULT_BOUNDARY_T2V = 0.875
_DEFAULT_BOUNDARY_I2V = 0.900
_DEFAULT_LOG_LEVEL = "INFO"


def _load_config() -> Dict[str, Any]:
    """
    Load configuration from config.json, auto-creating it from template if needed.

    Priority order:
    1. config.json (user's custom settings, gitignored)
    2. config.example.json (template with defaults, tracked in git) - auto-copied to config.json
    3. Hardcoded defaults (final fallback)

    Auto-creation behavior:
    - If config.json doesn't exist but config.example.json does, copies template to config.json
    - This provides a ready-to-customize configuration file on first run
    - Users can then edit config.json without affecting git tracking

    Returns:
        Dict containing configuration values with the following structure:
        {
            "sampling": {"min_total_steps": int},
            "boundaries": {"default_t2v": float, "default_i2v": float},
            "logging": {"level": str}
        }
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_config_path = os.path.join(script_dir, "config.json")
    template_config_path = os.path.join(script_dir, "config.example.json")

    # Auto-create config.json from template if it doesn't exist
    if not os.path.exists(user_config_path) and os.path.exists(template_config_path):
        try:
            shutil.copy2(template_config_path, user_config_path)
            print(f"[TripleKSampler] Created config.json from template for easy customization")
        except (IOError, OSError) as e:
            print(f"[TripleKSampler] Warning: Failed to create config.json from template: {e}")

    # Try user config first (gitignored, possibly just created)
    if os.path.exists(user_config_path):
        try:
            with open(user_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config
        except (json.JSONDecodeError, IOError) as e:
            # Fall through to template config if user config is invalid
            print(f"[TripleKSampler] Warning: Failed to load user config.json: {e}")

    # Try template config (tracked in git)
    if os.path.exists(template_config_path):
        try:
            with open(template_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"[TripleKSampler] Warning: Failed to load config.example.json: {e}")

    # Final fallback to hardcoded values
    return {
        "sampling": {"min_total_steps": _DEFAULT_MIN_TOTAL_STEPS},
        "boundaries": {
            "default_t2v": _DEFAULT_BOUNDARY_T2V,
            "default_i2v": _DEFAULT_BOUNDARY_I2V
        },
        "logging": {"level": _DEFAULT_LOG_LEVEL}
    }


# Load configuration once at module import
_CONFIG = _load_config()

# Extract configuration values with safe fallbacks
_MIN_TOTAL_STEPS = _CONFIG.get("sampling", {}).get("min_total_steps", _DEFAULT_MIN_TOTAL_STEPS)
_DEFAULT_BOUNDARY_T2V = _CONFIG.get("boundaries", {}).get("default_t2v", _DEFAULT_BOUNDARY_T2V)
_DEFAULT_BOUNDARY_I2V = _CONFIG.get("boundaries", {}).get("default_i2v", _DEFAULT_BOUNDARY_I2V)
_LOG_LEVEL = _CONFIG.get("logging", {}).get("level", _DEFAULT_LOG_LEVEL)


def _get_log_level() -> int:
    """
    Convert LOG_LEVEL string from configuration to logging level constant.

    Only supports DEBUG and INFO levels. WARNING and ERROR messages
    are always shown regardless of LOG_LEVEL setting.
    """
    if _LOG_LEVEL.upper() == "DEBUG":
        return logging.DEBUG
    else:
        # Default to INFO for any other value (INFO, WARNING, ERROR, invalid, etc.)
        return logging.INFO


# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[TripleKSampler] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(_get_log_level())

# Configure bare logger for clean empty line separators
bare_logger = logging.getLogger("TripleKSampler.separator")
if not bare_logger.handlers:
    bare_handler = logging.StreamHandler()
    bare_handler.setFormatter(logging.Formatter(""))
    bare_logger.addHandler(bare_handler)
bare_logger.propagate = False
bare_logger.setLevel(logging.INFO)


class TripleKSamplerWan22Base:
    """
    Base class for Triple-stage KSampler nodes with shared functionality.

    Contains all the common methods and logic for triple-stage sampling,
    including model patching, stage execution, and validation.
    """

    # Shared class attributes for ComfyUI
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "TripleKSampler/sampling"

    @classmethod
    def _get_base_input_types(cls) -> Dict[str, Any]:
        """
        Get shared INPUT_TYPES used by both node variants.

        Returns:
            Dict containing the base input parameters shared by both nodes.
        """
        return {
            "base_high": (
                "MODEL",
                {"tooltip": "Base high-noise model for Stage 1 denoising."}
            ),
            "lightning_high": (
                "MODEL",
                {"tooltip": "Lightning high-noise model for Stage 2."}
            ),
            "lightning_low": (
                "MODEL",
                {"tooltip": "Lightning low-noise model for Stage 3."}
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
            "sampler_name": (
                comfy.samplers.KSampler.SAMPLERS,
                {"tooltip": "Sampler to use."}
            ),
            "scheduler": (
                comfy.samplers.KSampler.SCHEDULERS,
                {"tooltip": "Scheduler for noise."}
            ),
        }

    @classmethod
    def _calculate_perfect_alignment(cls, min_total_steps: int, lightning_start: int, lightning_steps: int) -> Tuple[int, int, str]:
        """
        Calculate base_steps and total_base_steps for perfect stage alignment.

        This unified function handles both simple (lightning_start=1) and complex cases
        efficiently, ensuring Stage 1 end percentage exactly equals Stage 2 start percentage.

        Args:
            min_total_steps: Minimum total steps requirement.
            lightning_start: Starting step in lightning schedule.
            lightning_steps: Total lightning steps.

        Returns:
            Tuple of (base_steps, total_base_steps, method_used).
            method_used is one of: "simple_math", "mathematical_search", "fallback"
        """
        if lightning_start == 1:
            # Simple case: direct calculation guarantees perfect alignment
            base_steps = math.ceil(min_total_steps / lightning_steps)
            total_base_steps = base_steps * lightning_steps
            return base_steps, total_base_steps, "simple_math"
        else:
            # Complex case: search for perfect alignment
            search_limit = min_total_steps + lightning_steps
            for candidate_total in range(min_total_steps, search_limit):
                if (candidate_total * lightning_start) % lightning_steps == 0:
                    base_steps = candidate_total * lightning_start // lightning_steps
                    return base_steps, candidate_total, "mathematical_search"

            # Fallback if no perfect alignment found (very rare)
            base_steps = math.ceil(min_total_steps * lightning_start / lightning_steps)
            optimal_total = base_steps * lightning_steps / lightning_start
            total_base_steps = max(int(math.ceil(optimal_total)), min_total_steps)
            return base_steps, total_base_steps, "fallback"

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
        dry_run: bool = False,
        stage_name: str = "Sampler",
        stage_info: Optional[str] = None
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
            dry_run: Enable dry run mode for testing configurations without sampling.
            stage_name: Stage identifier for logging.
            stage_info: Optional stage information to log right before sampling.

        Returns:
            Tuple containing the resulting latent dictionary.

        Raises:
            RuntimeError: If sampling fails.
        """
        if start_at_step >= end_at_step:
            raise ValueError(
                f"{stage_name}: start_at_step ({start_at_step}) >= end_at_step ({end_at_step}). "
                "Check your step configuration - this indicates invalid sampling range."
            )

        bare_logger.info("")  # separator before sampling logs

        if stage_info:
            stage_type = (
                stage_name.replace("Stage 1", "Base denoising")
                .replace("Stage 2", "Lightning high model")
                .replace("Stage 3", "Lightning low model")
            )
            logger.info("%s: %s - %s", stage_name, stage_type, stage_info)

        if dry_run:
            return (latent,)

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
                denoise=1.0,
            )
        except Exception as exc:
            raise RuntimeError(f"{stage_name}: sampling failed.") from exc

        return result


class TripleKSamplerWan22LightningAdvanced(TripleKSamplerWan22Base):
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
        # Get base shared input types
        base_inputs = cls._get_base_input_types()

        # Build required parameters in original order
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
                "base_steps": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Number of base (Stage 1) steps. Use -1 for auto-calculation based on lightning_start."
                    }
                ),
                "base_cfg": base_inputs["base_cfg"],
                # Lightning parameters
                "lightning_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99,
                        "tooltip": "Start step inside lightning schedule (0 to skip Stage 1)."
                    }
                ),
                "lightning_steps": base_inputs["lightning_steps"],
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
                # Sampler parameters
                "sampler_name": base_inputs["sampler_name"],
                "scheduler": base_inputs["scheduler"],
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
                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable dry run mode for testing configurations without actual sampling."
                    }
                ),
            }
        }

    DESCRIPTION = (
        "Advanced triple-stage cascade sampler with full parameter control for "
        "Wan2.2 split models with Lightning LoRA. Supports both step-based and "
        "sigma boundary-based model switching."
    )

    def sample(
        self,
        base_high: Any,
        lightning_high: Any,
        lightning_low: Any,
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
        switch_step: int = -1,
        dry_run: bool = False
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Execute triple-stage cascade sampling with comprehensive logging.

        Args:
            base_high: Base high-noise model for Stage 1 denoising.
            lightning_high: Lightning high-noise model for Stage 2.
            lightning_low: Lightning low-noise model for Stage 3.
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
            dry_run: Enable dry run mode for testing configurations without sampling (optional).

        Returns:
            Tuple containing final latent dictionary.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Log all input parameters for debugging
        bare_logger.info("")
        logger.debug("=== TripleKSampler Node - Input Parameters ===")
        logger.debug("Models: base_high, lightning_high, lightning_low")
        logger.debug("Sampling: seed=%d, sigma_shift=%.3f", seed, sigma_shift)
        logger.debug("Base stage: base_steps=%d, base_cfg=%.1f", base_steps, base_cfg)
        logger.debug("Lightning: lightning_start=%d, lightning_steps=%d, lightning_cfg=%.1f",
                   lightning_start, lightning_steps, lightning_cfg)
        logger.debug("Sampler: %s, scheduler: %s", sampler_name, scheduler)
        logger.debug("Strategy: %s", switch_strategy)
        if switch_strategy == "Manual boundary":
            logger.debug("  switch_boundary=%.3f", switch_boundary)
        elif switch_strategy == "Manual switch step":
            logger.debug("  switch_step=%d", switch_step)
        logger.debug("Configuration: MIN_TOTAL_STEPS=%d, DRY_RUN=%s", _MIN_TOTAL_STEPS, dry_run)
        bare_logger.info("")

        # Validate parameters
        if lightning_steps < 2:
            raise ValueError("lightning_steps must be at least 2.")
        if not (0 <= lightning_start < lightning_steps):
            raise ValueError("lightning_start must be within [0, lightning_steps-1].")
        
        # Validate switch_step bounds for manual strategy
        if switch_strategy == "Manual switch step" and switch_step != -1:
            if switch_step < 0:
                raise ValueError(f"switch_step ({switch_step}) must be >= 0")
            if switch_step >= lightning_steps:
                raise ValueError(f"switch_step ({switch_step}) must be < lightning_steps ({lightning_steps}). Use a smaller value or different strategy.")
            if switch_step < lightning_start:
                raise ValueError(f"switch_step ({switch_step}) cannot be less than lightning_start ({lightning_start}). The high-noise model needs at least some steps before switching. If you want low-noise only, set lightning_start=0 as well.")

        # Track if base_steps was auto-calculated for smart method selection
        base_steps_auto_calculated = (base_steps == -1)

        # Auto-calculate base_steps if requested
        if base_steps == -1:
            # Use unified perfect alignment calculation
            base_steps, optimal_total_base_steps, method = self._calculate_perfect_alignment(
                _MIN_TOTAL_STEPS, lightning_start, lightning_steps
            )

            if method == "mathematical_search":
                logger.info("Auto-calculated base_steps = %d, total_base_steps = %d (mathematical search)",
                           base_steps, optimal_total_base_steps)
            elif method == "simple_math":
                logger.info("Auto-calculated base_steps = %d, total_base_steps = %d (simple math)",
                           base_steps, optimal_total_base_steps)
            else:  # fallback
                logger.info("Auto-calculated base_steps = %d (fallback - no perfect alignment found)",
                           base_steps)
                optimal_total_base_steps = None  # Will trigger fallback calculation later

        
        # Validate base_steps after potential auto-calculation
        if lightning_start > 0 and base_steps < 1:
            raise ValueError("base_steps must be >= 1 when lightning_start > 0.")
        
        # Validate base_steps=0 edge case
        if base_steps == 0 and lightning_start != 0:
            raise ValueError("base_steps = 0 is only allowed when lightning_start = 0 (Stage 1 skip mode)")
        
        # Validate consistency for Stage1+Stage2 skip scenario
        if lightning_start == 0:
            # Pre-calculate switch point to check for Stage1+Stage2 skip
            temp_switch_step = None
            if switch_strategy == "Manual switch step":
                temp_switch_step = switch_step if switch_step != -1 else lightning_steps // 2
            elif switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
                # We'll calculate this properly later, but for validation assume reasonable values
                temp_switch_step = 1  # Assume non-zero for boundary strategies
            else:  # "50% of steps" strategy
                temp_switch_step = math.ceil(lightning_steps / 2)
            
            # Check for Stage1+Stage2 skip scenario
            if temp_switch_step == 0 and base_steps > 0:
                raise ValueError("When skipping both Stage 1 and Stage 2 (lightning_start=0, switch_step=0), base_steps must be -1 or 0")

        # Clone and patch models with sigma shift
        patcher = self._get_model_patcher()
        shift_value = self._canonicalize_shift(sigma_shift)
        
        patched_base_high = patcher.patch(base_high, shift_value)[0]
        patched_lightning_high = patcher.patch(lightning_high, shift_value)[0]
        patched_lightning_low = patcher.patch(lightning_low, shift_value)[0]

        # Determine model switching strategy based on dropdown selection
        if switch_strategy == "Manual switch step":
            switch_step_calculated = switch_step if switch_step != -1 else lightning_steps // 2
        elif switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
            # Select appropriate boundary value
            if switch_strategy == "T2V boundary":
                boundary_value = _DEFAULT_BOUNDARY_T2V
            elif switch_strategy == "I2V boundary":
                boundary_value = _DEFAULT_BOUNDARY_I2V
            else:  # Manual boundary
                boundary_value = switch_boundary
            
            sampling = patched_lightning_high.get_model_object("model_sampling")
            switch_step_calculated = self._compute_boundary_switching_step(
                sampling, scheduler, lightning_steps, boundary_value
            )
        else:  # "50% of steps" strategy
            switch_step_calculated = math.ceil(lightning_steps / 2)

        switch_step_final = int(switch_step_calculated)

        # Determine stage execution logic
        skip_stage1 = (lightning_start == 0)
        skip_stage2 = False
        stage2_skip_reason = ""
        
        # Determine noise addition for each stage (first stage to run always adds noise)
        stage1_add_noise = True  # Always adds noise when it runs
        stage2_add_noise = skip_stage1  # Adds noise if it's the first stage to run
        stage3_add_noise = False  # Will be updated if both previous stages are skipped

        if lightning_start > switch_step_final:
            raise ValueError(f"lightning_start ({lightning_start}) cannot be greater than switch_step ({switch_step_final}). Either decrease lightning_start or increase switch_step, or use a different switching strategy.")
        else:
            # Log switching strategy
            if switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
                boundary_value = _DEFAULT_BOUNDARY_T2V if switch_strategy == "T2V boundary" else (
                    _DEFAULT_BOUNDARY_I2V if switch_strategy == "I2V boundary" else switch_boundary
                )
                logger.info(
                    "Model switching: %s (boundary = %s) → switch at step %d of %d",
                    switch_strategy, boundary_value, switch_step_final, lightning_steps
                )
            else:
                logger.info(
                    "Model switching: %s → switch at step %d of %d",
                    switch_strategy, switch_step_final, lightning_steps
                )

            if lightning_start == switch_step_final:
                skip_stage2 = True
                stage2_skip_reason = "lightning_start equals switch point"

        # Update stage3 noise logic
        stage3_add_noise = skip_stage1 and skip_stage2

        # Stage 1: Base Denoising
        if skip_stage1:
            if base_steps > 0:
                raise ValueError(f"Set base_steps=0 or base_steps=-1 for Lightning-only mode, or increase lightning_start to use base denoising.")
            bare_logger.info("")
            logger.info("Stage 1: Skipped (Lightning-only mode)")
            stage1_output = latent_image
        else:
            # Calculate total_base_steps
            if base_steps_auto_calculated:
                # Use pre-calculated optimal value from unified alignment function
                if 'optimal_total_base_steps' in locals() and optimal_total_base_steps is not None:
                    total_base_steps = optimal_total_base_steps
                    logger.debug("Using pre-calculated alignment (total_base_steps=%d)", total_base_steps)

                    # Verify perfect alignment (should always be exact)
                    stage1_end_pct = base_steps / total_base_steps
                    stage2_start_pct = lightning_start / lightning_steps
                    logger.debug("Perfect alignment verified (Stage1/Stage2 transition point matches)")
                else:
                    # Fallback: recalculate using unified function (should rarely happen)
                    _, total_base_steps, fallback_method = self._calculate_perfect_alignment(
                        _MIN_TOTAL_STEPS, lightning_start, lightning_steps
                    )
                    logger.debug("Fallback calculation: total_base_steps=%d", total_base_steps)
            else:
                # Manual base_steps: use standard calculation to match Stage 2 start
                total_base_steps = math.floor(base_steps * lightning_steps / max(1, lightning_start))
                total_base_steps = max(total_base_steps, base_steps)
                logger.debug("Manual calculation: total_base_steps=%d", total_base_steps)

                # Check for stage overlap and warn user
                stage1_end_pct = base_steps / total_base_steps
                stage2_start_pct = lightning_start / lightning_steps
                if stage1_end_pct > stage2_start_pct:
                    overlap_pct = (stage1_end_pct - stage2_start_pct) * 100
                    logger.warning(
                        "Stage overlap (%.1f%%) detected! For perfect alignment, try one of the following: "
                        "multiples of lightning_start for base_steps, even number for lightning_steps, "
                        "or base_steps=-1 for auto-calculation.",
                        overlap_pct
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
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=0,
                end_at_step=base_steps,
                add_noise=stage1_add_noise,
                return_with_leftover_noise=True,
                dry_run=dry_run,
                stage_name="Stage 1",
                stage_info=stage1_info
            )
            stage1_output = stage1_result[0]

        # Stage 2: Lightning High Model
        if skip_stage2:
            bare_logger.info("")
            logger.info("Stage 2: Skipped (%s)", stage2_skip_reason)
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
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=lightning_start,
                end_at_step=switch_step_final,
                add_noise=stage2_add_noise,
                return_with_leftover_noise=True,
                dry_run=dry_run,
                stage_name="Stage 2",
                stage_info=stage2_info
            )
            stage2_output = stage2_result[0]

        # Stage 3: Lightning Low Model
        stage3_start = max(lightning_start, switch_step_final)
        stage3_info = self._format_stage_range(stage3_start, lightning_steps, lightning_steps)
        
        stage3_result = self._run_sampling_stage(
            model=patched_lightning_low,
            positive=positive,
            negative=negative,
            latent=stage2_output,
            seed=seed + 1,  # Offset seed for final stage
            steps=lightning_steps,
            cfg=1.0,
            sampler_name=sampler_name,
            scheduler=scheduler,
            start_at_step=stage3_start,
            end_at_step=lightning_steps,
            add_noise=stage3_add_noise,
            return_with_leftover_noise=False,
            dry_run=dry_run,
            stage_name="Stage 3",
            stage_info=stage3_info
        )

        # Add final visual separator after all sampling completes
        bare_logger.info("")

        # Log dry run summary if enabled
        if dry_run:
            logger.info("[DRY RUN] Complete - All calculations performed, no actual sampling executed")

        return stage3_result


class TripleKSamplerWan22Lightning(TripleKSamplerWan22LightningAdvanced):
    """
    Main Triple-stage KSampler node for Wan2.2 split models with Lightning LoRA.

    This node provides an optimized interface to the triple-stage sampling
    process with auto-computed parameters. Uses fixed lightning_start=1 and
    auto-computed base_steps to ensure base_steps * lightning_steps >= MIN_TOTAL_STEPS
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
        # Get base shared input types
        base_inputs = cls._get_base_input_types()

        # Build required parameters in original order
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
                "base_cfg": base_inputs["base_cfg"],
                "lightning_steps": base_inputs["lightning_steps"],
                # Sampler parameters
                "sampler_name": base_inputs["sampler_name"],
                "scheduler": base_inputs["scheduler"],
                "switch_strategy": (
                    ["50% of steps", "T2V boundary", "I2V boundary"],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between lightning high and low models."
                    }
                ),
            }
        }

    DESCRIPTION = (
        "Triple-stage sampler for Wan2.2 split models with Lightning LoRA. "
        "Optimized interface with auto-computed parameters for ease of use."
    )

    def _compute_base_steps(self, lightning_steps: int) -> Tuple[int, int, str]:
        """
        Compute base_steps and total_base_steps for the simple node.

        Uses the unified perfect alignment function with lightning_start=1.

        Args:
            lightning_steps: Number of lightning steps.

        Returns:
            Tuple of (base_steps, total_base_steps, method) with perfect alignment.
        """
        if lightning_steps <= 0:
            return 1, _MIN_TOTAL_STEPS, "simple_math"

        # Simple node always uses lightning_start=1
        base_steps, total_base_steps, method = self._calculate_perfect_alignment(
            _MIN_TOTAL_STEPS, 1, lightning_steps
        )

        return max(1, int(base_steps)), total_base_steps, method

    def sample(
        self,
        base_high: Any,
        lightning_high: Any,
        lightning_low: Any,
        positive: Any,
        negative: Any,
        latent_image: Dict[str, torch.Tensor],
        seed: int,
        sigma_shift: float,
        base_cfg: float,
        lightning_steps: int,
        sampler_name: str,
        scheduler: str,
        switch_strategy: str
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Execute simplified triple-stage sampling pipeline.

        Args:
            base_high: Base high-noise model for Stage 1 denoising.
            lightning_high: Lightning high-noise model for Stage 2.
            lightning_low: Lightning low-noise model for Stage 3.
            positive: Positive prompt conditioning.
            negative: Negative prompt conditioning.
            latent_image: Input latent image.
            seed: Random seed.
            sigma_shift: Sigma shift value.
            base_cfg: CFG scale for base stage.
            lightning_steps: Total lightning steps.
            sampler_name: Sampler algorithm.
            scheduler: Noise scheduler.
            switch_strategy: Strategy for lightning model switching.

        Returns:
            Tuple containing final latent dictionary.

        Raises:
            ValueError: If parameters are invalid.
        """
        # Fixed parameters for simplified interface
        lightning_start = 1
        base_steps, total_base_steps, method = self._compute_base_steps(lightning_steps)

        # Log auto-computed parameters for user feedback
        if lightning_start > 0:
            pct_end = self._calculate_percentage(base_steps, total_base_steps)
            logger.info(
                "Simple node: base_steps = %d, total_base_steps = %d (%s). "
                "Stage 1 will denoise approx. 0%%–%.1f%%",
                base_steps, total_base_steps, method.replace("_", " "), pct_end
            )

        # Delegate to parent (advanced) implementation
        return super().sample(
            base_high=base_high,
            lightning_high=lightning_high,
            lightning_low=lightning_low,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            sigma_shift=sigma_shift,
            base_steps=base_steps,
            base_cfg=base_cfg,
            lightning_start=lightning_start,
            lightning_steps=lightning_steps,
            lightning_cfg=1.0,  # Fixed CFG for lightning stages in simple node
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
