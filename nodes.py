"""
Triple-stage KSampler for Wan2.2 split models with Lightning LoRA.

This module implements triple-stage sampling nodes for ComfyUI,
specifically designed for Wan2.2 split models with Lightning LoRA
integration.

The sampling process includes base denoising, lightning high-model
processing, and lightning low-model refinement stages.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import comfy.samplers
import nodes
import torch
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from server import PromptServer

# Hardcoded default values
_DEFAULT_BASE_QUALITY_THRESHOLD = 20
_DEFAULT_BOUNDARY_T2V = 0.875
_DEFAULT_BOUNDARY_I2V = 0.900
_DEFAULT_LOG_LEVEL = "INFO"


def _load_config() -> Dict[str, Any]:
    """Load configuration from config.toml or fallback to defaults.

    Priority:
      1. config.toml (user editable, gitignored)
      2. config.example.toml (template tracked in git)
      3. Hardcoded defaults

    If config.toml does not exist but config.example.toml does, it will
    copy the template to config.toml to provide an editable file for users.

    Returns:
        A dict with the keys "sampling", "boundaries", and "logging".
    """
    tomllib = None
    try:
        # Python 3.11+
        import tomllib as tomllib  # type: ignore
    except Exception:
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            logger = logging.getLogger(__name__)
            logger.warning(
                "[TripleKSampler] TOML parser not available. "
                "Install 'tomli' for Python < 3.11 to enable config.toml support."
            )
            return {
                "sampling": {"base_quality_threshold": _DEFAULT_BASE_QUALITY_THRESHOLD},
                "boundaries": {
                    "default_t2v": _DEFAULT_BOUNDARY_T2V,
                    "default_i2v": _DEFAULT_BOUNDARY_I2V,
                },
                "logging": {"level": _DEFAULT_LOG_LEVEL},
            }

    script_dir = Path(__file__).resolve().parent
    user_config_path = script_dir / "config.toml"
    template_config_path = script_dir / "config.example.toml"

    # Auto-create config.toml from template if necessary
    if not user_config_path.exists() and template_config_path.exists():
        try:
            shutil.copy2(template_config_path, user_config_path)
            logging.getLogger(__name__).info(
                "[TripleKSampler] Created config.toml from template"
            )
        except (IOError, OSError) as exc:
            logging.getLogger(__name__).warning(
                "[TripleKSampler] Failed to create config.toml from template: %s", exc
            )

    # Try user config first
    if user_config_path.exists():
        try:
            with user_config_path.open("rb") as f:
                return tomllib.load(f)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[TripleKSampler] Failed to load user config.toml: %s", exc
            )

    # Try template config
    if template_config_path.exists():
        try:
            with template_config_path.open("rb") as f:
                return tomllib.load(f)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "[TripleKSampler] Failed to load config.example.toml: %s", exc
            )

    # Final fallback to hardcoded defaults
    return {
        "sampling": {"base_quality_threshold": _DEFAULT_BASE_QUALITY_THRESHOLD},
        "boundaries": {
            "default_t2v": _DEFAULT_BOUNDARY_T2V,
            "default_i2v": _DEFAULT_BOUNDARY_I2V,
        },
        "logging": {"level": _DEFAULT_LOG_LEVEL},
    }


# Load configuration at import time
_CONFIG = _load_config()

# Extract configuration values with safe defaults
_BASE_QUALITY_THRESHOLD = _CONFIG.get("sampling", {}).get(
    "base_quality_threshold", _DEFAULT_BASE_QUALITY_THRESHOLD
)
_BOUNDARY_T2V = _CONFIG.get("boundaries", {}).get("default_t2v", _DEFAULT_BOUNDARY_T2V)
_BOUNDARY_I2V = _CONFIG.get("boundaries", {}).get("default_i2v", _DEFAULT_BOUNDARY_I2V)
_LOG_LEVEL = _CONFIG.get("logging", {}).get("level", _DEFAULT_LOG_LEVEL)


def _get_log_level() -> int:
    """
    Convert LOG_LEVEL string from configuration to logging level constant.

    Only supports DEBUG and INFO levels. WARNING and ERROR messages
    are always shown regardless of LOG_LEVEL setting.
    """
    if str(_LOG_LEVEL).upper() == "DEBUG":
        return logging.DEBUG
    return logging.INFO


# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[TripleKSampler] %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(_get_log_level())

# Bare logger for separator lines (keeps output tidy)
bare_logger = logging.getLogger("TripleKSampler.separator")
if not bare_logger.handlers:
    bare_handler = logging.StreamHandler()
    bare_handler.setFormatter(logging.Formatter(""))
    bare_logger.addHandler(bare_handler)
bare_logger.propagate = False
bare_logger.setLevel(logging.INFO)


class TripleKSamplerWan22Base:
    """Base class containing shared functionality for TripleKSampler nodes."""

    # ComfyUI required attributes
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "TripleKSampler/sampling"

    @classmethod
    def _get_base_input_types(cls) -> Dict[str, Any]:
        """Return the shared INPUT_TYPES mapping used by both nodes."""
        return {
            "base_high": ("MODEL", {"tooltip": "Base high-noise model for Stage 1."}),
            "lightning_high": ("MODEL", {"tooltip": "Lightning high-noise model."}),
            "lightning_low": ("MODEL", {"tooltip": "Lightning low-noise model."}),
            "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
            "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
            "latent_image": ("LATENT", {"tooltip": "Latent image to denoise."}),
            "seed": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for noise generation.",
                },
            ),
            "sigma_shift": (
                "FLOAT",
                {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "Sigma shift applied to model sampling.",
                },
            ),
            "base_cfg": (
                "FLOAT",
                {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "CFG scale for Stage 1.",
                },
            ),
            "lightning_steps": (
                "INT",
                {
                    "default": 8,
                    "min": 2,
                    "max": 100,
                    "tooltip": "Total steps for lightning stages.",
                },
            ),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler to use."}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler to use."}),
        }

    @classmethod
    def _calculate_perfect_alignment(
        cls, base_quality_threshold: int, lightning_start: int, lightning_steps: int
    ) -> Tuple[int, int, str]:
        """Calculate base_steps and total_base_steps for perfect alignment.

        Returns:
            (base_steps, total_base_steps, method_used) where method_used is one of:
            "simple_math", "mathematical_search", "fallback".
        """
        if lightning_start == 1:
            base_steps = math.ceil(base_quality_threshold / lightning_steps)
            total_base_steps = base_steps * lightning_steps
            return base_steps, total_base_steps, "simple_math"

        # Complex case: search for integer candidate that divides cleanly
        search_limit = base_quality_threshold + lightning_steps
        for candidate_total in range(base_quality_threshold, search_limit):
            if (candidate_total * lightning_start) % lightning_steps == 0:
                base_steps = (candidate_total * lightning_start) // lightning_steps
                return base_steps, candidate_total, "mathematical_search"

        # Fallback (no exact alignment found)
        base_steps = math.ceil(base_quality_threshold * lightning_start / lightning_steps)
        optimal_total = base_steps * lightning_steps / lightning_start
        total_base_steps = max(int(math.ceil(optimal_total)), base_quality_threshold)
        return base_steps, total_base_steps, "fallback"

    def _get_model_patcher(self) -> ModelSamplingSD3:
        """Return a ModelSamplingSD3 instance used to patch models."""
        return ModelSamplingSD3()

    def _canonicalize_shift(self, value: float) -> float:
        """Normalize shift value to a Python float."""
        return float(value)

    def _calculate_percentage(self, numerator: float, denominator: float) -> float:
        """Return a percentage (0.0–100.0) rounded to one decimal place."""
        if denominator == 0:
            return 0.0
        pct = (float(numerator) / float(denominator)) * 100.0
        return round(max(0.0, min(100.0, pct)), 1)

    def _format_stage_range(self, start: int, end: int, total: int) -> str:
        """Return a human-readable string describing step ranges and denoising pct."""
        start_safe = int(max(0, start))
        end_safe = int(max(start_safe, end))
        total_safe = int(max(1, total))
        pct_start = self._calculate_percentage(start_safe, total_safe)
        pct_end = self._calculate_percentage(end_safe, total_safe)
        return f"steps {start_safe}-{end_safe} of {total_safe} (denoising {pct_start:.1f}%–{pct_end:.1f}%)"

    def _compute_boundary_switching_step(
        self, sampling: Any, scheduler: str, steps: int, boundary: float
    ) -> int:
        """Compute the switch step index from sigmas and a boundary value.

        Args:
            sampling: model_sampling object returned by the patched model.
            scheduler: scheduler name.
            steps: number of lightning steps.
            boundary: boundary value between 0 and 1.

        Returns:
            int: step index in [0, steps-1] where sigma crosses the boundary.
        """
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
        timesteps: List[float] = []

        for sigma in sigmas:
            # convert tensor-like sigma values to float timesteps (defensive)
            try:
                sigma_val = float(sigma.item())
            except Exception:
                sigma_val = float(sigma)
            timestep = sampling.timestep(sigma_val) / 1000.0
            timesteps.append(timestep)

        switching_step = steps
        for i, timestep in enumerate(timesteps[1:], start=1):
            if timestep < float(boundary):
                switching_step = i
                break

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
        stage_info: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Run a single sampling stage using KSamplerAdvanced.

        Returns:
            A tuple whose first element is the resulting latent dict.

        Raises:
            ValueError: if start_at_step >= end_at_step.
            RuntimeError: if sampling fails.
        """
        if start_at_step >= end_at_step:
            raise ValueError(
                f"{stage_name}: start_at_step ({start_at_step}) >= end_at_step ({end_at_step})."
            )

        bare_logger.info("")  # visual separator before stage execution

        if stage_info:
            stage_type = (
                stage_name.replace("Stage 1", "Base high model")
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
    """Advanced triple-stage node with full parameter control."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
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
                "base_steps": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 100,
                        "tooltip": "Number of base (Stage 1) steps. Use -1 for auto-calculation.",
                    },
                ),
                "base_cfg": base_inputs["base_cfg"],
                # Lightning parameters
                "lightning_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99,
                        "tooltip": "Start step inside lightning schedule (0 to skip Stage 1).",
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
                        "tooltip": "CFG scale for Stage 2 and Stage 3.",
                    },
                ),
                # Sampler parameters
                "sampler_name": base_inputs["sampler_name"],
                "scheduler": base_inputs["scheduler"],
                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable dry run for config testing without sampling.",
                    },
                ),
                "switch_strategy": (
                    [
                        "50% of steps",
                        "Manual switch step",
                        "T2V boundary",
                        "I2V boundary",
                        "Manual boundary",
                    ],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between lightning models.",
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
                        "tooltip": "Manual step to switch models. -1 auto-calculates.",
                    },
                ),
                "switch_boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sigma boundary for switching. 0.875 (T2V) / 0.900 (I2V).",
                    },
                ),
            },
        }

    DESCRIPTION = (
        "Advanced triple-stage cascade sampler with full parameter control for "
        "Wan2.2 split models with Lightning LoRA."
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
        dry_run: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Perform the triple-stage sampling run and return final latent tuple."""
        if str(_LOG_LEVEL).upper() == "DEBUG":
            bare_logger.info("")
        logger.debug("=== TripleKSampler Node - Input Parameters ===")
        logger.debug("Sampling: seed=%d, sigma_shift=%.3f", seed, sigma_shift)
        logger.debug("Base stage: base_steps=%d, base_cfg=%.1f", base_steps, base_cfg)
        logger.debug(
            "Lightning: lightning_start=%d, lightning_steps=%d, lightning_cfg=%.1f",
            lightning_start,
            lightning_steps,
            lightning_cfg,
        )
        logger.debug("Sampler: %s, scheduler: %s", sampler_name, scheduler)
        logger.debug("Strategy: %s", switch_strategy)
        if switch_strategy == "Manual boundary":
            logger.debug("  switch_boundary=%.3f", switch_boundary)
        elif switch_strategy == "Manual switch step":
            logger.debug("  switch_step=%d", switch_step)
        logger.debug(
            "Configuration: BASE_QUALITY_THRESHOLD=%d, DRY_RUN=%s",
            _BASE_QUALITY_THRESHOLD,
            dry_run,
        )

        # Basic validation
        if lightning_steps < 2:
            raise ValueError("lightning_steps must be at least 2.")
        if not (0 <= lightning_start < lightning_steps):
            raise ValueError("lightning_start must be within [0, lightning_steps-1].")

        if switch_strategy == "Manual switch step" and switch_step != -1:
            if switch_step < 0:
                raise ValueError(f"switch_step ({switch_step}) must be >= 0")
            if switch_step >= lightning_steps:
                raise ValueError(
                    f"switch_step ({switch_step}) must be < lightning_steps ({lightning_steps})"
                )
            if switch_step < lightning_start:
                raise ValueError(
                    f"switch_step ({switch_step}) cannot be less than lightning_start ({lightning_start}). "
                    "If you want low-noise only, set lightning_start=0 as well."
                )

        bare_logger.info("")  # separator before calculation logs

        # Calculate base_steps and total_base_steps
        optimal_total_base_steps = None
        if base_steps == -1:
            base_steps, optimal_total_base_steps, method = self._calculate_perfect_alignment(
                _BASE_QUALITY_THRESHOLD, lightning_start, lightning_steps
            )
            if lightning_start > 0:
                if method == "mathematical_search":
                    logger.info(
                        "Auto-calculated base_steps = %d, total_base_steps = %d (mathematical search)",
                        base_steps,
                        optimal_total_base_steps,
                    )
                elif method == "simple_math":
                    logger.info(
                        "Auto-calculated base_steps = %d, total_base_steps = %d (simple math)",
                        base_steps,
                        optimal_total_base_steps,
                    )
                else:
                    logger.info(
                        "Auto-calculated base_steps = %d (fallback - no perfect alignment found)",
                        base_steps,
                    )
        else:
            if lightning_start == 0 and base_steps == 0:
                optimal_total_base_steps = 0
            else:
                # manual base_steps -> compute total_base_steps for alignment checks
                optimal_total_base_steps = math.floor(base_steps * lightning_steps / max(1, lightning_start))
                optimal_total_base_steps = max(optimal_total_base_steps, base_steps)
                logger.info(
                    "Auto-calculated total_base_steps = %d for manual base_steps = %d",
                    optimal_total_base_steps,
                    base_steps,
                )
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
                        # Send toast notification via ComfyUI message system
                        PromptServer.instance.send_sync("triple_ksampler_overlap", {
                            "severity": "warn",
                            "summary": "TripleKSampler: Stage overlap",
                            "detail": f"Stage 1 and Stage 2 overlap by {overlap_pct:.1f}%. Consider base_steps=-1 or adjust lightning parameters.",
                            "life": 8000,
                        })

        if lightning_start > 0 and base_steps < 1:
            raise ValueError("base_steps must be >= 1 when lightning_start > 0.")
        if base_steps == 0 and lightning_start != 0:
            raise ValueError("base_steps = 0 is only allowed when lightning_start = 0 (Stage 1 skip mode)")

        if lightning_start == 0:
            temp_switch_step = None
            if switch_strategy == "Manual switch step":
                temp_switch_step = switch_step if switch_step != -1 else lightning_steps // 2
            elif switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
                temp_switch_step = 1
            else:
                temp_switch_step = math.ceil(lightning_steps / 2)

            if temp_switch_step == 0 and base_steps > 0:
                raise ValueError("When skipping both Stage 1 and Stage 2, base_steps must be -1 or 0")

        # Patch models (non-mutating)
        patcher = self._get_model_patcher()
        shift_value = self._canonicalize_shift(sigma_shift)
        patched_base_high = patcher.patch(base_high, shift_value)[0]
        patched_lightning_high = patcher.patch(lightning_high, shift_value)[0]
        patched_lightning_low = patcher.patch(lightning_low, shift_value)[0]

        # Determine switch step based on strategy
        if switch_strategy == "Manual switch step":
            if switch_step == -1:
                switch_step_calculated = lightning_steps // 2
                effective_strategy = "50% of steps (auto)"
            else:
                switch_step_calculated = switch_step
                effective_strategy = "Manual switch step"
        elif switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
            if switch_strategy == "T2V boundary":
                boundary_value = _BOUNDARY_T2V
            elif switch_strategy == "I2V boundary":
                boundary_value = _BOUNDARY_I2V
            else:
                boundary_value = switch_boundary

            sampling = patched_lightning_high.get_model_object("model_sampling")
            switch_step_calculated = self._compute_boundary_switching_step(
                sampling, scheduler, lightning_steps, boundary_value
            )
            effective_strategy = switch_strategy
        else:
            switch_step_calculated = math.ceil(lightning_steps / 2)
            effective_strategy = switch_strategy

        switch_step_final = int(switch_step_calculated)

        # Stage execution logic flags
        skip_stage1 = (lightning_start == 0)
        skip_stage2 = False
        stage2_skip_reason = ""

        stage1_add_noise = True
        stage2_add_noise = skip_stage1
        stage3_add_noise = False

        if lightning_start > switch_step_final:
            raise ValueError("lightning_start cannot be greater than switch_step.")
        else:
            if switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
                boundary_value = (
                    _BOUNDARY_T2V
                    if switch_strategy == "T2V boundary"
                    else _BOUNDARY_I2V
                    if switch_strategy == "I2V boundary"
                    else switch_boundary
                )
                logger.info(
                    "Model switching: %s (boundary = %s) → switch at step %d of %d",
                    effective_strategy,
                    boundary_value,
                    switch_step_final,
                    lightning_steps,
                )
            else:
                logger.info(
                    "Model switching: %s → switch at step %d of %d",
                    effective_strategy,
                    switch_step_final,
                    lightning_steps,
                )

            if lightning_start == switch_step_final:
                skip_stage2 = True
                stage2_skip_reason = "lightning_start equals switch point"

        stage3_add_noise = skip_stage1 and skip_stage2

        # Stage 1: Base denoising
        if skip_stage1:
            if base_steps > 0:
                raise ValueError(
                    "Set base_steps=0 or base_steps=-1 for Lightning-only mode, "
                    "or increase lightning_start to use base denoising."
                )
            logger.info("Lightning-only mode: base_steps not applicable (Stage 1 skipped)")
            bare_logger.info("")  # separator before skipped stage log
            logger.info("Stage 1: Skipped (Lightning-only mode)")
            stage1_output = latent_image
        else:
            if optimal_total_base_steps is not None:
                total_base_steps = optimal_total_base_steps
            else:
                _, total_base_steps, _ = self._calculate_perfect_alignment(
                    _BASE_QUALITY_THRESHOLD, lightning_start, lightning_steps
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
                stage_info=stage1_info,
            )
            stage1_output = stage1_result[0]

        # Stage 2: Lightning high
        if skip_stage2:
            bare_logger.info("")  # separator before skipped stage log
            logger.info("Stage 2: Skipped (%s)", stage2_skip_reason)
            stage2_output = stage1_output
        else:
            stage2_info = self._format_stage_range(lightning_start, switch_step_final, lightning_steps)
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
            seed=seed + 1,
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
            stage_info=stage3_info,
        )

        bare_logger.info("")  # final separator after all sampling completes
        if dry_run:
            logger.info("[DRY RUN] Complete - calculations performed, no sampling executed")
            bare_logger.info("")

        # Always return tuple as expected by RETURN_TYPES
        return stage3_result


class TripleKSamplerWan22Lightning(TripleKSamplerWan22LightningAdvanced):
    """Simplified triple-stage sampler with sensible defaults."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """Return ComfyUI INPUT_TYPES mapping for simplified node."""
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
                "base_cfg": base_inputs["base_cfg"],
                "lightning_start": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 99,
                        "tooltip": "Start step inside lightning schedule (0 to skip Stage 1).",
                    },
                ),
                "lightning_steps": base_inputs["lightning_steps"],
                # Sampler params
                "sampler_name": base_inputs["sampler_name"],
                "scheduler": base_inputs["scheduler"],
                "switch_strategy": (
                    ["50% of steps", "T2V boundary", "I2V boundary"],
                    {
                        "default": "50% of steps",
                        "tooltip": "Strategy for switching between lightning models.",
                    },
                ),
            }
        }

    DESCRIPTION = (
        "Triple-stage sampler for Wan2.2 split models with Lightning LoRA. "
        "Simplified interface with auto-calculated parameters."
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
        base_steps: int = -1,
        base_cfg: float = 3.5,
        lightning_start: int = 1,
        lightning_steps: int = 8,
        lightning_cfg: float = 1.0,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        switch_strategy: str = "50% of steps",
        switch_boundary: float = 0.875,
        switch_step: int = -1,
        dry_run: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Delegate to the advanced implementation with simplified defaults."""
        # Unused parameters intentionally deleted to make intent clear
        del base_steps, lightning_cfg, switch_boundary, switch_step, dry_run  # type: ignore
        return super().sample(
            base_high=base_high,
            lightning_high=lightning_high,
            lightning_low=lightning_low,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
            sigma_shift=sigma_shift,
            base_steps=-1,
            base_cfg=base_cfg,
            lightning_start=lightning_start,
            lightning_steps=lightning_steps,
            lightning_cfg=1.0,
            sampler_name=sampler_name,
            scheduler=scheduler,
            switch_strategy=switch_strategy,
            switch_boundary=0.875,
            switch_step=-1,
            dry_run=False,
        )


# Node registration mapping (ComfyUI expects these names)
NODE_CLASS_MAPPINGS = {
    "TripleKSamplerWan22Lightning": TripleKSamplerWan22Lightning,
    "TripleKSamplerWan22LightningAdvanced": TripleKSamplerWan22LightningAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripleKSamplerWan22Lightning": "TripleKSampler (Wan2.2-Lightning)",
    "TripleKSamplerWan22LightningAdvanced": "TripleKSampler Advanced (Wan2.2-Lightning)",
}
