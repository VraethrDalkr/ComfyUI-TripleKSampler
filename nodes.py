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

# Conditional server import for testing compatibility
try:
    # Check if we're in testing mode
    import os
    if os.environ.get('COMFYUI_TESTING', '0') == '1':
        # Skip server import in testing mode
        PromptServer = None
    else:
        from server import PromptServer
except ImportError:
    # Fallback if server import fails
    PromptServer = None

# Configuration default values
_DEFAULT_BASE_QUALITY_THRESHOLD = 20
_DEFAULT_BOUNDARY_T2V = 0.875
_DEFAULT_BOUNDARY_I2V = 0.900
_DEFAULT_LOG_LEVEL = "INFO"

# Algorithm constants
STAGE3_SEED_OFFSET = 1  # Ensure Stage 3 uses different noise pattern
SIMPLE_NODE_LIGHTNING_CFG = 1.0  # Fixed CFG value for Simple node's lightning stages
MIN_LATENT_HEIGHT = 1   # Minimal latent height for dry run
MIN_LATENT_WIDTH = 8    # Minimal latent width for dry run (8x8 for VAE compatibility)
TOAST_LIFE_OVERLAP = 8000    # Toast notification duration for overlap warnings (ms)
TOAST_LIFE_DRY_RUN = 12000   # Toast notification duration for dry run results (ms)
SEARCH_LIMIT_MULTIPLIER = 1  # Additional steps to search beyond threshold for alignment


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
    """Convert LOG_LEVEL string from configuration to logging level constant.

    Only supports DEBUG and INFO levels. WARNING and ERROR messages
    are always shown regardless of LOG_LEVEL setting.

    Returns:
        int: logging level constant (DEBUG or INFO)
    """
    if str(_LOG_LEVEL).upper() == "DEBUG":
        return logging.DEBUG
    return logging.INFO


# ============================================================================
# Logging Configuration
# ============================================================================
#
# TripleKSampler uses a dual-logger system:
# 1. Main logger: Structured messages with [TripleKSampler] prefix
# 2. Bare logger: Clean visual separators without prefixes
#
# The log level is controlled by config.toml [logging] level setting:
# - "DEBUG": Shows all messages including detailed calculations
# - "INFO": Shows essential workflow information only
# - WARNING and ERROR always appear regardless of level
# ============================================================================

# Configure main logger for structured messages
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = "[TripleKSampler] %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
logger.propagate = False
logger.setLevel(_get_log_level())

# Configure bare logger for visual separators (clean empty lines)
bare_logger = logging.getLogger("TripleKSampler.separator")
if not bare_logger.handlers:
    bare_handler = logging.StreamHandler()
    bare_handler.setFormatter(logging.Formatter(""))  # No formatting for clean output
    bare_logger.addHandler(bare_handler)
bare_logger.propagate = False
bare_logger.setLevel(logging.INFO)  # Always show separators


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
                    "tooltip": "Sigma adjustment applied via ModelSamplingSD3 for model sampling.",
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

        Perfect alignment ensures Stage 1 end exactly matches Stage 2 start in the
        denoising schedule. This prevents gaps or overlaps between stages.

        Returns:
            (base_steps, total_base_steps, method_used) where method_used is one of:
            "simple_math", "mathematical_search", "fallback".
        """
        if lightning_start == 1:
            # Simple case: lightning starts at step 1, direct calculation possible
            # Formula: base_steps/total_base_steps = lightning_start/lightning_steps
            base_steps = math.ceil(base_quality_threshold / lightning_steps)
            total_base_steps = base_steps * lightning_steps
            return base_steps, total_base_steps, "simple_math"

        # Complex case: lightning_start > 1, need perfect integer alignment
        # Search for total_base_steps where (total_base_steps * lightning_start) is divisible by lightning_steps
        search_limit = base_quality_threshold + (lightning_steps * SEARCH_LIMIT_MULTIPLIER)
        for candidate_total in range(base_quality_threshold, search_limit):
            if (candidate_total * lightning_start) % lightning_steps == 0:
                base_steps = (candidate_total * lightning_start) // lightning_steps
                return base_steps, candidate_total, "mathematical_search"

        # Fallback: no perfect alignment found within search range
        # Use approximation to get as close as possible to optimal alignment
        base_steps = math.ceil(base_quality_threshold * lightning_start / lightning_steps)
        optimal_total = base_steps * lightning_steps / lightning_start
        total_base_steps = max(int(math.ceil(optimal_total)), base_quality_threshold)
        return base_steps, total_base_steps, "fallback"

    def _get_model_patcher(self) -> ModelSamplingSD3:
        """Return a ModelSamplingSD3 instance for sigma shift patching."""
        return ModelSamplingSD3()

    def _canonicalize_shift(self, value: float) -> float:
        """Convert shift value to Python float for consistent processing."""
        return float(value)

    def _calculate_percentage(self, numerator: float, denominator: float) -> float:
        """Calculate percentage with division-by-zero protection and bounds clamping."""
        if denominator == 0:
            return 0.0
        pct = (float(numerator) / float(denominator)) * 100.0
        return round(max(0.0, min(100.0, pct)), 1)

    def _format_stage_range(self, start: int, end: int, total: int) -> str:
        """Return a human-readable string describing step ranges and denoising percentages.

        Creates informative log messages showing both step ranges and corresponding
        denoising percentages for each sampling stage.

        Args:
            start: Starting step number
            end: Ending step number
            total: Total steps in the schedule

        Returns:
            str: Formatted string like "steps 0-5 of 20 (denoising 0.0%-25.0%)"
        """
        start_safe = int(max(0, start))
        end_safe = int(max(start_safe, end))
        total_safe = int(max(1, total))
        pct_start = self._calculate_percentage(start_safe, total_safe)
        pct_end = self._calculate_percentage(end_safe, total_safe)
        return f"steps {start_safe}-{end_safe} of {total_safe} (denoising {pct_start:.1f}%–{pct_end:.1f}%)"

    def _format_base_calculation_compact(self, base_calc_info: str) -> str:
        """Format base calculation info for compact toast display.

        Converts verbose calculation log messages into concise format suitable
        for UI toast notifications. Handles different calculation scenarios
        (auto-calculated, manual, fallback).

        Args:
            base_calc_info: Original verbose calculation message

        Returns:
            str: Compact formatted message for toast display
        """
        import re

        # Pattern: "Auto-calculated base_steps = X, total_base_steps = Y (method)"
        match1 = re.search(r'Auto-calculated base_steps = (\d+), total_base_steps = (\d+) \(([^)]+)\)', base_calc_info)
        if match1:
            base_steps, total_steps, method = match1.groups()
            return f"Base steps: {base_steps}, Total: {total_steps} ({method})"

        # Pattern: "Auto-calculated base_steps = X (fallback - no perfect alignment found)"
        match2 = re.search(r'Auto-calculated base_steps = (\d+) \(([^)]+)\)', base_calc_info)
        if match2:
            base_steps, _ = match2.groups()  # method_desc not used but preserved for clarity
            return f"Base steps: {base_steps} (fallback)"

        # Pattern: "Auto-calculated total_base_steps = X for manual base_steps = Y"
        match3 = re.search(r'Auto-calculated total_base_steps = (\d+) for manual base_steps = (\d+)', base_calc_info)
        if match3:
            total_steps, manual_steps = match3.groups()
            return f"Base steps: {manual_steps}, Total: {total_steps} (manual)"

        # Fallback: return original if no pattern matches
        return base_calc_info

    def _format_switch_info_compact(self, switch_info: str) -> str:
        """Format model switching info for compact toast display.

        Converts verbose switching strategy log messages into concise format
        suitable for UI toast notifications. Handles boundary-based and
        step-based switching strategies.

        Args:
            switch_info: Original verbose switching message

        Returns:
            str: Compact formatted message for toast display
        """
        import re

        # Pattern: "Model switching: STRATEGY (boundary = VALUE) → switch at step X of Y"
        match1 = re.search(r'Model switching: ([^(]+) \(boundary = ([^)]+)\) → switch at step (\d+) of (\d+)', switch_info)
        if match1:
            strategy, _, switch_step, total_steps = match1.groups()  # boundary not used but preserved
            return f"Switch: {strategy.strip()} → step {switch_step} of {total_steps}"

        # Pattern: "Model switching: STRATEGY → switch at step X of Y"
        match2 = re.search(r'Model switching: ([^→]+) → switch at step (\d+) of (\d+)', switch_info)
        if match2:
            strategy, switch_step, total_steps = match2.groups()
            return f"Switch: {strategy.strip()} → step {switch_step} of {total_steps}"

        # Fallback: return original if no pattern matches
        return switch_info

    def _send_dry_run_notification(
        self,
        stage1_info: str,
        stage2_info: str,
        stage3_info: str,
        base_calculation_info: str = "",
        model_switching_info: str = "",
    ) -> None:
        """Send a toast notification summarizing dry run results."""
        # Format the summary with calculated insights and stage information
        summary_lines = []

        # Add calculation insights if available
        if base_calculation_info or model_switching_info:
            summary_lines.append("Calculations:")

            if base_calculation_info:
                # Transform verbose base calculation info into compact format
                formatted_base = self._format_base_calculation_compact(base_calculation_info)
                summary_lines.append(f"• {formatted_base}")

            if model_switching_info:
                # Transform verbose model switching info into compact format
                formatted_switch = self._format_switch_info_compact(model_switching_info)
                summary_lines.append(f"• {formatted_switch}")

            summary_lines.append("")

        # Always add stage configuration
        summary_lines.extend([
            "Stage Configuration:",
            f"• {stage1_info}",
            f"• {stage2_info}",
            f"• {stage3_info}",
        ])

        detail_text = "\n".join(summary_lines)

        # Send toast notification via ComfyUI message system (if available)
        try:
            if PromptServer and hasattr(PromptServer, 'instance') and PromptServer.instance:
                PromptServer.instance.send_sync("triple_ksampler_dry_run", {
                    "severity": "info",
                    "summary": "TripleKSampler: Dry Run Complete",
                    "detail": detail_text,
                    "life": TOAST_LIFE_DRY_RUN,
                })
        except Exception:
            # Silently fail if PromptServer is not available (e.g., during testing)
            pass

    def _compute_boundary_switching_step(
        self, sampling: Any, scheduler: str, steps: int, boundary: float
    ) -> int:
        """Compute the switch step index from sigmas and a boundary value.

        This method implements sigma-based model switching, which is more accurate
        than step-based switching because it accounts for the actual noise schedule.
        Different schedulers have different sigma curves, so this ensures consistent
        switching behavior across schedulers.

        Args:
            sampling: model_sampling object returned by the patched model.
            scheduler: scheduler name.
            steps: number of lightning steps.
            boundary: boundary value between 0 and 1 (normalized timestep).

        Returns:
            int: step index in [0, steps-1] where sigma crosses the boundary.
        """
        # Calculate the sigma schedule for the given steps and scheduler
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
        timesteps: List[float] = []

        # Convert sigmas to normalized timesteps (0-1 range)
        for sigma in sigmas:
            # Handle both tensor and scalar sigma values defensively
            try:
                sigma_val = float(sigma.item())
            except Exception:
                sigma_val = float(sigma)
            # Convert to normalized timestep: sampling.timestep() returns 0-1000, normalize to 0-1
            timestep = sampling.timestep(sigma_val) / 1000.0
            timesteps.append(timestep)

        # Find the first step where timestep drops below the boundary
        # This identifies the transition point in the denoising schedule
        switching_step = steps
        for i, timestep in enumerate(timesteps[1:], start=1):
            if timestep < float(boundary):
                switching_step = i
                break

        # Ensure switching step is within valid range
        if switching_step >= steps:
            switching_step = steps - 1

        return int(switching_step)

    def _validate_basic_parameters(
        self,
        lightning_steps: int,
        lightning_start: int,
        switch_strategy: str,
        switch_step: int,
    ) -> None:
        """Validate basic input parameters that don't depend on auto-calculated values.

        Args:
            lightning_steps: Total steps for lightning stages
            lightning_start: Starting step within lightning schedule
            switch_strategy: Strategy for model switching
            switch_step: Manual switch step (when using manual strategy)

        Raises:
            ValueError: if any parameter validation fails
        """
        # Basic lightning parameters validation
        if lightning_steps < 2:
            raise ValueError("lightning_steps must be at least 2.")
        if not (0 <= lightning_start < lightning_steps):
            raise ValueError("lightning_start must be within [0, lightning_steps-1].")

        # Manual switch step validation
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

    def _validate_resolved_parameters(
        self,
        lightning_start: int,
        base_steps: int,
    ) -> None:
        """Validate parameters after auto-calculation has resolved base_steps.

        Args:
            lightning_start: Starting step within lightning schedule
            base_steps: Resolved steps for base model (no longer -1)

        Raises:
            ValueError: if resolved parameter validation fails
        """
        # Base steps and lightning_start relationship validation (after auto-calculation)
        if lightning_start > 0 and base_steps < 1:
            raise ValueError("base_steps must be >= 1 when lightning_start > 0.")
        if base_steps == 0 and lightning_start != 0:
            raise ValueError("base_steps = 0 is only allowed when lightning_start = 0 (Stage 1 skip mode)")

    def _validate_special_modes(
        self,
        lightning_start: int,
        lightning_steps: int,
        base_steps: int,
        switch_strategy: str,
        switch_step: int,
    ) -> None:
        """Validate special mode configurations (lightning-only, skip modes).

        Args:
            lightning_start: Starting step within lightning schedule
            lightning_steps: Total steps for lightning stages
            base_steps: Steps for base model
            switch_strategy: Strategy for model switching
            switch_step: Manual switch step

        Raises:
            ValueError: if special mode configuration is invalid
        """
        # Lightning-only mode validation
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

        # Stage 1 skip mode enforcement
        if lightning_start == 0 and base_steps > 0:
            raise ValueError(
                "Set base_steps=0 or base_steps=-1 for Lightning-only mode, "
                "or increase lightning_start to use base denoising."
            )

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
            exc_msg = str(exc).strip()
            if exc_msg:
                raise RuntimeError(f"{stage_name}: sampling failed - {type(exc).__name__}: {exc_msg}") from exc
            else:
                raise RuntimeError(f"{stage_name}: sampling failed - {type(exc).__name__}") from exc

        return result

    def _patch_models_for_sampling(
        self,
        base_high: Any,
        lightning_high: Any,
        lightning_low: Any,
        sigma_shift: float,
    ) -> Tuple[Any, Any, Any]:
        """Patch all models with sigma shift for sampling.

        Args:
            base_high: Base high-noise model
            lightning_high: Lightning high-noise model
            lightning_low: Lightning low-noise model
            sigma_shift: Sigma adjustment value

        Returns:
            Tuple of (patched_base_high, patched_lightning_high, patched_lightning_low)
        """
        # Apply sigma shift using ModelSamplingSD3 (non-mutating)
        patcher = self._get_model_patcher()
        shift_value = self._canonicalize_shift(sigma_shift)

        patched_base_high = patcher.patch(base_high, shift_value)[0]
        patched_lightning_high = patcher.patch(lightning_high, shift_value)[0]
        patched_lightning_low = patcher.patch(lightning_low, shift_value)[0]

        return patched_base_high, patched_lightning_high, patched_lightning_low

    def _calculate_switch_step_and_strategy(
        self,
        switch_strategy: str,
        switch_step: int,
        switch_boundary: float,
        lightning_steps: int,
        patched_lightning_high: Any,
        scheduler: str,
    ) -> Tuple[int, str, str]:
        """Calculate the switch step and effective strategy information.

        Args:
            switch_strategy: Strategy for model switching
            switch_step: Manual switch step value
            switch_boundary: Manual boundary value
            lightning_steps: Total lightning steps
            patched_lightning_high: Patched lightning high model for boundary calculations
            scheduler: Scheduler name

        Returns:
            Tuple of (switch_step_final, effective_strategy, model_switching_info)
        """
        model_switching_info = ""

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
            # Default to 50% of steps
            switch_step_calculated = math.ceil(lightning_steps / 2)
            effective_strategy = switch_strategy

        switch_step_final = int(switch_step_calculated)

        # Generate switching info for logging
        if switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
            boundary_value = (
                _BOUNDARY_T2V
                if switch_strategy == "T2V boundary"
                else _BOUNDARY_I2V
                if switch_strategy == "I2V boundary"
                else switch_boundary
            )
            model_switching_info = f"Model switching: {effective_strategy} (boundary = {boundary_value}) → switch at step {switch_step_final} of {lightning_steps}"
        else:
            model_switching_info = f"Model switching: {effective_strategy} → switch at step {switch_step_final} of {lightning_steps}"

        return switch_step_final, effective_strategy, model_switching_info

    def _create_dry_run_minimal_latent(self, original_latent: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Create minimal latent tensor for dry run mode.

        Generates a small latent tensor to speed up downstream VAE processing
        during dry run testing. Maintains device and dtype compatibility with
        the original latent.

        Args:
            original_latent: Original latent dict containing 'samples' tensor

        Returns:
            Tuple containing dict with minimal latent tensor
        """
        original_samples = original_latent.get("samples")
        if original_samples is not None:
            device = original_samples.device
            dtype = original_samples.dtype
            channels = original_samples.shape[1]
            # Create minimal 8x8 latent compatible with VAE processing
            small_latent_tensor = torch.zeros(
                (1, channels, MIN_LATENT_HEIGHT, MIN_LATENT_WIDTH, MIN_LATENT_WIDTH),
                device=device,
                dtype=dtype
            )
            return ({"samples": small_latent_tensor},)

        # Fallback if no samples found
        return (original_latent,)


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
                "base_quality_threshold": (
                    "INT",
                    {
                        "default": _DEFAULT_BASE_QUALITY_THRESHOLD,
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
                # Sampler parameters
                "sampler_name": base_inputs["sampler_name"],
                "scheduler": base_inputs["scheduler"],
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
                        "tooltip": "Strategy for switching between lightning high and low models.",
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
                        "tooltip": "Manual step to switch models. Use -1 for auto-calculation at 50% of lightning steps.",
                    },
                ),
                "switch_boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Sigma boundary for switching. Defaults to 0.875. T2V/I2V strategies use preset values.",
                    },
                ),
                "dry_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable dry run mode to bypass sampling and return a tiny latent for testing.",
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
        base_quality_threshold: int,
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
        # Check if dry run mode was requested via context menu or parameter
        context_dry_run = getattr(self, '_dry_run_requested', False)
        dry_run = dry_run or context_dry_run
        # Reset the context menu flag after checking
        if hasattr(self, '_dry_run_requested'):
            delattr(self, '_dry_run_requested')

        if str(_LOG_LEVEL).upper() == "DEBUG":
            bare_logger.info("")
        logger.debug("=== TripleKSampler Node - Input Parameters ===")
        logger.debug("Sampling: seed=%d, sigma_shift=%.3f", seed, sigma_shift)
        logger.debug("Base stage: base_steps=%d, base_quality_threshold=%d, base_cfg=%.1f", base_steps, base_quality_threshold, base_cfg)
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
            "Loaded config defaults: BASE_QUALITY_THRESHOLD=%d, DRY_RUN=%s",
            _BASE_QUALITY_THRESHOLD,
            dry_run,
        )

        # Variables to capture calculation info for dry run notification
        base_calculation_info = ""
        model_switching_info = ""

        # Early validation (before auto-calculation resolves base_steps)
        self._validate_basic_parameters(lightning_steps, lightning_start, switch_strategy, switch_step)

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
            if lightning_start == 0 and base_steps == 0:
                optimal_total_base_steps = 0
            else:
                # manual base_steps -> compute total_base_steps for alignment checks
                optimal_total_base_steps = math.floor(base_steps * lightning_steps / max(1, lightning_start))
                optimal_total_base_steps = max(optimal_total_base_steps, base_steps)
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
                        # Send toast notification via ComfyUI message system
                        try:
                            if PromptServer and hasattr(PromptServer, 'instance') and PromptServer.instance:
                                PromptServer.instance.send_sync("triple_ksampler_overlap", {
                                    "severity": "warn",
                                    "summary": "TripleKSampler: Stage overlap",
                                    "detail": f"Stage 1 and Stage 2 overlap by {overlap_pct:.1f}%. Consider base_steps=-1 or adjust lightning parameters.",
                                    "life": TOAST_LIFE_OVERLAP,
                                })
                        except Exception:
                            # Silently fail if PromptServer is not available
                            pass

        # Validate resolved parameters after auto-calculation
        self._validate_resolved_parameters(lightning_start, base_steps)

        # Validate special modes (lightning-only, skip configurations)
        self._validate_special_modes(lightning_start, lightning_steps, base_steps, switch_strategy, switch_step)

        # Patch all models with sigma shift (non-mutating)
        patched_base_high, patched_lightning_high, patched_lightning_low = self._patch_models_for_sampling(
            base_high, lightning_high, lightning_low, sigma_shift
        )

        # Calculate switch step and strategy information
        switch_step_final, _, model_switching_info = self._calculate_switch_step_and_strategy(
            switch_strategy, switch_step, switch_boundary, lightning_steps, patched_lightning_high, scheduler
        )

        # Stage execution logic flags
        skip_stage1 = (lightning_start == 0)
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
            stage2_info = f"Skipped ({stage2_skip_reason})"
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
            seed=seed + STAGE3_SEED_OFFSET,
            steps=lightning_steps,
            cfg=lightning_cfg,  # Use lightning_cfg parameter for advanced node flexibility
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

            # Send toast notification with dry run summary
            self._send_dry_run_notification(
                stage1_info=stage1_info,
                stage2_info=stage2_info,
                stage3_info=stage3_info,
                base_calculation_info=base_calculation_info,
                model_switching_info=model_switching_info
            )

            # Return minimal latent to speed up downstream VAE processing
            return self._create_dry_run_minimal_latent(latent_image)

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
                        "tooltip": "Starting step within lightning schedule. Set to 0 to skip Stage 1 entirely.",
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
                        "tooltip": "Strategy for switching between lightning high and low models.",
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
        base_quality_threshold: int = _DEFAULT_BASE_QUALITY_THRESHOLD,
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
        del base_steps, base_quality_threshold, lightning_cfg, switch_boundary, switch_step  # type: ignore
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
            lightning_cfg=SIMPLE_NODE_LIGHTNING_CFG,
            sampler_name=sampler_name,
            scheduler=scheduler,
            switch_strategy=switch_strategy,
            switch_boundary=0.875,
            switch_step=-1,
            base_quality_threshold=_DEFAULT_BASE_QUALITY_THRESHOLD,
            dry_run=dry_run,
        )

    def run_dry_run(self):
        """Method to be called from context menu to enable dry run mode."""
        self._dry_run_requested = True
        return ("Dry run mode enabled for next execution",)


# Node registration mapping (ComfyUI expects these names)
NODE_CLASS_MAPPINGS = {
    "TripleKSamplerWan22Lightning": TripleKSamplerWan22Lightning,
    "TripleKSamplerWan22LightningAdvanced": TripleKSamplerWan22LightningAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TripleKSamplerWan22Lightning": "TripleKSampler (Wan2.2-Lightning)",
    "TripleKSamplerWan22LightningAdvanced": "TripleKSampler Advanced (Wan2.2-Lightning)",
}
