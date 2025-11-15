"""Base class for triple-stage sampling.

This module contains TripleKSamplerBase, which provides the core sampling algorithm
and orchestration logic for all node variants. It delegates calculation, validation,
and formatting to shared modules while handling ComfyUI integration.

Design Pattern: Orchestrator - coordinates shared modules but contains minimal logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import comfy.model_management
import nodes
import torch

# Import shared logic modules
from triple_ksampler.shared import alignment as core_alignment
from triple_ksampler.shared import config as core_config
from triple_ksampler.shared import logging as core_logging
from triple_ksampler.shared import models as core_models
from triple_ksampler.shared import notifications as core_notifications
from triple_ksampler.shared import strategies as core_strategies
from triple_ksampler.shared import validation as core_validation

# Load configuration at import time (delegate to core.config)
_CONFIG = core_config.load_config(config_dir=Path(__file__).resolve().parent.parent)

# Extract configuration values using core.config
_BOUNDARY_T2V = core_config.get_boundary_t2v(_CONFIG)
_BOUNDARY_I2V = core_config.get_boundary_i2v(_CONFIG)

# Get logger and bare_logger
# These inherit configuration from the parent entry point (ksampler_nodes.py or wvsampler_nodes.py)
logger = logging.getLogger("triple_ksampler.base")
bare_logger = logging.getLogger("triple_ksampler.separator")


class TripleKSamplerBase:
    """Base class containing shared functionality for TripleKSampler nodes."""

    # ComfyUI required attributes
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "TripleKSampler/sampling"

    @classmethod
    def _get_base_input_types(cls) -> dict[str, Any]:
        """Return the shared INPUT_TYPES mapping used by both nodes."""
        return {
            "base_high": ("MODEL", {"tooltip": "Base high-noise model for Stage 1."}),
            "lightning_high": ("MODEL", {"tooltip": "Lightning high-noise model for Stage 2."}),
            "lightning_low": ("MODEL", {"tooltip": "Lightning low-noise model for Stage 3."}),
            "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning."}),
            "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning."}),
            "latent_image": ("LATENT", {"tooltip": "Latent image to denoise."}),
            "seed": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": "The random seed used for creating the noise.",
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
        }

    @classmethod
    def _calculate_perfect_alignment(
        cls, base_quality_threshold: int, lightning_start: int, lightning_steps: int
    ) -> tuple[int, int, str]:
        """Calculate base_steps and total_base_steps for perfect alignment.

        This method delegates to core.alignment.calculate_perfect_alignment().

        Returns:
            (base_steps, total_base_steps, method_used) where method_used is one of:
            "simple_math", "mathematical_search", "fallback".
        """
        return core_alignment.calculate_perfect_alignment(
            base_quality_threshold, lightning_start, lightning_steps
        )

    def _calculate_percentage(self, numerator: float, denominator: float) -> float:
        """Calculate percentage with division-by-zero protection and bounds clamping."""
        return core_logging.calculate_percentage(numerator, denominator)

    def _format_stage_range(self, start: int, end: int, total: int) -> str:
        """Return a human-readable string describing step ranges and denoising percentages."""
        return core_logging.format_stage_range(start, end, total)

    def _format_base_calculation_compact(self, base_calc_info: str) -> str:
        """Format base calculation info for compact toast display."""
        return core_logging.format_base_calculation_compact(base_calc_info)

    def _format_switch_info_compact(self, switch_info: str) -> str:
        """Format model switching info for compact toast display."""
        return core_logging.format_switch_info_compact(switch_info)

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
        summary_lines.extend(
            [
                "Stage Configuration:",
                f"• {stage1_info}",
                f"• {stage2_info}",
                f"• {stage3_info}",
            ]
        )

        detail_text = "\n".join(summary_lines)

        # Send toast notification via ComfyUI message system (delegate to core.notifications)
        core_notifications.send_dry_run_notification(detail_text)

    def _compute_boundary_switching_step(
        self, sampling: Any, scheduler: str, steps: int, boundary: float
    ) -> int:
        """Compute the switch step index from sigmas and a boundary value.

        This method delegates to core.strategies.compute_boundary_switching_step().

        Args:
            sampling: model_sampling object returned by the patched model.
            scheduler: scheduler name.
            steps: number of lightning steps.
            boundary: boundary value between 0 and 1 (normalized timestep).

        Returns:
            int: step index in [0, steps-1] where sigma crosses the boundary.
        """
        return core_strategies.compute_boundary_switching_step(sampling, scheduler, steps, boundary)

    def _get_target_boundary_from_strategy(
        self, switch_strategy: str, manual_boundary: float
    ) -> float:
        """Extract target boundary sigma from switch strategy.

        Args:
            switch_strategy: Strategy name (may include "(refined)" suffix)
            manual_boundary: User's manual boundary value

        Returns:
            Target sigma value for boundary alignment

        Raises:
            ValueError: If strategy is not boundary-based
        """
        # Strip "(refined)" suffix if present to get base strategy
        base_strategy = self._get_base_strategy(switch_strategy)

        if base_strategy == "T2V boundary":
            return _BOUNDARY_T2V
        if base_strategy == "I2V boundary":
            return _BOUNDARY_I2V
        if base_strategy == "Manual boundary":
            return manual_boundary
        raise ValueError(
            f"Strategy '{base_strategy}' is not boundary-based. "
            f"Refinement only supports: T2V boundary, I2V boundary, Manual boundary"
        )

    def _is_refined_strategy(self, strategy: str) -> bool:
        """Check if strategy is a refined variant.

        Args:
            strategy: Strategy name to check

        Returns:
            True if strategy ends with " (refined)", False otherwise

        Example:
            >>> node._is_refined_strategy("T2V boundary (refined)")
            True
            >>> node._is_refined_strategy("T2V boundary")
            False
        """
        return strategy.endswith(" (refined)")

    def _get_base_strategy(self, strategy: str) -> str:
        """Extract base strategy name by removing refined suffix.

        Args:
            strategy: Strategy name (may include "(refined)" suffix)

        Returns:
            Base strategy name with "(refined)" suffix removed

        Example:
            >>> node._get_base_strategy("T2V boundary (refined)")
            'T2V boundary'
            >>> node._get_base_strategy("T2V boundary")
            'T2V boundary'
        """
        return strategy.replace(" (refined)", "")

    def _validate_basic_parameters(
        self,
        lightning_steps: int,
        lightning_start: int,
        switch_strategy: str,
        switch_step: int,
    ) -> None:
        """Validate basic input parameters that don't depend on auto-calculated values.

        This method delegates to core.validation.validate_basic_parameters().

        Args:
            lightning_steps: Total steps for lightning stages
            lightning_start: Starting step within lightning schedule
            switch_strategy: Strategy for model switching
            switch_step: Manual switch step (when using manual strategy)

        Raises:
            ValueError: if any parameter validation fails
        """
        core_validation.validate_basic_parameters(
            lightning_steps, lightning_start, switch_strategy, switch_step
        )

    def _validate_resolved_parameters(
        self,
        lightning_start: int,
        base_steps: int,
    ) -> None:
        """Validate parameters after auto-calculation has resolved base_steps.

        This method delegates to core.validation.validate_resolved_parameters().

        Args:
            lightning_start: Starting step within lightning schedule
            base_steps: Resolved steps for base model (no longer -1)

        Raises:
            ValueError: if resolved parameter validation fails
        """
        core_validation.validate_resolved_parameters(lightning_start, base_steps)

    def _validate_special_modes(
        self,
        lightning_start: int,
        lightning_steps: int,
        base_steps: int,
        switch_strategy: str,
        switch_step: int,
    ) -> None:
        """Validate special mode configurations (lightning-only, skip modes).

        This method delegates to core.validation.validate_special_modes().

        Args:
            lightning_start: Starting step within lightning schedule
            lightning_steps: Total steps for lightning stages
            base_steps: Steps for base model
            switch_strategy: Strategy for model switching
            switch_step: Manual switch step

        Raises:
            ValueError: if special mode configuration is invalid
        """
        core_validation.validate_special_modes(
            lightning_start, lightning_steps, base_steps, switch_strategy, switch_step
        )

    def _run_sampling_stage(
        self,
        model: Any,
        positive: Any,
        negative: Any,
        latent: dict[str, torch.Tensor],
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
        stage_info: str | None = None,
    ) -> tuple[dict[str, torch.Tensor], ...]:
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
        except comfy.model_management.InterruptProcessingException:
            # User cancelled - re-raise without wrapping to allow graceful cancellation
            raise
        except Exception as exc:
            exc_msg = str(exc).strip()
            if exc_msg:
                raise RuntimeError(
                    f"{stage_name}: sampling failed - {type(exc).__name__}: {exc_msg}"
                ) from exc
            else:
                raise RuntimeError(f"{stage_name}: sampling failed - {type(exc).__name__}") from exc

        return result

    def _patch_models_for_sampling(
        self,
        base_high: Any,
        lightning_high: Any,
        lightning_low: Any,
        sigma_shift: float,
    ) -> tuple[Any, Any, Any]:
        """Patch all models with sigma shift for sampling (delegate to core.models)."""
        return core_models.patch_models_with_sigma_shift(
            base_high, lightning_high, lightning_low, sigma_shift
        )

    def _calculate_switch_step_and_strategy(
        self,
        switch_strategy: str,
        switch_step: int,
        switch_boundary: float,
        lightning_steps: int,
        patched_lightning_high: Any,
        scheduler: str,
    ) -> tuple[int, str, str]:
        """Calculate the switch step and effective strategy information.

        This method delegates to core.strategies.calculate_switch_step_and_strategy().

        Args:
            switch_strategy: Strategy for model switching
            switch_step: Manual switch step value
            switch_boundary: Manual boundary value
            lightning_steps: Total lightning steps
            patched_lightning_high: Patched lightning high model for boundary calculations
            scheduler: Lightning scheduler name (used for boundary-based switching)

        Returns:
            Tuple of (switch_step_final, effective_strategy, model_switching_info)
        """
        return core_strategies.calculate_switch_step_and_strategy(
            switch_strategy,
            switch_step,
            switch_boundary,
            lightning_steps,
            patched_lightning_high,
            scheduler,
            boundary_t2v=_BOUNDARY_T2V,
            boundary_i2v=_BOUNDARY_I2V,
        )
