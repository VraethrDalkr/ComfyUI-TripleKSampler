"""Simplified node with smart defaults and common strategies.

This module contains TripleKSampler, which provides a streamlined interface with:
- 5 strategies (50% of steps, T2V/I2V boundary + refined variants)
- Auto-calculated defaults for lightning_steps and lightning_start
- Simplified parameter set for ease of use

Recommended for users who want triple-stage sampling without advanced configuration.
"""

from __future__ import annotations

from typing import Any

import comfy.samplers
import torch

# Import shared logic modules
from triple_ksampler.shared import config as core_config

# Import advanced class to inherit from
from .advanced import TripleKSamplerAdvanced

# Algorithm constants
SIMPLE_NODE_LIGHTNING_CFG = 1.0  # Fixed CFG value for Simple node's lightning stages


class TripleKSampler(TripleKSamplerAdvanced):
    """Simplified triple-stage sampler with sensible defaults and auto-calculated parameters."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
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
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"tooltip": "Sampler to use for all stages."},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"tooltip": "Scheduler to use for all stages."},
                ),
                "switch_strategy": (
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
        latent_image: dict[str, torch.Tensor],
        seed: int,
        sigma_shift: float,
        base_steps: int = -1,
        base_quality_threshold: int = core_config.DEFAULT_BASE_QUALITY_THRESHOLD,
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
    ) -> tuple[dict[str, torch.Tensor], ...]:
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
            base_sampler=sampler_name,
            base_scheduler=scheduler,
            lightning_start=lightning_start,
            lightning_steps=lightning_steps,
            lightning_cfg=SIMPLE_NODE_LIGHTNING_CFG,
            lightning_sampler=sampler_name,
            lightning_scheduler=scheduler,
            switch_strategy=switch_strategy,
            switch_boundary=0.875,
            switch_step=-1,
            base_quality_threshold=core_config.DEFAULT_BASE_QUALITY_THRESHOLD,
            dry_run=dry_run,
        )
