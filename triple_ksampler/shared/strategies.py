"""Strategy calculation logic for model switching.

This module contains all strategy-related calculations for determining when to switch
between Lightning high and low models during the denoising process.

Supported Strategies:
- Manual switch step: User-specified step number
- T2V boundary: Text-to-video optimized sigma boundary (0.875)
- I2V boundary: Image-to-video optimized sigma boundary (0.900)
- Manual boundary: User-specified sigma boundary
- Percentage-based: 50% of total steps
"""

import math
from typing import Any

try:
    import comfy.samplers
except ImportError:
    # Fallback for testing without ComfyUI
    comfy = None  # type: ignore

# Default boundary values for video generation
DEFAULT_BOUNDARY_T2V = 0.875
DEFAULT_BOUNDARY_I2V = 0.900


def calculate_switch_step_and_strategy(
    switch_strategy: str,
    switch_step: int,
    switch_boundary: float,
    lightning_steps: int,
    patched_lightning_high: Any,
    scheduler: str,
    boundary_t2v: float = DEFAULT_BOUNDARY_T2V,
    boundary_i2v: float = DEFAULT_BOUNDARY_I2V,
) -> tuple[int, str, str]:
    """Calculate the switch step and effective strategy information.

    Determines the step at which to switch from Lightning high model to Lightning low
    model based on the selected strategy. Different strategies provide different levels
    of control and optimization for various use cases.

    Args:
        switch_strategy: Strategy for model switching. Options:
            - "Manual switch step": Use explicit step number
            - "T2V boundary": Use T2V-optimized sigma boundary
            - "I2V boundary": Use I2V-optimized sigma boundary
            - "Manual boundary": Use user-specified sigma boundary
            - Other: Default to 50% of steps
        switch_step: Manual switch step value (used when strategy is "Manual switch step").
            Set to -1 for auto-calculation at 50% of lightning_steps.
        switch_boundary: Manual boundary value (used when strategy is "Manual boundary").
            Should be between 0.0 and 1.0 (normalized timestep).
        lightning_steps: Total number of lightning steps in the schedule
        patched_lightning_high: Patched lightning high model for boundary calculations.
            Used to access model_sampling for sigma schedule computation.
        scheduler: Lightning scheduler name (e.g., "normal", "karras", "exponential").
            Used for sigma schedule generation in boundary-based switching.
        boundary_t2v: T2V boundary value (default: 0.875)
        boundary_i2v: I2V boundary value (default: 0.900)

    Returns:
        Tuple of (switch_step_final, effective_strategy, model_switching_info) where:
            - switch_step_final: Final calculated switch step (int)
            - effective_strategy: Human-readable strategy description (str)
            - model_switching_info: Formatted info string for logging (str)

    Raises:
        ValueError: If boundary values are invalid or lightning_steps < 1

    Example:
        >>> # Manual step switching
        >>> step, strategy, info = calculate_switch_step_and_strategy(
        ...     "Manual switch step", 5, 0.875, 10, model, "normal"
        ... )
        >>> print(f"Switch at step {step}")
        Switch at step 5

        >>> # Auto-calculate 50%
        >>> step, strategy, info = calculate_switch_step_and_strategy(
        ...     "Manual switch step", -1, 0.875, 10, model, "normal"
        ... )
        >>> print(f"Switch at step {step}")
        Switch at step 5
    """
    if lightning_steps < 1:
        raise ValueError(f"lightning_steps must be at least 1, got {lightning_steps}")

    model_switching_info = ""

    if switch_strategy == "Manual switch step":
        if switch_step == -1:
            # Auto-calculate 50% of steps
            switch_step_calculated = lightning_steps // 2
            effective_strategy = "50% of steps (auto)"
        else:
            # Use manual step value
            switch_step_calculated = switch_step
            effective_strategy = "Manual switch step"
    elif switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
        # Sigma boundary-based switching
        if switch_strategy == "T2V boundary":
            boundary_value = boundary_t2v
        elif switch_strategy == "I2V boundary":
            boundary_value = boundary_i2v
        else:
            boundary_value = switch_boundary

        # Get model sampling object for sigma schedule calculation
        sampling = patched_lightning_high.get_model_object("model_sampling")
        switch_step_calculated = compute_boundary_switching_step(
            sampling, scheduler, lightning_steps, boundary_value
        )
        effective_strategy = switch_strategy
    else:
        # Default strategy: 50% of steps
        switch_step_calculated = math.ceil(lightning_steps / 2)
        effective_strategy = switch_strategy

    switch_step_final = int(switch_step_calculated)

    # Generate switching info for logging
    if switch_strategy in ["T2V boundary", "I2V boundary", "Manual boundary"]:
        boundary_value = (
            boundary_t2v
            if switch_strategy == "T2V boundary"
            else boundary_i2v
            if switch_strategy == "I2V boundary"
            else switch_boundary
        )
        model_switching_info = f"Model switching: {effective_strategy} (boundary = {boundary_value}) → switch at step {switch_step_final} of {lightning_steps}"
    else:
        model_switching_info = f"Model switching: {effective_strategy} → switch at step {switch_step_final} of {lightning_steps}"

    return switch_step_final, effective_strategy, model_switching_info


def compute_boundary_switching_step(
    sampling: Any, scheduler: str, steps: int, boundary: float
) -> int:
    """Compute the switch step index from sigmas and a boundary value.

    This function implements sigma-based model switching, which is more accurate
    than step-based switching because it accounts for the actual noise schedule.
    Different schedulers have different sigma curves, so this ensures consistent
    switching behavior across schedulers.

    The boundary value represents a normalized timestep (0-1 range), where 0 is
    the start of denoising (high noise) and 1 is the end (low noise). The function
    finds the first step where the normalized timestep drops below the boundary,
    indicating the transition point in the denoising schedule.

    Args:
        sampling: model_sampling object returned by the patched model.
            Must have timestep() method for sigma-to-timestep conversion.
        scheduler: Scheduler name (e.g., "normal", "karras", "exponential").
            Determines the sigma curve shape.
        steps: Number of lightning steps (must be >= 1).
        boundary: Boundary value between 0 and 1 (normalized timestep).
            Typical values: 0.875 for T2V, 0.900 for I2V.

    Returns:
        Step index in [0, steps-1] where sigma crosses the boundary.
        If boundary is never crossed, returns steps-1 (last step).

    Raises:
        ValueError: If steps < 1 or boundary not in [0, 1]
        ImportError: If ComfyUI's comfy.samplers module is not available

    Example:
        >>> # Find switching point at 87.5% timestep
        >>> sampling = model.get_model_object("model_sampling")
        >>> switch_step = compute_boundary_switching_step(
        ...     sampling, "normal", 10, 0.875
        ... )
        >>> print(f"Switch at step {switch_step}")
        Switch at step 7
    """
    if comfy is None:
        raise ImportError(
            "ComfyUI's comfy.samplers module is not available. "
            "This function requires ComfyUI to be installed."
        )

    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")

    if not 0.0 <= boundary <= 1.0:
        raise ValueError(
            f"boundary must be between 0.0 and 1.0, got {boundary}. "
            f"Recommended: T2V=0.875, I2V=0.900"
        )

    # Calculate the sigma schedule for the given steps and scheduler
    sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
    timesteps: list[float] = []

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


def validate_strategy(switch_strategy: str) -> None:
    """Validate that the strategy name is recognized.

    Args:
        switch_strategy: Strategy name to validate

    Raises:
        ValueError: If strategy is not recognized

    Example:
        >>> validate_strategy("Manual switch step")  # OK
        >>> validate_strategy("invalid")  # Raises ValueError
    """
    valid_strategies = [
        "Manual switch step",
        "T2V boundary",
        "I2V boundary",
        "Manual boundary",
        "50% of steps",
    ]

    if switch_strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy: {switch_strategy}. Valid strategies: {', '.join(valid_strategies)}"
        )
