"""Model manipulation and sigma shift application for TripleKSampler.

This module encapsulates all model patching logic using ComfyUI's ModelSamplingSD3 class.
All operations are non-mutating - models are cloned internally by ModelSamplingSD3.patch().

Includes automatic sigma_shift refinement algorithm for boundary-based strategies,
inspired by ComfyUI-WanMoEScheduler's iterative search approach (MIT License).
"""

from typing import Any

try:
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
except ImportError:
    # Fallback for testing without ComfyUI
    ModelSamplingSD3 = None  # type: ignore

try:
    import comfy.samplers
    import torch
except ImportError:
    # Fallback for testing without ComfyUI
    comfy = None  # type: ignore
    torch = None  # type: ignore

# Search interval for sigma shift refinement (0.01 sigma units per iteration)
_SEARCH_INTERVAL = 0.01


def patch_models_with_sigma_shift(
    base_high: Any,
    lightning_high: Any,
    lightning_low: Any,
    sigma_shift: float,
) -> tuple[Any, Any, Any]:
    """Patch all three models with sigma shift using ModelSamplingSD3.

    Applies sigma shift adjustment to all three models (base, lightning high, lightning low)
    using ComfyUI's ModelSamplingSD3 class. This operation is non-mutating - the original
    models are not modified. ModelSamplingSD3.patch() clones models internally before
    applying the sigma shift.

    Args:
        base_high: Base high-noise model
        lightning_high: Lightning high-noise model
        lightning_low: Lightning low-noise model
        sigma_shift: Sigma adjustment value (typically 3.0-5.0 for Wan2.2 models)

    Returns:
        Tuple of (patched_base_high, patched_lightning_high, patched_lightning_low)

    Raises:
        ImportError: If ModelSamplingSD3 is not available (ComfyUI not installed)

    Example:
        >>> # Apply sigma shift to all three models
        >>> base_patched, high_patched, low_patched = patch_models_with_sigma_shift(
        ...     base_model, lightning_high_model, lightning_low_model, sigma_shift=5.0
        ... )
        >>> # Original models unchanged, patched models ready for sampling
    """
    if ModelSamplingSD3 is None:
        raise ImportError(
            "ComfyUI's ModelSamplingSD3 is not available. "
            "This function requires ComfyUI to be installed."
        )

    # Create patcher instance
    patcher = ModelSamplingSD3()

    # Convert sigma_shift to Python float for consistent processing
    shift_value = float(sigma_shift)

    # Apply sigma shift to all three models (non-mutating)
    # ModelSamplingSD3.patch() returns tuple (patched_model, ...)
    patched_base_high = patcher.patch(base_high, shift_value)[0]
    patched_lightning_high = patcher.patch(lightning_high, shift_value)[0]
    patched_lightning_low = patcher.patch(lightning_low, shift_value)[0]

    return patched_base_high, patched_lightning_high, patched_lightning_low


def _get_sigma_at_step_comfy(
    model: Any,
    scheduler: str,
    total_steps: int,
    target_step: int,
    shift_value: float,
) -> float:
    """Get sigma value at target_step using ComfyUI samplers.

    Args:
        model: ComfyUI MODEL object
        scheduler: Scheduler name (e.g., "sgm_uniform", "karras")
        total_steps: Total steps in schedule
        target_step: Step index to retrieve sigma for
        shift_value: Sigma shift to apply via ModelSamplingSD3

    Returns:
        Sigma value at target_step

    Raises:
        ImportError: If required ComfyUI modules not available
    """
    if ModelSamplingSD3 is None or comfy is None:
        raise ImportError("ComfyUI modules not available for sigma calculation")

    # Apply shift temporarily
    patcher = ModelSamplingSD3()
    patched = patcher.patch(model, shift_value)[0]

    # Get sigmas from ComfyUI
    sigmas = comfy.samplers.calculate_sigmas(
        patched.get_model_object("model_sampling"), scheduler, total_steps
    )

    return float(sigmas[target_step])


def _get_sigma_at_step_wanvideo(
    model: Any,
    scheduler: str,
    total_steps: int,
    target_step: int,
    shift_value: float,
) -> float:
    """Get sigma value at target_step using WanVideo schedulers.

    Uses WanVideo's get_scheduler() function to generate sigmas.
    Note: WanVideo sigmas are normalized timesteps (0-1 range).

    Args:
        model: WanVideo WANVIDEOMODEL object
        scheduler: WanVideo scheduler name (e.g., "euler", "euler/beta")
        total_steps: Total steps in schedule
        target_step: Step index to retrieve sigma for
        shift_value: Sigma shift to apply

    Returns:
        Sigma value at target_step (normalized timestep)

    Raises:
        ImportError: If WanVideoWrapper components not available
    """
    if torch is None:
        raise ImportError("PyTorch not available for WanVideo sigma calculation")

    # Import WanVideo utilities
    from triple_ksampler.wvsampler.utils import get_wanvideo_components

    _, _, get_scheduler = get_wanvideo_components()

    # Get transformer_dim from model (default 5120 for Wan 14B models)
    transformer_dim = getattr(model, "transformer_dim", 5120)

    # Get WanVideo scheduler with shift
    device = torch.device("cpu")
    sample_scheduler, _, _, _ = get_scheduler(
        scheduler,
        total_steps,
        start_step=0,
        end_step=total_steps,
        shift=shift_value,
        device=device,
        transformer_dim=transformer_dim,
    )

    return float(sample_scheduler.sigmas[target_step])


def calculate_perfect_shift_for_step(
    model: Any,
    scheduler: str,
    total_steps: int,
    target_step: int,
    target_sigma: float,
    initial_shift: float,
    is_wanvideo: bool = False,
) -> tuple[float, str]:
    """Find optimal sigma_shift for perfect boundary alignment.

    Uses adaptive bidirectional search algorithm (inspired by ComfyUI-WanMoEScheduler)
    starting from user's initial shift value for optimal performance.

    Algorithm:
    1. Test initial shift to get baseline sigma
    2. Test one step in each direction (+/- 0.01 sigma units)
    3. Choose direction that reduces error from target_sigma
    4. Linear search in that direction until overshoot detected (finds actual closest match)

    Args:
        model: MODEL (ComfyUI) or WANVIDEOMODEL object
        scheduler: Scheduler name
        total_steps: Total steps in sigma schedule
        target_step: Step where boundary should align (from discovery phase)
        target_sigma: Target sigma value at target_step
        initial_shift: User's provided shift (starting point for search)
        is_wanvideo: True for WanVideo schedulers, False for ComfyUI samplers

    Returns:
        Tuple of (optimal_shift, info_message)

    Raises:
        ValueError: If search exceeds 10000 iterations (safety limit)
        ImportError: If required modules not available

    Performance:
        - Typical: 20-50 iterations (user's shift close to optimal)
        - Average: 100-200 iterations (user ±1.0 off)
        - Worst: ~1000 iterations (user far off, but still 5x faster than zero-start)

    Example:
        >>> shift, info = calculate_perfect_shift_for_step(
        ...     lightning_high, "sgm_uniform", 8, 4, 0.875, 5.0
        ... )
        >>> # shift might be 5.23, info describes the search
    """
    # Choose sigma getter based on node type
    get_sigma_fn = _get_sigma_at_step_wanvideo if is_wanvideo else _get_sigma_at_step_comfy

    # Step 1: Test initial guess
    baseline_sigma = get_sigma_fn(model, scheduler, total_steps, target_step, initial_shift)
    baseline_diff = abs(baseline_sigma - target_sigma)

    # Step 2: Empirically determine search direction
    test_up_shift = min(initial_shift + _SEARCH_INTERVAL, 100.0)
    test_down_shift = max(initial_shift - _SEARCH_INTERVAL, 0.0)

    sigma_up = get_sigma_fn(model, scheduler, total_steps, target_step, test_up_shift)
    sigma_down = get_sigma_fn(model, scheduler, total_steps, target_step, test_down_shift)

    diff_up = abs(sigma_up - target_sigma)
    diff_down = abs(sigma_down - target_sigma)

    # Step 3: Choose direction that reduces error
    if diff_up < baseline_diff:
        direction = +1
    elif diff_down < baseline_diff:
        direction = -1
    else:
        # Local optimum at initial shift
        return (
            initial_shift,
            f"Local optimum at initial shift {initial_shift:.2f} (diff={baseline_diff:.4f})",
        )

    # Step 4: Linear search in determined direction
    shift = initial_shift
    iterations = 0
    prev_diff = baseline_diff

    while True:
        shift += direction * _SEARCH_INTERVAL

        # Bounds check
        if shift < 0.0 or shift > 100.0:
            # Hit boundary - return closest match
            final_shift = shift - direction * _SEARCH_INTERVAL
            return (
                final_shift,
                f"Boundary reached after {iterations} iterations (closest: {final_shift:.2f})",
            )

        test_sigma = get_sigma_fn(model, scheduler, total_steps, target_step, shift)
        current_diff = abs(test_sigma - target_sigma)
        iterations += 1

        # Overshoot detection - finds actual closest match
        if current_diff > prev_diff:
            # Overshot - return previous value
            final_shift = shift - direction * _SEARCH_INTERVAL
            return (
                final_shift,
                f"Converged after {iterations} iterations at {final_shift:.2f} (diff={prev_diff:.4f})",
            )

        prev_diff = current_diff

        # Safety limit
        if iterations > 10000:
            raise ValueError(
                f"Search exceeded 10000 iterations. "
                f"Started at {initial_shift:.2f}, target_sigma={target_sigma:.3f}"
            )
