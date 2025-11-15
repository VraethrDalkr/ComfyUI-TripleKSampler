"""Input validation logic for triple-stage sampling.

This module contains all parameter validation functions for TripleKSampler nodes.
Validation is split into three phases:
1. Basic validation - Check inputs before any auto-calculation
2. Resolved validation - Check after auto-calculated values are resolved
3. Special modes validation - Validate lightning-only and skip modes
"""

import math


def validate_basic_parameters(
    lightning_steps: int,
    lightning_start: int,
    switch_strategy: str,
    switch_step: int,
) -> None:
    """Validate basic input parameters that don't depend on auto-calculated values.

    Checks fundamental constraints on lightning parameters and manual switch step
    before any auto-calculation or resolution occurs. This catches obvious input
    errors early in the sampling process.

    Args:
        lightning_steps: Total steps in lightning schedule. Must be >= 2.
        lightning_start: Starting step within lightning schedule (0-based index).
            Must be in range [0, lightning_steps-1].
        switch_strategy: Strategy for model switching. Used to determine if
            switch_step validation is required.
        switch_step: Manual switch step value. If -1, will be auto-calculated.
            If specified (not -1), must be >= 0, < lightning_steps, and >= lightning_start.

    Raises:
        ValueError: If any validation constraint is violated

    Example:
        >>> # Valid parameters
        >>> validate_basic_parameters(10, 4, "Manual switch step", 7)
        >>> # No exception raised

        >>> # Invalid: lightning_steps too low
        >>> validate_basic_parameters(1, 0, "50% of steps", -1)
        ValueError: lightning_steps must be at least 2.

        >>> # Invalid: switch_step < lightning_start
        >>> validate_basic_parameters(10, 4, "Manual switch step", 2)
        ValueError: switch_step (2) cannot be less than lightning_start (4)...
    """
    # Basic lightning parameters validation
    if lightning_steps < 2:
        raise ValueError("lightning_steps must be at least 2.")

    if not (0 <= lightning_start < lightning_steps):
        raise ValueError(
            f"lightning_start must be within [0, lightning_steps-1]. "
            f"Got lightning_start={lightning_start}, lightning_steps={lightning_steps}"
        )

    # Manual switch step validation
    if switch_strategy == "Manual switch step" and switch_step != -1:
        if switch_step < 0:
            raise ValueError(
                f"switch_step ({switch_step}) must be >= 0. "
                f"Use switch_step=-1 for auto-calculation."
            )

        if switch_step >= lightning_steps:
            raise ValueError(
                f"switch_step ({switch_step}) must be < lightning_steps ({lightning_steps})"
            )

        if switch_step < lightning_start:
            raise ValueError(
                f"switch_step ({switch_step}) cannot be less than lightning_start ({lightning_start}). "
                "If you want low-noise only, set lightning_start=0 as well."
            )


def validate_resolved_parameters(
    lightning_start: int,
    base_steps: int,
) -> None:
    """Validate parameters after auto-calculation has resolved base_steps.

    This validation phase occurs after base_steps has been auto-calculated or
    user-specified. Ensures the relationship between lightning_start and base_steps
    is consistent with valid sampling configurations.

    Args:
        lightning_start: Starting step within lightning schedule (0-based index).
        base_steps: Number of base model denoising steps (Stage 1).
            May be auto-calculated or user-specified.

    Raises:
        ValueError: If base_steps is invalid for the given lightning_start

    Example:
        >>> # Valid: Stage 1 active with base_steps > 0
        >>> validate_resolved_parameters(4, 10)
        >>> # No exception raised

        >>> # Valid: Stage 1 skip mode (both zero)
        >>> validate_resolved_parameters(0, 0)
        >>> # No exception raised

        >>> # Invalid: base_steps=0 but lightning_start > 0
        >>> validate_resolved_parameters(4, 0)
        ValueError: base_steps = 0 is only allowed when lightning_start = 0...

        >>> # Invalid: lightning_start > 0 but base_steps < 1
        >>> validate_resolved_parameters(1, 0)
        ValueError: base_steps must be >= 1 when lightning_start > 0.
    """
    # Base steps and lightning_start relationship validation
    if lightning_start > 0 and base_steps < 1:
        raise ValueError(
            f"base_steps must be >= 1 when lightning_start > 0. "
            f"Got base_steps={base_steps}, lightning_start={lightning_start}"
        )

    if base_steps == 0 and lightning_start != 0:
        raise ValueError(
            f"base_steps = 0 is only allowed when lightning_start = 0 (Stage 1 skip mode). "
            f"Got base_steps=0, lightning_start={lightning_start}"
        )


def validate_special_modes(
    lightning_start: int,
    lightning_steps: int,
    base_steps: int,
    switch_strategy: str,
    switch_step: int,
) -> None:
    """Validate special mode configurations (lightning-only, skip modes).

    Special modes have additional constraints beyond basic validation:
    - Lightning-only mode: No base stage, only lightning models (base_steps=0, lightning_start=0)
    - Stage 1 skip mode: Skip base model entirely (lightning_start=0)

    These modes require careful validation to ensure consistent configuration.

    Args:
        lightning_start: Starting step within lightning schedule (0-based index).
        lightning_steps: Total steps in lightning schedule.
        base_steps: Number of base model denoising steps (Stage 1).
        switch_strategy: Strategy for model switching between high/low models.
        switch_step: Manual switch step value (only relevant for Manual strategy).

    Raises:
        ValueError: If special mode configuration is inconsistent

    Example:
        >>> # Valid: Lightning-only mode (base_steps=0, lightning_start=0)
        >>> validate_special_modes(0, 10, 0, "50% of steps", -1)
        >>> # No exception raised

        >>> # Invalid: Lightning-only mode requires base_steps=0
        >>> validate_special_modes(0, 10, 5, "50% of steps", -1)
        ValueError: Set base_steps=0 or base_steps=-1 for Lightning-only mode...
    """
    # Lightning-only mode validation
    if lightning_start == 0:
        # Calculate temp_switch_step to check for skipping both Stage 1 and Stage 2
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
