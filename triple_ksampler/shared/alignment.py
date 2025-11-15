"""Perfect alignment algorithm for triple-stage sampling.

This module implements the Perfect Alignment Algorithm that calculates base_steps
and total_base_steps to ensure Stage 1 end exactly matches Stage 2 start in the
denoising schedule. This prevents gaps or overlaps between stages.

The algorithm uses three methods in priority order:
1. simple_math: Direct calculation when lightning_start=1
2. mathematical_search: Search for perfect integer alignment
3. fallback: Approximation when perfect alignment not found
"""

import math

# Algorithm constants
SEARCH_LIMIT_MULTIPLIER = 1  # Additional steps to search beyond threshold for alignment


def calculate_perfect_alignment(
    base_quality_threshold: int, lightning_start: int, lightning_steps: int
) -> tuple[int, int, str]:
    """Calculate base_steps and total_base_steps for perfect alignment.

    Perfect alignment ensures Stage 1 end exactly matches Stage 2 start in the
    denoising schedule. This prevents gaps or overlaps between stages by finding
    integer values where the denoising percentages align precisely.

    The algorithm tries three methods in order:
    1. **Simple Math** (lightning_start=1): Direct calculation using the formula
       base_steps/total_base_steps = lightning_start/lightning_steps
    2. **Mathematical Search** (lightning_start>1): Search for total_base_steps
       where (total_base_steps * lightning_start) is divisible by lightning_steps
    3. **Fallback**: Approximation when perfect alignment not found within search range

    Args:
        base_quality_threshold: Minimum total steps for base stage. The algorithm
            will find base_steps >= threshold that achieves perfect alignment.
        lightning_start: Starting step within lightning schedule (0-based index).
            Determines where Stage 2 begins in the lightning denoising process.
        lightning_steps: Total steps in lightning schedule. Must be > 0.

    Returns:
        Tuple of (base_steps, total_base_steps, method_used) where:
            - base_steps: Calculated steps for base model Stage 1
            - total_base_steps: Total theoretical steps for alignment calculation
            - method_used: One of "simple_math", "mathematical_search", "fallback"

    Raises:
        ValueError: If parameters are invalid (steps < 1, start >= steps, etc.)

    Example:
        >>> # Simple case: lightning_start=1
        >>> base, total, method = calculate_perfect_alignment(20, 1, 10)
        >>> print(f"base={base}, total={total}, method={method}")
        base=2, total=20, method=simple_math

        >>> # Complex case: lightning_start=4
        >>> base, total, method = calculate_perfect_alignment(20, 4, 10)
        >>> print(f"base={base}, total={total}, method={method}")
        base=8, total=20, method=mathematical_search

        >>> # Verify alignment
        >>> stage1_end_pct = base / total
        >>> stage2_start_pct = 4 / 10
        >>> print(f"Stage 1 end: {stage1_end_pct:.1%}, Stage 2 start: {stage2_start_pct:.1%}")
        Stage 1 end: 40.0%, Stage 2 start: 40.0%  # Perfect alignment!
    """
    # Special case: lightning_start=0 means skip base stage entirely
    # Check this BEFORE validation to allow base_quality_threshold=0
    if lightning_start == 0:
        return 0, 0, "simple_math"

    # Validation
    if base_quality_threshold < 1:
        raise ValueError(f"base_quality_threshold must be at least 1, got {base_quality_threshold}")

    if lightning_steps < 1:
        raise ValueError(f"lightning_steps must be at least 1, got {lightning_steps}")

    if not 0 <= lightning_start < lightning_steps:
        raise ValueError(
            f"lightning_start ({lightning_start}) must be between 0 and {lightning_steps - 1}"
        )

    if lightning_start == 1:
        # Simple case: lightning starts at step 1, direct calculation possible
        # Formula: base_steps/total_base_steps = lightning_start/lightning_steps
        # Therefore: base_steps = total_base_steps / lightning_steps
        # To meet threshold: total_base_steps >= base_quality_threshold
        # So: base_steps = ceil(threshold / lightning_steps)
        # And: total_base_steps = base_steps * lightning_steps
        base_steps = math.ceil(base_quality_threshold / lightning_steps)
        total_base_steps = base_steps * lightning_steps
        return base_steps, total_base_steps, "simple_math"

    # Complex case: lightning_start > 1, need perfect integer alignment
    # We need: base_steps/total_base_steps = lightning_start/lightning_steps
    # Rearranging: base_steps = (total_base_steps * lightning_start) / lightning_steps
    # For integer base_steps, need (total_base_steps * lightning_start) % lightning_steps == 0
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


def calculate_manual_base_steps_alignment(
    base_steps: int, lightning_start: int, lightning_steps: int
) -> int:
    """Calculate total_base_steps for manual base_steps with alignment checks.

    When user provides manual base_steps (not auto-calculated), we still need
    to calculate optimal total_base_steps for alignment checking and overlap detection.

    Formula: total_base_steps = floor(base_steps * lightning_steps / lightning_start)
    Then: total_base_steps = max(total_base_steps, base_steps)

    Args:
        base_steps: Manual base steps value (user-provided)
        lightning_start: Starting step within lightning schedule
        lightning_steps: Total steps in lightning schedule

    Returns:
        Calculated total_base_steps for alignment checks

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> # Manual base_steps=10, lightning_start=4, lightning_steps=7
        >>> total = calculate_manual_base_steps_alignment(10, 4, 7)
        >>> print(f"total_base_steps = {total}")
        total_base_steps = 17

        >>> # Verify overlap calculation
        >>> stage1_end_pct = 10 / total
        >>> stage2_start_pct = 4 / 7
        >>> print(f"Stage 1: {stage1_end_pct:.1%}, Stage 2: {stage2_start_pct:.1%}")
        Stage 1: 58.8%, Stage 2: 57.1%  # Small overlap
    """
    if base_steps < 0:
        raise ValueError(f"base_steps must be >= 0, got {base_steps}")

    if lightning_steps < 1:
        raise ValueError(f"lightning_steps must be at least 1, got {lightning_steps}")

    if not 0 <= lightning_start < lightning_steps:
        raise ValueError(
            f"lightning_start ({lightning_start}) must be between 0 and {lightning_steps - 1}"
        )

    # Special cases
    if lightning_start == 0 and base_steps == 0:
        return 0

    if lightning_start == 0:
        # If lightning starts at 0, no base stage, total equals base
        return base_steps

    # General formula
    total_base_steps = math.floor(base_steps * lightning_steps / lightning_start)
    # Ensure total is at least as large as base_steps
    total_base_steps = max(total_base_steps, base_steps)

    return total_base_steps
