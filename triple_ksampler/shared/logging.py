"""Logging formatters and utilities for TripleKSampler.

This module provides pure functions for formatting log messages, calculating percentages,
and creating compact toast notification messages. All functions are stateless and have no
side effects, making them easy to test and reuse.
"""

import re


def calculate_percentage(numerator: float, denominator: float) -> float:
    """Calculate percentage with division-by-zero protection and bounds clamping.

    Safely converts a numerator/denominator pair to a percentage value, handling
    edge cases like division by zero and clamping results to the valid range [0, 100].

    Args:
        numerator: Numerator value (can be any numeric type)
        denominator: Denominator value (can be any numeric type)

    Returns:
        float: Percentage value rounded to 1 decimal place, clamped to [0.0, 100.0]

    Example:
        >>> calculate_percentage(5, 20)
        25.0
        >>> calculate_percentage(15, 10)
        100.0
        >>> calculate_percentage(5, 0)
        0.0
    """
    if denominator == 0:
        return 0.0
    pct = (float(numerator) / float(denominator)) * 100.0
    return round(max(0.0, min(100.0, pct)), 1)


def format_stage_range(start: int, end: int, total: int) -> str:
    """Format a human-readable string describing step ranges and denoising percentages.

    Creates informative log messages showing both step ranges and corresponding
    denoising percentages for each sampling stage. Safely handles edge cases by
    clamping values to valid ranges.

    Args:
        start: Starting step number (will be clamped to >= 0)
        end: Ending step number (will be clamped to >= start)
        total: Total steps in the schedule (will be clamped to >= 1)

    Returns:
        str: Formatted string like "steps 0-5 of 20 (denoising 0.0%–25.0%)"

    Example:
        >>> format_stage_range(0, 5, 20)
        'steps 0-5 of 20 (denoising 0.0%–25.0%)'
        >>> format_stage_range(10, 20, 20)
        'steps 10-20 of 20 (denoising 50.0%–100.0%)'
    """
    start_safe = int(max(0, start))
    end_safe = int(max(start_safe, end))
    total_safe = int(max(1, total))
    pct_start = calculate_percentage(start_safe, total_safe)
    pct_end = calculate_percentage(end_safe, total_safe)
    return (
        f"steps {start_safe}-{end_safe} of {total_safe} (denoising {pct_start:.1f}%–{pct_end:.1f}%)"
    )


def format_base_calculation_compact(base_calc_info: str) -> str:
    """Format base calculation info for compact toast display.

    Converts verbose calculation log messages into concise format suitable
    for UI toast notifications. Handles different calculation scenarios:
    - Auto-calculated with perfect alignment (simple_math, mathematical_search)
    - Auto-calculated fallback (no perfect alignment)
    - Manual base_steps with auto-calculated total

    Args:
        base_calc_info: Original verbose calculation message from alignment module

    Returns:
        str: Compact formatted message for toast display. If no pattern matches,
            returns the original string unchanged.

    Example:
        >>> msg = "Auto-calculated base_steps = 10, total_base_steps = 50 (simple math)"
        >>> format_base_calculation_compact(msg)
        'Base steps: 10, Total: 50 (simple math)'
        >>> msg = "Auto-calculated total_base_steps = 17 for manual base_steps = 10"
        >>> format_base_calculation_compact(msg)
        'Base steps: 10, Total: 17 (manual)'
    """
    # Pattern: "Auto-calculated base_steps = X, total_base_steps = Y (method)"
    match1 = re.search(
        r"Auto-calculated base_steps = (\d+), total_base_steps = (\d+) \(([^)]+)\)",
        base_calc_info,
    )
    if match1:
        base_steps, total_steps, method = match1.groups()
        return f"Base steps: {base_steps}, Total: {total_steps} ({method})"

    # Pattern: "Auto-calculated base_steps = X (fallback - no perfect alignment found)"
    match2 = re.search(r"Auto-calculated base_steps = (\d+) \(([^)]+)\)", base_calc_info)
    if match2:
        base_steps, _ = match2.groups()  # method_desc not used but preserved for clarity
        return f"Base steps: {base_steps} (fallback)"

    # Pattern: "Auto-calculated total_base_steps = X for manual base_steps = Y"
    match3 = re.search(
        r"Auto-calculated total_base_steps = (\d+) for manual base_steps = (\d+)",
        base_calc_info,
    )
    if match3:
        total_steps, manual_steps = match3.groups()
        return f"Base steps: {manual_steps}, Total: {total_steps} (manual)"

    # Fallback: return original if no pattern matches
    return base_calc_info


def format_switch_info_compact(switch_info: str) -> str:
    """Format model switching info for compact toast display.

    Converts verbose switching strategy log messages into concise format
    suitable for UI toast notifications. Handles both boundary-based and
    step-based switching strategies, including sigma shift refinement info.

    Args:
        switch_info: Original verbose switching message from strategies module

    Returns:
        str: Compact formatted message for toast display. If no pattern matches,
            returns the original string unchanged.

    Example:
        >>> msg = "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10"
        >>> format_switch_info_compact(msg)
        'Switch: T2V boundary → step 7 of 10'
        >>> msg = "Model switching: 50% of steps → switch at step 5 of 10"
        >>> format_switch_info_compact(msg)
        'Switch: 50% of steps → step 5 of 10'
        >>> msg = "Model switching: T2V boundary (boundary = 0.875) → switch at step 7 of 10 [Refined shift: 5.00→6.94]"
        >>> format_switch_info_compact(msg)
        'Switch: T2V boundary → step 7 of 10 (σ-shift: 5.00→6.94)'
    """
    # Extract optional refinement suffix first
    refinement_suffix = ""
    refinement_match = re.search(r"\[Refined shift: ([\d.]+)→([\d.]+)\]", switch_info)
    if refinement_match:
        initial_shift, refined_shift = refinement_match.groups()
        refinement_suffix = f"\n  (σ-shift refined: {initial_shift} → {refined_shift})"
        # Remove refinement bracket from main string for pattern matching
        switch_info = re.sub(r"\s*\[Refined shift: [\d.]+→[\d.]+\]", "", switch_info)

    # Pattern: "Model switching: STRATEGY (boundary = VALUE) → switch at step X of Y"
    match1 = re.search(
        r"Model switching: ([^(]+) \(boundary = ([^)]+)\) → switch at step (\d+) of (\d+)",
        switch_info,
    )
    if match1:
        strategy, _, switch_step, total_steps = match1.groups()  # boundary not used but preserved
        return (
            f"Switch: {strategy.strip()} → step {switch_step} of {total_steps}{refinement_suffix}"
        )

    # Pattern: "Model switching: STRATEGY → switch at step X of Y"
    match2 = re.search(r"Model switching: ([^→]+) → switch at step (\d+) of (\d+)", switch_info)
    if match2:
        strategy, switch_step, total_steps = match2.groups()
        return (
            f"Switch: {strategy.strip()} → step {switch_step} of {total_steps}{refinement_suffix}"
        )

    # Fallback: return original if no pattern matches
    return switch_info
