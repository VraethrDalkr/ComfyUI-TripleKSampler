"""
Standalone unit tests for TripleKSampler parameter validation logic.

Tests the validation functions without requiring ComfyUI dependencies.
"""

import pytest


def test_parameter_validation_logic():
    """Test core parameter validation logic without ComfyUI dependencies."""

    # Test lightning_steps validation
    def validate_lightning_steps(lightning_steps):
        if lightning_steps < 2:
            raise ValueError("lightning_steps must be at least 2.")

    with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
        validate_lightning_steps(1)

    # Should not raise for valid values
    validate_lightning_steps(2)
    validate_lightning_steps(8)

    # Test lightning_start range validation
    def validate_lightning_start(lightning_start, lightning_steps):
        if not (0 <= lightning_start < lightning_steps):
            raise ValueError("lightning_start must be within [0, lightning_steps-1].")

    with pytest.raises(ValueError, match="lightning_start must be within"):
        validate_lightning_start(-1, 8)

    with pytest.raises(ValueError, match="lightning_start must be within"):
        validate_lightning_start(8, 8)

    # Should not raise for valid values
    validate_lightning_start(0, 8)
    validate_lightning_start(7, 8)

    # Test switch_step validation for manual strategy
    def validate_manual_switch_step(switch_strategy, switch_step, lightning_steps, lightning_start):
        if switch_strategy == "Manual switch step" and switch_step != -1:
            if switch_step < 0:
                raise ValueError(f"switch_step ({switch_step}) must be >= 0")
            if switch_step >= lightning_steps:
                raise ValueError(
                    f"switch_step ({switch_step}) must be < lightning_steps ({lightning_steps}). Use a smaller value or different strategy."
                )
            if switch_step < lightning_start:
                raise ValueError(
                    f"switch_step ({switch_step}) cannot be less than lightning_start ({lightning_start}). The high-noise model needs at least some steps before switching. If you want low-noise only, set lightning_start=0 as well."
                )

    with pytest.raises(ValueError, match="switch_step \\(-5\\) must be >= 0"):
        validate_manual_switch_step("Manual switch step", -5, 8, 1)

    with pytest.raises(ValueError, match="switch_step \\(10\\) must be < lightning_steps \\(8\\)"):
        validate_manual_switch_step("Manual switch step", 10, 8, 1)

    with pytest.raises(ValueError) as exc_info:
        validate_manual_switch_step("Manual switch step", 2, 8, 5)
    assert "If you want low-noise only, set lightning_start=0 as well" in str(exc_info.value)

    # Should not raise for valid values
    validate_manual_switch_step("Manual switch step", 5, 8, 1)  # Valid
    validate_manual_switch_step("50% of steps", 2, 8, 5)  # Non-manual strategy
    validate_manual_switch_step("Manual switch step", -1, 8, 1)  # Auto-calculate

    # Test base_steps validation
    def validate_base_steps(base_steps, lightning_start):
        if lightning_start > 0 and base_steps == 0:
            raise ValueError(
                "base_steps = 0 is only allowed when lightning_start = 0 (Stage 1 skip mode)"
            )
        if lightning_start > 0 and base_steps != -1 and base_steps < 1:
            raise ValueError("base_steps must be >= 1 when lightning_start > 0.")

    with pytest.raises(ValueError, match="base_steps = 0 is only allowed when lightning_start = 0"):
        validate_base_steps(0, 1)

    with pytest.raises(ValueError, match="base_steps must be >= 1 when lightning_start > 0"):
        validate_base_steps(-2, 2)

    # Should not raise for valid values
    validate_base_steps(0, 0)  # Lightning-only mode
    validate_base_steps(3, 1)  # Normal mode
    validate_base_steps(-1, 2)  # Auto-calculate

    # Test Stage1+Stage2 skip validation
    def validate_stage_skip_scenario(lightning_start, switch_step, base_steps):
        temp_switch_step = switch_step if switch_step != -1 else 4  # Simulate auto-calc
        if lightning_start == 0 and temp_switch_step == 0 and base_steps > 0:
            raise ValueError(
                "When skipping both Stage 1 and Stage 2 (lightning_start=0, switch_step=0), base_steps must be -1 or 0"
            )

    with pytest.raises(
        ValueError, match="When skipping both Stage 1 and Stage 2.*base_steps must be -1 or 0"
    ):
        validate_stage_skip_scenario(0, 0, 5)

    # Should not raise for valid values
    validate_stage_skip_scenario(0, 0, 0)  # Valid: both stages skipped, no base steps
    validate_stage_skip_scenario(0, 0, -1)  # Valid: auto-calculate
    validate_stage_skip_scenario(1, 0, 5)  # Valid: only Stage 2 skipped

    # Test lightning_start=0 with positive base_steps
    def validate_lightning_only_conflict(lightning_start, base_steps):
        if lightning_start == 0 and base_steps > 0:
            raise ValueError(
                f"base_steps ({base_steps}) is ignored when lightning_start=0. Set base_steps=0 or base_steps=-1 for Lightning-only mode, or increase lightning_start to use base denoising."
            )

    with pytest.raises(ValueError) as exc_info:
        validate_lightning_only_conflict(0, 5)
    error_msg = str(exc_info.value)
    assert "base_steps (5) is ignored when lightning_start=0" in error_msg
    assert "Set base_steps=0 or base_steps=-1 for Lightning-only mode" in error_msg

    # Should not raise for valid values
    validate_lightning_only_conflict(0, 0)  # Valid: Lightning-only mode
    validate_lightning_only_conflict(0, -1)  # Valid: auto-calculate
    validate_lightning_only_conflict(1, 5)  # Valid: normal mode

    # Test lightning_start > switch_point
    def validate_switch_point_conflict(lightning_start, switch_step):
        if lightning_start > switch_step:
            raise ValueError(
                f"lightning_start ({lightning_start}) cannot be greater than switch_step ({switch_step}). Either decrease lightning_start or increase switch_step, or use a different switching strategy."
            )

    with pytest.raises(ValueError) as exc_info:
        validate_switch_point_conflict(6, 2)
    error_msg = str(exc_info.value)
    assert "cannot be greater than switch_step" in error_msg
    assert "Either decrease lightning_start or increase switch_step" in error_msg

    # Should not raise for valid values
    validate_switch_point_conflict(2, 6)  # Valid
    validate_switch_point_conflict(3, 3)  # Valid: equal


def test_error_message_quality():
    """Test that error messages are helpful and actionable."""

    # Test that all error messages provide actionable guidance
    def validate_with_suggestion(switch_step, lightning_start):
        if switch_step < lightning_start:
            raise ValueError(
                f"switch_step ({switch_step}) cannot be less than lightning_start ({lightning_start}). The high-noise model needs at least some steps before switching. If you want low-noise only, set lightning_start=0 as well."
            )

    with pytest.raises(ValueError) as exc_info:
        validate_with_suggestion(2, 5)

    error_msg = str(exc_info.value)
    # Check that error message contains the suggestion
    assert "If you want low-noise only, set lightning_start=0 as well" in error_msg
    # Check that it explains the problem
    assert "The high-noise model needs at least some steps before switching" in error_msg
    # Check that it includes the specific values
    assert "switch_step (2)" in error_msg
    assert "lightning_start (5)" in error_msg


def test_edge_case_combinations():
    """Test complex edge case combinations."""

    def validate_all_parameters(
        lightning_steps, lightning_start, base_steps, switch_strategy, switch_step
    ):
        """Simplified version of the full validation logic."""

        # Basic validation
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
                    f"switch_step ({switch_step}) cannot be less than lightning_start ({lightning_start}). The high-noise model needs at least some steps before switching. If you want low-noise only, set lightning_start=0 as well."
                )

        # Base steps validation (-1 means auto-calculate, which is always valid)
        if lightning_start > 0 and base_steps == 0:
            raise ValueError(
                "base_steps = 0 is only allowed when lightning_start = 0 (Stage 1 skip mode)"
            )
        if lightning_start > 0 and base_steps != -1 and base_steps < 1:
            raise ValueError("base_steps must be >= 1 when lightning_start > 0.")

        # Lightning-only conflict
        if lightning_start == 0 and base_steps > 0:
            raise ValueError(
                f"base_steps ({base_steps}) is ignored when lightning_start=0. Set base_steps=0 or base_steps=-1 for Lightning-only mode, or increase lightning_start to use base denoising."
            )

    # Test various valid combinations
    validate_all_parameters(8, 1, 3, "50% of steps", -1)  # Normal case
    validate_all_parameters(8, 0, 0, "50% of steps", -1)  # Lightning-only
    validate_all_parameters(8, 2, -1, "T2V boundary", -1)  # Auto-calculate
    validate_all_parameters(8, 1, 5, "Manual switch step", 4)  # Manual switching

    # Test invalid combinations
    with pytest.raises(ValueError):
        validate_all_parameters(1, 0, 0, "50% of steps", -1)  # lightning_steps too small

    with pytest.raises(ValueError):
        validate_all_parameters(8, 8, 3, "50% of steps", -1)  # lightning_start out of range

    with pytest.raises(ValueError):
        validate_all_parameters(8, 0, 5, "50% of steps", -1)  # Lightning-only conflict

    with pytest.raises(ValueError):
        validate_all_parameters(8, 1, 0, "50% of steps", -1)  # base_steps=0 with lightning_start>0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
