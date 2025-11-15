"""Unit tests for shared/validation.py.

Tests validation logic independently of ComfyUI integration.
"""

import pytest

from triple_ksampler.shared import validation


class TestValidateBasicParameters:
    """Tests for validate_basic_parameters function."""

    def test_valid_parameters_no_error(self):
        """Test that valid parameters pass without raising."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=4,
            switch_strategy="Manual switch step",
            switch_step=7,
        )

    def test_valid_parameters_auto_switch_step(self):
        """Test valid parameters with auto switch step (-1)."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=4,
            switch_strategy="Manual switch step",
            switch_step=-1,  # Auto-calculate
        )

    def test_valid_parameters_non_manual_strategy(self):
        """Test valid parameters with non-manual strategy."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=4,
            switch_strategy="50% of steps",
            switch_step=-1,
        )

    def test_lightning_steps_too_low_raises_error(self):
        """Test that lightning_steps < 2 raises ValueError."""
        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            validation.validate_basic_parameters(
                lightning_steps=1,
                lightning_start=0,
                switch_strategy="50% of steps",
                switch_step=-1,
            )

    def test_lightning_steps_zero_raises_error(self):
        """Test that lightning_steps = 0 raises ValueError."""
        with pytest.raises(ValueError, match="lightning_steps must be at least 2"):
            validation.validate_basic_parameters(
                lightning_steps=0,
                lightning_start=0,
                switch_strategy="50% of steps",
                switch_step=-1,
            )

    def test_lightning_start_negative_raises_error(self):
        """Test that negative lightning_start raises ValueError."""
        with pytest.raises(ValueError, match="lightning_start must be within"):
            validation.validate_basic_parameters(
                lightning_steps=10,
                lightning_start=-1,
                switch_strategy="50% of steps",
                switch_step=-1,
            )

    def test_lightning_start_equals_steps_raises_error(self):
        """Test that lightning_start >= lightning_steps raises ValueError."""
        with pytest.raises(ValueError, match="lightning_start must be within"):
            validation.validate_basic_parameters(
                lightning_steps=10,
                lightning_start=10,  # Equal to steps
                switch_strategy="50% of steps",
                switch_step=-1,
            )

    def test_lightning_start_exceeds_steps_raises_error(self):
        """Test that lightning_start > lightning_steps raises ValueError."""
        with pytest.raises(ValueError, match="lightning_start must be within"):
            validation.validate_basic_parameters(
                lightning_steps=10,
                lightning_start=15,
                switch_strategy="50% of steps",
                switch_step=-1,
            )

    def test_switch_step_negative_with_manual_strategy_raises_error(self):
        """Test that switch_step < 0 (not -1) raises ValueError with manual strategy."""
        with pytest.raises(ValueError, match="switch_step.*must be >= 0"):
            validation.validate_basic_parameters(
                lightning_steps=10,
                lightning_start=4,
                switch_strategy="Manual switch step",
                switch_step=-5,  # Invalid negative
            )

    def test_switch_step_exceeds_lightning_steps_raises_error(self):
        """Test that switch_step >= lightning_steps raises ValueError."""
        with pytest.raises(ValueError, match="switch_step.*must be < lightning_steps"):
            validation.validate_basic_parameters(
                lightning_steps=10,
                lightning_start=4,
                switch_strategy="Manual switch step",
                switch_step=10,  # Equal to lightning_steps
            )

    def test_switch_step_less_than_lightning_start_raises_error(self):
        """Test that switch_step < lightning_start raises ValueError."""
        with pytest.raises(ValueError, match="cannot be less than lightning_start"):
            validation.validate_basic_parameters(
                lightning_steps=10,
                lightning_start=4,
                switch_strategy="Manual switch step",
                switch_step=2,  # Less than lightning_start
            )

    def test_boundary_case_lightning_start_zero(self):
        """Test boundary case: lightning_start = 0."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=0,
            switch_strategy="50% of steps",
            switch_step=-1,
        )

    def test_boundary_case_lightning_start_max(self):
        """Test boundary case: lightning_start = lightning_steps - 1."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=9,  # Max valid value
            switch_strategy="50% of steps",
            switch_step=-1,
        )

    def test_boundary_case_switch_step_equals_lightning_start(self):
        """Test boundary case: switch_step = lightning_start."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=4,
            switch_strategy="Manual switch step",
            switch_step=4,  # Equal to lightning_start
        )

    def test_boundary_case_switch_step_max_valid(self):
        """Test boundary case: switch_step = lightning_steps - 1."""
        # Should not raise
        validation.validate_basic_parameters(
            lightning_steps=10,
            lightning_start=0,
            switch_strategy="Manual switch step",
            switch_step=9,  # Max valid value
        )


class TestValidateResolvedParameters:
    """Tests for validate_resolved_parameters function."""

    def test_valid_parameters_stage1_active(self):
        """Test valid parameters with Stage 1 active."""
        # Should not raise
        validation.validate_resolved_parameters(
            lightning_start=4,
            base_steps=10,
        )

    def test_valid_parameters_stage1_skip_mode(self):
        """Test valid parameters in Stage 1 skip mode (both zero)."""
        # Should not raise
        validation.validate_resolved_parameters(
            lightning_start=0,
            base_steps=0,
        )

    def test_base_steps_zero_with_nonzero_lightning_start_raises_error(self):
        """Test that base_steps=0 with lightning_start > 0 raises ValueError."""
        with pytest.raises(ValueError, match="base_steps must be >= 1 when lightning_start > 0"):
            validation.validate_resolved_parameters(
                lightning_start=4,
                base_steps=0,
            )

    def test_base_steps_negative_with_nonzero_lightning_start_raises_error(self):
        """Test that base_steps < 1 with lightning_start > 0 raises ValueError."""
        with pytest.raises(ValueError, match="base_steps must be >= 1 when lightning_start > 0"):
            validation.validate_resolved_parameters(
                lightning_start=1,
                base_steps=0,
            )

    def test_boundary_case_lightning_start_one_base_steps_one(self):
        """Test boundary case: lightning_start=1, base_steps=1."""
        # Should not raise
        validation.validate_resolved_parameters(
            lightning_start=1,
            base_steps=1,
        )

    def test_boundary_case_large_values(self):
        """Test boundary case with large values."""
        # Should not raise
        validation.validate_resolved_parameters(
            lightning_start=100,
            base_steps=200,
        )


class TestValidateSpecialModes:
    """Tests for validate_special_modes function."""

    def test_valid_lightning_only_mode_auto_switch(self):
        """Test valid lightning-only mode with auto switch step."""
        # Should not raise
        validation.validate_special_modes(
            lightning_start=0,
            lightning_steps=10,
            base_steps=0,
            switch_strategy="50% of steps",
            switch_step=-1,
        )

    def test_valid_lightning_only_mode_manual_switch(self):
        """Test valid lightning-only mode with manual switch step."""
        # Should not raise
        validation.validate_special_modes(
            lightning_start=0,
            lightning_steps=10,
            base_steps=0,
            switch_strategy="Manual switch step",
            switch_step=5,
        )

    def test_stage1_skip_mode_with_base_steps_raises_error(self):
        """Test Stage 1 skip mode (lightning_start=0) with base_steps > 0 raises error."""
        # Should raise error - lightning-only mode requires base_steps=0 or -1
        with pytest.raises(
            ValueError, match="Set base_steps=0 or base_steps=-1 for Lightning-only mode"
        ):
            validation.validate_special_modes(
                lightning_start=0,
                lightning_steps=10,
                base_steps=5,
                switch_strategy="50% of steps",
                switch_step=-1,
            )

    def test_normal_mode_no_special_validation(self):
        """Test normal mode (no special mode) passes validation."""
        # Should not raise
        validation.validate_special_modes(
            lightning_start=4,
            lightning_steps=10,
            base_steps=10,
            switch_strategy="Manual switch step",
            switch_step=7,
        )

    def test_boundary_case_lightning_only_switch_step_zero(self):
        """Test lightning-only mode with switch_step=0."""
        # Should not raise
        validation.validate_special_modes(
            lightning_start=0,
            lightning_steps=10,
            base_steps=0,
            switch_strategy="Manual switch step",
            switch_step=0,
        )

    def test_boundary_case_lightning_only_switch_step_max(self):
        """Test lightning-only mode with switch_step at maximum valid value."""
        # Should not raise
        validation.validate_special_modes(
            lightning_start=0,
            lightning_steps=10,
            base_steps=0,
            switch_strategy="Manual switch step",
            switch_step=9,  # lightning_steps - 1
        )

    def test_temp_switch_step_zero_with_base_steps_raises_error(self):
        """Test that temp_switch_step=0 with base_steps > 0 raises error."""
        # switch_step=0 means skipping both Stage 1 and Stage 2, but base_steps > 0
        with pytest.raises(ValueError, match="When skipping both Stage 1 and Stage 2"):
            validation.validate_special_modes(
                lightning_start=0,
                lightning_steps=10,
                base_steps=5,
                switch_strategy="Manual switch step",
                switch_step=0,  # Will use 0 as temp_switch_step
            )
