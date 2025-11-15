"""Unit tests for shared/models.py.

Tests model patching and sigma shift application.
"""

from unittest.mock import MagicMock, patch

import pytest

from triple_ksampler.shared import models


class TestPatchModelsWithSigmaShift:
    """Tests for patch_models_with_sigma_shift function."""

    def test_patch_models_successfully(self, mock_model_factory):
        """Test patching all three models successfully."""
        # Arrange
        mock_base = mock_model_factory("base_model")
        mock_high = mock_model_factory("lightning_high")
        mock_low = mock_model_factory("lightning_low")
        sigma_shift = 5.0

        # Mock ModelSamplingSD3
        mock_patcher = MagicMock()
        mock_base_patched = MagicMock(name="base_patched")
        mock_high_patched = MagicMock(name="high_patched")
        mock_low_patched = MagicMock(name="low_patched")

        # Configure mock to return different patched models for each call
        mock_patcher.patch.side_effect = [
            (mock_base_patched,),  # First call (base)
            (mock_high_patched,),  # Second call (lightning_high)
            (mock_low_patched,),  # Third call (lightning_low)
        ]

        with patch("triple_ksampler.shared.models.ModelSamplingSD3", return_value=mock_patcher):
            # Act
            base_result, high_result, low_result = models.patch_models_with_sigma_shift(
                mock_base, mock_high, mock_low, sigma_shift
            )

        # Assert
        assert base_result is mock_base_patched
        assert high_result is mock_high_patched
        assert low_result is mock_low_patched

        # Verify all three models were patched
        assert mock_patcher.patch.call_count == 3
        mock_patcher.patch.assert_any_call(mock_base, 5.0)
        mock_patcher.patch.assert_any_call(mock_high, 5.0)
        mock_patcher.patch.assert_any_call(mock_low, 5.0)

    def test_patch_models_converts_sigma_to_float(self, mock_model_factory):
        """Test that sigma_shift is converted to float."""
        # Arrange
        mock_base = mock_model_factory()
        mock_high = mock_model_factory()
        mock_low = mock_model_factory()

        mock_patcher = MagicMock()
        mock_patcher.patch.return_value = (MagicMock(),)

        with patch("triple_ksampler.shared.models.ModelSamplingSD3", return_value=mock_patcher):
            # Act - Pass integer sigma_shift
            models.patch_models_with_sigma_shift(mock_base, mock_high, mock_low, sigma_shift=3)

        # Assert - Should be converted to float
        for call_args in mock_patcher.patch.call_args_list:
            _, kwargs_or_args = call_args
            if len(kwargs_or_args) > 1:
                sigma_value = kwargs_or_args[1]
            else:
                sigma_value = call_args[0][1]
            assert isinstance(sigma_value, float)
            assert sigma_value == 3.0

    def test_patch_models_raises_import_error_when_unavailable(self, mock_model_factory):
        """Test that ImportError is raised when ModelSamplingSD3 is unavailable."""
        # Arrange
        mock_base = mock_model_factory()
        mock_high = mock_model_factory()
        mock_low = mock_model_factory()

        with patch("triple_ksampler.shared.models.ModelSamplingSD3", None):
            # Act & Assert
            with pytest.raises(ImportError, match="ModelSamplingSD3 is not available"):
                models.patch_models_with_sigma_shift(mock_base, mock_high, mock_low, 5.0)

    def test_patch_models_with_different_sigma_values(self, mock_model_factory):
        """Test patching with different sigma shift values."""
        # Arrange
        mock_base = mock_model_factory()
        mock_high = mock_model_factory()
        mock_low = mock_model_factory()

        mock_patcher = MagicMock()
        mock_patcher.patch.return_value = (MagicMock(),)

        test_sigmas = [3.0, 5.0, 7.5, 10.0]

        for sigma in test_sigmas:
            mock_patcher.reset_mock()

            with patch("triple_ksampler.shared.models.ModelSamplingSD3", return_value=mock_patcher):
                # Act
                models.patch_models_with_sigma_shift(mock_base, mock_high, mock_low, sigma)

            # Assert - All three models patched with same sigma
            assert mock_patcher.patch.call_count == 3
            for call_args in mock_patcher.patch.call_args_list:
                assert call_args[0][1] == sigma

    def test_patch_models_returns_tuple_of_three(self, mock_model_factory):
        """Test that function returns exactly three values."""
        # Arrange
        mock_base = mock_model_factory()
        mock_high = mock_model_factory()
        mock_low = mock_model_factory()

        mock_patcher = MagicMock()
        mock_patcher.patch.return_value = (MagicMock(),)

        with patch("triple_ksampler.shared.models.ModelSamplingSD3", return_value=mock_patcher):
            # Act
            result = models.patch_models_with_sigma_shift(mock_base, mock_high, mock_low, 5.0)

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_patch_models_extracts_first_element_from_tuple(self, mock_model_factory):
        """Test that function correctly extracts first element from patch() tuple."""
        # Arrange
        mock_base = mock_model_factory()
        mock_high = mock_model_factory()
        mock_low = mock_model_factory()

        mock_patcher = MagicMock()
        # patch() returns tuple (model, extra_data)
        mock_patched = MagicMock()
        mock_extra = MagicMock()
        mock_patcher.patch.return_value = (mock_patched, mock_extra)

        with patch("triple_ksampler.shared.models.ModelSamplingSD3", return_value=mock_patcher):
            # Act
            base_result, high_result, low_result = models.patch_models_with_sigma_shift(
                mock_base, mock_high, mock_low, 5.0
            )

        # Assert - Should extract first element [0] from tuple
        assert base_result is mock_patched
        assert high_result is mock_patched
        assert low_result is mock_patched
