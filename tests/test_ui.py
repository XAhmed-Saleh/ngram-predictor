"""Unit tests for the PredictorUI class (src/ui/app.py)."""

import pytest
from unittest.mock import MagicMock

# Import PredictorUI — streamlit top-level code is guarded
from src.ui.app import PredictorUI


@pytest.fixture
def mock_predictor():
    """Create a mock Predictor that returns fixed predictions."""
    predictor = MagicMock()
    predictor.predict_next.return_value = ["watson", "holmes", "the"]
    predictor.model = MagicMock()
    predictor.model.ngram_order = 4
    predictor.model.vocab = {"the", "watson", "holmes", "<UNK>"}
    predictor.normalizer = MagicMock()
    return predictor


@pytest.fixture
def ui(mock_predictor):
    """Provide a PredictorUI wired to the mock predictor."""
    return PredictorUI(mock_predictor)


class TestGetPredictions:
    def test_returns_list_of_strings(self, ui):
        result = ui.get_predictions("holmes looked at")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_handles_empty_input_without_crashing(self, ui):
        result = ui.get_predictions("")
        assert isinstance(result, list)
        assert result == []

    def test_handles_none_input_without_crashing(self, ui):
        result = ui.get_predictions(None)
        assert isinstance(result, list)
        assert result == []

    def test_handles_whitespace_only_input(self, ui):
        result = ui.get_predictions("   ")
        assert isinstance(result, list)
        assert result == []
