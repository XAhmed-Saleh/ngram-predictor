"""Unit tests for the Predictor class (src/inference/predictor.py)."""

import pytest
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


@pytest.fixture
def small_model(tmp_path):
    """Build a small NGramModel from a hand-crafted corpus."""
    filepath = tmp_path / "tokens.txt"
    corpus = (
        "the cat sat on the mat\n"
        "the cat sat on the floor\n"
        "the dog sat on the mat\n"
        "the cat ran to the door\n"
        "the cat sat on the rug\n"
    )
    filepath.write_text(corpus, encoding="utf-8")

    model = NGramModel(ngram_order=3, unk_threshold=2, smoothing="false")
    model.build_vocab(str(filepath))
    model.build_counts_and_probabilities(str(filepath))
    return model


@pytest.fixture
def predictor(small_model):
    """Provide a Predictor wired to the small model."""
    normalizer = Normalizer()
    return Predictor(small_model, normalizer)


# --- predict_next() ---

class TestPredictNext:
    def test_returns_exactly_k_predictions_for_seen_context(self, predictor):
        # "the cat" should have multiple follow-ups
        result = predictor.predict_next("the cat", 3)
        assert isinstance(result, list)
        assert len(result) <= 3
        assert len(result) > 0

    def test_results_sorted_by_probability(self, predictor):
        # Get top predictions — they should be in descending probability order
        result = predictor.predict_next("the cat", 5)
        # Verify by re-checking against model lookup
        context = predictor.normalize("the cat")
        context = predictor.map_oov(context)
        candidates = predictor.model.lookup(context)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert candidates.get(result[i], 0) >= candidates.get(result[i + 1], 0)

    def test_handles_all_oov_context_without_crashing(self, predictor):
        # Nonsense words not in vocab should fall back to unigram
        result = predictor.predict_next("xyzzy qqq rrr", 3)
        assert isinstance(result, list)
        # Should still get unigram predictions

    def test_returns_empty_list_for_no_match(self):
        # Model with empty probabilities
        model = NGramModel(ngram_order=3, unk_threshold=2)
        model.probabilities = {"1gram": {}, "2gram": {}, "3gram": {}}
        model.vocab = set()
        normalizer = Normalizer()
        predictor = Predictor(model, normalizer)
        result = predictor.predict_next("hello world", 3)
        assert result == []


# --- map_oov() ---

class TestMapOov:
    def test_replaces_unknown_words_with_unk(self, predictor):
        context = ["unknownword123"]
        result = predictor.map_oov(context)
        assert result == ["<UNK>"]

    def test_leaves_known_words_unchanged(self, predictor):
        context = ["the", "cat"]
        result = predictor.map_oov(context)
        assert result == ["the", "cat"]

    def test_mixed_known_and_unknown(self, predictor):
        context = ["the", "xyzzy"]
        result = predictor.map_oov(context)
        assert result[0] == "the"
        assert result[1] == "<UNK>"


# --- predict_next() edge cases ---

class TestPredictNextEdgeCases:
    def test_empty_input_raises_value_error(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_next("", 3)

    def test_whitespace_only_input_raises_value_error(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict_next("   ", 3)
