"""Unit tests for the NGramModel class (src/model/ngram_model.py)."""

import os
import tempfile
import pytest
from src.model.ngram_model import NGramModel


@pytest.fixture
def sample_token_file(tmp_path):
    """Create a small tokenized corpus for testing."""
    filepath = tmp_path / "tokens.txt"
    # Corpus of 5 sentences with repeated patterns
    corpus = (
        "the cat sat on the mat\n"
        "the cat sat on the floor\n"
        "the dog sat on the mat\n"
        "the cat ran to the door\n"
        "a rare word appeared here\n"
    )
    filepath.write_text(corpus, encoding="utf-8")
    return str(filepath)


@pytest.fixture
def trained_model(sample_token_file):
    """Return a trained NGramModel on the small sample corpus."""
    model = NGramModel(ngram_order=3, unk_threshold=2, smoothing="false")
    model.build_vocab(sample_token_file)
    model.build_counts_and_probabilities(sample_token_file)
    return model


# --- build_vocab() ---

class TestBuildVocab:
    def test_replaces_low_frequency_words_with_unk(self, sample_token_file):
        model = NGramModel(ngram_order=3, unk_threshold=2, smoothing="false")
        model.build_vocab(sample_token_file)
        # "rare", "appeared", "here", "word" appear only once → should not be in vocab
        assert "rare" not in model.vocab
        assert "appeared" not in model.vocab

    def test_unk_in_vocabulary(self, sample_token_file):
        model = NGramModel(ngram_order=3, unk_threshold=2, smoothing="false")
        model.build_vocab(sample_token_file)
        assert "<UNK>" in model.vocab

    def test_frequent_words_kept(self, sample_token_file):
        model = NGramModel(ngram_order=3, unk_threshold=2, smoothing="false")
        model.build_vocab(sample_token_file)
        assert "the" in model.vocab
        assert "cat" in model.vocab
        assert "sat" in model.vocab


# --- lookup() ---

class TestLookup:
    def test_returns_non_empty_for_seen_context(self, trained_model):
        # "the cat" is a frequent bigram context
        result = trained_model.lookup(["the", "cat"])
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_returns_non_empty_for_unseen_context_via_backoff(self, trained_model):
        # "zzz" won't be a valid context at any higher order,
        # should fall back to unigram
        result = trained_model.lookup(["<UNK>", "<UNK>"])
        assert isinstance(result, dict)
        assert len(result) > 0  # Should have unigram fallback

    def test_returns_empty_only_when_all_orders_fail(self):
        # Empty model with no probabilities
        model = NGramModel(ngram_order=3, unk_threshold=2)
        model.probabilities = {"1gram": {}, "2gram": {}, "3gram": {}}
        model.vocab = set()
        result = model.lookup(["nonexistent"])
        assert result == {}

    def test_probabilities_sum_to_approximately_one(self, trained_model):
        # For a seen context, the candidates should sum to ~1
        result = trained_model.lookup(["the"])
        if result:
            total = sum(result.values())
            assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"


# --- save and load ---

class TestSaveLoad:
    def test_save_and_load_round_trip(self, trained_model, tmp_path):
        model_path = str(tmp_path / "model.json")
        vocab_path = str(tmp_path / "vocab.json")

        trained_model.save_model(model_path)
        trained_model.save_vocab(vocab_path)

        loaded = NGramModel(ngram_order=3, unk_threshold=2)
        loaded.load(model_path, vocab_path)

        assert loaded.vocab == trained_model.vocab
        assert set(loaded.probabilities.keys()) == set(trained_model.probabilities.keys())

    def test_load_missing_model_raises_file_not_found(self):
        model = NGramModel()
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/model.json", "/nonexistent/vocab.json")
