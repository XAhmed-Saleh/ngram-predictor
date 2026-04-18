"""Unit tests for the Evaluator class (src/evaluation/evaluator.py)."""

import pytest
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.evaluation.evaluator import Evaluator


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
def evaluator(small_model):
    """Provide an Evaluator wired to the small model."""
    normalizer = Normalizer()
    return Evaluator(small_model, normalizer)


@pytest.fixture
def eval_file(tmp_path):
    """Create a small evaluation file."""
    filepath = tmp_path / "eval.txt"
    corpus = (
        "the cat sat on the mat\n"
        "the dog ran to the door\n"
    )
    filepath.write_text(corpus, encoding="utf-8")
    return str(filepath)


class TestScoreWord:
    def test_returns_negative_float_for_seen_word(self, evaluator):
        # "the" should have a unigram probability < 1, so log2 < 0
        score = evaluator.score_word("the", [])
        assert score is not None
        assert isinstance(score, float)
        assert score < 0

    def test_returns_none_for_zero_probability_word(self, evaluator):
        # A word that has zero probability at all orders
        score = evaluator.score_word("xyznonexistent", ["also_nonexistent"])
        assert score is None

    def test_returns_float_for_word_with_context(self, evaluator):
        # "cat" after "the" should have probability
        score = evaluator.score_word("cat", ["the"])
        assert score is not None
        assert isinstance(score, float)
        assert score < 0


class TestComputePerplexity:
    def test_returns_positive_float_greater_than_one(self, evaluator, eval_file):
        perplexity, evaluated, skipped = evaluator.compute_perplexity(eval_file)
        assert isinstance(perplexity, float)
        assert perplexity > 1.0

    def test_words_evaluated_is_positive(self, evaluator, eval_file):
        perplexity, evaluated, skipped = evaluator.compute_perplexity(eval_file)
        assert evaluated > 0

    def test_skipped_is_non_negative(self, evaluator, eval_file):
        perplexity, evaluated, skipped = evaluator.compute_perplexity(eval_file)
        assert skipped >= 0
