"""
Evaluator — Model evaluation module for the N-Gram Next-Word Predictor.

Computes perplexity on a held-out evaluation corpus using the trained
n-gram model's backoff lookup. Extra credit module.
"""

import math
import os
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates n-gram model quality via perplexity on a held-out corpus.

    Accepts a pre-loaded NGramModel and Normalizer via dependency
    injection. Computes cross-entropy and perplexity by scoring each
    word in the evaluation corpus using the model's backoff lookup.
    """

    def __init__(self, model, normalizer):
        """Accept a pre-loaded NGramModel and Normalizer instance.

        Parameters
        ----------
        model : NGramModel
            A fully loaded n-gram language model.
        normalizer : Normalizer
            A Normalizer instance for text preprocessing.
        """
        self.model = model
        self.normalizer = normalizer

    def score_word(self, word, context):
        """Return log2 P(word | context) via NGramModel.lookup().

        Parameters
        ----------
        word : str
            The target word to score.
        context : list[str]
            The preceding context words (already OOV-mapped).

        Returns
        -------
        float or None
            log2 of the probability if the word is found in the
            lookup results, otherwise None (zero probability at all orders).
        """
        # Use stupid backoff: try progressively shorter contexts with 0.4 discount per level
        _BACKOFF_FACTOR = 0.4
        for step, length in enumerate(range(len(context), -1, -1)):
            sub_context = context[-length:] if length > 0 else []
            candidates = self.model.lookup(sub_context)
            if candidates and word in candidates:
                prob = candidates[word] * (_BACKOFF_FACTOR ** step)
                if prob <= 0:
                    return None
                return math.log2(prob)
        return None

    def compute_perplexity(self, eval_file):
        """Compute perplexity over the full evaluation corpus.

        For each word in the eval file, builds a context from preceding
        words in the same sentence, scores the word, and accumulates
        log-probabilities. Skips words with zero probability. Warns
        if more than 20% of words are skipped.

        Parameters
        ----------
        eval_file : str
            Path to the tokenized evaluation file (one sentence per line).

        Returns
        -------
        tuple[float, int, int]
            (perplexity, words_evaluated, words_skipped)
        """
        total_log_prob = 0.0
        words_evaluated = 0
        words_skipped = 0
        max_context_len = self.model.ngram_order - 1

        with open(eval_file, "r", encoding="utf-8") as fh:
            for line in fh:
                words = line.strip().split()
                if not words:
                    continue

                # Map OOV words
                mapped_words = []
                for w in words:
                    if w in self.model.vocab:
                        mapped_words.append(w)
                    else:
                        mapped_words.append("<UNK>")

                for i, word in enumerate(mapped_words):
                    # Build context from preceding words in this sentence
                    start = max(0, i - max_context_len)
                    context = mapped_words[start:i]

                    score = self.score_word(word, context)
                    if score is not None:
                        total_log_prob += score
                        words_evaluated += 1
                    else:
                        words_skipped += 1

        total_words = words_evaluated + words_skipped
        if total_words > 0:
            skip_ratio = words_skipped / total_words
            if skip_ratio > 0.20:
                logger.warning(
                    "More than 20%% of words skipped (%.1f%%). "
                    "Model may have poor coverage on eval corpus.",
                    skip_ratio * 100,
                )

        if words_evaluated == 0:
            logger.error("No words evaluated — cannot compute perplexity.")
            return float("inf"), 0, words_skipped

        cross_entropy = -total_log_prob / words_evaluated
        perplexity = 2 ** cross_entropy

        return perplexity, words_evaluated, words_skipped

    def run(self, eval_file):
        """Orchestrate perplexity computation and print the result.

        Parameters
        ----------
        eval_file : str
            Path to the tokenized evaluation file.

        Returns
        -------
        None
        """
        logger.info("Evaluating model on %s ...", eval_file)
        perplexity, evaluated, skipped = self.compute_perplexity(eval_file)

        print(f"Perplexity: {perplexity:.2f}")
        print(f"Words evaluated: {evaluated:,}")
        print(f"Words skipped (zero probability): {skipped:,}")

        logger.info("Evaluation complete. Perplexity=%.2f", perplexity)


def main():
    """Run the evaluator as a standalone module."""
    from dotenv import load_dotenv

    load_dotenv("config/.env", override=True)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NGramModel

    model_path = os.getenv("MODEL", "data/model/model.json")
    vocab_path = os.getenv("VOCAB", "data/model/vocab.json")
    eval_tokens = os.getenv("EVAL_TOKENS", "data/processed/eval_tokens.txt")
    ngram_order = int(os.getenv("NGRAM_ORDER", "4"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD", "3"))
    smoothing = os.getenv("SMOOTHING", "false")

    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold, smoothing=smoothing)
    model.load(model_path, vocab_path)

    evaluator = Evaluator(model, normalizer)
    evaluator.run(eval_tokens)


if __name__ == "__main__":
    main()
