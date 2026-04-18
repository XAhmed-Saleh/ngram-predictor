"""
NGramModel — N-gram language model with MLE probabilities and stupid backoff.

Responsible for building, storing, and exposing n-gram probability tables
and a backoff lookup across all orders from 1 up to NGRAM_ORDER. Supports
optional Laplace add-one smoothing.
"""

import json
import os
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class NGramModel:
    """Builds, stores, and queries n-gram probability tables with backoff.

    Counts n-grams at all orders from 1 to ngram_order, computes MLE
    probabilities (with optional Laplace smoothing), and provides a
    lookup method that tries the highest-order context first and falls
    back to lower orders when the context is unseen.
    """

    def __init__(self, ngram_order=4, unk_threshold=3, smoothing="false"):
        """Initialize the NGramModel.

        Parameters
        ----------
        ngram_order : int
            Maximum n-gram order (e.g. 4 means up to 4-grams).
        unk_threshold : int
            Words appearing fewer than this many times are replaced with <UNK>.
        smoothing : str
            Smoothing method. 'laplace' for Laplace add-one, 'false' for none.
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold
        self.smoothing = smoothing.lower()
        self.vocab = set()
        self.vocab_list = []
        self.word_counts = defaultdict(int)
        self.probabilities = {}  # {"1gram": {...}, "2gram": {...}, ...}

    def build_vocab(self, token_file):
        """Build vocabulary from the tokenized training file.

        Reads the file, counts every word, replaces words with count
        below unk_threshold with <UNK>, and stores the vocabulary.

        Parameters
        ----------
        token_file : str
            Path to the tokenized training file (one sentence per line).

        Returns
        -------
        None
        """
        self.word_counts = defaultdict(int)

        with open(token_file, "r", encoding="utf-8") as fh:
            for line in fh:
                for word in line.strip().split():
                    self.word_counts[word] += 1

        logger.info("Total unique words before UNK replacement: %d", len(self.word_counts))

        # Build vocab: keep words at or above threshold
        self.vocab = set()
        for word, count in self.word_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(word)

        self.vocab.add("<UNK>")
        self.vocab_list = sorted(self.vocab)

        logger.info("Vocabulary size (after UNK threshold=%d): %d",
                     self.unk_threshold, len(self.vocab))

    def _map_word(self, word):
        """Map a word to <UNK> if it is not in the vocabulary.

        Parameters
        ----------
        word : str
            A single word token.

        Returns
        -------
        str
            The original word if it is in vocab, otherwise '<UNK>'.
        """
        return word if word in self.vocab else "<UNK>"

    def build_counts_and_probabilities(self, token_file):
        """Count all n-grams and compute MLE probabilities.

        Slides a window across every sentence for orders 1 through
        ngram_order. Probabilities are computed together with counts
        to avoid hidden ordering bugs. Supports optional Laplace
        add-one smoothing.

        Parameters
        ----------
        token_file : str
            Path to the tokenized training file (one sentence per line).

        Returns
        -------
        None
        """
        # Counts: {order: {context_key: {next_word: count}}}
        counts = {}
        for order in range(1, self.ngram_order + 1):
            counts[order] = defaultdict(lambda: defaultdict(int))

        total_tokens = 0

        with open(token_file, "r", encoding="utf-8") as fh:
            for line in fh:
                words = [self._map_word(w) for w in line.strip().split()]
                if not words:
                    continue
                total_tokens += len(words)

                for order in range(1, self.ngram_order + 1):
                    for i in range(len(words) - order + 1):
                        ngram = words[i:i + order]
                        if order == 1:
                            # Unigram: context is empty string, word is the token
                            counts[1][""][ngram[0]] += 1
                        else:
                            context_key = " ".join(ngram[:-1])
                            next_word = ngram[-1]
                            counts[order][context_key][next_word] += 1

        logger.info("Total tokens: %d", total_tokens)

        # Compute probabilities
        vocab_size = len(self.vocab)
        self.probabilities = {}

        for order in range(1, self.ngram_order + 1):
            key = f"{order}gram"
            self.probabilities[key] = {}

            if order == 1:
                # Unigram probabilities
                unigram_counts = counts[1][""]
                if self.smoothing == "laplace":
                    for word, count in unigram_counts.items():
                        self.probabilities[key][word] = (count + 1) / (total_tokens + vocab_size)
                    # Add smoothed probability for words not seen as unigrams
                    for word in self.vocab:
                        if word not in self.probabilities[key]:
                            self.probabilities[key][word] = 1 / (total_tokens + vocab_size)
                else:
                    for word, count in unigram_counts.items():
                        self.probabilities[key][word] = count / total_tokens

                logger.debug("1gram: %d entries", len(self.probabilities[key]))
            else:
                for context_key, next_words in counts[order].items():
                    context_total = sum(next_words.values())
                    self.probabilities[key][context_key] = {}

                    if self.smoothing == "laplace":
                        for word, count in next_words.items():
                            self.probabilities[key][context_key][word] = (
                                (count + 1) / (context_total + vocab_size)
                            )
                    else:
                        for word, count in next_words.items():
                            self.probabilities[key][context_key][word] = (
                                count / context_total
                            )

                logger.debug("%s: %d contexts", key, len(self.probabilities[key]))

        logger.info("Built probabilities for orders 1 through %d", self.ngram_order)

    def lookup(self, context):
        """Backoff lookup: try the highest-order context first, fall back.

        Iterates from the highest order down to 1-gram. Returns the
        probability dict from the first order where the context is found.

        Parameters
        ----------
        context : list[str]
            A list of context words (already mapped for OOV).

        Returns
        -------
        dict[str, float]
            A dictionary mapping candidate next words to their
            probabilities. Empty dict if no match at any order.
        """
        for order in range(self.ngram_order, 0, -1):
            key = f"{order}gram"

            if key not in self.probabilities:
                continue

            if order == 1:
                # Unigram: return all unigram probabilities
                result = self.probabilities.get(key, {})
                if result:
                    logger.debug("Backoff reached 1gram, returning %d candidates", len(result))
                    return dict(result)
            else:
                # Higher order: use last (order-1) words as context
                ctx_len = order - 1
                if len(context) >= ctx_len:
                    context_key = " ".join(context[-ctx_len:])
                else:
                    context_key = " ".join(context)

                result = self.probabilities.get(key, {}).get(context_key, {})
                if result:
                    logger.debug("Found match at %s for context '%s': %d candidates",
                                 key, context_key, len(result))
                    return dict(result)

        logger.debug("No match found at any order for context: %s", context)
        return {}

    def save_model(self, model_path):
        """Save all probability tables to a JSON file.

        Parameters
        ----------
        model_path : str
            Path to the output model.json file.

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as fh:
            json.dump(self.probabilities, fh, ensure_ascii=False)
        logger.info("Saved model to %s", model_path)

    def save_vocab(self, vocab_path):
        """Save the vocabulary list to a JSON file.

        Parameters
        ----------
        vocab_path : str
            Path to the output vocab.json file.

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as fh:
            json.dump(self.vocab_list, fh, ensure_ascii=False)
        logger.info("Saved vocab (%d words) to %s", len(self.vocab_list), vocab_path)

    def load(self, model_path, vocab_path):
        """Load model.json and vocab.json into this instance.

        Called once in main() before passing the model to Predictor.

        Parameters
        ----------
        model_path : str
            Path to model.json.
        vocab_path : str
            Path to vocab.json.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If model.json or vocab.json is not found.
        json.JSONDecodeError
            If model.json or vocab.json is malformed.
        """
        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                self.probabilities = json.load(fh)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"model.json not found at {model_path}. Run the Model module first."
            )
        except json.JSONDecodeError:
            raise json.JSONDecodeError(
                f"model.json is malformed at {model_path}. Re-run the Model module.",
                "", 0
            )

        try:
            with open(vocab_path, "r", encoding="utf-8") as fh:
                self.vocab_list = json.load(fh)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"vocab.json not found at {vocab_path}. Run the Model module first."
            )
        except json.JSONDecodeError:
            raise json.JSONDecodeError(
                f"vocab.json is malformed at {vocab_path}. Re-run the Model module.",
                "", 0
            )

        self.vocab = set(self.vocab_list)

        # Determine ngram_order from loaded keys
        orders = []
        for key in self.probabilities:
            order_str = key.replace("gram", "")
            try:
                orders.append(int(order_str))
            except ValueError:
                pass
        if orders:
            self.ngram_order = max(orders)

        logger.info("Loaded model from %s (order=%d, vocab=%d)",
                     model_path, self.ngram_order, len(self.vocab))


def main():
    """Run the model building pipeline as a standalone module."""
    from dotenv import load_dotenv

    load_dotenv("config/.env", override=True)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    token_file = os.getenv("TRAIN_TOKENS", "data/processed/train_tokens.txt")
    model_path = os.getenv("MODEL", "data/model/model.json")
    vocab_path = os.getenv("VOCAB", "data/model/vocab.json")
    ngram_order = int(os.getenv("NGRAM_ORDER", "4"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD", "3"))
    smoothing = os.getenv("SMOOTHING", "false")

    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold, smoothing=smoothing)
    model.build_vocab(token_file)
    model.build_counts_and_probabilities(token_file)
    model.save_model(model_path)
    model.save_vocab(vocab_path)

    print(f"Model built. Saved to {model_path} and {vocab_path}")


if __name__ == "__main__":
    main()
