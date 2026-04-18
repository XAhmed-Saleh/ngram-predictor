"""
Predictor — Inference module for the N-Gram Next-Word Predictor.

Accepts a pre-loaded NGramModel and Normalizer via the constructor,
normalizes input text, and returns the top-k predicted next words
sorted by probability. Backoff lookup is delegated to NGramModel.lookup().
"""

import os
import logging

logger = logging.getLogger(__name__)


class Predictor:
    """Normalizes user input and returns top-k next-word predictions.

    Uses dependency injection: receives a pre-loaded NGramModel and
    Normalizer at construction time. Does not load any files itself.
    Backoff logic lives entirely in NGramModel.lookup().
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

    def normalize(self, text):
        """Normalize the input text and extract the context words.

        Calls Normalizer.normalize(text) and extracts the last
        (NGRAM_ORDER - 1) words as the context for lookup.

        Parameters
        ----------
        text : str
            Raw user input text.

        Returns
        -------
        list[str]
            The last (ngram_order - 1) words of the normalized text.
        """
        normalized = self.normalizer.normalize(text)
        words = normalized.split()
        max_context = self.model.ngram_order - 1
        context = words[-max_context:] if len(words) >= max_context else words
        return context

    def map_oov(self, context):
        """Replace out-of-vocabulary words with <UNK>.

        Any word in the context that is not in the model's vocabulary
        is replaced with the <UNK> token.

        Parameters
        ----------
        context : list[str]
            A list of context words.

        Returns
        -------
        list[str]
            Context with OOV words replaced by '<UNK>'.
        """
        mapped = []
        for word in context:
            if word in self.model.vocab:
                mapped.append(word)
            else:
                logger.warning("OOV word encountered: '%s' → mapped to <UNK>", word)
                mapped.append("<UNK>")
        if mapped and all(w == "<UNK>" for w in mapped):
            logger.warning(
                "All context words are out-of-vocabulary. "
                "Predictions will be generic (based on most frequent words)."
            )
        return mapped

    def predict_next(self, text, k):
        """Predict the top-k most probable next words for the given text.

        Orchestrates: normalize → map_oov → NGramModel.lookup() →
        sort by probability descending → return top-k words.

        Parameters
        ----------
        text : str
            The user's input text.
        k : int
            Number of top predictions to return.

        Returns
        -------
        list[str]
            A list of up to k predicted next words, sorted by
            probability (highest first). Empty list if no predictions.

        Raises
        ------
        ValueError
            If the input text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty. Please type at least one word.")

        context = self.normalize(text)
        context = self.map_oov(context)

        # Stupid backoff: try progressively shorter contexts, discounting by 0.4 per level
        _BACKOFF_FACTOR = 0.4
        candidates = {}
        for step, length in enumerate(range(len(context), -1, -1)):
            sub_context = context[-length:] if length > 0 else []
            result = self.model.lookup(sub_context)
            discount = _BACKOFF_FACTOR ** step
            for word, prob in result.items():
                if word not in candidates:
                    candidates[word] = prob * discount
            non_unk = [w for w in candidates if w != "<UNK>"]
            if len(non_unk) >= k:
                break

        if not candidates:
            logger.debug("No predictions found for context: %s", context)
            return []

        # Sort by probability descending, filter out <UNK>, take top-k
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        top_k = [word for word, prob in sorted_candidates if word != "<UNK>"][:k]

        logger.debug("Top-%d predictions for context %s: %s", k, context, top_k)
        return top_k


def main():
    """Run inference as a standalone module for testing."""
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
    ngram_order = int(os.getenv("NGRAM_ORDER", "4"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD", "3"))
    smoothing = os.getenv("SMOOTHING", "false")
    top_k = int(os.getenv("TOP_K", "3"))

    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold, smoothing=smoothing)
    model.load(model_path, vocab_path)

    predictor = Predictor(model, normalizer)

    print("Predictor ready. Type a phrase (or 'quit' to exit):")
    while True:
        try:
            user_input = input("\n> ")
            if user_input.strip().lower() == "quit":
                print("Goodbye.")
                break
            predictions = predictor.predict_next(user_input, top_k)
            print(f"Predictions: {predictions}")
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
