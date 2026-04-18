"""
Normalizer — Data preparation module for the N-Gram Next-Word Predictor.

Responsible for loading raw text files from Project Gutenberg, stripping
headers/footers, normalizing text (lowercase, remove punctuation, remove
numbers, remove extra whitespace), tokenizing into sentences and words,
and saving the processed output.

Dual-use class:
  - Module 1 (Data Prep): processes whole raw files end-to-end.
  - Module 3 (Inference): normalizes a single input string via normalize().
"""

import os
import re
import logging

import nltk

logger = logging.getLogger(__name__)


class Normalizer:
    """Loads, cleans, tokenizes, and saves the corpus.

    Provides a consistent normalization pipeline used both during
    training data preparation and at inference time. All text that
    the model will ever see passes through normalize() to ensure
    consistent treatment.
    """

    def load(self, folder_path):
        """Load all .txt files from a folder and concatenate their contents.

        Parameters
        ----------
        folder_path : str
            Path to the directory containing .txt files.

        Returns
        -------
        str
            The concatenated raw text of every .txt file in the folder.

        Raises
        ------
        FileNotFoundError
            If the folder does not exist.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env."
            )

        texts = []
        files = sorted(f for f in os.listdir(folder_path) if f.endswith(".txt"))
        logger.info("Found %d .txt file(s) in %s", len(files), folder_path)

        for filename in files:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as fh:
                content = fh.read()
                texts.append(content)
                logger.info("Loaded %s (%d characters)", filename, len(content))

        return "\n".join(texts)

    def strip_gutenberg(self, text):
        """Remove Project Gutenberg header and footer from the text.

        Removes all text before and including each line matching
        ``*** START OF THE PROJECT GUTENBERG EBOOK ... ***`` and all text
        from and including ``*** END OF THE PROJECT GUTENBERG EBOOK ... ***``.
        Handles multiple concatenated books by extracting the content
        between every START/END pair found.

        Parameters
        ----------
        text : str
            Raw text that may contain Gutenberg boilerplate.

        Returns
        -------
        str
            The text with all headers and footers removed.
        """
        start_pattern = r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
        end_pattern = r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"

        starts = [m.end() for m in re.finditer(start_pattern, text, re.IGNORECASE | re.DOTALL)]
        ends = [m.start() for m in re.finditer(end_pattern, text, re.IGNORECASE | re.DOTALL)]

        if starts and ends and len(starts) == len(ends):
            # Extract content between each START/END pair
            parts = []
            for s, e in zip(starts, ends):
                parts.append(text[s:e].strip())
            return "\n".join(parts)
        elif starts and ends:
            # Fallback: use first START and last END
            return text[starts[0]:ends[-1]].strip()
        elif starts:
            return text[starts[0]:].strip()
        elif ends:
            return text[:ends[0]].strip()
        else:
            return text.strip()

    def lowercase(self, text):
        """Convert all characters in the text to lowercase.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Lowercased text.
        """
        return text.lower()

    def remove_punctuation(self, text):
        """Remove all punctuation characters from the text.

        Replaces any character that is not a word character or whitespace
        with an empty string.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with punctuation removed.
        """
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def remove_numbers(self, text):
        """Remove all numeric digits from the text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with digits removed.
        """
        return re.sub(r"\b\w*\d+\w*\b", "", text)

    def remove_whitespace(self, text):
        """Collapse runs of whitespace into single spaces and strip.

        Removes extra spaces, tabs, and blank lines so that the result
        contains only single spaces between words.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Text with normalized whitespace.
        """
        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, text):
        """Apply all normalization steps in order.

        Pipeline: lowercase → remove_punctuation → remove_numbers →
        remove_whitespace.  This is the single method that other modules
        call to normalize text consistently.

        Parameters
        ----------
        text : str
            Raw or user-supplied text.

        Returns
        -------
        str
            Fully normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        """Split text into a list of sentences using NLTK.

        Parameters
        ----------
        text : str
            Normalized text (should already be cleaned).

        Returns
        -------
        list[str]
            A list of sentence strings.
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            logger.error(
                "NLTK 'punkt_tab' data not found. "
                "Run: python -c \"import nltk; nltk.download('punkt_tab')\""
            )
            raise
        logger.debug("Sentence tokenization produced %d sentences", len(sentences))
        return sentences

    def word_tokenize(self, sentence):
        """Split a single sentence into a list of word tokens.

        Parameters
        ----------
        sentence : str
            A single sentence string.

        Returns
        -------
        list[str]
            A list of word tokens with no empty strings.
        """
        tokens = sentence.split()
        return [t for t in tokens if t]

    def save(self, sentences, filepath):
        """Write tokenized sentences to an output file.

        Each sentence is written on its own line with tokens separated
        by single spaces.

        Parameters
        ----------
        sentences : list[list[str]]
            A list of tokenized sentences, where each sentence is a
            list of word strings.
        filepath : str
            Path to the output file.

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            for tokens in sentences:
                line = " ".join(tokens)
                if line.strip():
                    fh.write(line + "\n")
        logger.info("Saved %d sentences to %s", len(sentences), filepath)


def main():
    """Run the data preparation pipeline as a standalone module."""
    from dotenv import load_dotenv

    load_dotenv("config/.env", override=True)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    normalizer = Normalizer()

    train_raw_dir = os.getenv("TRAIN_RAW_DIR", "data/raw/train/")
    train_tokens = os.getenv("TRAIN_TOKENS", "data/processed/train_tokens.txt")

    raw_text = normalizer.load(train_raw_dir)
    stripped = normalizer.strip_gutenberg(raw_text)
    sentences = normalizer.sentence_tokenize(stripped)
    tokenized = [normalizer.word_tokenize(normalizer.normalize(s)) for s in sentences]
    tokenized = [t for t in tokenized if t]
    normalizer.save(tokenized, train_tokens)

    print(f"Data prep complete. Wrote {len(tokenized)} sentences to {train_tokens}")


if __name__ == "__main__":
    main()
