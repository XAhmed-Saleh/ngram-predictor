"""
main.py — Single entry point for the N-Gram Next-Word Predictor.

Loads configuration from config/.env, parses --step arguments, wires
all modules via dependency injection, and runs the requested pipeline
step. Supports: dataprep, model, inference, all, evaluate.
"""

import argparse
import logging
import os
import sys
import nltk

from dotenv import load_dotenv


def get_config():
    """Read all configuration variables from the environment.

    Returns
    -------
    dict
        A dictionary of configuration values cast to appropriate types.

    Raises
    ------
    KeyError
        If a required configuration variable is missing.
    """
    required_keys = [
        "TRAIN_RAW_DIR", "TRAIN_TOKENS", "MODEL", "VOCAB",
        "UNK_THRESHOLD", "TOP_K", "NGRAM_ORDER",
    ]
    for key in required_keys:
        if os.getenv(key) is None:
            raise KeyError(f"Missing config variable: {key}. Check config/.env.")

    return {
        "train_raw_dir": os.getenv("TRAIN_RAW_DIR"),
        "eval_raw_dir": os.getenv("EVAL_RAW_DIR", "data/raw/eval/"),
        "train_tokens": os.getenv("TRAIN_TOKENS"),
        "eval_tokens": os.getenv("EVAL_TOKENS", "data/processed/eval_tokens.txt"),
        "model_path": os.getenv("MODEL"),
        "vocab_path": os.getenv("VOCAB"),
        "unk_threshold": int(os.getenv("UNK_THRESHOLD")),
        "top_k": int(os.getenv("TOP_K")),
        "ngram_order": int(os.getenv("NGRAM_ORDER")),
        "smoothing": os.getenv("SMOOTHING", "false"),
    }


def run_dataprep(config):
    """Run the data preparation step.

    Loads raw text, strips Gutenberg headers, normalizes, tokenizes,
    and saves train_tokens.txt (and eval_tokens.txt if eval data exists).

    Parameters
    ----------
    config : dict
        Configuration dictionary from get_config().
    """
    from src.data_prep.normalizer import Normalizer

    logger = logging.getLogger(__name__)
    normalizer = Normalizer()

    # Process training corpus
    logger.info("=== Data Prep: Training Corpus ===")
    raw_text = normalizer.load(config["train_raw_dir"])
    stripped = normalizer.strip_gutenberg(raw_text)
    # Sentence-tokenize BEFORE removing punctuation (periods needed)
    sentences = normalizer.sentence_tokenize(stripped)
    tokenized = [normalizer.word_tokenize(normalizer.normalize(s)) for s in sentences]
    # Filter out empty sentences
    tokenized = [t for t in tokenized if t]
    normalizer.save(tokenized, config["train_tokens"])
    logger.info("Training data prep complete: %d sentences", len(tokenized))

    # Process evaluation corpus if the folder exists and has files
    eval_dir = config["eval_raw_dir"]
    if os.path.isdir(eval_dir) and any(f.endswith(".txt") for f in os.listdir(eval_dir)):
        logger.info("=== Data Prep: Evaluation Corpus ===")
        eval_raw = normalizer.load(eval_dir)
        eval_stripped = normalizer.strip_gutenberg(eval_raw)
        eval_sentences = normalizer.sentence_tokenize(eval_stripped)
        eval_tokenized = [normalizer.word_tokenize(normalizer.normalize(s)) for s in eval_sentences]
        eval_tokenized = [t for t in eval_tokenized if t]
        normalizer.save(eval_tokenized, config["eval_tokens"])
        logger.info("Eval data prep complete: %d sentences", len(eval_tokenized))


def run_model(config):
    """Run the model building step.

    Builds vocabulary, counts n-grams, computes probabilities, and
    saves model.json and vocab.json.

    Parameters
    ----------
    config : dict
        Configuration dictionary from get_config().
    """
    from src.model.ngram_model import NGramModel

    logger = logging.getLogger(__name__)
    logger.info("=== Building N-Gram Model ===")

    model = NGramModel(
        ngram_order=config["ngram_order"],
        unk_threshold=config["unk_threshold"],
        smoothing=config["smoothing"],
    )
    model.build_vocab(config["train_tokens"])
    model.build_counts_and_probabilities(config["train_tokens"])
    model.save_model(config["model_path"])
    model.save_vocab(config["vocab_path"])

    logger.info("Model building complete.")


def run_inference(config):
    """Run the interactive CLI prediction loop.

    Loads the model and enters a loop where the user types text and
    receives top-k predictions.

    Parameters
    ----------
    config : dict
        Configuration dictionary from get_config().
    """
    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NGramModel
    from src.inference.predictor import Predictor

    logger = logging.getLogger(__name__)
    logger.info("=== Starting Inference CLI ===")

    normalizer = Normalizer()
    model = NGramModel(
        ngram_order=config["ngram_order"],
        unk_threshold=config["unk_threshold"],
        smoothing=config["smoothing"],
    )
    model.load(config["model_path"], config["vocab_path"])

    predictor = Predictor(model, normalizer)

    print("\nN-Gram Next-Word Predictor")
    print("Type a phrase to get predictions (or 'quit' to exit).\n")

    while True:
        try:
            user_input = input("> ")
            if user_input.strip().lower() == "quit":
                print("Goodbye.")
                break
            predictions = predictor.predict_next(user_input, config["top_k"])
            print(f"Predictions: {predictions}\n")
        except ValueError as e:
            print(f"Error: {e}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break


def run_evaluate(config):
    """Run the model evaluation step (extra credit).

    Computes perplexity on the held-out evaluation corpus.

    Parameters
    ----------
    config : dict
        Configuration dictionary from get_config().
    """
    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NGramModel
    from src.evaluation.evaluator import Evaluator

    logger = logging.getLogger(__name__)
    logger.info("=== Evaluating Model ===")

    normalizer = Normalizer()
    model = NGramModel(
        ngram_order=config["ngram_order"],
        unk_threshold=config["unk_threshold"],
        smoothing=config["smoothing"],
    )
    model.load(config["model_path"], config["vocab_path"])

    evaluator = Evaluator(model, normalizer)
    evaluator.run(config["eval_tokens"])


def main():
    """Entry point: parse CLI arguments and run the requested step."""
    # Load config/.env FIRST, before any other operations
    load_dotenv("config/.env", override=True)

    # Configure logging
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # --- FINAL NLTK CHECK ---
    logger.info("Checking NLTK resources...")
    # 'punkt' is the model, 'punkt_tab' is the data table
    for resource in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except (LookupError, OSError):
            logger.info(f"Downloading NLTK resource: {resource}...")
            nltk.download(resource)
    # -----------------------------------------------------

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="N-Gram Next-Word Predictor — Capstone Project"
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all", "evaluate"],
        required=True,
        help="Pipeline step to run: dataprep, model, inference, all, or evaluate.",
    )
    args = parser.parse_args()

    try:
        config = get_config()
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        if args.step == "dataprep":
            run_dataprep(config)
        elif args.step == "model":
            run_model(config)
        elif args.step == "inference":
            run_inference(config)
        elif args.step == "evaluate":
            run_evaluate(config)
        elif args.step == "all":
            run_dataprep(config)
            run_model(config)
            run_inference(config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
