"""
main.py — Single entry point for the N-Gram Next-Word Predictor.

Loads configuration from config/.env, parses --step arguments, wires
all modules via dependency injection, and runs the requested pipeline
step. Supports: dataprep, model, inference, all, evaluate.
"""


def main():
    """Entry point: parse CLI arguments and run the requested step."""
    # Load config/.env FIRST, before any other operations


if __name__ == "__main__":
    main()
