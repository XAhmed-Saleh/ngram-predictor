# N-Gram Next-Word Predictor

A next-word prediction system built from scratch using an n-gram language model with MLE probabilities and stupid backoff. Trained on four Sherlock Holmes novels by Arthur Conan Doyle (Project Gutenberg), the system takes the last few words typed by the user and returns the top-k most probable next words. Supports optional Laplace smoothing, a Streamlit UI, model evaluation via perplexity, structured logging, exception handling, and unit tests.

## Requirements

- **Python 3.9+**
- **Anaconda** (recommended for environment management)
- All dependencies listed in `requirements.txt`

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ngram-predictor
   ```

2. **Create and activate an Anaconda environment**
   ```bash
   conda create -n ngram python=3.10
   conda activate ngram
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab')"
   ```

5. **Configure environment variables**
   Edit `config/.env` with the correct paths and settings. The default values work out of the box for the standard project structure.

6. **Download raw text files**
   Place the four training books (`.txt`) in `data/raw/train/`:
   - [The Adventures of Sherlock Holmes (1661)](https://www.gutenberg.org/files/1661/1661-0.txt)
   - [The Memoirs of Sherlock Holmes (834)](https://www.gutenberg.org/files/834/834-0.txt)
   - [The Return of Sherlock Holmes (108)](https://www.gutenberg.org/files/108/108.txt)
   - [The Hound of the Baskervilles (2852)](https://www.gutenberg.org/files/2852/2852-0.txt)

   For the Model Evaluator (extra credit), also place:
   - [The Valley of Fear (3289)](https://www.gutenberg.org/files/3289/3289-0.txt) in `data/raw/eval/`

## Usage

### Run the full pipeline
```bash
python main.py --step all
```

### Run individual steps
```bash
# Step 1: Data preparation — produce train_tokens.txt
python main.py --step dataprep

# Step 2: Model training — produce model.json and vocab.json
python main.py --step model

# Step 3: Interactive CLI prediction loop
python main.py --step inference

# Extra credit: Evaluate model perplexity on held-out corpus
python main.py --step evaluate
```

### Launch the Streamlit UI (extra credit)
```bash
streamlit run src/ui/app.py
```

### Run unit tests (extra credit)
```bash
pytest tests/
```

## Example Session

```
$ python main.py --step inference

> holmes looked at
Predictions: ['the', 'him', 'his']

> the game is
Predictions: ['afoot', 'up', 'over']

> quit
Goodbye.
```

## Configuration

All configuration is managed via `config/.env`:

| Variable | Default | Description |
|---|---|---|
| `TRAIN_RAW_DIR` | `data/raw/train/` | Folder containing training `.txt` files |
| `EVAL_RAW_DIR` | `data/raw/eval/` | Folder containing evaluation `.txt` files |
| `TRAIN_TOKENS` | `data/processed/train_tokens.txt` | Output path for tokenized training data |
| `EVAL_TOKENS` | `data/processed/eval_tokens.txt` | Output path for tokenized eval data |
| `MODEL` | `data/model/model.json` | Path to save/load the model |
| `VOCAB` | `data/model/vocab.json` | Path to save/load the vocabulary |
| `UNK_THRESHOLD` | `3` | Words appearing fewer times are replaced with `<UNK>` |
| `TOP_K` | `3` | Number of top predictions to return |
| `NGRAM_ORDER` | `4` | Maximum n-gram order (supports 1 to n) |
| `LOG_LEVEL` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `SMOOTHING` | `false` | Set to `laplace` to enable Laplace add-one smoothing |

## Project Structure

```
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/          # Four training books (.txt)
│   │   └── eval/           # One evaluation book (.txt) — extra credit only
│   ├── processed/
│   │   ├── train_tokens.txt
│   │   └── eval_tokens.txt # Extra credit only
│   └── model/
│       ├── model.json      # Generated — do not commit
│       └── vocab.json      # Generated — do not commit
├── src/
│   ├── data_prep/
│   │   └── normalizer.py      # Normalizer class
│   ├── model/
│   │   └── ngram_model.py     # NGramModel class
│   ├── inference/
│   │   └── predictor.py       # Predictor class
│   ├── ui/
│   │   └── app.py             # PredictorUI class          # Extra credit
│   └── evaluation/
│       └── evaluator.py       # Evaluator class            # Extra credit
├── main.py                    # Single entry point — CLI and wiring
├── tests/
│   ├── test_data_prep.py      # Extra credit
│   ├── test_model.py          # Extra credit
│   ├── test_inference.py      # Extra credit
│   ├── test_ui.py             # Extra credit
│   └── test_evaluation.py     # Extra credit
├── .gitignore
├── requirements.txt
└── README.md
```

## Reference

Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. *Computer Speech & Language, 13*(4), 359–394.
