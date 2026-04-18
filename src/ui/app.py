"""
PredictorUI — Streamlit-based web UI for the N-Gram Next-Word Predictor.

Provides a browser-based interface alongside the CLI. Accepts a Predictor
instance via dependency injection and displays top-k predictions. Extra
credit module.
"""

import os
import logging

logger = logging.getLogger(__name__)


class PredictorUI:
    """Streamlit web interface for next-word prediction.

    Receives a Predictor instance via the constructor and exposes
    a text input field with prediction results displayed in real time.
    """

    def __init__(self, predictor):
        """Accept a Predictor instance.

        Parameters
        ----------
        predictor : Predictor
            A fully wired Predictor (with loaded model and normalizer).
        """
        self.predictor = predictor
        self.top_k = int(os.getenv("TOP_K", "3"))

    def get_predictions(self, text=""):
        """Get next-word predictions for the given text.

        Parameters
        ----------
        text : str
            The user's input text from the UI. If empty, returns an
            empty list without crashing.

        Returns
        -------
        list[str]
            A list of predicted next words, or an empty list.
        """
        if not text or not text.strip():
            return []
        try:
            return self.predictor.predict_next(text, self.top_k)
        except ValueError:
            return []

    def run(self):
        """Launch the Streamlit UI with text input and prediction display.

        Returns
        -------
        None
        """
        import streamlit as st

        st.set_page_config(
            page_title="N-Gram Next-Word Predictor",
            page_icon="📖",
            layout="centered",
        )

        st.title("📖 N-Gram Next-Word Predictor")
        st.markdown(
            "Type a sequence of words and the model will predict the "
            "most likely next words using an n-gram language model "
            "trained on Sherlock Holmes novels."
        )

        st.divider()

        user_input = st.text_input(
            "Enter text:",
            placeholder="e.g. holmes looked at",
            key="user_input",
        )

        predict_clicked = st.button("Predict", type="primary")
        if predict_clicked:
            predictions = self.get_predictions(user_input)

            if predictions:
                st.subheader("Top Predictions")
                for i, word in enumerate(predictions, 1):
                    st.markdown(f"**{i}.** `{word}`")
            elif user_input and user_input.strip():
                st.warning("No predictions found for this input.")
            else:
                st.info("Please type at least one word above.")

        st.divider()
        st.caption(
            "N-Gram model with stupid backoff · "
            "Trained on four Sherlock Holmes novels · "
            "ADI Egypt AI Training Program"
        )


def _build_predictor():
    """Wire up the Predictor for the Streamlit app.

    Returns
    -------
    Predictor
        A fully wired Predictor instance.
    """
    from dotenv import load_dotenv

    load_dotenv("config/.env", override=True)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NGramModel
    from src.inference.predictor import Predictor

    model_path = os.getenv("MODEL", "data/model/model.json")
    vocab_path = os.getenv("VOCAB", "data/model/vocab.json")
    ngram_order = int(os.getenv("NGRAM_ORDER", "4"))
    unk_threshold = int(os.getenv("UNK_THRESHOLD", "3"))
    smoothing = os.getenv("SMOOTHING", "false")

    normalizer = Normalizer()
    model = NGramModel(ngram_order=ngram_order, unk_threshold=unk_threshold, smoothing=smoothing)
    model.load(model_path, vocab_path)

    return Predictor(model, normalizer)


# Streamlit entry point — only runs when launched via `streamlit run src/ui/app.py`
def _streamlit_main():
    """Entry point for Streamlit execution."""
    import streamlit as st

    @st.cache_resource
    def get_predictor():
        """Cache the Predictor so it is only built once per session."""
        return _build_predictor()

    predictor = get_predictor()
    ui = PredictorUI(predictor)
    ui.run()


# Guard: only execute when run via Streamlit (not when imported by tests)
try:
    import streamlit as _st_check
    # If streamlit is importable and we're in a streamlit runtime, run the app
    if hasattr(_st_check, "runtime") and _st_check.runtime.exists():
        _streamlit_main()
except (ImportError, ModuleNotFoundError):
    pass
