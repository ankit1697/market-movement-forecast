"""
finbert_news_classifier.py

News classification using FinBERT embeddings + cosine similarity
to semantic category descriptions.

Designed for use in an MLOps pipeline (e.g., Databricks):
- Import this module in notebooks / jobs.
- Initialize the classifier once.
- Call `classify_texts` or `classify_dataframe` on new data.

Dependencies:
    pip install pandas numpy torch transformers scikit-learn tqdm
"""

import warnings
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Device selection (CPU / CUDA / MPS)
# ----------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ MPS (Mac GPU) is available - using GPU acceleration!")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ CUDA is available - using GPU acceleration!")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (no GPU available)")


# ----------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------
class FinBERTNewsClassifier:
    """
    FinBERT-based news topic classifier.

    Uses:
      - ProsusAI/finbert as an encoder
      - Natural language descriptions for categories
      - Cosine similarity + softmax to produce probabilities

    Example
    -------
    >>> clf = FinBERTNewsClassifier()
    >>> texts = ["Tesla shares rally after strong delivery numbers."]
    >>> results = clf.classify_texts(texts)
    >>> results[0]["category"]
    'Automobile'
    """

    DEFAULT_CATEGORIES = [
        "Corporate",         # Earnings, mergers, acquisitions, corporate news
        "Technology",        # Tech sector, software, hardware, innovation
        "Geo-Political",     # Geopolitical events, international relations, global conflicts
        "US Politics",       # US political events, government policies, elections
        "Economy",           # Economic indicators, GDP, employment, monetary policy, inflation
        "Energy",            # Oil, gas, renewable energy, energy sector
        "Healthcare",        # Pharma, healthcare companies, medical news
        "Automobile",        # Automotive industry, car manufacturers, EVs
        "Airlines",          # Airline industry, aviation, air travel
    ]

    CATEGORY_DESCRIPTIONS = {
        "Corporate": (
            "News about corporate earnings, quarterly results, company revenue and profits, "
            "mergers and acquisitions, CEO announcements, corporate strategy, "
            "business performance, and general corporate news."
        ),
        "Technology": (
            "News about technology companies, software, hardware, digital platforms, "
            "artificial intelligence, cloud computing, cybersecurity, tech innovation, "
            "startups, tech sector stocks, and technology industry developments."
        ),
        "Geo-Political": (
            "News about geopolitical events, international relations, global conflicts, "
            "diplomatic relations between countries, trade wars, sanctions, military actions, "
            "international tensions, foreign policy, global security issues, and geopolitical "
            "developments that affect global markets and economies."
        ),
        "US Politics": (
            "News about US government policies, US political events, US elections, US legislation, "
            "US regulatory changes, US federal government actions, US political parties, "
            "US presidential or congressional actions, US political developments, and US domestic "
            "political news affecting markets."
        ),
        "Economy": (
            "News about economic indicators, GDP growth, inflation, employment data, "
            "unemployment rates, monetary policy, Federal Reserve decisions, economic growth, "
            "recession, economic recovery, macroeconomic trends, international trade, "
            "global economic conditions, and economic developments affecting markets."
        ),
        "Energy": (
            "News about energy companies, oil and gas industry, renewable energy, "
            "solar power, wind energy, crude oil prices, energy sector stocks, "
            "fossil fuels, and energy market developments."
        ),
        "Healthcare": (
            "News about pharmaceutical companies, healthcare providers, medical treatments, "
            "drug development, biotech companies, FDA approvals, clinical trials, "
            "healthcare stocks, medical devices, and healthcare industry news."
        ),
        "Automobile": (
            "News about the automotive industry, car manufacturers, vehicle sales, "
            "electric vehicles, autonomous vehicles, automotive stocks, "
            "vehicle recalls, and auto industry trends."
        ),
        "Airlines": (
            "News about the airline industry, aviation companies, air travel, airline stocks, "
            "airline earnings, flight operations, aircraft orders, airline mergers, "
            "air travel demand, and aviation industry news."
        ),
    }

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        model_name: str = "ProsusAI/finbert",
        temperature: float = 10.0,
    ):
        """
        Parameters
        ----------
        categories : list of str, optional
            Categories to classify into. If None, use default set.
        model_name : str
            HuggingFace model name for FinBERT encoder.
        temperature : float
            Temperature for softmax over similarities (higher = sharper distribution).
        """
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.model_name = model_name
        self.temperature = temperature

        # Load model + tokenizer
        print(f"\nLoading FinBERT model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        print("✓ FinBERT model loaded.")

        # Precompute category embeddings
        self.category_embeddings = self._build_category_embeddings()
        print(f"✓ Category embeddings built for {len(self.categories)} categories.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text into a FinBERT embedding (mean pooled)."""
        text_str = str(text)
        inputs = self.tokenizer(
            text_str,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze()

        if emb.device.type in ("mps", "cuda"):
            emb = emb.cpu()

        return emb.numpy()

    def _build_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for each category from its semantic description."""
        cat_embeds = {}
        for cat in self.categories:
            desc = self.CATEGORY_DESCRIPTIONS.get(cat, f"News about {cat.lower()}.")
            emb = self._encode_text(desc)
            cat_embeds[cat] = emb
        return cat_embeds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify_texts(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Classify a list of raw texts.

        Returns a list of dicts:
            {
              "category": str,
              "confidence": float,
              "all_scores": {category: prob, ...}
            }
        """
        results = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Classifying texts")

        for i in iterator:
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    text_emb = self._encode_text(text)

                    sims = {}
                    for cat in self.categories:
                        cat_emb = self.category_embeddings[cat]
                        sim = cosine_similarity(
                            text_emb.reshape(1, -1),
                            cat_emb.reshape(1, -1),
                        )[0][0]
                        sims[cat] = float(sim)

                    # normalize similarities to [0,1]
                    min_sim, max_sim = min(sims.values()), max(sims.values())
                    if max_sim > min_sim:
                        norm_sims = {
                            k: (v - min_sim) / (max_sim - min_sim)
                            for k, v in sims.items()
                        }
                    else:
                        norm_sims = sims

                    # softmax with temperature
                    exp_sims = {
                        k: np.exp(v * self.temperature)
                        for k, v in norm_sims.items()
                    }
                    total = sum(exp_sims.values())
                    probs = {k: v / total for k, v in exp_sims.items()}

                    best_cat = max(probs, key=probs.get)
                    results.append(
                        {
                            "category": best_cat,
                            "confidence": float(probs[best_cat]),
                            "all_scores": probs,
                        }
                    )
                except Exception as e:
                    print(f"[WARN] Error classifying text: {e}")
                    results.append(
                        {"category": "Unknown", "confidence": 0.0, "all_scores": {}}
                    )

        return results

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_prefix: str = "topic_",
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """
        Classify a DataFrame of news articles.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a column with text (headline/description/combined).
        text_col : str
            Name of column containing text.
        output_prefix : str
            Prefix for output columns: category column will be f"{output_prefix}category"
        batch_size : int
            Batch size for encodings.

        Returns
        -------
        df_out : pd.DataFrame
            Original df with added:
                f"{output_prefix}category"
                f"{output_prefix}confidence"
        """
        texts = df[text_col].fillna("").astype(str).tolist()
        results = self.classify_texts(texts, batch_size=batch_size)

        df_out = df.copy()
        df_out[f"{output_prefix}category"] = [r["category"] for r in results]
        df_out[f"{output_prefix}confidence"] = [r["confidence"] for r in results]
        return df_out


# ----------------------------------------------------------------------
# Simple CLI entry-point (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify news articles into topics using FinBERT."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with a 'text' column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="classified_news.csv",
        help="Path to save classified CSV.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Column name containing article text.",
    )
    args = parser.parse_args()

    df_input = pd.read_csv(args.input)
    classifier = FinBERTNewsClassifier()
    df_out = classifier.classify_dataframe(df_input, text_col=args.text_col)
    df_out.to_csv(args.output, index=False)
    print(f"✓ Saved classified news to {args.output}")
