"""
gpt4o_sentiment.py

Sentiment analysis for news articles using GPT-4o-mini.

Outputs:
- sentiment: Positive / Negative / Neutral
- confidence: float (0–1)

Dependencies:
  pip install openai pandas tqdm
"""

from typing import List, Tuple, Optional
import itertools
import re

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# -------------------------------------------------------------------
# Setup OpenAI client (requires OPENAI_API_KEY in env)
# -------------------------------------------------------------------
client = OpenAI()

# -------------------------------------------------------------------
# Prompt template
# -------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a financial sentiment analyst.
Given a news headline or short article, classify its market sentiment.

Return STRICTLY in this format:

Sentiment: Positive/Negative/Neutral
Confidence: 0.00–1.00

Rules:
- Positive → likely to increase market/stock confidence
- Negative → harmful, pessimistic, risky, or loss-related
- Neutral → unclear, mixed, or no directional impact
"""

# -------------------------------------------------------------------
# Core function for a single text
# -------------------------------------------------------------------
def analyze_single_text(text: str) -> Tuple[str, float, str]:
    """
    Returns (sentiment, confidence, raw_output)
    """
    if not isinstance(text, str) or not text.strip():
        return ("Neutral", 0.50, "")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"{SYSTEM_PROMPT}\n\nText:\n{text}",
        max_output_tokens=50,
    )

    output = response.output_text

    # Extract sentiment + confidence
    match = re.search(
        r"Sentiment:\s*(Positive|Negative|Neutral)\s*Confidence:\s*([\d.]+)",
        output,
        re.IGNORECASE,
    )

    if match:
        sent = match.group(1).capitalize()
        conf = float(match.group(2))
        return sent, conf, output
    else:
        return ("Neutral", 0.50, output)


# -------------------------------------------------------------------
# Batch sentiment for a list of texts
# -------------------------------------------------------------------
def analyze_texts(texts: List[str]) -> pd.DataFrame:
    results = []
    for t in tqdm(texts, desc="Sentiment analysis (GPT-4o-mini)"):
        sentiment, confidence, raw = analyze_single_text(t)
        results.append(
            {
                "sentiment": sentiment,
                "confidence": confidence,
                "raw_output": raw,
            }
        )
    return pd.DataFrame(results)


# -------------------------------------------------------------------
# DataFrame wrapper
# -------------------------------------------------------------------
def analyze_sentiment_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    date_col: Optional[str] = None,
):
    """
    Adds sentiment + confidence columns to DF.

    Returns
    -------
    df_out : pd.DataFrame
        Original df with added:
            - 'sentiment'
            - 'confidence'
            - 'raw_output'
    daily_topic_sentiment : pd.DataFrame or None
        If date_col and 'topic_category' exist:
          index: date
          columns: <topic_category>_<sentiment> for all permutations
                   (e.g. 'Corporate_Positive', 'Corporate_Negative', ...)
          values: counts (0 if never present)
          plus 'Total_Articles' column.
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    sent_df = analyze_texts(texts)

    df_out = df.copy()
    df_out["sentiment"] = sent_df["sentiment"]
    df_out["confidence"] = sent_df["confidence"]
    df_out["raw_output"] = sent_df["raw_output"]

    # -------------------------------------------------------------
    # Enhanced daily aggregation by (date × topic_category × sentiment)
    # with all permutations as columns (missing combos → 0)
    # -------------------------------------------------------------
    daily_topic_sentiment = None

    if date_col and date_col in df_out.columns and "topic_category" in df_out.columns:
        # 1. Count news by (date, topic_category, sentiment)
        grouped = (
            df_out.groupby([date_col, "topic_category", "sentiment"])
            .size()
            .reset_index(name="count")
        )

        # 2. Pivot to wide format based on observed combos
        daily_topic_sentiment = grouped.pivot_table(
            index=date_col,
            columns=["topic_category", "sentiment"],
            values="count",
            fill_value=0,
        )

        # 3. Build full set of (category × sentiment) permutations
        categories = (
            df_out["topic_category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        sentiments = ["Positive", "Negative", "Neutral"]

        full_cols = pd.MultiIndex.from_product(
            [categories, sentiments],
            names=["topic_category", "sentiment"],
        )

        # 4. Reindex to ensure every combo exists as a column
        daily_topic_sentiment = daily_topic_sentiment.reindex(
            columns=full_cols,
            fill_value=0,
        )

        # 5. Flatten MultiIndex columns into "<category>_<sentiment>"
        daily_topic_sentiment.columns = [
            f"{topic}_{sentiment}"
            for (topic, sentiment) in daily_topic_sentiment.columns
        ]

        # 6. Add total articles per day
        daily_topic_sentiment["Total_Articles"] = daily_topic_sentiment.sum(axis=1)

    return df_out, daily_topic_sentiment
