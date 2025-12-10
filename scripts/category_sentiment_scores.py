"""
category_sentiment_scores.py

Computes a sentiment score for each news category per day:

    score = (pos - neg) / (pos + neg + neutral)

Requires daily_topic_sentiment DataFrame with columns like:
    Corporate_Positive, Corporate_Negative, Corporate_Neutral, ... etc.

If a category from the provided list does NOT exist in the dataframe,
its score is defined as 0 for all dates.
"""

import pandas as pd
from typing import List, Optional



def compute_category_sentiment_scores(
    daily_topic_sentiment: pd.DataFrame,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:

    """
    Given a DataFrame with columns <category>_<sentiment>,
    compute sentiment scores for each category.

    Parameters
    ----------
    daily_topic_sentiment : pd.DataFrame
        Output from analyze_sentiment_dataframe(...)[1]
        Columns like '<category>_Positive', '<category>_Negative', '<category>_Neutral'.
    categories : list of str, optional
        List of category names to compute scores for
        (e.g., from finbert_news_classifier).
        If None, categories are inferred from the dataframe columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns sentiment_<category> for each category.
        If a category has no data at all, its score is 0 for all dates.
    """

    if daily_topic_sentiment is None or daily_topic_sentiment.empty:
        raise ValueError("daily_topic_sentiment is empty or None.")

    df = daily_topic_sentiment.copy()

    sentiment_suffixes = ["Positive", "Negative", "Neutral"]

    # -------------------------------------------------------------
    # 1. Determine categories
    # -------------------------------------------------------------
    if categories is None:
        # Infer categories from existing columns
        cat_set = set()
        for col in df.columns:
            for suffix in sentiment_suffixes:
                if col.endswith(suffix):
                    # strip "_<suffix>"
                    cat_set.add(col[: -len(suffix) - 1])
                    break
        categories = sorted(cat_set)
    else:
        # Use the provided list exactly as-is
        categories = list(categories)

    # -------------------------------------------------------------
    # 2. Compute sentiment score per category
    # -------------------------------------------------------------
    score_data = {}

    for cat in categories:
        pos_col = f"{cat}_Positive"
        neg_col = f"{cat}_Negative"
        neu_col = f"{cat}_Neutral"

        # If category columns are missing, get() returns a 0-series
        pos = df.get(pos_col, pd.Series(0, index=df.index))
        neg = df.get(neg_col, pd.Series(0, index=df.index))
        neu = df.get(neu_col, pd.Series(0, index=df.index))

        denominator = pos + neg + neu

        # Avoid divide-by-zero: when denominator = 0 → score = 0
        denom_safe = denominator.replace(0, 1)
        raw_score = (pos - neg) / denom_safe

        # But explicitly set score to 0 where denominator was 0
        score = raw_score.where(denominator != 0, 0.0)

        score_data[f"sentiment_{cat}"] = score

    # -------------------------------------------------------------
    # 3. Return scores as a DataFrame indexed by date
    # -------------------------------------------------------------
    score_df = pd.DataFrame(score_data, index=df.index)

    return score_df
