"""
ollama_news_sentiment.py

News sentiment analysis using an Ollama-hosted Gemma model.

- Uses an Agent abstraction with a compact prompt.
- Designed to work on new DataFrames of news.
- Returns both row-level and daily-aggregated sentiment.

Dependencies:
    pip install pandas tqdm
    # plus a running Ollama server with the chosen model pulled:
    #   ollama pull gemma3:4b
"""

from typing import Optional, Tuple

import logging
import multiprocessing
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
OLLAMA_MODEL = "gemma3:4b"  # fast, good-enough default


# ----------------------------------------------------------------------
# Agent definitions
# ----------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        name: str,
        context_prompt: str,
        example_prompt: str = "",
        model_name: str = OLLAMA_MODEL,
        temperature: float = 0.0,
    ):
        self.name = name
        self.context_prompt = context_prompt
        self.example_prompt = example_prompt
        self.model_name = model_name
        self.temperature = temperature

    def build_full_prompt(self, text: str) -> str:
        return (
            f"{self.context_prompt}\n\n"
            f"{self.example_prompt}\n\n"
            f"News to analyze:\n{text}"
        )

    def process(self, text: str):
        """Call Ollama and return raw model output."""
        try:
            full_prompt = self.build_full_prompt(str(text) if text else "")
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": 50,  # short output
                    "num_ctx": 4096,
                },
            )
            return {"result": response["response"], "status": "success"}
        except Exception as e:
            logger.error(f"[Agent:{self.name}] Error processing text: {e}")
            return {"result": None, "status": f"error: {e}"}


sentiment_context = """Analyze news sentiment for stock prediction. Return ONLY:
Sentiment: Positive/Negative/Neutral
Confidence: 0.00-1.00

Guidelines:
- Positive: Favorable, optimistic, growth, beneficial outcomes
- Negative: Unfavorable, pessimistic, decline, risks, harmful outcomes  
- Neutral: No clear signal, balanced, or cannot determine

If uncertain, use Neutral with Confidence: 0.50."""

sentiment_agent = Agent(
    name="Sentiment Agent",
    context_prompt=sentiment_context,
    example_prompt="",
    model_name=OLLAMA_MODEL,
    temperature=0.0,
)


class AgentOrchestrator:
    def __init__(self):
        self.agents = {}

    def register_agent(self, key: str, agent: Agent):
        self.agents[key] = agent

    def run_agent(self, key: str, text: str):
        if key not in self.agents:
            raise ValueError(f"Agent '{key}' not registered.")
        return self.agents[key].process(text)


# ----------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------
def extract_sentiment_confidence(raw_output: str):
    """
    Extract sentiment + confidence from raw LLM output.

    Expected format in the response:
        Sentiment: Positive
        Confidence: 0.87
    """
    match = re.search(
        r"Sentiment:\s*(\w+)\s*Confidence:\s*([\d.]+)", str(raw_output)
    )
    if match:
        return pd.Series([match.group(1), float(match.group(2))])
    else:
        return pd.Series([None, None])


def _process_single_row(row_data, orchestrator: AgentOrchestrator, text_col: str):
    """Internal helper used in parallel map."""
    idx, row = row_data
    text = row.get(text_col, "")
    result = orchestrator.run_agent("sentiment", text)
    return {
        "row_index": idx,
        "raw_output": result["result"],
        "status": result["status"],
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def analyze_sentiment_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    date_col: Optional[str] = "date",
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run sentiment analysis on a DataFrame of news articles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a text column.
    text_col : str
        Name of the column containing news text.
    date_col : str or None
        Name of column with date; if provided, daily aggregation is returned.
    max_workers : int or None
        Number of threads for parallel calls. None => auto (5 * CPU cores, max 50).

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with one row per input article, containing:
          - 'row_index'
          - 'raw_output'
          - 'status'
          - 'sentiment'
          - 'confidence'
        joined back to the original index.
    daily_sentiment : pd.DataFrame or None
        If date_col is provided and exists, pivoted daily sentiment counts.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Nothing to analyze.")
        return df.copy(), None

    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("sentiment", sentiment_agent)

    # Determine workers
    if max_workers is None:
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(cpu_count * 5, 50)
    logger.info(
        f"Starting sentiment analysis on {len(df):,} rows "
        f"using {max_workers} workers and model '{OLLAMA_MODEL}'."
    )

    start_time = time.time()
    results_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_row, (idx, row), orchestrator, text_col
            ): idx
            for idx, row in df.iterrows()
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing news for sentiment",
        ):
            result = future.result()
            if result:
                results_list.append(result)

    elapsed = time.time() - start_time
    logger.info(
        f"Sentiment analysis complete in {elapsed:.2f}s "
        f"({elapsed/len(df):.2f}s / article)."
    )

    # Build results DF
    results_df = pd.DataFrame(results_list)
    results_df[["sentiment", "confidence"]] = results_df["raw_output"].apply(
        extract_sentiment_confidence
    )

    # Join back to original data if you want to keep all cols aligned
    df_with_sent = df.copy()
    # ensure row_index is an integer index
    results_df = results_df.set_index("row_index").sort_index()
    df_with_sent = df_with_sent.join(results_df[["raw_output", "status", "sentiment", "confidence"]])

    # Daily aggregation if date_col available
    daily_sentiment = None
    if date_col is not None and date_col in df_with_sent.columns:
        tmp = (
            df_with_sent.groupby([date_col, "sentiment"])
            .size()
            .reset_index(name="count")
        )
        daily_pivot = (
            tmp.pivot(index=date_col, columns="sentiment", values="count")
            .fillna(0)
            .astype(int)
        )
        daily_pivot["Total"] = daily_pivot.sum(axis=1)
        daily_sentiment = daily_pivot

    return df_with_sent, daily_sentiment


# ----------------------------------------------------------------------
# Simple CLI entry-point (optional)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Gemma/Ollama sentiment analysis on a CSV of news."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CSV file with a text column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="news_with_sentiment.csv",
        help="Path to save row-level sentiment outputs.",
    )
    parser.add_argument(
        "--daily-output",
        type=str,
        default="daily_sentiment.csv",
        help="Path to save daily sentiment aggregation (requires 'date' column).",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text",
        help="Column name containing article text.",
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default="date",
        help="Column name containing date (optional).",
    )
    args = parser.parse_args()

    df_in = pd.read_csv(args.input)
    df_out, daily = analyze_sentiment_dataframe(
        df_in, text_col=args.text_col, date_col=args.date_col
    )
    df_out.to_csv(args.output, index=False)
    print(f"✓ Saved row-level sentiment to {args.output}")

    if daily is not None:
        daily.to_csv(args.daily_output)
        print(f"✓ Saved daily sentiment aggregation to {args.daily_output}")
