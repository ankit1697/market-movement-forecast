"""
News Sentiment Analysis using Ollama
Analyzes news articles to determine sentiment (Positive/Negative/Neutral) and confidence scores.
"""

import pandas as pd
import logging
import time
import re
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ollama


# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ========== Ollama Configuration ==========
# Note: Ollama model names typically use format like "gemma2:12b"
# gemma3:4b is faster but less accurate than gemma3:12b
OLLAMA_MODEL = "gemma3:4b"  # Using smaller model for faster processing


# ========== Agent Base Class ==========
class Agent:
    def __init__(self, name, context_prompt, example_prompt,
                 model_name=OLLAMA_MODEL, temperature=0.0):
        self.name = name
        self.context_prompt = context_prompt
        self.example_prompt = example_prompt
        self.model_name = model_name
        self.temperature = temperature
    
    def build_full_prompt(self, text):
        """Combine context, examples, and actual news text"""
        return f"{self.context_prompt}\n\n{self.example_prompt}\n\nNews to analyze:\n{text}"
    
    def process(self, text):
        """Process text and return sentiment result using Ollama"""
        try:
            # Use full text (no truncation)
            full_prompt = self.build_full_prompt(str(text) if text else "")
            
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": 50,  # Limit output tokens (we only need short response)
                    "num_ctx": 4096,     # Context window (increased since we're using full text)
                }
            )
            return {"result": response["response"], "status": "success"}
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {"result": None, "status": f"error: {str(e)}"}


# ========== Sentiment Agent ==========
# Shorter, more concise prompt for faster processing
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
    example_prompt=" ",
    model_name=OLLAMA_MODEL,
    temperature=0.0
)


# ========== Agent Orchestrator ==========
class AgentOrchestrator:
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, agent_key, agent):
        self.agents[agent_key] = agent
    
    def run_agent(self, agent_key, text):
        if agent_key not in self.agents:
            raise ValueError(f"Agent '{agent_key}' not found")
        return self.agents[agent_key].process(text)


# ========== Row Processing ==========
def process_single_row(row_data, orchestrator):
    """Process a single row of news data"""
    idx, row = row_data
    text = row["text"]
    result = orchestrator.run_agent("sentiment", text)
    return {
        "id": row.get("id", idx),
        "time": row["time"],
        "source": row["source"],
        "date": row["date"],
        "text": text,
        "raw_output": result["result"]
    }


# ========== Sentiment & Confidence Extraction ==========
def extract_sentiment_confidence(raw_output):
    """Extract sentiment and confidence from raw API output"""
    match = re.search(r"Sentiment:\s*(\w+)\s*Confidence:\s*([\d.]+)", str(raw_output))
    if match:
        return pd.Series([match.group(1), float(match.group(2))])
    else:
        return pd.Series([None, None])


def main(input_file="data/all_news.csv", output_file="data/news_sentiment.csv", 
         daily_output_file="data/daily_sentiment.csv", sample_size=None, max_workers=None,
         checkpoint_file="data/sentiment_checkpoint.csv", resume=True, checkpoint_interval=50):
    """
    Main function to perform sentiment analysis on news articles.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save sentiment results
        daily_output_file: Path to save daily sentiment aggregation
        sample_size: Number of articles to process (None for all)
        max_workers: Number of parallel workers (None for auto-detect)
        checkpoint_file: Path to save checkpoint for resuming
        resume: Whether to resume from checkpoint if it exists
        checkpoint_interval: Save checkpoint every N articles (default: 50)
    """
    print("="*70)
    print("NEWS SENTIMENT ANALYSIS")
    print("="*70)
    
    # ========== Load Data ==========
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    if sample_size:
        df = df.head(sample_size).copy()
        print(f"Processing sample of {len(df):,} articles")
    else:
        print(f"Processing {len(df):,} articles")
    
    # ========== Check for existing checkpoint ==========
    processed_ids = set()
    if resume and os.path.exists(checkpoint_file):
        print(f"\nFound checkpoint file: {checkpoint_file}")
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            processed_ids = set(checkpoint_df['id'].astype(str))
            print(f"  Found {len(processed_ids):,} already processed articles")
            # Filter out already processed
            df = df[~df['id'].astype(str).isin(list(processed_ids))].copy()
            print(f"  Remaining to process: {len(df):,} articles")
        except Exception as e:
            print(f"  Error reading checkpoint: {e}. Starting fresh.")
            processed_ids = set()
    
    if len(df) == 0:
        print("\nAll articles already processed!")
        return None, None
    
    # ========== Setup Orchestrator ==========
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("sentiment", sentiment_agent)
    
    # ========== Determine optimal workers ==========
    if max_workers is None:
        # Use CPU count, but cap at reasonable limit for Ollama
        cpu_count = multiprocessing.cpu_count()
        # Use 5x CPU cores (aggressive), cap at 100 for Ollama server limits
        max_workers = min(cpu_count * 5, 100)  # 5x CPU cores, max 100
        print(f"\nAuto-detected {cpu_count} CPU cores, using {max_workers} workers")
        print(f"Note: You can manually set more workers via command line (e.g., python script.py 0 50)")
    
    # ========== Parallel Processing ==========
    print(f"\nStarting sentiment analysis with {max_workers} workers...")
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    print(f"Optimizations: Limited output tokens (50), Context window (4096), Auto-detected workers ({max_workers})")
    print(f"Checkpointing: Every {checkpoint_interval} articles to {checkpoint_file}")
    start_time = time.time()
    results_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_row, (idx, row), orchestrator): idx
            for idx, row in df.iterrows()
        }
        
        # Save checkpoint periodically
        last_checkpoint_size = 0
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing news")):
            result = future.result()
            if result:
                results_list.append(result)
                
                # Periodic checkpoint save (every N articles)
                current_size = len(results_list)
                if resume and (current_size - last_checkpoint_size) >= checkpoint_interval:
                    checkpoint_df = pd.DataFrame(results_list)
                    checkpoint_df.to_csv(checkpoint_file, index=False)
                    last_checkpoint_size = current_size
                    print(f"\n  ✓ Checkpoint saved: {current_size:,} articles processed")
                    print(f"\n  ✓ Checkpoint saved: {current_size:,} articles processed")
    
    # Final checkpoint save
    if resume and results_list:
        checkpoint_df = pd.DataFrame(results_list)
        checkpoint_df.to_csv(checkpoint_file, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average time per row: {elapsed_time/len(df):.2f} seconds")
    
    # ========== Sentiment & Confidence Extraction ==========
    print("\nExtracting sentiment and confidence scores...")
    results_df = pd.DataFrame(results_list)
    results_df[["sentiment", "confidence"]] = results_df["raw_output"].apply(extract_sentiment_confidence)
    
    # ========== Merge with checkpoint if resuming ==========
    if resume and processed_ids and os.path.exists(checkpoint_file):
        try:
            checkpoint_df = pd.read_csv(checkpoint_file)
            checkpoint_df[["sentiment", "confidence"]] = checkpoint_df["raw_output"].apply(extract_sentiment_confidence)
            # Combine new results with checkpoint
            results_df = pd.concat([checkpoint_df, results_df], ignore_index=True)
            print(f"  Merged with checkpoint: {len(results_df):,} total articles")
        except Exception as e:
            print(f"  Warning: Could not merge checkpoint: {e}")
    
    # ========== Save Results ==========
    print(f"\nSaving results to {output_file}...")
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(results_df):,} analyzed articles to {output_file}")
    
    # Clean up checkpoint file if processing complete
    if resume and os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"✓ Removed checkpoint file (processing complete)")
        except:
            pass
    
    # ========== Daily Sentiment Aggregation ==========
    print(f"\nCreating daily sentiment aggregation...")
    daily_sentiment_counts = (
        results_df.groupby(['date', 'sentiment'])
                  .size()
                  .reset_index()
    )
    daily_sentiment_counts.columns = ['date', 'sentiment', 'count']
    
    # Pivot so each sentiment is a column
    daily_sentiment_pivot = daily_sentiment_counts.pivot(
        index='date',
        columns='sentiment',
        values='count'
    ).fillna(0).astype(int)
    
    # Add a total column
    daily_sentiment_pivot['Total'] = daily_sentiment_pivot.sum(axis=1)
    
    # Save daily aggregation
    daily_sentiment_pivot.to_csv(daily_output_file)
    print(f"✓ Saved daily sentiment aggregation to {daily_output_file}")
    
    # ========== Summary Statistics ==========
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nSentiment Distribution:")
    sentiment_counts = results_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {sentiment:10s}: {count:6,} ({pct:5.2f}%)")
    
    print("\nConfidence Statistics:")
    print(f"  Mean confidence:   {results_df['confidence'].mean():.4f}")
    print(f"  Median confidence: {results_df['confidence'].median():.4f}")
    print(f"  Min confidence:    {results_df['confidence'].min():.4f}")
    print(f"  Max confidence:    {results_df['confidence'].max():.4f}")
    
    print("\n" + "="*70)
    print("Sentiment analysis complete!")
    print("="*70)
    
    return results_df, daily_sentiment_pivot


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    sample_size = None
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            if sample_size <= 0:
                sample_size = None
        except ValueError:
            pass
    
    max_workers = None  # Auto-detect optimal workers
    if len(sys.argv) > 2:
        try:
            max_workers = int(sys.argv[2])
            if max_workers <= 0:
                max_workers = None  # Auto-detect
        except ValueError:
            pass
    
    main(sample_size=sample_size, max_workers=max_workers)

