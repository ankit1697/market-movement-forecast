"""
News Classification using FinBERT with MPS (Mac GPU) acceleration
Uses FinBERT embeddings with keyword-based category matching for better confidence scores.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Check for MPS (Mac GPU) availability
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ MPS (Mac GPU) is available - using GPU acceleration!")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ CUDA is available - using GPU acceleration!")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU (no GPU available)")


def load_data(data_path='data/all_news.csv', sample_size=None):
    """Load news data from CSV file."""
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    if sample_size:
        df = df.head(sample_size).copy()
        print(f"Loaded sample of {len(df):,} articles")
    else:
        print(f"Loaded {len(df):,} articles")
    return df


def load_finbert_model():
    """Load FinBERT model and move to device."""
    print("\nLoading FinBERT model (ProsusAI/finbert)...")
    print("This may take a minute on first run (downloading model)...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model = model.to(DEVICE)
    model.eval()
    print("✓ FinBERT model loaded successfully!")
    return tokenizer, model


def get_category_embeddings(tokenizer, model, categories):
    """
    Get embeddings for each category using natural language descriptions.
    This approach is more flexible and comprehensive than keyword matching,
    as it captures semantic meaning rather than exact word matches.
    """
    print("\nGenerating category embeddings from descriptions...")
    
    # Natural language descriptions that capture the semantic meaning of each category
    category_descriptions = {
        "Corporate": (
            "News about corporate earnings, quarterly results, company revenue and profits, "
            "mergers and acquisitions, M&A deals, CEO announcements, corporate strategy, "
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
            "News about automotive industry, car manufacturers, vehicle sales, automobile companies, "
            "car production, electric vehicles, autonomous vehicles, automotive stocks, "
            "auto industry trends, vehicle recalls, and automotive sector developments."
        ),
        "Airlines": (
            "News about airline industry, aviation companies, air travel, airline stocks, "
            "airline earnings, flight operations, aircraft orders, airline mergers, "
            "aviation sector, air travel demand, airline routes, and aviation industry news."
        )
    }
    
    category_embeddings = {}
    
    for category in categories:
        # Use the natural language description for the category
        description = category_descriptions.get(category, f"News about {category.lower()}.")
        
        # Tokenize and get embeddings
        inputs = tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            # Move to CPU for numpy
            if embeddings.device.type in ["mps", "cuda"]:
                embeddings = embeddings.cpu()
            category_embeddings[category] = embeddings.numpy()
    
    print(f"✓ Generated embeddings for {len(category_embeddings)} categories using semantic descriptions")
    return category_embeddings


def classify_texts(tokenizer, model, texts, category_embeddings, categories, batch_size=32):
    """
    Classify texts using FinBERT embeddings and cosine similarity.
    
    Args:
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        texts: List of texts to classify
        category_embeddings: Dictionary of category embeddings
        categories: List of category names
        batch_size: Batch size for processing
        
    Returns:
        List of classification results
    """
    print(f"\nClassifying {len(texts):,} articles...")
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing articles"):
        batch_texts = texts[i:i+batch_size]
        
        for text in batch_texts:
            try:
                # Truncate text if too long
                text_str = str(text)[:512]
                
                # Get text embedding
                inputs = tokenizer(
                    text_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    # Move to CPU for numpy
                    if text_embedding.device.type in ["mps", "cuda"]:
                        text_embedding = text_embedding.cpu()
                    text_embedding = text_embedding.numpy()
                
                # Calculate cosine similarity with each category
                similarities = {}
                for category in categories:
                    cat_embedding = category_embeddings[category]
                    # Reshape for cosine_similarity
                    sim = cosine_similarity(
                        text_embedding.reshape(1, -1),
                        cat_embedding.reshape(1, -1)
                    )[0][0]
                    similarities[category] = float(sim)
                
                # Convert similarities to probabilities using softmax
                # Use a higher temperature to sharpen the distribution
                # Also normalize similarities to [0, 1] range first
                min_sim = min(similarities.values())
                max_sim = max(similarities.values())
                if max_sim > min_sim:
                    normalized_sims = {k: (v - min_sim) / (max_sim - min_sim) for k, v in similarities.items()}
                else:
                    normalized_sims = similarities
                
                # Apply temperature scaling (higher = sharper distribution)
                temperature = 10.0
                exp_sims = {k: np.exp(v * temperature) for k, v in normalized_sims.items()}
                total = sum(exp_sims.values())
                probabilities = {k: v / total for k, v in exp_sims.items()}
                
                # Get best category
                best_category = max(probabilities, key=probabilities.get)
                best_confidence = probabilities[best_category]
                
                results.append({
                    'category': best_category,
                    'confidence': best_confidence,
                    'all_scores': probabilities
                })
                
            except Exception as e:
                print(f"\nError classifying article: {e}")
                results.append({
                    'category': 'Unknown',
                    'confidence': 0.0,
                    'all_scores': {}
                })
    
    return results


def analyze_results(df, results):
    """Analyze and display classification results."""
    df['category'] = [r['category'] for r in results]
    df['category_confidence'] = [r['confidence'] for r in results]
    
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)
    
    # Category distribution
    print("\nCategory Distribution:")
    category_counts = df['category'].value_counts()
    for category in category_counts.index:
        count = category_counts[category]
        pct = count / len(df) * 100
        print(f"  {category:20s}: {count:6,} ({pct:5.2f}%)")
    
    # Confidence statistics
    print("\nConfidence Statistics:")
    print(f"  Mean confidence:   {df['category_confidence'].mean():.4f}")
    print(f"  Median confidence: {df['category_confidence'].median():.4f}")
    print(f"  Min confidence:    {df['category_confidence'].min():.4f}")
    print(f"  Max confidence:    {df['category_confidence'].max():.4f}")
    
    # Low confidence articles
    low_conf = df[df['category_confidence'] < 0.3]
    print(f"\nArticles with low confidence (<0.3): {len(low_conf):,} ({len(low_conf)/len(df)*100:.2f}%)")
    
    # High confidence articles
    high_conf = df[df['category_confidence'] >= 0.7]
    print(f"Articles with high confidence (>=0.7): {len(high_conf):,} ({len(high_conf)/len(df)*100:.2f}%)")
    
    # Category by source (if available)
    if 'source' in df.columns:
        print("\nCategory Distribution by Source:")
        source_category = pd.crosstab(df['source'], df['category'], margins=True)
        print(source_category)
    
    print("\n" + "="*70)
    
    return df


def main(sample_size=None):
    """
    Main execution function.
    
    Args:
        sample_size: Number of articles to process (None for all)
    """
    categories = [
        "Corporate",         # Earnings, mergers, acquisitions, corporate news
        "Technology",        # Tech sector, software, hardware, innovation
        "Geo-Political",     # Geopolitical events, international relations, global conflicts
        "US Politics",       # US political events, government policies, elections
        "Economy",           # Economic indicators, GDP, employment, monetary policy, inflation
        "Energy",            # Oil, gas, renewable energy, energy sector
        "Healthcare",        # Pharmaceuticals, healthcare companies, medical news
        "Automobile",        # Automotive industry, car manufacturers, vehicle sales
        "Airlines"           # Airline industry, aviation, air travel, airline stocks
    ]
    
    print("="*70)
    print("NEWS CLASSIFICATION USING FinBERT (with MPS acceleration)")
    print("="*70)
    print(f"\nCategories: {', '.join(categories)}")
    
    # Load data
    df = load_data('data/all_news.csv', sample_size=sample_size)
    
    # Prepare texts
    print("\nPreparing texts...")
    texts = []
    for idx, row in df.iterrows():
        text = row.get('text', '')
        if pd.isna(text) or len(str(text).strip()) == 0:
            text = f"{row.get('headline', '')} {row.get('description', '')}"
        texts.append(str(text))
    
    # Load FinBERT model
    tokenizer, model = load_finbert_model()
    
    # Get category embeddings
    category_embeddings = get_category_embeddings(tokenizer, model, categories)
    
    # Classify articles
    results = classify_texts(tokenizer, model, texts, category_embeddings, categories, batch_size=16)
    
    # Analyze results
    df_classified = analyze_results(df, results)
    
    # Save results
    output_path = 'data/all_news_classified.csv'
    print(f"\nSaving classified data to {output_path}...")
    df_classified.to_csv(output_path, index=False)
    print(f"✓ Saved {len(df_classified):,} classified articles to {output_path}")
    
    # Save summary
    summary_path = 'data/classification_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("NEWS CLASSIFICATION SUMMARY (FinBERT)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total articles classified: {len(df_classified):,}\n\n")
        f.write("Category Distribution:\n")
        for category, count in df_classified['category'].value_counts().items():
            pct = count / len(df_classified) * 100
            f.write(f"  {category:20s}: {count:6,} ({pct:5.2f}%)\n")
        f.write(f"\nMean confidence: {df_classified['category_confidence'].mean():.4f}\n")
        f.write(f"Median confidence: {df_classified['category_confidence'].median():.4f}\n")
        f.write(f"High confidence (>=0.7): {len(df_classified[df_classified['category_confidence'] >= 0.7]):,}\n")
    
    print(f"✓ Summary saved to {summary_path}")
    print("\n" + "="*70)
    print("Classification complete!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Default to sample of 200 articles
    sample_size = 200
    if len(sys.argv) > 1:
        try:
            sample_size = int(sys.argv[1])
            if sample_size <= 0:
                sample_size = None  # Process all
        except ValueError:
            sample_size = 200
    
    main(sample_size=sample_size)

