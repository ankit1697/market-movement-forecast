import os
import sys
import json
import requests
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# Path to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to project root (one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add project root to PYTHONPATH
sys.path.append(PROJECT_ROOT)

# Path to feature_names.json (located at project root)
FEATURE_PATH = os.path.join(SCRIPT_DIR, "..", "feature_names.json")

# Normalize the path
FEATURE_PATH = os.path.abspath(FEATURE_PATH)

# Python script imports
from finbert_news_classifier import FinBERTNewsClassifier
from gpt4o_sentiment import analyze_sentiment_dataframe
from category_sentiment_scores import compute_category_sentiment_scores

#Import config
from config.configs import CATEGORY_LIST

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Path to the .env file
env_path = os.path.join(project_root, ".env")

# Load the .env file
load_dotenv(env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")

sample_df = pd.DataFrame(
    {
        "date": [
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
        ],
        "text": [
            "Tesla shares rally after record vehicle deliveries and strong Q4 guidance.",
            "Oil prices tumble as OPEC signals potential production increase.",
            "U.S. President announces new trade measures targeting Chinese tech firms.",
            "U.S. economy crashes",
            "Apple and Google to invest 500 million in AI research partnership in the US",
            "U.S. economy shows no signs of recovery amid rising unemployment rates.",
            "U.S. Federal Reserve hints at possible interest rate hike in upcoming meeting.",
            "U.S. federal reserve hints at positive economic outlook despite inflation concerns.",
        ],
    }
)

# News classification
classifier = FinBERTNewsClassifier()  # uses default categories
df_with_topics = classifier.classify_dataframe(sample_df, text_col="text")

# News sentiment
df_sent, daily = analyze_sentiment_dataframe(df_with_topics, text_col="text", date_col="date")

# Create scores
score_df = compute_category_sentiment_scores(daily, categories=CATEGORY_LIST)

score_df["overall_sentiment"] = (
    (df_sent[df_sent['sentiment'] == 'Positive'].shape[0] - df_sent[df_sent['sentiment'] == 'Negative'].shape[0]) / df_sent.shape[0]
)

# Inference via REST API
with open(FEATURE_PATH, "r") as f:
    feature_list = json.load(f)

df_new = score_df.copy(deep=True)
X_new = df_new[feature_list]

payload = {
    "dataframe_split": X_new.to_dict(orient="split")
}

url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status:", response.status_code)
print("Raw response text:", response.text)

try:
    print("Predictions:", response.json())
except Exception:
    print("Response is not valid JSON.")