# Forecasting Stock Movement Using News Sentiment

This project builds a complete pipeline to **predict whether the S&P500 will go up or down the next day** using **news articles, sentiment scores, topic clustering, and machine learning models**.  

The goal is to demonstrate an end-to-end **MLOps workflow** including data processing, feature engineering, model training, MLflow tracking, deployment, and model monitoring.

---
## 1. Introduction
Financial markets respond continuously to information, and news media remains one of the primary channels through which new information is disseminated. Each day thousands of headlines report developments on the economy, corporate earnings, politics, global risks, and more. This project explores whether the sentiment embedded in these news articles can be transformed into predictive features for forecasting next-day S&P 500 returns.

However, the focus is broader than building a predictive model. The project aims to construct a **complete MLOps system** that addresses:

-   Ingestion and preprocessing of external text data
-   Sentiment extraction using multiple NLP models
-   Transformation of noisy text features into structured financial signals
-   Training and comparing a wide range of models using AutoML
-   Tracking every experiment using MLflow
-   Deploying the best-performing model for  inference
-   Monitoring for data drift and model performance decay using Evidently AI
    
The result is a reproducible, extensible workflow that represents how a financial ML system would operate in a production environment.

## 2. Data Sources and Topic classification

### **2.1. Data sources**
The project collects and processes more than **53,000 news articles** from _CNBC_, _The Guardian_, and _Reuters_ between late 2017 and late 2020.

The raw datasets differ in structure, formatting, and text quality, so a comprehensive preprocessing pipeline was implemented.

Key steps include:
- Normalizing timestamp formats to ensure consistent daily aggregation
- Cleaning text fields for whitespace, line breaks, and formatting inconsistencies
- Resolving duplicated articles
- Creating a canonical structure containing: time, date, headline, description, and source

This produces a unified news dataset that can be reliably merged with market data.

### **2.2. Topic Classification (Sector Labeling)**
Each article is assigned to a topic or sector, allowing the system to compute sector-level sentiment indicators.

**Categories include:**
**Technology, Economy, Energy, Healthcare, Geo-Political, Automobile, Airlines, US Politics,  and Corporate.**

The classification approach combines keyword-based mapping and language-model-assisted evaluation. This structured labeling enriches the sentiment features by making it possible to evaluate whether certain sectors carry more predictive power than others.

## 3. Sentiment Analysis Framework
A major component of the project is transforming unstructured text into measurable sentiment signals. To improve robustness, the project employs three independent sentiment engines, each offering a different perspective:

### **3.1 VADER (NLTK)**
A lexicon-based sentiment analyzer optimized for social media and general news.
It provides quick polarity scores (positive, neutral, negative) and is useful as a starting point.
### **3.2 GPT-4o-Mini (LLM Sentiment Evaluation)**
An advanced large language model is used to classify sentiment and provide a confidence score. The model is prompted with a consistent evaluation format to ensure deterministic output. This LLM-based signal often captures tone, implication, and narrative context beyond simple polarity.
### 3.3 Daily Sentiment Aggregation

After sentiment is assigned at the article level, signals are aggregated **per day** and **per category**.

For each date, the pipeline computes:

- The number of positive, negative, and neutral articles  
- Sector-level sentiment scores  
- An overall market sentiment score defined as:

$$
\text{sentiment\-score} =
\frac{\text{positive} - \text{negative}}
{\text{positive} + \text{negative} + \text{neutral}}
$$

The output is a structured time-series dataset that can be merged directly with financial market data.

## 4. Market Data and Label Creation
Daily S&P 500 data (Open, High, Low, Close, Volume) is merged with the news-derived features.
Several target variables are created to support different modeling tasks:
1.  **Regression target**:
    Next-day percentage return (return_t_plus_1)
  
2.  **Four-class classification target**:
    Categorizing returns into
    _up_, _slightly_up_, _slightly_down_, _down_
    
3.  **Binary classification target**:
   Simply _up_ vs _down_
    
  This variety of targets allows the project to evaluate:

-   Whether sentiment can predict direction even if it cannot predict magnitude
    
-   Whether coarser target definitions yield more stable patterns

## **5. Modeling and AutoML**
The project uses **PyCaret AutoML** to systematically evaluate a wide range of models for each target type. This includes tree-based models, linear models, ensemble methods, gradient boosting, and more.

For each modeling task:
1.  The dataset is split into train and test sets (random, not time-based).
2.  PyCaret’s setup() initializes preprocessing and feature handling.
3.  compare_models() evaluates many models and selects the best one.
4.  The best model is finalized and evaluated on the test set.

Metrics include:
-   **Regression**: MAE, RMSE, R², and directional accuracy
-   **Classification**: F1 score, precision/recall, and accuracy

All results are considered to refine and ultimately we will derive the chosen algorithm.

## **6. Experiment Tracking with MLflow**

MLflow is used extensively to record:
-   Model configuration and parameters
-   Training and validation metrics
-   Feature set definitions
-   Prediction artifacts
  
The experiment tracking interface enables:
-   Visual comparison of model performance
-   Re-running and reproducing historical experiments
-   Selecting the best model for deployment based on results

## **7. Deployment Pipeline**

The best performing model is registered and deployed using **MLflow Models**, enabling simple, portable serving.
- Deploys the model using:
`mlflow models serve --model-uri <run_id> --port 5001 --no-conda`
- Includes an inference client script to send new feature rows for prediction using the code below

```
import json
import pandas as pd
import requests

# Load feature order
with open("../feature_names.json", "r") as f:
    feature_list = json.load(f)

# Prepare inference dataframe
df_new = score_df.copy(deep=True)

X_new = df_new[feature_list]

# MLflow serving payload
payload = {
    "dataframe_split": X_new.to_dict(orient="split")
}

# MLflow endpoint
url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

# Send request
response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status:", response.status_code)
print("Raw response text:", response.text)

# Try to parse JSON predictions
try:
    print("Predictions:", response.json())
except Exception:
    print("Response is not valid JSON.")
```

## **8. Model Monitoring with Evidently AI**
To simulate a production monitoring environment, the project integrates **Evidently** to track:
-   Data drift
-   Target drift
-   Prediction drift
-   Feature distribution changes over time

This provides early detection of model degradation, which is particularly important in financial markets where relationships change rapidly.

Monitoring reports are generated in both notebook and HTML formats and stored for analysis.

##  **9. Repository Structure**

```text
market-movement-forecast/
│
├── data/                           # Raw & processed data (ignored in repo)
│
├── scripts/
|   ├── inference.py                s# Model inference script
│   ├── gpt4o_sentiment.py          # GPT-4o sentiment scoring
│   ├── ollama_news_sentiment.py    # Gemma 3 sentiment scoring
│   ├── finbert_news_classifier.py  # Topic/category assignment
│   └── category_sentiment_scores.py# Daily sentiment aggregation
│   │
│   ├── config/
│   │   └── configs.py              # Config settings
│   │
│   └── notebooks/
│       └── modelling_training_random_forest.ipynb   # Model training with MLflow logging
│
├── testing/
│   └── modelling_training_automl.ipynb # Exploratory model notebook with AutoML testing
|
├── mlruns/                             # To log and track MLFlow runs
│
├── feature_names.json                  # Saved feature order for inference
├── sp500_monitoring.txt                # Monitoring dashboard
├── requirements.txt                    # Python dependencies
└── README.md
```

## **10. Findings and Key Insights**

-   Stock market returns are notoriously close to a **random walk**, and news sentiment is only **weakly correlated** with next-day returns in a linear sense. However, non-linear models still uncover meaningful structure: our four-bucket return classifier achieves **~30% F1/accuracy**, and binary up/down classification exceeds **50% accuracy**, suggesting the market may not be entirely random after all.
    
-   Sector-level sentiment features show long-term promise. They help identify **which categories of news influence the market most**, offering interpretability beyond aggregate sentiment.
    
-   Predicting the exact next-day return (regression) remains very difficult, but **directional classification** consistently performs better than random baselines and provides actionable signals.
    
-   Combining multiple sentiment engines **VADER, FinBERT, and GPT-based sentiment**, leads to a more stable and robust feature set over time, reducing dependence on a single model’s biases.
    
-   **MLflow** is useful for structured experiment tracking, especially in workflows involving many models, feature subsets, and parameter variations.
    
-   **Continuous monitoring** is mandatory in financial machine learning. Market and news distributions shift constantly, making drift detection and periodic model recalibration a critical part of the pipeline.

---

