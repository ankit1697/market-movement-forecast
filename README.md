# Forecasting Stock Movement Using News Sentiment

This project builds a complete pipeline to **predict whether the S&P500 will go up or down the next day** using **news articles, sentiment scores, topic clustering, and machine learning models**.  
The goal is to demonstrate an end-to-end **MLOps workflow** including data processing, feature engineering, model training, MLflow tracking, and deployment.

---

## Project Overview

The workflow consists of four major components:

### **1. News Processing & Classification**
- Merges news from multiple sources (CNBC, Guardian, Reuters).
- Cleans and normalizes text.
- Assigns each article to a **topic category** (Corporate, Economy, Technology, Politics, etc.).
- Optional: clustering experiments (BERTopic, MPNet embeddings, Hierarchical Agglomerative Clustering).

### **2. Sentiment Analysis**
- Uses a **GPT-4o-mini** based prompt to classify each article as:
  - Positive  
  - Negative  
  - Neutral  
- Returns a **confidence score** for each prediction.
- Aggregates daily sentiment signals:
  - Category-level positive/negative/neutral counts  
  - Total positive, negative, neutral counts  
  - Category sentiment scores:  
    $score = \frac{pos - neg}{pos + neg + neutral}$

### **3. Market Data (S&P500)**
- Downloads daily S&P500 data (Open, High, Low, Close, Volume).
- Computes **daily returns**.
- Converts returns into 3 classes:
  - **Up**
  - **Down**
  - **Flat**
- Merges market data with sentiment-based features by date.

### **4. Predictive Modeling**
- Trains a **Random Forest classifier** to predict next-day market movement.
- Tracks everything with **MLflow**, including:
  - Parameters  
  - Metrics  
  - Confusion matrix  
  - Saved model artifacts  
  - Feature order (JSON)
- Deploys the model using:
`mlflow models serve --model-uri <run_id> --port 5001 --no-conda`
- Includes an inference client script to send new feature rows for prediction.

---

## Repository Structure
market-movement-forecast/
│
├── data/ # Raw & processed data (ignored in repo)
│
├── src/
│ ├── data/
│ │ ├── gpt4o_sentiment.py # GPT-4o sentiment scoring
│ │ ├── finbert_news_classifier.py # Topic/category assignment
│ │ └── category_sentiment_scores.py # Daily sentiment scores
│ │
│ ├── models/
│ │ └── inference.py # Client for deployed MLflow model
│ │
│ ├── training/
│ │ └── train_model.py # Model training with MLflow logging
│
├── notebooks/
│ └── model_training.ipynb
│
├── feature_names.json # Saved feature order for inference
├── requirements.txt
└── README.md



---

## Key Features

- Multi-source news ingestion  
- Topic classification and clustering  
- GPT-4o-powered sentiment analysis  
- Rich daily sentiment feature engineering  
- S&P500 market movement labeling  
- Random Forest prediction model  
- MLflow experiment tracking  
- MLflow model deployment + inference client  

---

## Purpose

This project demonstrates how **news sentiment can be transformed into predictive features** and how a complete **MLOps pipeline** can operationalize a financial prediction system.

---

