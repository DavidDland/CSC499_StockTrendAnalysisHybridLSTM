# LSTM-FinBERT Stock Trend Prediction

## Overview

This project explores the effectiveness of combining **historical stock price data** with **news sentiment analysis** to predict short-term stock market trends. The hybrid model leverages:

- **LSTM (Long Short-Term Memory)** for sequential modeling of historical stock prices.
- **FinBERT** (a domain-specific BERT variant for financial text) for sentiment classification of stock-related news articles.

---

## Features

- Data acquisition from **Yahoo Finance (`yfinance`)** for historical stock prices.
- Financial news sentiment extraction using **Kaggle + FinBERT**.
- Sliding window approach for time series input.
- Training pipeline with **PyTorch**.
- Model evaluation using metrics such as:
  - **Binary Cross-Entropy Loss (BCE)**
  - **Accuracy**
  - **RMSE (Root Mean Squared Error)**
  - **MAE (Mean Absolute Error)**

---

## Repository Structure and File Descriptions

### 1. Data Processing (`src/data/`)
These scripts handle data cleaning, feature engineering, and preparation for modeling.

- **`download_prices.py`**: Downloads historical stock price data using `yfinance`.
- **`timecleaning.py`**: Cleans and formats date columns in the dataset.
- **`clean_OHLCdata.py`**: Cleans OHLC (Open, High, Low, Close) data and handles missing values.
- **`calc_%avg.py`**: Calculates daily return percentages and intraday range percentages.
- **`data_organization.py`**: Merges and organizes datasets for further processing.
- **`data_split.py`**: Splits the dataset into train, validation, and test sets.
- **`prepare_targets.py`**: Creates binary target labels for stock price movement prediction.
- **`generate_sequences.py`**: Converts tabular data into fixed-length sequences for LSTM input.
- **`dataset.py`**: Defines PyTorch `Dataset` and `DataLoader` utilities for training and evaluation.

### 2. Modeling (`src/models/`)
These scripts define the machine learning models used in the project.

- **`lstm_classifier.py`**: Implements a simple LSTM-based binary classifier.
- **`hybrid_classifier.py`**: Combines LSTM sequence encoding with an auxiliary sentiment vector.
- **`utils.py`**: Utility functions for model parameter counting and debugging.

### 3. Configuration and Utilities
- **`configs/`**: Contains configuration files such as feature lists.
- **`utils/`**: General utility scripts for data handling and visualization.

---

## Workflow

### Step 1: Data Collection
1. Use `download_prices.py` to fetch historical stock price data.
2. Process and clean the data using `timecleaning.py` and `clean_OHLCdata.py`.

### Step 2: Feature Engineering
1. Use `calc/%avg.py` to calculate daily return percentages and intraday range percentages.
2. Merge datasets and organize them using `data_organization.py`.

### Step 3: Data Preparation
1. Split the data into train, validation, and test sets using `data_split.py`.
2. Generate binary target labels with `prepare_targets.py`.
3. Create fixed-length sequences for LSTM input using `generate_sequences.py`.

### Step 4: Modeling
1. Train the baseline LSTM model using `lstm_classifier.py`.
2. Train the hybrid LSTM-FinBERT model using `hybrid_classifier.py`.

---

## Results

- **Baseline LSTM**: Achieved accuracy of ~.56% (stock price only). With a F1 Score of .67
- **Hybrid LSTM-FinBERT**: Achieved accuracy of ~.56% with a F1 Score of .60 showing that while the accuracies maybe the same, the base model scored true positives and false negatives more accurately.

---

## Future Work

- Aggregate FinBERT data.
- Improve backpropagation techniques.
- Experiment with additional hybrid architectures.
- Train models on larger datasets.

---

## 🔗 References

- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [FinBERT](https://github.com/ProsusAI/finBERT)
- [DataSet](https://www.kaggle.com/datasets/frankossai/apple-stock-aapl-historical-financial-news-data)


# CSC499_StockTrendAnalysisHybridLSTM
This project explores the effectiveness of combining **historical stock price data** with **news sentiment analysis** to predict short-term stock market trends. The hybrid model leverages: LSTM, FinBERT
