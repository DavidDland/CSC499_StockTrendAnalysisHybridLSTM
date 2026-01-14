"""
Clean and normalize OHLCV price data for a given CSV file.
"""
import pandas as pd
import os

def clean_prices(df: pd.DataFrame):
    """
    Clean price DataFrame: timezone-naive, sorted, drop NA, and normalize Volume column.
    """
    df = df.copy()
    # Ensure index is datetime and timezone-naive
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz_localize'):
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            pass
    df = df.sort_index()
    df = df.dropna()
    # Do not normalize Volume column -- will be added in OHLCfeatures.py
    return df

if __name__ == "__main__":
    # Path to raw AAPL prices
    raw_path = os.path.join(os.path.dirname(__file__), '../../data/raw/AAPL_prices.csv')
    df = pd.read_csv(raw_path, index_col=0)
    cleaned = clean_prices(df)
    # Save cleaned data to interim folder
    interim_dir = os.path.join(os.path.dirname(__file__), '../../data/interim')
    os.makedirs(interim_dir, exist_ok=True)
    cleaned_path = os.path.join(interim_dir, 'AAPL_prices_cleaned.csv')
    cleaned.to_csv(cleaned_path)
    print(f"Cleaned data saved to {cleaned_path}")
