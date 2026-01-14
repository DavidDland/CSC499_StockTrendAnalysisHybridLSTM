"""
Add basic derived features to OHLCV price data.
"""
import numpy as np
import pandas as pd
import os

def add_price_features(df: pd.DataFrame):
    """
    Add log_ret, range, vol_z (volume z-score) to DataFrame.
    No future leakage: use shift where needed.
    """
    df = df.copy()
    # Log return (using previous close)
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    # Daily range (high-low)
    df['range'] = df['High'] - df['Low']
    # Volume z-score (normalize volume)
    vol_mean = df['Volume'].mean()
    vol_std = df['Volume'].std(ddof=0)
    df['vol_z'] = (df['Volume'] - vol_mean) / vol_std if vol_std != 0 else 0
    return df

if __name__ == "__main__":
    # Example usage: add features to cleaned AAPL data
    cleaned_path = os.path.join(os.path.dirname(__file__), '../../data/raw/AAPL_prices.csv')
    df = pd.read_csv(cleaned_path, index_col=0)
    featured = add_price_features(df)
    # Save to interim folder
    featured_path = os.path.join(os.path.dirname(__file__), '../../data/interim/AAPL_prices_features.csv')
    featured.to_csv(featured_path)
    print(f"Feature data saved to {featured_path}")
