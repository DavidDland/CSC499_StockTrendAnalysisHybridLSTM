"""
Download and save raw price data using yfinance.
"""
import pandas as pd
import yfinance as yf
import os

def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch price data for a ticker using yfinance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start (str): Start date for historical data (YYYY-MM-DD).
        end (str): End date for historical data (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                      indexed by 'Date'.
    """
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start, end=end, interval="1d", auto_adjust=True)
    # Ensure columns are present and index is named 'Date'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index.name = 'Date'
    return df

def save_raw_prices(df: pd.DataFrame, ticker: str):
    """
    Save raw price DataFrame to a CSV file in the 'data/raw' directory.

    Args:
        df (pd.DataFrame): DataFrame containing stock price data.
        ticker (str): Stock ticker symbol (used in the filename).

    Returns:
        str: Path to the saved CSV file.
    """
    raw_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f'{ticker}_prices.csv')
    df.to_csv(out_path)
    return out_path
