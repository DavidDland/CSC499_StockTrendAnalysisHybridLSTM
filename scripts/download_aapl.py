"""
David Durland CSMC 499
08/30/2025
Script to download AAPL daily price data (2022-2025) and save to data/raw/AAPL_prices.csv
"""

import sys
import os
# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.data.download_prices import fetch_prices, save_raw_prices

def main():
    df = fetch_prices('AAPL', '2022-01-01', '2025-01-01')
    save_raw_prices(df, 'AAPL')
    print("AAPL prices downloaded and saved to data/raw/AAPL_prices.csv")

if __name__ == "__main__":
    main()
