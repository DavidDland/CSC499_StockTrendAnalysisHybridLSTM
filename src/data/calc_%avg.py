import pandas as pd

def calculate_daily_return(input_path="../../data/interim/merged_prices_with_sentiment.csv",
                           output_path="../../data/interim/merged_with_daily_return.csv"):
    """
    Calculate:
      - Daily percentage change based on consecutive closing prices
      - Intraday range percentage (High - Low) relative to Open

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the output CSV file with new features.
    """
    try:
        # Load dataset
        df = pd.read_csv(input_path)

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Input file must contain the following columns: {missing}")

        # 1. Daily return (% change in Close)
        df['daily_return_%'] = df['Close'].pct_change() * 100

        # 2. Range percentage relative to Open
        df['range_%'] = ((df['High'] - df['Low']) / df['Open']) * 100

        # Drop the first row (NaN daily return)
        df = df.dropna(subset=['daily_return_%'])

        # Save updated dataset
        df.to_csv(output_path, index=False)
        print(f"Daily % change and range % calculated and saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    calculate_daily_return()
