import pandas as pd
import numpy as np

def prepare_features(input_path="../../data/interim/merged_with_daily_return.csv",
                     output_path="../../data/processed/merged_prices_with_sentiment_ready.csv"):
    """y
    Defines model features and creates the binary target column.
    """
    # Load your merged stock + sentiment dataset
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return
    except Exception as e:
        print(f"Error reading the file {input_path}: {e}")
        return

    # Ensure the 'date' column is parsed as datetime and sort chronologically
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values("date", ascending=True).reset_index(drop=True)
    except KeyError:
        print("Error: 'date' column not found in the dataset.")
        return
    except Exception as e:
        print(f"Error processing the 'date' column: {e}")
        return

    # Define input feature columns
    feature_cols = [
        "daily_return_%", "range_%",
        "vol_z",
        "positive", "negative", "neutral"
    ]

    # Check if all required columns are present
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the input file: {missing_cols}")
        return

    # Create target column (1 if next day's close > today's close)
    try:
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    except KeyError as e:
        print(f"Error creating target column: {e}")
        return

    # Drop the final row since it has no "next day"
    df = df[:-1]

    # Save feature column list for later reuse (e.g., in configs)
    with open("../../configs/features.txt", "w") as f:
        f.write("\n".join(feature_cols))

    # Save processed dataset
    try:
        df.to_csv(output_path, index=False)
        print(f"[+] Features prepared and saved to {output_path}")
        print(f"[+] Total rows: {len(df)} | Features: {len(feature_cols)} | Target unique: {df['target'].nunique()}")
    except Exception as e:
        print(f"Error saving the processed dataset: {e}")

    return df, feature_cols

if __name__ == "__main__":
    prepare_features()