import pandas as pd

def split_data():
    """
    Split the dataset into train, validation, and test sets based on date ranges.

    Saves the splits as separate CSV files.
    """
    input_file = "../../data/processed/merged_prices_with_sentiment_ready.csv"
    train_file = "../../data/processed/train.csv"
    val_file = "../../data/processed/val.csv"
    test_file = "../../data/processed/test.csv"

    try:
        # Load the dataset
        df = pd.read_csv(input_file, parse_dates=["date"])
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except Exception as e:
        print(f"Error reading the file {input_file}: {e}")
        return

    # Sort the dataset by date
    df = df.sort_values("date").reset_index(drop=True)
    assert "target" in df.columns, "Run prepare_targets.py first."

    # Define date ranges for splits
    train_df = df[(df['date'] >= '2022-01-03') & (df['date'] <= '2023-12-31')].copy()
    val_df   = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-07-15')].copy()
    test_df  = df[(df['date'] >= '2024-07-16') & (df['date'] <= '2024-11-15')].copy()

    # Prevent cross-split label lookahead
    if len(train_df) > 0: train_df = train_df.iloc[:-1]
    if len(val_df)   > 0: val_df   = val_df.iloc[:-1]

    # Save the splits
    for (dfi, path) in [(train_df, train_file), (val_df, val_file), (test_df, test_file)]:
        if len(dfi) == 0:
            print(f"Warning: {path} would be empty. Check your date ranges.")
        dfi.to_csv(path, index=False)

    print("Train, validation, and test sets saved.")

if __name__ == "__main__":
    split_data()
