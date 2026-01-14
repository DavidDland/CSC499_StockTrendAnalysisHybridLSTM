import pandas as pd

def merge_sentiment_with_prices():
    # File paths
    sentiment_file = "daily_sent_analysis_averages.csv"
    prices_file = "AAPL_prices_features_cleaned.csv"
    output_file = "merged_prices_with_sentiment.csv"

    # Read the sentiment scores
    try:
        sentiment_df = pd.read_csv(sentiment_file)
    except FileNotFoundError:
        print(f"Error: File {sentiment_file} not found.")
        return
    except Exception as e:
        print(f"Error reading the file {sentiment_file}: {e}")
        return

    # Read the prices data
    try:
        prices_df = pd.read_csv(prices_file)
    except FileNotFoundError:
        print(f"Error: File {prices_file} not found.")
        return
    except Exception as e:
        print(f"Error reading the file {prices_file}: {e}")
        return

    # Merge the data on the 'date' column
    try:
        merged_df = pd.merge(prices_df, sentiment_df[['date', 'positive', 'neutral', 'negative']], on='date', how='left')
    except KeyError:
        print("Error: The 'date' column is missing in one of the files.")
        return

    # Save the merged data to a new CSV file
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged data saved as {output_file}")
    except Exception as e:
        print(f"Error saving the file {output_file}: {e}")

if __name__ == "__main__":
    merge_sentiment_with_prices()