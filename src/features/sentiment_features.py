import pandas as pd
from transformers import pipeline, set_seed  # type: ignore

# Set a fixed seed for deterministic results
set_seed(42)

# Load the FinBERT sentiment analysis pipeline
# Type ignore is used to suppress type checker errors for custom models/tasks
sentiment_pipeline = pipeline(
    "sentiment-analysis",  # type: ignore
    model="ProsusAI/finbert"
)

def score_sentiment(texts):
    """
    Given a list of texts, return a DataFrame with sentiment probabilities and label for each text.
    Args:
        texts (list of str): Input texts.
    Returns:
        pd.DataFrame: Columns: ['label', 'score', 'positive', 'neutral', 'negative']
    """
    results = sentiment_pipeline(texts, return_all_scores=True)
    records = []
    for scores in results:
        # scores: list of dicts [{'label': 'positive', 'score': ...}, ...]
        row = {d['label'].lower(): d['score'] for d in scores}
        # Get max label
        max_label = max(scores, key=lambda d: d['score'])['label'].lower()
        row['label'] = max_label
        row['score'] = row[max_label]
        records.append(row)
    df = pd.DataFrame(records)
    # Ensure all columns present
    for col in ['positive', 'neutral', 'negative']:
        if col not in df:
            df[col] = 0.0
    return df[['label', 'score', 'positive', 'neutral', 'negative']]


# Test block
if __name__ == "__main__":
    import sys

    # Example usage: python sentiment_features.py input.csv text_column output.csv
    if len(sys.argv) == 4:
        input_csv = sys.argv[1]
        text_column = sys.argv[2]
        output_csv = sys.argv[3]

        # Try reading with latin1 encoding to avoid UnicodeDecodeError
        df = pd.read_csv(input_csv, encoding='latin1')
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 5 rows:\n{df.head()}")

        # Clean up the DataFrame
        df = df.drop_duplicates()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        # Drop rows where the text column is missing or empty
        df = df[df[text_column].notna() & (df[text_column].str.strip() != "")]

        texts = df[text_column].astype(str).tolist()
        # Truncate each text after 15 tokens (words)
        texts = [' '.join(t.split()[:15]) for t in texts]
        print(f"Scoring {len(texts)} rows in batch...")
        sentiment_df = score_sentiment(texts)
        print("Batch sentiment scoring complete.")
        result = pd.concat([df, sentiment_df], axis=1)
        result.to_csv(output_csv, index=False)
        print(f"Sentiment scores saved to {output_csv}")
    else:
        # Default test block
        test_texts = [
            "The stock price rose significantly after the positive earnings report.",
            "The company faces uncertainty due to ongoing legal issues.",
            "The quarterly results were in line with expectations."
        ]
        print(score_sentiment(test_texts))
