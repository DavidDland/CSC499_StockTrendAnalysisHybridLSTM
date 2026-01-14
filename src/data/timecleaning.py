import pandas as pd

def clean_date_column(file_path, output_path):
    """
    Clean and format the 'date' column in a CSV file.

    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the cleaned CSV file.
    """
    try:
        # Read the CSV file with specified encoding to handle special characters
        df = pd.read_csv(file_path, encoding='latin1')

        # Ensure the 'date' column exists
        if 'date' not in df.columns:
            print(f"Error: 'date' column not found in {file_path}.")
            return

        # Remove time and timezone information from the 'date' column
        df['date'] = df['date'].str.split(' ').str[0]

        # Format the 'date' column to `YYYY-MM-DD`
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Save the cleaned data to a new file
        df.to_csv(output_path, index=False)
        print(f"Cleaned file saved as {output_path}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"Error processing the file {file_path}: {e}")

if __name__ == "__main__":
    clean_date_column("data/AAPL_prices_features.csv", "data/AAPL_prices_features_cleaned.csv")