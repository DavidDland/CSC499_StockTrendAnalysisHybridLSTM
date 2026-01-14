import pandas as pd

# Load the CSV file
input_file = "output_sentiment_context_cleaned.csv"
output_file = "output_sentiment_context_with_avg.csv"

def calculate_average_sentiment(input_file, output_file):
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Ensure the required columns exist
    required_columns = ['date', 'positive', 'neutral', 'negative']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing in the input file.")
            return

    # Group by 'date' and calculate the mean for sentiment columns
    sentiment_avg = df.groupby('date')[['positive', 'neutral', 'negative']].mean().reset_index()

    # Merge the average sentiment back to the original dataframe
    df = pd.merge(df, sentiment_avg, on='date', suffixes=('', '_avg'))

    # Save the updated dataframe to a new CSV file
    try:
        df.to_csv(output_file, index=False)
        print(f"Output file saved as {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")

def calculate_daily_averages(input_file, output_file):
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Ensure the required columns exist
    required_columns = ['date', 'positive', 'neutral', 'negative']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' is missing in the input file.")
            return

    # Group by 'date' and calculate the mean for sentiment columns
    daily_averages = df.groupby('date')[['positive', 'neutral', 'negative']].mean().reset_index()

    # Ensure the 'date' column is parsed as datetime, replacing invalid dates with a default value
    daily_averages['date'] = pd.to_datetime(daily_averages['date'], errors='coerce').fillna(pd.Timestamp('1970-01-01'))

    # Sort the daily averages by date in true chronological order
    daily_averages = daily_averages.sort_values(by='date')

    # Save the daily averages to a new CSV file
    try:
        daily_averages.to_csv(output_file, index=False)
        print(f"Daily averages file saved as {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")

if __name__ == "__main__":
    calculate_daily_averages("output_sentiment_context_cleaned.csv", "daily_averages.csv")