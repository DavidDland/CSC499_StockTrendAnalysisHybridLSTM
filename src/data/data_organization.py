import pandas as pd

def sort_by_date():
    # File paths
    input_file = "/data/merged_prices_with_sentiment.csv"
    output_file = "/data/sorted_prices_with_sentiment.csv"

    # Read the merged data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except Exception as e:
        print(f"Error reading the file {input_file}: {e}")
        return

    # Ensure the 'date' column is parsed as datetime
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        print(f"Error parsing the 'date' column: {e}")
        return

    # Sort the data by date in ascending order
    df = df.sort_values(by='date', ascending=True)

    # Save the sorted data to a new CSV file
    try:
        df.to_csv(output_file, index=False)
        print(f"Sorted data saved as {output_file}")
    except Exception as e:
        print(f"Error saving the file {output_file}: {e}")

if __name__ == "__main__":
    sort_by_date()