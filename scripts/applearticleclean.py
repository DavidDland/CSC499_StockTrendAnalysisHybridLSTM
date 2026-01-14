import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV file
input_csv = os.path.join(script_dir, '../data/raw/testing.csv')
output_csv = os.path.join(script_dir, 'testing_filtered.csv')

# Read the CSV file with encoding that handles Windows/Excel files
df = pd.read_csv(input_csv, encoding='latin1')

# Define keywords (case-insensitive)
keywords = ['apple', 'Apple', 'APPL', 'iPhone', 'iPad', 'MacBook', 'iMac', 'AirPods', 'Apple Watch']

# Check if any keyword is in columns B or C (columns 1 and 2, zero-indexed)
mask = (
    df.iloc[:, 1].astype(str).str.contains('|'.join(keywords), case=False, na=False) |
    df.iloc[:, 2].astype(str).str.contains('|'.join(keywords), case=False, na=False)
)

# Keep only rows where mask is True
filtered_df = df[mask]

# Save to new CSV
filtered_df.to_csv(output_csv, index=False)
print(f"Filtered data saved to {output_csv}")