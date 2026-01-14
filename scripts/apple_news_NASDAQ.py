import pandas as pd

csv_file = 'nasdaq_exteral_data.csv'
output_file = 'aapl_news_nasdaq.csv'

# List all columns except the last 4
all_columns = pd.read_csv(csv_file, nrows=0).columns.tolist()
columns_to_read = all_columns[:-4]

chunk_size = 100000
aapl_rows = []

for chunk in pd.read_csv(csv_file, usecols=columns_to_read, chunksize=chunk_size):
    filtered = chunk[chunk['Stock_symbol'] == 'AAPL']
    if not filtered.empty:
        aapl_rows.append(filtered)

if aapl_rows:
    result = pd.concat(aapl_rows)
    result.to_csv(output_file, index=False)
    print(f"Extracted {len(result)} rows with Stock_symbol 'AAPL' to {output_file}")
else:
    print("No rows found with Stock_symbol 'AAPL'.")