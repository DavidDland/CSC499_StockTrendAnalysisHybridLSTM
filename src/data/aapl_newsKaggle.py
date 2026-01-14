import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load (empty for default)
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "frankossai/apple-stock-aapl-historical-financial-news-data",
    file_path
)

print("First 5 records:", df.head())