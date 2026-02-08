import pandas as pd

# Load the parquet file
df = pd.read_parquet("C:\\Users\\Tanish\\Desktop\\TradingMLModel\\data\\raw\\btcusdt_1m.parquet")

# Show first 5 rows
print(df.head())

# Show basic info
print("\nData info:")
print(df.info())
