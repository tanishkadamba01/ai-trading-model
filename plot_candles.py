import pandas as pd
import mplfinance as mpf

# Load the parquet file
df = pd.read_parquet("data/raw/btcusdt_1m.parquet")

# Set timestamp as index
df = df.set_index("timestamp")

# Ensure correct column order
df = df[["open", "high", "low", "close", "volume"]]

df_last = df.tail(500)

mpf.plot(
    df_last,
    type="candle",
    volume=True,
    style="charles",
    title="BTCUSDT 1-minute Candles",
    ylabel="Price (USDT)",
    ylabel_lower="Volume"
)
