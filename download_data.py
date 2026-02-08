import ccxt
import pandas as pd
import time
from tqdm import tqdm

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"  # VERY IMPORTANT
    }
})

symbol = "BTC/USDT"
timeframe = "1m"

months_of_data = 6  # change to 12 later if you want

since_timestamp = exchange.parse8601(
    (pd.Timestamp.utcnow() - pd.DateOffset(months=months_of_data))
    .strftime("%Y-%m-%d %H:%M:%S")
)

limit = 1500  # max Binance allows per request
all_candles = []

current_since = since_timestamp

with tqdm(desc="Downloading BTCUSDT 1m data") as progress:
    while True:
        candles = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=current_since,
            limit=limit
        )

        if not candles:
            break

        all_candles.extend(candles)

        # move forward 1 minute
        current_since = candles[-1][0] + 60_000

        progress.update(len(candles))

        if current_since >= exchange.milliseconds():
            break

        time.sleep(exchange.rateLimit / 1000)

df = pd.DataFrame(
    all_candles,
    columns=["timestamp", "open", "high", "low", "close", "volume"]
)

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df.drop_duplicates(subset="timestamp")
df = df.sort_values("timestamp")
df = df.reset_index(drop=True)

expected_times = pd.date_range(
    start=df["timestamp"].iloc[0],
    end=df["timestamp"].iloc[-1],
    freq="1min",
    tz="UTC"
)

missing = expected_times.difference(df["timestamp"])

print("Total candles:", len(df))
print("Missing candles:", len(missing))

output_path = "data/raw/btcusdt_1m.parquet"
df.to_parquet(output_path, engine="pyarrow")

print("Saved file to:", output_path)
