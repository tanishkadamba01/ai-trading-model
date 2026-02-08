import pandas as pd
import numpy as np
import os
import sys

os.makedirs("data/labeled", exist_ok=True)

df = pd.read_parquet("data/raw/btcusdt_1m.parquet")
df = df.set_index("timestamp").sort_index()

X = pd.read_parquet("data/features/btcusdt_features.parquet")

TP_PCT = float(sys.argv[1])   
SL_PCT = 0.0008   # -0.08%
MAX_HOLD = 5      # candles (minutes)

def label_trade(df, entry_time):
    entry_price = df.loc[entry_time, "close"]

    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)

    future = df.loc[entry_time:].iloc[1:MAX_HOLD+1]

    for _, row in future.iterrows():
        if row["high"] >= tp_price:
            return 1   # TP hit first
        if row["low"] <= sl_price:
            return 0   # SL hit first

    return -1  # timeout

labels = []

for ts in X.index:
    label = label_trade(df, ts)
    labels.append(label)

y = pd.Series(labels, index=X.index, name="label")

print(y.value_counts())
print(y.value_counts(normalize=True))

mask = y != -1
X_labeled = X.loc[mask]
y_labeled = y.loc[mask]

data = X_labeled.copy()
data["label"] = y_labeled

data.to_parquet("data/labeled/btcusdt_labeled.parquet")

print("Labeled dataset saved")
print("Shape:", data.shape)
print(data["label"].value_counts())
