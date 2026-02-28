import pandas as pd
import os
import sys

SL_PCT = 0.0008   # -0.08%
MAX_HOLD = 5      # candles (minutes)


def label_trade(df, entry_time, tp_pct):
    entry_price = df.loc[entry_time, "close"]

    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - SL_PCT)

    future = df.loc[entry_time:].iloc[1:MAX_HOLD + 1]
    if future.empty:
        return -1

    for _, row in future.iterrows():
        if row["high"] >= tp_price:
            return 1   # TP hit first
        if row["low"] <= sl_price:
            return 0   # SL hit first

    return -1  # timeout


def main():
    os.makedirs("data/labeled", exist_ok=True)

    tp_pct = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0023
    print(f"Labeling with TP_PCT={tp_pct}")

    df = pd.read_parquet("data/raw/btcusdt_1m.parquet")
    df = df.set_index("timestamp").sort_index()

    X = pd.read_parquet("data/features/btcusdt_features.parquet")

    labels = []
    for ts in X.index:
        labels.append(label_trade(df, ts, tp_pct))

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


if __name__ == "__main__":
    main()
