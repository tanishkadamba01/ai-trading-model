import pandas as pd
import numpy as np

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    df = pd.read_parquet("data/raw/btcusdt_1m.parquet")

    # Set timestamp as index
    df = df.set_index("timestamp")
    df = df.sort_index()

    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_ret_5"] = np.log(df["close"] / df["close"].shift(5))

    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]

    df["rsi_5"] = compute_rsi(df["close"], 5)
    df["rsi_9"] = compute_rsi(df["close"], 9)
    df["rsi_14"] = compute_rsi(df["close"], 14)

    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["ema9_dist"] = df["close"] - df["ema_9"]
    df["ema21_dist"] = df["close"] - df["ema_21"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["atr_7"] = true_range.rolling(7).mean()
    df["atr_14"] = true_range.rolling(14).mean()

    df["ret_std_5"] = df["log_ret_1"].rolling(5).std()
    df["ret_std_15"] = df["log_ret_1"].rolling(15).std()

    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()

    df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std
    df["vol_spike"] = (df["vol_zscore"] > 2).astype(int)

    feature_columns = [
        "log_ret_1", "log_ret_3", "log_ret_5",
        "candle_body", "candle_range",
        "rsi_5", "rsi_9", "rsi_14",
        "ema9_dist", "ema21_dist",
        "atr_7", "atr_14",
        "ret_std_5", "ret_std_15",
        "vol_zscore", "vol_spike"
    ]

    X = df[feature_columns].dropna()
    X.to_parquet("data/features/btcusdt_features.parquet")

    print("Feature matrix saved")
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()
