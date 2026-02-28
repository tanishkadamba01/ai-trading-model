import os
from datetime import datetime

import ccxt
import joblib
import numpy as np
import pandas as pd


SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
LOOKBACK = 200
PROB_THRESHOLD = 0.65
MODEL_PATH = "data/models/xgb_tp_sl_model.pkl"

FEATURE_COLUMNS = [
    "log_ret_1", "log_ret_3", "log_ret_5",
    "candle_body", "candle_range",
    "rsi_5", "rsi_9", "rsi_14",
    "ema9_dist", "ema21_dist",
    "atr_7", "atr_14",
    "ret_std_5", "ret_std_15",
    "vol_zscore", "vol_spike",
]

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})


def fetch_latest_candles(limit=LOOKBACK):
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def compute_features(df):
    df = df.copy()

    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_3"] = np.log(df["close"] / df["close"].shift(3))
    df["log_ret_5"] = np.log(df["close"] / df["close"].shift(5))

    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]

    def compute_rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["rsi_5"] = compute_rsi(df["close"], 5)
    df["rsi_9"] = compute_rsi(df["close"], 9)
    df["rsi_14"] = compute_rsi(df["close"], 14)

    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema9_dist"] = df["close"] - df["ema_9"]
    df["ema21_dist"] = df["close"] - df["ema_21"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    df["atr_7"] = tr.rolling(7).mean()
    df["atr_14"] = tr.rolling(14).mean()

    df["ret_std_5"] = df["log_ret_1"].rolling(5).std()
    df["ret_std_15"] = df["log_ret_1"].rolling(15).std()

    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std
    df["vol_spike"] = (df["vol_zscore"] > 2).astype(int)

    return df


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_xgboost.py first."
        )
    return joblib.load(MODEL_PATH)


def run_paper_trade():
    now = datetime.utcnow()
    model = load_model()

    df = fetch_latest_candles()
    df = compute_features(df)

    X_live = df[FEATURE_COLUMNS].dropna()
    if X_live.empty:
        print("Not enough candles to compute all features yet.")
        return

    latest_X = X_live.iloc[[-1]]
    prob = model.predict_proba(latest_X)[0, 1]
    signal = "LONG" if prob >= PROB_THRESHOLD else "NO_TRADE"

    print(f"[{now}] {SYMBOL} | Prob={prob:.4f} | Signal={signal}")


if __name__ == "__main__":
    run_paper_trade()
