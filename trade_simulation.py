import pandas as pd
import numpy as np
import joblib
import sys


# Load market data
df = pd.read_parquet("data/raw/btcusdt_1m.parquet")
df = df.set_index("timestamp").sort_index()

# Load test features
X_test = pd.read_parquet("data/splits/X_test.parquet")

# Load trained model
model = joblib.load("data/models/xgb_tp_sl_model.pkl")

high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()

true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df["atr_14"] = true_range.rolling(14).mean()

# Rolling median ATR (volatility regime filter)
df["atr_med"] = df["atr_14"].rolling(100).median()

probs = model.predict_proba(X_test)[:, 1]

signals = pd.DataFrame(index=X_test.index)
signals["prob"] = probs

# Command-line arguments
TP_PCT = float(sys.argv[1])
PROB_THRESHOLD = float(sys.argv[2])


signals["atr"] = df.loc[signals.index, "atr_14"]
signals["atr_med"] = df.loc[signals.index, "atr_med"]

signals["enter"] = (
    (signals["prob"] > PROB_THRESHOLD) &
    (signals["atr"] > 1.2 * signals["atr_med"])
)

SL_PCT = 0.0008
MAX_HOLD = 5
FEE_PCT = 0.0004   # 0.04% per side (Binance-like)

trades = []

for entry_time in signals[signals["enter"]].index:

    entry_price = df.loc[entry_time, "close"]
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)

    future = df.loc[entry_time:].iloc[1:MAX_HOLD+1]

    exit_price = None
    exit_time = None
    result = "timeout"

    for ts, row in future.iterrows():
        if row["high"] >= tp_price:
            exit_price = tp_price
            exit_time = ts
            result = "tp"
            break
        if row["low"] <= sl_price:
            exit_price = sl_price
            exit_time = ts
            result = "sl"
            break

    if exit_price is None:
        exit_price = future.iloc[-1]["close"]
        exit_time = future.index[-1]

    gross_ret = (exit_price - entry_price) / entry_price
    fees = 2 * FEE_PCT
    net_ret = gross_ret - fees

    trades.append({
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "result": result,
        "gross_return": gross_ret,
        "net_return": net_ret,
        "pnl": net_ret
    })

trades_df = pd.DataFrame(trades)

print("Total trades:", len(trades_df))

if trades_df.empty:
    print("No trades executed")

    # Create empty but valid outputs
    avg_return = 0.0
    win_rate = 0.0
    final_equity = 1.0
    max_drawdown = 0.0

    print("\nAverage net return:", avg_return)
    print("Win rate:", win_rate)
    print("\nFinal equity:", final_equity)
    print("Max drawdown:", max_drawdown)

    # Save empty trades file so pipeline does not break
    trades_df.to_parquet("data/results/trades.parquet")

    print("Trades saved")
    sys.exit(0)   # ⬅️ VERY IMPORTANT



print("\nAverage net return:", trades_df["net_return"].mean())
print("Win rate:", (trades_df["net_return"] > 0).mean())

trades_df["equity"] = (1 + trades_df["net_return"]).cumprod()

print("\nFinal equity:", trades_df["equity"].iloc[-1])

trades_df["peak"] = trades_df["equity"].cummax()
trades_df["drawdown"] = trades_df["equity"] / trades_df["peak"] - 1

print("Max drawdown:", trades_df["drawdown"].min())

trades_df.to_parquet("data/results/trades.parquet")

print("Trades saved")
