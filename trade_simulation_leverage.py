import pandas as pd
import joblib
import sys
import os


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
TP_PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0023
PROB_THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
LEVERAGE = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
if LEVERAGE <= 0:
    raise ValueError("LEVERAGE must be > 0")

signals["atr"] = df.loc[signals.index, "atr_14"]
signals["atr_med"] = df.loc[signals.index, "atr_med"]

signals["enter"] = (
    (signals["prob"] > PROB_THRESHOLD)
    & (signals["atr"] > 1.2 * signals["atr_med"])
)

SL_PCT = 0.0008
MAX_HOLD = 5
FEE_PCT = 0.0004  # 0.04% per side (Binance-like)
INITIAL_CAPITAL = float(sys.argv[4]) if len(sys.argv) > 4 else 1000.0
CAPITAL_FRACTION = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
if INITIAL_CAPITAL <= 0:
    raise ValueError("INITIAL_CAPITAL must be > 0")
if not (0 < CAPITAL_FRACTION <= 1):
    raise ValueError("CAPITAL_FRACTION must be in (0, 1]")

trades = []
current_capital = INITIAL_CAPITAL

for entry_time in signals[signals["enter"]].index:
    if current_capital <= 0:
        break

    entry_price = df.loc[entry_time, "close"]
    tp_price = entry_price * (1 + TP_PCT)
    sl_price = entry_price * (1 - SL_PCT)

    future = df.loc[entry_time:].iloc[1 : MAX_HOLD + 1]
    if future.empty:
        continue

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

    # Position sizing based on current capital and leverage (notional exposure).
    capital_before = current_capital
    margin_used = capital_before * CAPITAL_FRACTION
    entry_notional = margin_used * LEVERAGE
    quantity = entry_notional / entry_price
    exit_notional = abs(quantity * exit_price)

    gross_pnl_usd = (exit_price - entry_price) * quantity
    entry_fee_usd = entry_notional * FEE_PCT
    exit_fee_usd = exit_notional * FEE_PCT
    total_fees_usd = entry_fee_usd + exit_fee_usd
    net_pnl_usd = gross_pnl_usd - total_fees_usd

    capital_after = max(capital_before + net_pnl_usd, 0.0)
    pnl_return = (capital_after - capital_before) / capital_before
    gross_ret = (exit_price - entry_price) / entry_price
    net_ret = net_pnl_usd / capital_before

    trades.append(
        {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "result": result,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "leveraged_return": pnl_return,
            "leverage": LEVERAGE,
            "margin_used": margin_used,
            "position_notional": entry_notional,
            "quantity": quantity,
            "gross_pnl_usd": gross_pnl_usd,
            "entry_fee_usd": entry_fee_usd,
            "exit_fee_usd": exit_fee_usd,
            "fees_usd": total_fees_usd,
            "pnl_usd": net_pnl_usd,
            "capital_before": capital_before,
            "capital_after": capital_after,
            "pnl": pnl_return,
            "equity_capital": capital_after,
            "equity": capital_after / INITIAL_CAPITAL,
        }
    )

    current_capital = capital_after

trades_df = pd.DataFrame(trades)

print("Total trades:", len(trades_df))
os.makedirs("data/results", exist_ok=True)

if trades_df.empty:
    print("No trades executed")

    avg_return = 0.0
    win_rate = 0.0
    final_equity = 1.0
    final_capital = INITIAL_CAPITAL
    max_drawdown = 0.0

    print("\nAverage leveraged return:", avg_return)
    print("Win rate:", win_rate)
    print("\nFinal equity:", final_equity)
    print("Final capital:", final_capital)
    print("Max drawdown:", max_drawdown)

    trades_df.to_parquet("data/results/trades_leverage.parquet")

    print("Leveraged trades saved")
    sys.exit(0)

print("\nAverage leveraged return:", trades_df["pnl"].mean())
print("Win rate:", (trades_df["pnl"] > 0).mean())
print("\nFinal equity:", trades_df["equity"].iloc[-1])
print("Final capital:", trades_df["equity_capital"].iloc[-1])

trades_df["peak"] = trades_df["equity"].cummax()
trades_df["drawdown"] = trades_df["equity"] / trades_df["peak"] - 1

print("Max drawdown:", trades_df["drawdown"].min())

trades_df.to_parquet("data/results/trades_leverage.parquet")

print("Leveraged trades saved")
