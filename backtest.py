import pandas as pd
import numpy as np

# Load trade log from Phase 6
trades = pd.read_parquet("data/results/trades.parquet")

TICK_SIZE = 0.5       # $0.5 per tick
SLIPPAGE_TICKS = 2

trades["entry_slippage"] = SLIPPAGE_TICKS * TICK_SIZE
trades["exit_slippage"] = SLIPPAGE_TICKS * TICK_SIZE

trades["entry_price_real"] = trades["entry_price"] + trades["entry_slippage"]
trades["exit_price_real"] = trades["exit_price"] - trades["exit_slippage"]

trades["gross_return_real"] = (
    (trades["exit_price_real"] - trades["entry_price_real"])
    / trades["entry_price_real"]
)

FEE_PCT = 0.0005  # 0.04% per side

TOTAL_FEE = 2 * FEE_PCT

trades["net_return_real"] = trades["gross_return_real"] - TOTAL_FEE

POSITION_SIZE = 1.0

trades["pnl"] = trades["net_return_real"] * POSITION_SIZE

trades["equity"] = (1 + trades["pnl"]).cumprod()

win_rate = (trades["pnl"] > 0).mean()

gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()

profit_factor = gross_profit / gross_loss

trades["peak"] = trades["equity"].cummax()
trades["drawdown"] = trades["equity"] / trades["peak"] - 1

max_dd = trades["drawdown"].min()

expectancy = trades["pnl"].mean()

print("===== BACKTEST RESULTS =====")
print("Total trades:", len(trades))
print("Win rate:", round(win_rate, 3))
print("Profit factor:", round(profit_factor, 2))
print("Expectancy:", round(expectancy, 5))
print("Final equity:", round(trades['equity'].iloc[-1], 4))
print("Max drawdown:", round(max_dd, 4))

trades.to_parquet("data/results/trades_realistic.parquet")
print("Realistic backtest saved")

def run_backtest(tp_pct, prob_threshold):
    # load trades
    trades = pd.read_parquet("data/results/trades.parquet")

    total_trades = len(trades)

    if total_trades == 0:
        return {
            "Total trades": 0,
            "Win rate": 0,
            "Profit factor": 0,
            "Expectancy": 0,
            "Final Equity": 1.0,
            "Max drawdown": 0
        }

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]

    win_rate = len(wins) / total_trades

    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    if gross_loss == 0:
        profit_factor = float("inf")
    else:
        profit_factor = gross_profit / gross_loss

    expectancy = trades["pnl"].mean()
    final_equity = trades["equity"].iloc[-1]
    max_dd = trades["drawdown"].min()

    return {
        "Total trades": total_trades,
        "Win rate": win_rate,
        "Profit factor": profit_factor,
        "Expectancy": expectancy,
        "Final Equity": final_equity,
        "Max drawdown": max_dd
    }
