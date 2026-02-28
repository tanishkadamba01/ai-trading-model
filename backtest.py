import os
import argparse
import pandas as pd

def run_realistic_backtest(
    trades_path="data/results/trades.parquet",
    output_path="data/results/trades_realistic.parquet",
):
    # Load trade log from Phase 6
    trades = pd.read_parquet(trades_path)

    tick_size = 0.5  # $0.5 per tick
    slippage_ticks = 2
    fee_pct = 0.0005  # 0.05% per side
    total_fee = 2 * fee_pct
    position_size = 1.0

    trades["entry_slippage"] = slippage_ticks * tick_size
    trades["exit_slippage"] = slippage_ticks * tick_size

    trades["entry_price_real"] = trades["entry_price"] + trades["entry_slippage"]
    trades["exit_price_real"] = trades["exit_price"] - trades["exit_slippage"]

    trades["gross_return_real"] = (
        (trades["exit_price_real"] - trades["entry_price_real"])
        / trades["entry_price_real"]
    )

    trades["net_return_real"] = trades["gross_return_real"] - total_fee
    trades["pnl"] = trades["net_return_real"] * position_size
    trades["equity"] = (1 + trades["pnl"]).cumprod()

    win_rate = (trades["pnl"] > 0).mean()
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    trades["peak"] = trades["equity"].cummax()
    trades["drawdown"] = trades["equity"] / trades["peak"] - 1

    max_dd = trades["drawdown"].min()
    expectancy = trades["pnl"].mean()

    print("===== BACKTEST RESULTS =====")
    print("Total trades:", len(trades))
    print("Win rate:", round(win_rate, 3))
    print("Profit factor:", round(profit_factor, 2))
    print("Expectancy:", round(expectancy, 5))
    print("Final equity:", round(trades["equity"].iloc[-1], 4))
    print("Max drawdown:", round(max_dd, 4))

    trades.to_parquet(output_path)
    print("Realistic backtest saved")

def _resolve_leverage_used(trades, leverage_hint):
    if "leverage" in trades.columns and trades["leverage"].notna().any():
        unique = sorted(trades["leverage"].dropna().astype(float).unique())
        if len(unique) == 1:
            return float(unique[0])
        return "mixed(" + ",".join(f"{x:g}x" for x in unique) + ")"
    return float(leverage_hint)


def _print_backtest_results(metrics, label=None):
    print("===== BACKTEST RESULTS =====")
    if label:
        print("Mode:", label)
    print("Leverage used:", metrics["Leverage Used"])
    print("Total trades:", metrics["Total trades"])
    print("Win rate:", round(metrics["Win rate"], 3))
    print("Profit factor:", round(metrics["Profit factor"], 2))
    print("Expectancy:", round(metrics["Expectancy"], 5))
    print("Final equity:", round(metrics["Final Equity"], 4))
    print("Max drawdown:", round(metrics["Max drawdown"], 4))


def run_backtest(
    tp_pct=None,
    prob_threshold=None,
    trades_path="data/results/trades.parquet",
    leverage_hint=1.0,
    print_results=True,
    label=None,
):
    _ = (tp_pct, prob_threshold)  # kept for call parity with existing sweep code

    trades = pd.read_parquet(trades_path)
    total_trades = len(trades)
    leverage_used = _resolve_leverage_used(trades, leverage_hint)

    if total_trades == 0:
        metrics = {
            "Total trades": 0,
            "Win rate": 0,
            "Profit factor": 0,
            "Expectancy": 0,
            "Final Equity": 1.0,
            "Max drawdown": 0,
            "Leverage Used": leverage_used,
        }
        if print_results:
            _print_backtest_results(metrics, label=label)
        return metrics

    if "equity" not in trades.columns:
        trades["equity"] = (1 + trades["pnl"]).cumprod()

    if "drawdown" not in trades.columns:
        trades["peak"] = trades["equity"].cummax()
        trades["drawdown"] = trades["equity"] / trades["peak"] - 1

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]

    win_rate = len(wins) / total_trades
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    profit_factor = float("inf") if gross_loss == 0 else gross_profit / gross_loss
    expectancy = trades["pnl"].mean()
    final_equity = trades["equity"].iloc[-1]
    max_dd = trades["drawdown"].min()

    metrics = {
        "Total trades": total_trades,
        "Win rate": win_rate,
        "Profit factor": profit_factor,
        "Expectancy": expectancy,
        "Final Equity": final_equity,
        "Max drawdown": max_dd,
        "Leverage Used": leverage_used,
    }

    if print_results:
        _print_backtest_results(metrics, label=label)

    return metrics


def run_latest_backtest():
    normal_path = "data/results/trades.parquet"
    leveraged_path = "data/results/trades_leverage.parquet"

    candidates = []
    if os.path.exists(normal_path):
        candidates.append((os.path.getmtime(normal_path), normal_path, "No Leverage", 1.0))
    if os.path.exists(leveraged_path):
        candidates.append((os.path.getmtime(leveraged_path), leveraged_path, "With Leverage", 3.0))

    if not candidates:
        print("No trades file found. Run trade_simulation.py or trade_simulation_leverage.py first.")
        return

    _, latest_path, label, leverage_hint = max(candidates, key=lambda x: x[0])
    run_backtest(
        trades_path=latest_path,
        leverage_hint=leverage_hint,
        print_results=True,
        label=label,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Backtest trades from non-leveraged or leveraged simulation outputs."
    )
    parser.add_argument(
        "--trades-path",
        default=None,
        help="Optional path to trades parquet. If omitted, the most recently updated trades file is used.",
    )
    parser.add_argument(
        "--leverage-hint",
        type=float,
        default=1.0,
        help="Fallback leverage value if leverage column is missing in trades file.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label printed in output.",
    )
    args = parser.parse_args()

    if args.trades_path:
        run_backtest(
            trades_path=args.trades_path,
            leverage_hint=args.leverage_hint,
            print_results=True,
            label=args.label,
        )
    else:
        run_latest_backtest()


if __name__ == "__main__":
    main()
