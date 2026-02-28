import argparse
import subprocess
import sys

import pandas as pd

from backtest import run_backtest


DEFAULT_TP_VALUES = [0.0016, 0.0017, 0.0018, 0.0019, 0.0020, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025]
DEFAULT_PROB_VALUES = [0.65, 0.70, 0.75]


def run_cmd(command):
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_float_csv(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run TP/probability parameter sweep.")
    parser.add_argument(
        "--tp-values",
        type=parse_float_csv,
        default=DEFAULT_TP_VALUES,
        help="Comma-separated TP values. Example: 0.0017,0.0019,0.0021",
    )
    parser.add_argument(
        "--prob-values",
        type=parse_float_csv,
        default=DEFAULT_PROB_VALUES,
        help="Comma-separated probability thresholds. Example: 0.65,0.70,0.75",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="Leverage used for leveraged simulation.",
    )
    parser.add_argument(
        "--excel-path",
        default="tpandprobanalysis.xlsx",
        help="Output Excel file path.",
    )
    args = parser.parse_args()

    if args.leverage <= 0:
        raise ValueError("--leverage must be > 0")

    py = sys.executable
    excel_path = args.excel_path

    try:
        results_df = pd.read_excel(excel_path)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=[
            "Mode",
            "Leverage",
            "Take Profit",
            "Probability",
            "Total trades",
            "Win rate",
            "Profit factor",
            "Expectancy",
            "Final equity",
            "Max drawdown",
        ])

    rows = []

    for tp in args.tp_values:
        print(f"\n=== Running full pipeline for TP={tp} ===")

        run_cmd([py, "labeling.py", str(tp)])
        run_cmd([py, "train_test_split.py"])
        run_cmd([py, "train_xgboost.py"])

        for prob in args.prob_values:
            print(f"\n--- Non-leverage simulation: TP={tp}, Prob={prob} ---")
            run_cmd([py, "trade_simulation.py", str(tp), str(prob)])

            metrics = run_backtest(
                tp,
                prob,
                trades_path="data/results/trades.parquet",
                leverage_hint=1.0,
                print_results=True,
                label="No Leverage",
            )

            rows.append({
                "Mode": "No Leverage",
                "Leverage": metrics["Leverage Used"],
                "Take Profit": tp,
                "Probability": prob,
                "Total trades": metrics["Total trades"],
                "Win rate": metrics["Win rate"],
                "Profit factor": metrics["Profit factor"],
                "Expectancy": metrics["Expectancy"],
                "Final equity": metrics["Final Equity"],
                "Max drawdown": metrics["Max drawdown"],
            })

            print(f"\n--- Leveraged simulation: TP={tp}, Prob={prob}, Leverage={args.leverage} ---")
            run_cmd([py, "trade_simulation_leverage.py", str(tp), str(prob), str(args.leverage)])

            metrics_leverage = run_backtest(
                tp,
                prob,
                trades_path="data/results/trades_leverage.parquet",
                leverage_hint=args.leverage,
                print_results=True,
                label=f"With Leverage ({args.leverage}x)",
            )

            rows.append({
                "Mode": "With Leverage",
                "Leverage": metrics_leverage["Leverage Used"],
                "Take Profit": tp,
                "Probability": prob,
                "Total trades": metrics_leverage["Total trades"],
                "Win rate": metrics_leverage["Win rate"],
                "Profit factor": metrics_leverage["Profit factor"],
                "Expectancy": metrics_leverage["Expectancy"],
                "Final equity": metrics_leverage["Final Equity"],
                "Max drawdown": metrics_leverage["Max drawdown"],
            })

    final_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
    final_df.to_excel(excel_path, index=False)
    print(f"\nParameter sweep complete. Results saved to: {excel_path}")


if __name__ == "__main__":
    main()
