import argparse
import subprocess
import sys


def run_step(step_name, command):
    print(f"\n===== {step_name} =====", flush=True)
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete AI Trading Model workflow end-to-end."
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Run download_data.py before feature generation.",
    )
    parser.add_argument(
        "--tp",
        type=float,
        default=0.0023,
        help="Take-profit percentage used in trade simulations.",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.65,
        help="Probability threshold used in trade simulations.",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="Leverage value used in leveraged simulation.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000.0,
        help="Initial capital used by leveraged simulation.",
    )
    parser.add_argument(
        "--capital-fraction",
        type=float,
        default=1.0,
        help="Fraction of capital allocated per leveraged trade (0, 1].",
    )
    args = parser.parse_args()

    if args.leverage <= 0:
        raise ValueError("--leverage must be > 0")
    if args.initial_capital <= 0:
        raise ValueError("--initial-capital must be > 0")
    if not (0 < args.capital_fraction <= 1):
        raise ValueError("--capital-fraction must be in (0, 1]")

    py = sys.executable

    if args.download_data:
        run_step("Download Data", [py, "download_data.py"])

    run_step("Feature Engineering", [py, "features.py"])
    run_step("Labeling", [py, "labeling.py", str(args.tp)])
    run_step("Train/Test Split", [py, "train_test_split.py"])
    run_step("Model Training", [py, "train_xgboost.py"])

    run_step(
        "Trade Simulation (No Leverage)",
        [py, "trade_simulation.py", str(args.tp), str(args.prob)],
    )
    run_step(
        "Backtest (No Leverage)",
        [py, "backtest.py", "--trades-path", "data/results/trades.parquet", "--label", "No Leverage"],
    )

    run_step(
        "Trade Simulation (With Leverage)",
        [
            py,
            "trade_simulation_leverage.py",
            str(args.tp),
            str(args.prob),
            str(args.leverage),
            str(args.initial_capital),
            str(args.capital_fraction),
        ],
    )
    run_step(
        "Backtest (With Leverage)",
        [
            py,
            "backtest.py",
            "--trades-path",
            "data/results/trades_leverage.parquet",
            "--leverage-hint",
            str(args.leverage),
            "--label",
            f"With Leverage ({args.leverage}x)",
        ],
    )

    print("\n===== WORKFLOW COMPLETE =====")


if __name__ == "__main__":
    main()
