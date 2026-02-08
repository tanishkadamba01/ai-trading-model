import subprocess
import pandas as pd
from backtest import run_backtest

# Parameters to test
tp_values = [0.0017, 0.0019, 0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.0030]
prob_values = [0.65, 0.70, 0.75, 0.80]

excel_path = "tpandprobanalysis.xlsx"

# Load existing Excel (or create empty)
try:
    results_df = pd.read_excel(excel_path)
except FileNotFoundError:
    results_df = pd.DataFrame(columns=[
        "Take Profit",
        "Probability",
        "Total trades",
        "Win rate",
        "Profit factor",
        "Expectancy",
        "Final equity",
        "Max drawdown"
    ])

rows = []

for tp in tp_values:
    print(f"\n🔁 Running full pipeline for TP = {tp}")

    # 1️⃣ Relabel
    subprocess.run(["python", "labeling.py", str(tp)], check=True)

    # 2️⃣ Resplit
    subprocess.run(["python", "train_test_split.py"], check=True)

    # 3️⃣ Retrain
    subprocess.run(["python", "train_xgboost.py"], check=True)

    # 4️⃣ Loop over probability thresholds
    for prob in prob_values:
        print(f"Running trade_simulation with TP={tp}, Prob={prob}")

        # Run trade simulation FIRST
        subprocess.run(
            ["python", "trade_simulation.py", str(tp), str(prob)],
            check=True
        )

        # Run backtest AFTER simulation
        metrics = run_backtest(tp, prob)

        # Save results for THIS (tp, prob)
        rows.append({
            "Take Profit": tp,
            "Probability": prob,
            "Total trades": metrics["Total trades"],
            "Win rate": metrics["Win rate"],
            "Profit factor": metrics["Profit factor"],
            "Expectancy": metrics["Expectancy"],
            "Final equity": metrics["Final Equity"],
            "Max drawdown": metrics["Max drawdown"]
        })

# Append to Excel
final_df = pd.concat([results_df, pd.DataFrame(rows)], ignore_index=True)
final_df.to_excel(excel_path, index=False)

print("\n✅ FULL PARAMETER SWEEP COMPLETE")
