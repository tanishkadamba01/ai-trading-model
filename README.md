# AI Trading Model (v0.2)

AI Trading Model is a Python-based research project for building and evaluating an ML-driven BTC/USDT trading pipeline on 1-minute candles.

> This repository is for research and learning. It is **not** production trading software.

## What is new in v0.2

- Unified backtest engine for both non-leverage and leverage trade logs.
- Realistic leverage simulation based on capital, notional exposure, and fee impact.
- One-command full workflow runner (`run_full_workflow.py`).
- Improved parameter sweep runner with clean CLI options.
- Cleaned scripts and publish-ready documentation.

## Repository Structure

```text
.
|-- backtest.py
|-- download_data.py
|-- features.py
|-- labeling.py
|-- train_test_split.py
|-- train_xgboost.py
|-- trade_simulation.py
|-- trade_simulation_leverage.py
|-- run_full_workflow.py
|-- run_parameter_sweep.py
|-- paper_trade.py
|-- plot_candles.py
|-- open_data.py
|-- live/
|   `-- live_trading.py
|-- data/
|   |-- raw/
|   |-- features/
|   |-- labeled/
|   |-- splits/
|   |-- models/
|   `-- results/
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10+
- pip
- Internet access (for `download_data.py`, `paper_trade.py`, and `live/live_trading.py`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/tanishkadamba01/ai-trading-model.git
cd ai-trading-model
```

### 2. Create virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### Option A: Full workflow (recommended)

Runs feature generation, labeling, splits, training, both simulations, and both backtests:

```bash
python run_full_workflow.py
```

With custom parameters:

```bash
python run_full_workflow.py --tp 0.0023 --prob 0.65 --leverage 3.0 --initial-capital 1000 --capital-fraction 1.0
```

Include fresh market download:

```bash
python run_full_workflow.py --download-data
```

### Option B: Manual pipeline

```bash
python download_data.py
python features.py
python labeling.py 0.0023
python train_test_split.py
python train_xgboost.py
python trade_simulation.py 0.0023 0.65
python backtest.py --trades-path data/results/trades.parquet --label "No Leverage"
python trade_simulation_leverage.py 0.0023 0.65 3.0
python backtest.py --trades-path data/results/trades_leverage.parquet --leverage-hint 3.0 --label "With Leverage"
```

### Option C: Parameter sweep

```bash
python run_parameter_sweep.py
```

Custom sweep example:

```bash
python run_parameter_sweep.py --tp-values 0.0018,0.0020,0.0022 --prob-values 0.65,0.70 --leverage 3.0
```

Results are saved to `tpandprobanalysis.xlsx`.

## Main Outputs

- `data/features/btcusdt_features.parquet`
- `data/labeled/btcusdt_labeled.parquet`
- `data/splits/*.parquet`
- `data/models/xgb_tp_sl_model.pkl`
- `data/results/trades.parquet`
- `data/results/trades_leverage.parquet`
- `tpandprobanalysis.xlsx`

## Script Reference

- `download_data.py`: Downloads BTC/USDT 1m candles from Binance (via `ccxt`).
- `features.py`: Builds technical and statistical features.
- `labeling.py`: Creates TP/SL outcome labels (supports TP CLI argument).
- `train_test_split.py`: Time-order-preserving split into train/val/test.
- `train_xgboost.py`: Trains and evaluates XGBoost model.
- `trade_simulation.py`: Non-leverage signal and trade simulation.
- `trade_simulation_leverage.py`: Leverage simulation with capital/notional fee modeling.
- `backtest.py`: Unified backtest for both trade outputs.
- `run_full_workflow.py`: One-command end-to-end workflow runner.
- `run_parameter_sweep.py`: TP/probability sweep for comparative analysis.

## Risk Disclaimer

- Historical backtest performance does not guarantee live performance.
- Trading involves substantial financial risk.

## Author

Built by Tanish Kadamba as a machine learning and quantitative trading research project.
