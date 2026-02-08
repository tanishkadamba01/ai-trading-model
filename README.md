# ai-trading-model

An end-to-end **AI-powered trading system** built in Python, designed to explore machine learning applications in algorithmic trading.  
This project focuses on building a **reproducible research pipeline** covering data processing, feature engineering, model training, signal generation, and realistic backtesting with risk management.

> 📌 This is a research and learning project, not a production trading system.

---

## 🔍 Overview

This project implements a complete workflow:  
from raw market data → machine learning predictions → trade execution simulation → performance evaluation.

The goal is **not** to maximize short-term profit, but to:
- design a clean ML pipeline
- study probabilistic trade signals
- evaluate strategies using realistic backtesting
- understand risk, drawdowns, and robustness

The repository intentionally excludes datasets and trained models to keep it lightweight and reproducible.

---

## 🧠 Project Pipeline

1. **Data Collection**
   - Market data download and preprocessing
   - Candle-based time series handling

2. **Feature Engineering**
   - Technical indicators
   - Volatility measures
   - Rolling statistical features

3. **Labeling**
   - Supervised labels based on future price movement
   - Risk-aware target construction

4. **Train / Test Split**
   - Time-series–aware splitting
   - Avoids data leakage

5. **Model Training**
   - Gradient-boosted decision trees (XGBoost)
   - Probabilistic output (`predict_proba`)

6. **Signal Generation**
   - Probability thresholding
   - Volatility regime filtering

7. **Backtesting**
   - Event-driven trade simulation
   - Take-profit & stop-loss logic
   - Equity tracking and drawdown analysis

8. **Experimentation**
   - Parameter sweeps for probability thresholds and TP levels
   - Result aggregation for comparison

---

## 📊 Baseline Backtest Results

**Selected Baseline Configuration**
- Probability Threshold: **0.70**
- Take Profit: **0.0017**
- Stop Loss: Fixed (risk-defined)
- Leverage: **None**

**Performance Summary**
- Total Trades: ~37
- Win Rate: ~78%
- Profit Factor: ~2.0
- Maximum Drawdown: ~0.32%
- Final Equity: Positive growth without leverage

📌 Results are based on historical backtests only and do **not** represent live trading performance.

---

## 🗂 Repository Structure

ai-trading-model/
│
├── backtest.py
├── trade_simulation.py
├── run_parameter_sweep.py
├── train_xgboost.py
├── train_test_split.py
├── features.py
├── labeling.py
├── download_data.py
├── open_data.py
├── plot_candles.py
│
├── live/
│ └── live_trading.py
│
├── README.md
├── requirements.txt
└── .gitignore


---

## ⚙️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/tanishkadamba01/ai-trading-model.git
cd ai-trading-model

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

python train_xgboost.py
python backtest.py
python run_parameter_sweep.py

👤 Author

Built by a first-year Computer Science Engineering student exploring machine learning, quantitative research, and system design through hands-on projects.

If you’re interested in AI, ML, or quantitative trading, feel free to connect or share feedback.
