# Intraday Predictor 📈

An ML-powered intraday stock prediction system for Indian markets, built for
beginners trading on **Groww** with **₹3,500 capital (5× margin → ₹17,500)**.

> ⚠️ **RISK WARNING** — Trading stocks involves significant financial risk.
> This tool is for **educational and paper-trading purposes only**.
> Past performance does not guarantee future results.
> **Never invest money you cannot afford to lose.**

---

## Stocks Covered
| Stock | NSE Symbol | Description |
|-------|-----------|-------------|
| Godfrey Phillips | GODFRYPHLP.NS | Consumer goods |
| State Bank of India | SBIN.NS | Banking |

---

## Project Structure
```
intraday-predictor/
├── README.md                   ← You are here
├── requirements.txt            ← Python dependencies
├── config.py                   ← Capital, stocks, risk settings
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py         ← Yahoo Finance OHLCV data
│   ├── features.py             ← Technical indicator calculations
│   ├── model.py                ← XGBoost / Random Forest ML model
│   ├── predictor.py            ← Main prediction orchestration
│   ├── risk_manager.py         ← Position sizing, stop-loss, targets
│   └── utils.py                ← Logging & helper functions
├── main.py                     ← Daily prediction script ⭐
├── train_model.py              ← Train / retrain models
├── backtest.py                 ← Strategy backtesting
└── paper_trade.py              ← Paper trading tracker
```

---

## Installation

### Prerequisites
- Python 3.8 or newer
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yashMishra16/intraday-predictor.git
cd intraday-predictor

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start (Daily Workflow)

### Step 1 — Train the model (once, then weekly)
```bash
python train_model.py
```
This downloads ~60 days of 5-minute candle data and trains an XGBoost
classifier for each stock. Models are saved to the `models/` folder.

### Step 2 — Get today's predictions (every morning)
```bash
python main.py
```

Example output:
```
╔══════════════════════════════════════════════════════════════╗
║           INTRADAY PREDICTIONS - March 8, 2026               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📈 GODFREY PHILLIPS (GODFRYPHLP.NS)                         ║
║  ────────────────────────────────────                        ║
║  Current Price: ₹5,450.00                                    ║
║  Signal: BUY 🟢                                              ║
║  Confidence: 67%                                             ║
║                                                              ║
║  ➤ Entry Price:      ₹5,450.00                               ║
║  ➤ Stop Loss:        ₹5,422.75 (-0.5%)                       ║
║  ➤ Target:           ₹5,504.50 (+1.0%)                       ║
║  ➤ Quantity:         3 shares                                ║
║  ➤ Max Loss:         ₹81.75                                  ║
║  ➤ Potential Profit: ₹163.50                                 ║
║  ➤ Risk/Reward:      1:2.0                                   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🏦 STATE BANK OF INDIA (SBIN.NS)                            ║
║  ────────────────────────────────────                        ║
║  Current Price: ₹750.00                                      ║
║  Signal: WAIT ⚪                                             ║
║  Confidence: 52%                                             ║
║                                                              ║
║  ➤ Reason: Confidence too low, skip this trade               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  💡 TODAY'S RECOMMENDATION                                   ║
║  ────────────────────────────────────                        ║
║  Trade GODFRYPHLP (GODFREY PHILLIPS) with the levels above.  ║
║  Set stop-loss IMMEDIATELY after buying!                     ║
║  ⚠️  Never risk more than you can afford to lose.            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Step 3 — Log paper trades (optional but recommended)
```bash
# After seeing a BUY signal for SBIN at ₹750
python paper_trade.py log SBIN BUY 750.00

# After you would have exited at ₹758
python paper_trade.py outcome SBIN 758.00

# Check your performance
python paper_trade.py summary
```

### Step 4 — Backtest the strategy
```bash
python backtest.py
```

---

## Configuration

Edit `config.py` to adjust your settings:

```python
CAPITAL = 3500              # Your actual capital in INR
MARGIN_MULTIPLIER = 5       # Broker's intraday margin multiplier
STOP_LOSS_PERCENT = 0.005   # 0.5% stop-loss
TARGET_PERCENT = 0.01       # 1.0% target
CONFIDENCE_THRESHOLD = 0.60 # Only signal if model is >60% confident
```

---

## Features Used by the Model

The model uses 30+ technical indicators including:

| Category | Indicators |
|----------|-----------|
| Momentum | 1/3/5/10-period returns, ROC |
| Moving Averages | EMA 9, EMA 21, SMA 20 |
| VWAP | Volume-weighted average price |
| RSI | 14-period relative strength index |
| MACD | MACD line, signal, histogram |
| Bollinger Bands | Upper/lower bands, width, position |
| Volume | Volume ratio vs average, volume trend |
| Volatility | ATR (normalised), rolling std-dev |
| Time | Minutes since open, first/last 30 min flags |

---

## Understanding Signals

| Signal | Meaning | Action |
|--------|---------|--------|
| 🟢 BUY | Model predicts price rises >0.5% in next 15 min | Consider buying |
| 🔴 SELL | Model predicts price falls >0.5% in next 15 min | Consider shorting* |
| ⚪ WAIT | Confidence < 60% | Skip this trade |

*Intraday shorting requires margin — only do this if your broker supports it.

---

## Risk Management

The risk manager enforces:
- **Max loss per trade**: 2% of ₹3,500 = **₹70**
- **Stop-loss**: 0.5% below entry price
- **Target**: 1.0% above entry price
- **Max 2 trades per day** to preserve capital

Always set your stop-loss on Groww **immediately** after buying.

---

## ⚠️ Important Limitations

1. **Data delay**: Yahoo Finance provides ~15-minute delayed data for free.
   You are trading on stale information — use this system for learning only.

2. **No guarantee**: ML models can and do fail, especially in volatile markets.

3. **Manual execution**: You must place trades manually on Groww.
   There is no automatic order placement.

4. **Taxes & charges**: Brokerage, STT, exchange fees, and short-term capital
   gains tax are NOT accounted for in P&L calculations.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `No trained model found` | Run `python train_model.py` first |
| `No data returned` | Check your internet connection; Yahoo Finance may be temporarily unavailable |
| `XGBoost not installed` | Run `pip install xgboost`; the system will fall back to Random Forest automatically |
| Empty prediction output | Market may be closed — NSE trades Mon–Fri, 9:15 AM–3:30 PM IST |

---

## Disclaimer

This software is provided **"as is"** for educational purposes only.
The authors are not registered financial advisers.
Do not make real financial decisions based solely on this tool's output.
Always consult a SEBI-registered investment adviser before investing.
