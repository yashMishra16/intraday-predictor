"""
Configuration file for the Intraday Predictor.
Edit this file to adjust capital, stocks, and risk settings.
"""

# ─────────────────────────────────────────────────────────────
# User Settings
# ─────────────────────────────────────────────────────────────
CAPITAL = 3500                      # Your actual capital in INR
MARGIN_MULTIPLIER = 5               # 5x intraday margin offered by broker
EFFECTIVE_CAPITAL = CAPITAL * MARGIN_MULTIPLIER  # ₹17,500 buying power

# ─────────────────────────────────────────────────────────────
# Stocks to Trade
# ─────────────────────────────────────────────────────────────
STOCKS = {
    'GODFRYPHLP': 'GODFRYPHLP.NS',
    'SBIN': 'SBIN.NS',
}

# ─────────────────────────────────────────────────────────────
# Risk Management
# ─────────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE = 0.02           # 2% of capital = ₹70 max loss per trade
STOP_LOSS_PERCENT = 0.005           # 0.5% stop-loss below entry price
TARGET_PERCENT = 0.01               # 1.0% target above entry price
MAX_TRADES_PER_DAY = 2              # Maximum simultaneous trades

# Minimum confidence required to show a BUY/SELL signal (0–1 scale)
CONFIDENCE_THRESHOLD = 0.60

# ─────────────────────────────────────────────────────────────
# Data Settings
# ─────────────────────────────────────────────────────────────
INTERVAL = '5m'                     # 5-minute candles
LOOKBACK_PERIOD = '5d'              # Last 5 trading days of minute data
TRAINING_PERIOD = '60d'             # Data used for model training

# ─────────────────────────────────────────────────────────────
# Model Settings
# ─────────────────────────────────────────────────────────────
PREDICTION_HORIZON = 3              # Predict price direction 3 periods (15 min) ahead
MODEL_DIR = 'models'                # Directory where trained models are saved

# ─────────────────────────────────────────────────────────────
# Paper Trading
# ─────────────────────────────────────────────────────────────
PAPER_TRADE_LOG = 'paper_trades.csv'

# ─────────────────────────────────────────────────────────────
# Indian Market Timing (IST)
# ─────────────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
