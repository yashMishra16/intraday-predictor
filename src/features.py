"""
features.py — Calculate technical indicators used as ML features.

All indicators are computed using the `ta` library plus custom logic.
NaN rows produced by look-back windows are dropped before returning.
"""

import logging

import numpy as np
import pandas as pd
import ta

import config

logger = logging.getLogger(__name__)

# Market open time used for time-of-day features (minutes since open)
_MARKET_OPEN_MINUTES = config.MARKET_OPEN_HOUR * 60 + config.MARKET_OPEN_MINUTE


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator features to an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns [Open, High, Low, Close, Volume]
         and a DatetimeIndex (IST-aware preferred).

    Returns
    -------
    DataFrame with all original columns plus feature columns.
    Rows with NaN values (from indicator warm-up) are dropped.
    """
    if df.empty or len(df) < 30:
        logger.warning("Not enough data to calculate features (need ≥30 rows)")
        return pd.DataFrame()

    df = df.copy()

    # ── Momentum / Returns ───────────────────────────────────────────────────
    for period in [1, 3, 5, 10]:
        df[f'return_{period}'] = df['Close'].pct_change(period)

    # Rate of Change
    df['roc_5'] = ta.momentum.ROCIndicator(df['Close'], window=5).roc()
    df['roc_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()

    # ── Moving Averages ───────────────────────────────────────────────────────
    df['ema_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
    df['ema_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()

    # Price relative to MAs (normalised)
    df['price_vs_ema9'] = (df['Close'] - df['ema_9']) / df['ema_9']
    df['price_vs_ema21'] = (df['Close'] - df['ema_21']) / df['ema_21']
    df['ema9_vs_ema21'] = (df['ema_9'] - df['ema_21']) / df['ema_21']

    # ── VWAP ──────────────────────────────────────────────────────────────────
    # Typical price × volume cumsum / volume cumsum, reset each day
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_vol'] = df['typical_price'] * df['Volume']
    df['cum_vol'] = df.groupby(df.index.date)['Volume'].transform('cumsum')
    df['cum_tp_vol'] = df.groupby(df.index.date)['tp_vol'].transform('cumsum')
    df['vwap'] = df['cum_tp_vol'] / df['cum_vol']
    df['price_vs_vwap'] = (df['Close'] - df['vwap']) / df['vwap']
    # Drop helper columns
    df.drop(columns=['typical_price', 'tp_vol', 'cum_vol', 'cum_tp_vol'], inplace=True)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    # Normalise RSI to 0–1 range
    df['rsi_norm'] = df['rsi'] / 100.0

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd_ind = ta.trend.MACD(df['Close'])
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_hist'] = macd_ind.macd_diff()
    # Normalise by closing price
    df['macd_norm'] = df['macd'] / df['Close']
    df['macd_signal_norm'] = df['macd_signal'] / df['Close']

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (
        df['bb_upper'] - df['bb_lower'] + 1e-10
    )

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_avg = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (vol_avg + 1e-10)
    df['volume_trend'] = df['Volume'].pct_change(5)

    # ── Volatility ────────────────────────────────────────────────────────────
    # Average True Range (normalised)
    df['atr'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close'], window=14
    ).average_true_range()
    df['atr_norm'] = df['atr'] / df['Close']

    # Rolling standard deviation of returns (annualised not needed here)
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()

    # ── Time Features ─────────────────────────────────────────────────────────
    if df.index.tz is not None:
        minutes = df.index.hour * 60 + df.index.minute
    else:
        minutes = pd.Series(df.index).apply(lambda t: t.hour * 60 + t.minute).values

    df['mins_since_open'] = minutes - _MARKET_OPEN_MINUTES
    df['is_first_30min'] = (df['mins_since_open'] <= 30).astype(int)
    df['is_last_30min'] = (df['mins_since_open'] >= (6 * 60 + 15 - 30)).astype(int)

    # ── Target Variable ───────────────────────────────────────────────────────
    # 1 if price is higher N periods in the future, else 0
    horizon = config.PREDICTION_HORIZON
    df['target'] = (
        df['Close'].shift(-horizon) > df['Close']
    ).astype(int)

    # Drop rows with NaN (indicator warm-up + last N rows without target)
    df.dropna(inplace=True)

    logger.info(f"Features calculated: {len(df)} rows, {len(df.columns)} columns")
    return df


# Columns that are *not* input features (they will be excluded from X)
NON_FEATURE_COLS = {
    'Open', 'High', 'Low', 'Close', 'Volume',
    'vwap', 'ema_9', 'ema_21', 'sma_20',
    'bb_upper', 'bb_lower', 'atr', 'rsi',
    'macd', 'macd_signal',
    'target',
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes raw OHLCV and target)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
