"""
data_fetcher.py — Fetch OHLCV intraday data from Yahoo Finance.

Yahoo Finance provides data with a ~15-minute delay for free users.
This data is suitable for paper trading and learning, NOT for live trading.
"""

import logging
from datetime import datetime

import pandas as pd
import pytz
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# Indian Standard Time timezone
IST = pytz.timezone('Asia/Kolkata')


def fetch_intraday_data(
    ticker: str,
    period: str = config.LOOKBACK_PERIOD,
    interval: str = config.INTERVAL,
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV data for a single ticker.

    Parameters
    ----------
    ticker  : Yahoo Finance ticker symbol, e.g. 'SBIN.NS'
    period  : How far back to fetch, e.g. '5d', '60d'
    interval: Candle size, e.g. '5m', '15m', '1d'

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume]
    indexed by datetime (IST-aware).  Returns empty DataFrame on error.
    """
    try:
        logger.info(f"Fetching {interval} data for {ticker} (period={period})")
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Ensure the index is tz-aware in IST
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        # Keep only core OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.dropna(inplace=True)

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as exc:
        logger.error(f"Error fetching data for {ticker}: {exc}")
        return pd.DataFrame()


def fetch_all_stocks(
    period: str = config.LOOKBACK_PERIOD,
    interval: str = config.INTERVAL,
) -> dict[str, pd.DataFrame]:
    """
    Fetch intraday data for all stocks defined in config.STOCKS.

    Returns
    -------
    dict mapping short_name → DataFrame, e.g. {'SBIN': <df>, ...}
    """
    result = {}
    for short_name, ticker in config.STOCKS.items():
        df = fetch_intraday_data(ticker, period=period, interval=interval)
        if not df.empty:
            result[short_name] = df
        else:
            logger.warning(f"Skipping {short_name} — no data available")
    return result


def get_latest_price(ticker: str) -> float | None:
    """
    Return the most recent closing price for a ticker.

    Returns None if data cannot be fetched.
    """
    df = fetch_intraday_data(ticker, period='1d', interval='1m')
    if df.empty:
        return None
    return float(df['Close'].iloc[-1])


def is_market_open() -> bool:
    """
    Return True if the Indian stock market (NSE/BSE) is currently open.
    Market hours: Monday–Friday, 9:15 AM – 3:30 PM IST.
    """
    now = datetime.now(IST)
    if now.weekday() >= 5:          # Saturday (5) or Sunday (6)
        return False
    market_open = now.replace(
        hour=config.MARKET_OPEN_HOUR,
        minute=config.MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0,
    )
    market_close = now.replace(
        hour=config.MARKET_CLOSE_HOUR,
        minute=config.MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0,
    )
    return market_open <= now <= market_close
