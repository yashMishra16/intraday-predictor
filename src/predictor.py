"""
predictor.py — Orchestrates data fetching, feature calculation, model prediction,
and risk management for all configured stocks.
"""

import logging
from dataclasses import dataclass

import pandas as pd

import config
from src.data_fetcher import fetch_intraday_data
from src.model import IntradayModel
from src.risk_manager import TradeParameters, calculate_trade_parameters

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Holds the complete prediction output for one stock.
    """
    short_name: str
    ticker: str
    current_price: float
    signal: str               # 'BUY', 'SELL', or 'WAIT'
    confidence: float         # 0–1
    trade_params: TradeParameters | None
    error: str = ""


def predict_stock(
    short_name: str,
    ticker: str,
    model: IntradayModel,
) -> PredictionResult:
    """
    Run end-to-end prediction for a single stock.

    Parameters
    ----------
    short_name : e.g. 'SBIN'
    ticker     : Yahoo Finance symbol e.g. 'SBIN.NS'
    model      : Trained IntradayModel instance

    Returns
    -------
    PredictionResult
    """
    # 1. Fetch recent data
    df = fetch_intraday_data(ticker, period=config.LOOKBACK_PERIOD, interval=config.INTERVAL)
    if df.empty:
        return PredictionResult(
            short_name=short_name,
            ticker=ticker,
            current_price=0.0,
            signal='WAIT',
            confidence=0.0,
            trade_params=None,
            error=f"Could not fetch data for {ticker}",
        )

    current_price = float(df['Close'].iloc[-1])

    # 2. Get ML signal
    try:
        signal, confidence = model.predict(df)
    except RuntimeError as e:
        return PredictionResult(
            short_name=short_name,
            ticker=ticker,
            current_price=current_price,
            signal='WAIT',
            confidence=0.0,
            trade_params=None,
            error=str(e),
        )

    # 3. Calculate risk parameters (only for actionable signals)
    trade_params = None
    if signal in ('BUY', 'SELL'):
        trade_params = calculate_trade_parameters(ticker, current_price, signal)

    return PredictionResult(
        short_name=short_name,
        ticker=ticker,
        current_price=current_price,
        signal=signal,
        confidence=confidence,
        trade_params=trade_params,
    )


def run_all_predictions() -> list[PredictionResult]:
    """
    Run predictions for every stock in config.STOCKS.

    Loads each stock's saved model from disk.

    Returns
    -------
    list of PredictionResult, one per stock.
    """
    results = []
    for short_name, ticker in config.STOCKS.items():
        model = IntradayModel(short_name)
        loaded = model.load()
        if not loaded:
            logger.warning(
                f"No trained model for {short_name}. "
                "Run train_model.py first."
            )
            results.append(PredictionResult(
                short_name=short_name,
                ticker=ticker,
                current_price=0.0,
                signal='WAIT',
                confidence=0.0,
                trade_params=None,
                error="Model not trained. Run: python train_model.py",
            ))
            continue

        result = predict_stock(short_name, ticker, model)
        results.append(result)
        logger.info(
            f"[{short_name}] Signal={result.signal} "
            f"Confidence={result.confidence:.1%} "
            f"Price=₹{result.current_price:,.2f}"
        )

    return results
