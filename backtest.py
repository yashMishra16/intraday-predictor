#!/usr/bin/env python3
"""
backtest.py — Backtest the prediction strategy on historical data.

This simulates how the model would have performed in the past.
It does NOT account for slippage, brokerage, or taxes.

Usage:
    python backtest.py
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

import config
from src.data_fetcher import fetch_intraday_data
from src.features import add_features, get_feature_columns
from src.model import IntradayModel
from src.risk_manager import calculate_trade_parameters
from src.utils import setup_logging

logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')


def backtest_stock(short_name: str, ticker: str) -> dict:
    """
    Run a walk-forward backtest for one stock.

    Strategy
    --------
    * For each 5-min candle (after warm-up), use the trained model to predict UP/DOWN.
    * If confidence >= threshold AND it's not the last 30 min of day:
      - Enter at next candle's open.
      - Exit at stop-loss OR target OR end of day.
    * Track P&L per trade.

    Returns
    -------
    dict of backtest metrics.
    """
    print(f"\n{'─'*60}")
    print(f"  Backtesting: {short_name} ({ticker})")

    # Load trained model
    model = IntradayModel(short_name)
    if not model.load():
        print(f"  ❌ No trained model found. Run: python train_model.py")
        return {}

    # Fetch recent data for backtesting (use LOOKBACK since training data
    # was on TRAINING_PERIOD; here we use a longer period)
    df_raw = fetch_intraday_data(ticker, period='30d', interval=config.INTERVAL)
    if df_raw.empty or len(df_raw) < 50:
        print(f"  ❌ Not enough data to backtest.")
        return {}

    df = add_features(df_raw)
    if df.empty:
        return {}

    feature_cols = model.feature_cols

    trades = []
    i = 0
    rows = df.reset_index()  # so we can access by integer index

    while i < len(rows) - config.PREDICTION_HORIZON - 1:
        row = rows.iloc[i]
        ts = row['index'] if 'index' in rows.columns else row.name

        # Time of day check — avoid last 30 min
        if hasattr(ts, 'hour'):
            mins = ts.hour * 60 + ts.minute
            market_close_mins = config.MARKET_CLOSE_HOUR * 60 + config.MARKET_CLOSE_MINUTE
            if mins >= market_close_mins - 30:
                i += 1
                continue

        # Build feature vector for this candle
        feat_vec = row[feature_cols].values.reshape(1, -1)
        try:
            X_scaled = model.scaler.transform(feat_vec)
            proba = model.clf.predict_proba(X_scaled)[0]
        except Exception:
            i += 1
            continue

        p_up = float(proba[1])
        p_down = float(proba[0])

        if p_up >= config.CONFIDENCE_THRESHOLD:
            signal = 'BUY'
            confidence = p_up
        elif p_down >= config.CONFIDENCE_THRESHOLD:
            signal = 'SELL'
            confidence = p_down
        else:
            i += 1
            continue

        # Simulate entry at next candle open
        entry_idx = i + 1
        if entry_idx >= len(rows):
            break

        entry_price = float(rows.iloc[entry_idx]['Open'])
        tp = calculate_trade_parameters(ticker, entry_price, signal)
        if not tp.is_valid or tp.quantity == 0:
            i += 1
            continue

        # Walk forward until stop-loss, target, or end of day hit
        entry_ts = rows.iloc[entry_idx]['index'] if 'index' in rows.columns else rows.iloc[entry_idx].name
        trade_result = 'OPEN'
        exit_price = entry_price
        exit_idx = entry_idx

        # Maximum candles per trading day based on market hours and interval
        interval_minutes = int(config.INTERVAL.replace('m', ''))
        market_minutes = (
            (config.MARKET_CLOSE_HOUR - config.MARKET_OPEN_HOUR) * 60
            + (config.MARKET_CLOSE_MINUTE - config.MARKET_OPEN_MINUTE)
        )
        max_candles_per_day = market_minutes // interval_minutes

        for j in range(entry_idx + 1, min(entry_idx + max_candles_per_day + 1, len(rows))):
            future_row = rows.iloc[j]
            future_ts = future_row['index'] if 'index' in rows.columns else future_row.name
            hi = float(future_row['High'])
            lo = float(future_row['Low'])

            # Check if same trading day
            if hasattr(future_ts, 'date') and hasattr(entry_ts, 'date'):
                if future_ts.date() != entry_ts.date():
                    # Close at day's end
                    exit_price = float(future_row['Open'])
                    trade_result = 'EOD'
                    exit_idx = j
                    break

            if signal == 'BUY':
                if lo <= tp.stop_loss_price:
                    exit_price = tp.stop_loss_price
                    trade_result = 'STOP'
                    exit_idx = j
                    break
                if hi >= tp.target_price:
                    exit_price = tp.target_price
                    trade_result = 'TARGET'
                    exit_idx = j
                    break
            else:  # SELL
                if hi >= tp.stop_loss_price:
                    exit_price = tp.stop_loss_price
                    trade_result = 'STOP'
                    exit_idx = j
                    break
                if lo <= tp.target_price:
                    exit_price = tp.target_price
                    trade_result = 'TARGET'
                    exit_idx = j
                    break

        if trade_result == 'OPEN':
            eod_idx = min(entry_idx + max_candles_per_day, len(rows) - 1)
            exit_price = float(rows.iloc[eod_idx]['Close'])
            trade_result = 'EOD'

        if signal == 'BUY':
            pnl = (exit_price - entry_price) * tp.quantity
        else:
            pnl = (entry_price - exit_price) * tp.quantity

        trades.append({
            'entry_time': entry_ts,
            'signal': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': tp.quantity,
            'pnl': round(pnl, 2),
            'result': trade_result,
        })

        i = exit_idx + 1  # skip to after the trade exits

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not trades:
        print("  No trades generated in the backtest period.")
        return {}

    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    wins = (df_trades['pnl'] > 0).sum()
    losses = (df_trades['pnl'] < 0).sum()
    win_rate = wins / total_trades if total_trades else 0

    total_pnl = df_trades['pnl'].sum()
    avg_profit = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins else 0
    avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losses else 0

    # Max drawdown
    cumulative = df_trades['pnl'].cumsum()
    rolling_max = cumulative.cummax()
    drawdown = rolling_max - cumulative
    max_drawdown = float(drawdown.max())

    # Sharpe-like ratio (daily P&L)
    pnl_std = df_trades['pnl'].std()
    sharpe = (df_trades['pnl'].mean() / pnl_std * np.sqrt(total_trades)) if pnl_std else 0

    metrics = {
        'ticker': ticker,
        'total_trades': total_trades,
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': win_rate,
        'total_pnl': round(total_pnl, 2),
        'avg_profit': round(avg_profit, 2),
        'avg_loss': round(avg_loss, 2),
        'max_drawdown': round(max_drawdown, 2),
        'sharpe': round(sharpe, 2),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print(f"  Backtest Results — {short_name}")
    print(f"  {'─'*40}")
    print(f"  Total trades    : {total_trades}")
    print(f"  Wins            : {wins}  |  Losses: {losses}")
    print(f"  Win Rate        : {win_rate:.1%}")
    print(f"  Total P&L       : ₹{total_pnl:,.2f}")
    print(f"  Avg Profit/trade: ₹{avg_profit:,.2f}")
    print(f"  Avg Loss/trade  : ₹{avg_loss:,.2f}")
    print(f"  Max Drawdown    : ₹{max_drawdown:,.2f}")
    print(f"  Sharpe (approx) : {sharpe:.2f}")
    print(f"  {'─'*40}")
    print(f"  ⚠️  Past performance does not guarantee future results.")

    return metrics


def main() -> None:
    setup_logging(logging.WARNING)
    print("\n  INTRADAY PREDICTOR — Backtesting")
    print("  " + "─" * 50)

    all_metrics = []
    for short_name, ticker in config.STOCKS.items():
        m = backtest_stock(short_name, ticker)
        if m:
            all_metrics.append(m)

    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("  Combined Portfolio Summary")
        print(f"{'='*60}")
        total_pnl = sum(m['total_pnl'] for m in all_metrics)
        total_trades = sum(m['total_trades'] for m in all_metrics)
        avg_win_rate = np.mean([m['win_rate'] for m in all_metrics])
        print(f"  Total P&L       : ₹{total_pnl:,.2f}")
        print(f"  Total trades    : {total_trades}")
        print(f"  Avg win rate    : {avg_win_rate:.1%}")

    print()


if __name__ == '__main__':
    main()
