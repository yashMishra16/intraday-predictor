#!/usr/bin/env python3
"""
paper_trade.py — Paper trading tracker.

Log your intended trades (from main.py), then record the actual outcome
to track your win/loss ratio and cumulative P&L without risking real money.

Usage:
    # Log a new trade prediction
    python paper_trade.py log SBIN BUY 750.00

    # Record the outcome of a trade
    python paper_trade.py outcome SBIN 760.00

    # Show your trading summary
    python paper_trade.py summary
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import pytz

import config
from src.utils import setup_logging

LOG_FILE = config.PAPER_TRADE_LOG
IST = pytz.timezone('Asia/Kolkata')
FIELDNAMES = [
    'trade_id', 'date', 'ticker', 'signal',
    'entry_price', 'stop_loss', 'target', 'quantity',
    'exit_price', 'pnl', 'result', 'notes',
]


def _now_ist() -> str:
    return datetime.now(IST).strftime('%Y-%m-%d %H:%M')


def _read_trades() -> list[dict]:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, newline='') as f:
        return list(csv.DictReader(f))


def _write_trades(trades: list[dict]) -> None:
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(trades)


def _next_id(trades: list[dict]) -> int:
    if not trades:
        return 1
    return max(int(t['trade_id']) for t in trades) + 1


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_log(args: argparse.Namespace) -> None:
    """Log a new paper trade based on a signal from main.py."""
    from src.risk_manager import calculate_trade_parameters

    ticker_short = args.ticker.upper()
    ticker = config.STOCKS.get(ticker_short, ticker_short + '.NS')
    signal = args.signal.upper()
    entry_price = float(args.entry_price)

    tp = calculate_trade_parameters(ticker, entry_price, signal)

    trades = _read_trades()
    trade_id = _next_id(trades)

    trade = {
        'trade_id': trade_id,
        'date': _now_ist(),
        'ticker': ticker_short,
        'signal': signal,
        'entry_price': entry_price,
        'stop_loss': tp.stop_loss_price,
        'target': tp.target_price,
        'quantity': tp.quantity,
        'exit_price': '',
        'pnl': '',
        'result': 'OPEN',
        'notes': '',
    }
    trades.append(trade)
    _write_trades(trades)

    print(f"\n  ✅ Paper trade #{trade_id} logged!")
    print(f"     Ticker    : {ticker_short}")
    print(f"     Signal    : {signal}")
    print(f"     Entry     : ₹{entry_price:,.2f}")
    print(f"     Stop Loss : ₹{tp.stop_loss_price:,.2f}")
    print(f"     Target    : ₹{tp.target_price:,.2f}")
    print(f"     Quantity  : {tp.quantity} shares")
    print(f"\n  Now enter this on Groww with these levels!")
    print(f"  When you exit, run: python paper_trade.py outcome {ticker_short} <exit_price>\n")


def cmd_outcome(args: argparse.Namespace) -> None:
    """Record the actual exit price for the most recent open trade."""
    ticker_short = args.ticker.upper()
    exit_price = float(args.exit_price)

    trades = _read_trades()

    # Find the most recent open trade for this ticker
    open_trades = [
        (i, t) for i, t in enumerate(trades)
        if t['ticker'] == ticker_short and t['result'] == 'OPEN'
    ]

    if not open_trades:
        print(f"  ❌ No open paper trades found for {ticker_short}.")
        return

    idx, trade = open_trades[-1]  # most recent open

    entry_price = float(trade['entry_price'])
    quantity = int(trade['quantity'])
    signal = trade['signal']

    if signal == 'BUY':
        pnl = (exit_price - entry_price) * quantity
    else:
        pnl = (entry_price - exit_price) * quantity

    result = 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BREAKEVEN')

    trades[idx]['exit_price'] = exit_price
    trades[idx]['pnl'] = round(pnl, 2)
    trades[idx]['result'] = result

    _write_trades(trades)

    emoji = '🟢' if pnl > 0 else ('🔴' if pnl < 0 else '⚪')
    print(f"\n  {emoji} Trade #{trade['trade_id']} closed!")
    print(f"     Exit Price : ₹{exit_price:,.2f}")
    print(f"     P&L        : ₹{pnl:+,.2f}")
    print(f"     Result     : {result}\n")


def cmd_summary(args: argparse.Namespace) -> None:
    """Display P&L summary and win/loss statistics."""
    trades = _read_trades()

    if not trades:
        print(f"\n  No paper trades logged yet.")
        print(f"  Start by running: python paper_trade.py log SBIN BUY 750.00\n")
        return

    closed = [t for t in trades if t['result'] not in ('OPEN', '')]
    open_trades = [t for t in trades if t['result'] == 'OPEN']

    total = len(closed)
    wins = sum(1 for t in closed if t['result'] == 'WIN')
    losses = sum(1 for t in closed if t['result'] == 'LOSS')
    win_rate = (wins / total * 100) if total else 0

    total_pnl = sum(float(t['pnl']) for t in closed if t['pnl'] != '')
    avg_pnl = total_pnl / total if total else 0

    print(f"\n  {'='*50}")
    print(f"  Paper Trading Summary")
    print(f"  {'='*50}")
    print(f"  Closed trades   : {total}")
    print(f"  Wins            : {wins}  |  Losses: {losses}")
    print(f"  Win Rate        : {win_rate:.1f}%")
    print(f"  Total P&L       : ₹{total_pnl:+,.2f}")
    print(f"  Avg P&L/trade   : ₹{avg_pnl:+,.2f}")

    if open_trades:
        print(f"\n  Open trades     : {len(open_trades)}")
        for t in open_trades:
            print(f"    #{t['trade_id']}  {t['ticker']}  {t['signal']}  "
                  f"Entry ₹{float(t['entry_price']):,.2f}  "
                  f"SL ₹{float(t['stop_loss']):,.2f}  "
                  f"Target ₹{float(t['target']):,.2f}")

    print(f"\n  📄 Full log: {LOG_FILE}")
    print(f"  {'='*50}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description='Paper trading tracker for the Intraday Predictor'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # log
    p_log = sub.add_parser('log', help='Log a new paper trade')
    p_log.add_argument('ticker', help='Stock ticker short name, e.g. SBIN')
    p_log.add_argument('signal', choices=['BUY', 'SELL', 'buy', 'sell'])
    p_log.add_argument('entry_price', type=float, help='Entry price in INR')

    # outcome
    p_out = sub.add_parser('outcome', help='Record exit price for an open trade')
    p_out.add_argument('ticker', help='Stock ticker short name, e.g. SBIN')
    p_out.add_argument('exit_price', type=float, help='Actual exit price in INR')

    # summary
    sub.add_parser('summary', help='Show P&L summary')

    args = parser.parse_args()

    if args.command == 'log':
        cmd_log(args)
    elif args.command == 'outcome':
        cmd_outcome(args)
    elif args.command == 'summary':
        cmd_summary(args)


if __name__ == '__main__':
    main()
