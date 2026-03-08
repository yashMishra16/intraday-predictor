#!/usr/bin/env python3
"""
main.py — Daily prediction script.

Run this every morning before 9:15 AM IST (or during market hours)
to get today's trading signals.

Usage:
    python main.py
"""

from datetime import datetime

import pytz

import config
from src.predictor import PredictionResult, run_all_predictions
from src.risk_manager import risk_reward_ratio
from src.utils import setup_logging, today_ist

# ── Pretty-print helpers ──────────────────────────────────────────────────────

WIDTH = 62  # Total box width (inner content width)


def _line(text: str = "") -> str:
    """Left-pad text inside a box line."""
    return f"║  {text:<{WIDTH - 4}}║"


def _divider() -> str:
    return "╠" + "═" * (WIDTH) + "╣"


def _top() -> str:
    return "╔" + "═" * (WIDTH) + "╗"


def _bottom() -> str:
    return "╚" + "═" * (WIDTH) + "╝"


def _title(text: str) -> str:
    padded = text.center(WIDTH)
    return f"║{padded}║"


def _signal_emoji(signal: str) -> str:
    return {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪"}.get(signal, "⚪")


def _stock_emoji(short_name: str) -> str:
    emojis = {
        "GODFRYPHLP": "📈",
        "SBIN": "🏦",
    }
    return emojis.get(short_name, "📊")


FULL_NAMES = {
    "GODFRYPHLP": "GODFREY PHILLIPS",
    "SBIN": "STATE BANK OF INDIA",
}

# ── Section printers ──────────────────────────────────────────────────────────

def _print_stock_section(result: PredictionResult) -> None:
    emoji = _stock_emoji(result.short_name)
    full = FULL_NAMES.get(result.short_name, result.short_name)
    ticker = result.ticker

    print(_line())
    print(_line(f"  {emoji} {full} ({ticker})"))
    print(_line("  " + "─" * (WIDTH - 8)))

    if result.error and result.current_price == 0:
        print(_line(f"  ⚠️  {result.error}"))
        print(_line())
        return

    # Basic info
    print(_line(f"  Current Price: ₹{result.current_price:,.2f}"))
    sig_emoji = _signal_emoji(result.signal)
    print(_line(f"  Signal: {result.signal} {sig_emoji}"))
    print(_line(f"  Confidence: {result.confidence * 100:.0f}%"))
    print(_line())

    tp = result.trade_params
    if result.signal == 'WAIT' or tp is None:
        if result.error:
            reason = result.error
        elif result.confidence < config.CONFIDENCE_THRESHOLD:
            reason = "Confidence too low, skip this trade"
        else:
            reason = "No clear signal right now"
        print(_line(f"  ➤ Reason: {reason}"))
    else:
        sl_pct = config.STOP_LOSS_PERCENT * 100
        tgt_pct = config.TARGET_PERCENT * 100
        rr = risk_reward_ratio(tp)
        print(_line(f"  ➤ Entry Price:     ₹{tp.entry_price:,.2f}"))
        print(_line(f"  ➤ Stop Loss:       ₹{tp.stop_loss_price:,.2f} (-{sl_pct:.1f}%)"))
        print(_line(f"  ➤ Target:          ₹{tp.target_price:,.2f} (+{tgt_pct:.1f}%)"))
        print(_line(f"  ➤ Quantity:        {tp.quantity} shares"))
        print(_line(f"  ➤ Max Loss:        ₹{tp.max_loss:,.2f}"))
        print(_line(f"  ➤ Potential Profit:₹{tp.potential_profit:,.2f}"))
        print(_line(f"  ➤ Risk/Reward:     1:{rr}"))

    print(_line())


def _print_recommendation(results: list[PredictionResult]) -> None:
    """Print today's overall recommendation at the bottom of the box."""
    print(_line())
    print(_line("  💡 TODAY'S RECOMMENDATION"))
    print(_line("  " + "─" * (WIDTH - 8)))

    actionable = [r for r in results if r.signal in ('BUY', 'SELL') and not r.error]

    if not actionable:
        print(_line("  No high-confidence signals today."))
        print(_line("  Consider staying in CASH and waiting for tomorrow."))
    else:
        for r in actionable[:config.MAX_TRADES_PER_DAY]:
            full = FULL_NAMES.get(r.short_name, r.short_name)
            print(_line(f"  Trade {r.short_name} ({full}) with the levels above."))
        print(_line("  Set stop-loss IMMEDIATELY after buying!"))
        print(_line("  ⚠️  Never risk more than you can afford to lose."))

    print(_line())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging()

    print()
    print(_top())
    print(_title(f"INTRADAY PREDICTIONS - {today_ist()}"))

    results = run_all_predictions()

    for i, result in enumerate(results):
        print(_divider())
        _print_stock_section(result)

    print(_divider())
    _print_recommendation(results)
    print(_bottom())
    print()

    # Reminder about data delay
    ist = pytz.timezone('Asia/Kolkata')
    now_str = datetime.now(ist).strftime('%I:%M %p IST')
    print(f"  ℹ️  Generated at {now_str}  |  Data is ~15 min delayed (Yahoo Finance)")
    print(f"  ⚠️  THIS IS FOR EDUCATIONAL/PAPER TRADING ONLY — NOT FINANCIAL ADVICE")
    print()


if __name__ == '__main__':
    main()
