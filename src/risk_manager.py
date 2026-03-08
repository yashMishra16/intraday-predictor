"""
risk_manager.py — Position sizing, stop-loss, and target calculation.

All amounts are in Indian Rupees (INR).
"""

import logging
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)


@dataclass
class TradeParameters:
    """
    Holds all trade-level risk/sizing parameters for a single trade idea.
    """
    ticker: str
    entry_price: float
    stop_loss_price: float
    target_price: float
    quantity: int
    max_loss: float
    potential_profit: float
    position_value: float
    is_valid: bool
    reason: str = ""


def calculate_trade_parameters(
    ticker: str,
    entry_price: float,
    signal: str = 'BUY',
) -> TradeParameters:
    """
    Calculate position size, stop-loss, and target for a trade.

    Rules
    -----
    * Stop-loss is STOP_LOSS_PERCENT below entry (for BUY).
    * Target is TARGET_PERCENT above entry (for BUY).
    * Number of shares = floor(EFFECTIVE_CAPITAL / entry_price).
    * Max loss per share = entry_price * STOP_LOSS_PERCENT.
    * Total max loss must not exceed MAX_RISK_PER_TRADE * CAPITAL.

    Parameters
    ----------
    ticker       : e.g. 'SBIN.NS'
    entry_price  : Current / expected entry price in INR.
    signal       : 'BUY' or 'SELL'

    Returns
    -------
    TradeParameters dataclass.
    """
    if entry_price <= 0:
        return TradeParameters(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss_price=0,
            target_price=0,
            quantity=0,
            max_loss=0,
            potential_profit=0,
            position_value=0,
            is_valid=False,
            reason="Invalid entry price",
        )

    # Calculate stop-loss and target levels
    if signal == 'BUY':
        stop_loss_price = round(entry_price * (1 - config.STOP_LOSS_PERCENT), 2)
        target_price = round(entry_price * (1 + config.TARGET_PERCENT), 2)
    else:  # SELL / short
        stop_loss_price = round(entry_price * (1 + config.STOP_LOSS_PERCENT), 2)
        target_price = round(entry_price * (1 - config.TARGET_PERCENT), 2)

    # Risk per share
    risk_per_share = abs(entry_price - stop_loss_price)

    # Maximum allowable total loss
    max_allowed_loss = config.CAPITAL * config.MAX_RISK_PER_TRADE  # e.g. ₹70

    # Max shares based on capital (with margin)
    max_shares_capital = int(config.EFFECTIVE_CAPITAL // entry_price)

    # Max shares based on risk limit
    if risk_per_share > 0:
        max_shares_risk = int(max_allowed_loss // risk_per_share)
    else:
        max_shares_risk = max_shares_capital

    quantity = min(max_shares_capital, max_shares_risk)

    if quantity <= 0:
        return TradeParameters(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            target_price=target_price,
            quantity=0,
            max_loss=0,
            potential_profit=0,
            position_value=0,
            is_valid=False,
            reason=f"Entry price ₹{entry_price:,.2f} exceeds available capital",
        )

    max_loss = round(risk_per_share * quantity, 2)
    reward_per_share = abs(target_price - entry_price)
    potential_profit = round(reward_per_share * quantity, 2)
    position_value = round(entry_price * quantity, 2)

    # Final validation — max_loss must not exceed allowed limit
    is_valid = max_loss <= (config.CAPITAL * config.MAX_RISK_PER_TRADE)
    reason = "" if is_valid else f"Max loss ₹{max_loss} exceeds 2% risk limit (₹{config.CAPITAL * config.MAX_RISK_PER_TRADE:.2f})"

    return TradeParameters(
        ticker=ticker,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        target_price=target_price,
        quantity=quantity,
        max_loss=max_loss,
        potential_profit=potential_profit,
        position_value=position_value,
        is_valid=is_valid,
        reason=reason,
    )


def risk_reward_ratio(params: TradeParameters) -> float:
    """
    Return the risk/reward ratio.
    A value >= 2.0 means the trade offers at least 2× reward for every 1× risk.
    """
    if params.max_loss == 0:
        return 0.0
    return round(params.potential_profit / params.max_loss, 2)


def summarise(params: TradeParameters) -> str:
    """Return a short human-readable summary of the trade parameters."""
    if not params.is_valid:
        return f"❌ Trade not valid: {params.reason}"
    sl_pct = config.STOP_LOSS_PERCENT * 100
    tgt_pct = config.TARGET_PERCENT * 100
    rr = risk_reward_ratio(params)
    return (
        f"Entry: ₹{params.entry_price:,.2f} | "
        f"SL: ₹{params.stop_loss_price:,.2f} (-{sl_pct:.1f}%) | "
        f"Target: ₹{params.target_price:,.2f} (+{tgt_pct:.1f}%) | "
        f"Qty: {params.quantity} | "
        f"Max Loss: ₹{params.max_loss:,.2f} | "
        f"Potential Profit: ₹{params.potential_profit:,.2f} | "
        f"R:R = 1:{rr}"
    )
