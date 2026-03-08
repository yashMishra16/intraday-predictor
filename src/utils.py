"""
utils.py — Shared helper functions used across the project.
"""

import logging
import os
import sys
from datetime import datetime

import pytz

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Formatting ────────────────────────────────────────────────────────────────

def format_inr(amount: float) -> str:
    """Return a human-readable INR string, e.g. '₹5,450.00'."""
    return f"₹{amount:,.2f}"


def format_pct(value: float) -> str:
    """Return a percentage string, e.g. '67%'."""
    return f"{value * 100:.0f}%"


def today_ist() -> str:
    """Return today's date in Indian Standard Time as a readable string."""
    ist = pytz.timezone('Asia/Kolkata')
    # Use %d then strip the leading zero for cross-platform compatibility
    return datetime.now(ist).strftime('%B %d, %Y').replace(' 0', ' ')


# ── File Utilities ────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def project_root() -> str:
    """Return the absolute path of the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
