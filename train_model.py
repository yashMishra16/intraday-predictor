#!/usr/bin/env python3
"""
train_model.py — Train (or retrain) prediction models for all configured stocks.

Run this script once before using main.py, and again periodically
(e.g., weekly) to keep the models fresh.

Usage:
    python train_model.py
"""

import logging

import config
from src.data_fetcher import fetch_intraday_data
from src.model import IntradayModel
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def train_stock(short_name: str, ticker: str) -> None:
    """Train and save the model for a single stock."""
    print(f"\n{'='*60}")
    print(f"  Training model for: {short_name} ({ticker})")
    print(f"{'='*60}")

    # Fetch training data — use a longer period for more samples
    print(f"  Fetching {config.TRAINING_PERIOD} of 5-minute data …")
    df = fetch_intraday_data(
        ticker,
        period=config.TRAINING_PERIOD,
        interval=config.INTERVAL,
    )

    if df.empty:
        print(f"  ❌ No data available for {ticker}. Skipping.")
        return

    print(f"  Got {len(df)} rows of data.")

    # Train model
    model = IntradayModel(short_name)
    try:
        metrics = model.train(df)
    except ValueError as e:
        print(f"  ❌ Training failed: {e}")
        return

    # Display results
    print(f"\n  ✅ Training complete!")
    print(f"     Samples     : {metrics['n_samples']}")
    print(f"     Train acc   : {metrics['train_accuracy']:.1%}")
    print(f"     CV accuracy : {metrics['cv_accuracy']:.1%}")
    print(f"\n  Classification report (training set):")
    for line in metrics['report'].splitlines():
        print(f"     {line}")

    # Feature importances
    imp = model.feature_importance()
    if imp is not None:
        print(f"\n  Top-10 most important features:")
        for _, row in imp.head(10).iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"     {row['feature']:<22} {row['importance']:.4f}  {bar}")

    # Save to disk
    path = model.save()
    print(f"\n  💾 Model saved to: {path}")


def main() -> None:
    setup_logging(logging.WARNING)  # suppress verbose logs during training
    print("\n  INTRADAY PREDICTOR — Model Training")
    print("  " + "─" * 50)
    print(f"  Stocks  : {', '.join(config.STOCKS.keys())}")
    print(f"  Period  : {config.TRAINING_PERIOD}")
    print(f"  Interval: {config.INTERVAL}")

    for short_name, ticker in config.STOCKS.items():
        train_stock(short_name, ticker)

    print(f"\n{'='*60}")
    print("  All models trained. You can now run: python main.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
