#!/usr/bin/env python3
"""
run_backtest.py – Entry-point for the Sentiment-Positioning Hybrid backtest.

Usage
-----
    python run_backtest.py                          # defaults
    python run_backtest.py --bars 2000 --capital 50000
    python run_backtest.py --help

The script:
    1. Generates synthetic hourly OHLCV + funding-rate + sentiment data.
    2. Computes all indicators and produces entry/exit signals.
    3. Runs the backtest engine (risk management, position sizing, PnL).
    4. Prints a summary report to stdout.
"""

from __future__ import annotations

import argparse

from backtest.data import generate_synthetic_data
from backtest.engine import BacktestEngine
from backtest.signals import generate_signals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentiment-Positioning Hybrid Crypto Backtest"
    )
    p.add_argument("--bars", type=int, default=1000,
                   help="Number of hourly bars to simulate (default: 1000)")
    p.add_argument("--capital", type=float, default=10_000.0,
                   help="Starting capital in USD (default: 10000)")
    p.add_argument("--risk", type=float, default=0.01,
                   help="Risk per trade as a fraction (default: 0.01 = 1%%)")
    p.add_argument("--stop-loss", type=float, default=0.02,
                   help="Stop-loss as a fraction of entry price (default: 0.02 = 2%%)")
    p.add_argument("--tp-ratio", type=float, default=1.5,
                   help="Take-profit / stop-loss ratio (default: 1.5)")
    p.add_argument("--max-positions", type=int, default=5,
                   help="Max concurrent open positions (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for data generation (default: 42)")
    p.add_argument("--start-price", type=float, default=30_000.0,
                   help="Starting price for synthetic data (default: 30000)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nGenerating {args.bars} bars of synthetic data (seed={args.seed}) …")
    df = generate_synthetic_data(
        n_bars=args.bars,
        start_price=args.start_price,
        seed=args.seed,
    )

    print("Computing signals …")
    df_signals = generate_signals(df)

    n_long = int((df_signals["signal"] == 1).sum())
    n_short = int((df_signals["signal"] == -1).sum())
    print(f"  Long signals : {n_long}")
    print(f"  Short signals: {n_short}")

    print("Running backtest …")
    engine = BacktestEngine(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        stop_loss_pct=args.stop_loss,
        take_profit_ratio=args.tp_ratio,
        max_positions=args.max_positions,
    )
    result = engine.run(df_signals)

    print("\n" + result.summary())

    if result.trades:
        reasons: dict[str, int] = {}
        for t in result.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        print("\n  Exit reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"    {reason:<20} {count}")
        print()


if __name__ == "__main__":
    main()
