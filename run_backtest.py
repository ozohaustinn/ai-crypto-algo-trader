#!/usr/bin/env python3
"""
run_backtest.py – Entry-point for the Sentiment-Positioning Hybrid backtest.

Usage
-----
    python run_backtest.py                                              # synthetic, defaults
    python run_backtest.py --bars 2000 --capital 50000                  # bigger synthetic run
    python run_backtest.py --mode historical \
        --symbol BTC/USDT --start 2022-01-01 --end 2025-01-01           # real Binance data
    python run_backtest.py --help

The script:
    1. Loads either synthetic or real historical OHLCV + funding + sentiment data.
    2. Computes all indicators and produces entry/exit signals.
    3. Runs the backtest engine (risk management, position sizing, PnL).
    4. Prints a summary report to stdout.
"""
from __future__ import annotations

import argparse

from backtest.data import get_data
from backtest.engine import BacktestEngine
from backtest.signals import generate_signals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentiment-Positioning Hybrid Crypto Backtest"
    )

    # --- data source ---------------------------------------------------------
    p.add_argument("--mode", choices=["synthetic", "historical"], default="synthetic",
                   help="Data source: synthetic GBM or real historical (default: synthetic)")

    # synthetic-only options
    p.add_argument("--bars", type=int, default=1000,
                   help="[synthetic] Number of hourly bars to simulate (default: 1000)")
    p.add_argument("--seed", type=int, default=42,
                   help="[synthetic] Random seed for data generation (default: 42)")
    p.add_argument("--start-price", type=float, default=30_000.0,
                   help="[synthetic] Starting price (default: 30000)")

    # historical-only options
    p.add_argument("--symbol", type=str, default="BTC/USDT",
                   help="[historical] ccxt symbol (default: BTC/USDT)")
    p.add_argument("--timeframe", type=str, default="1h",
                   help="[historical] Bar timeframe (default: 1h)")
    p.add_argument("--start", type=str, default="2023-01-01",
                   help="[historical] Start date YYYY-MM-DD (default: 2023-01-01)")
    p.add_argument("--end", type=str, default="2025-01-01",
                   help="[historical] End date YYYY-MM-DD (default: 2025-01-01)")
    p.add_argument("--exchange", type=str, default="binance",
                   help="[historical] ccxt exchange name (default: binance)")
    p.add_argument("--no-cache", action="store_true",
                   help="[historical] Bypass the local parquet cache")

    # --- engine / risk -------------------------------------------------------
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

    return p.parse_args()


def load_data(args: argparse.Namespace):
    if args.mode == "historical":
        print(f"\nLoading historical data: {args.symbol} {args.timeframe} "
              f"{args.start} → {args.end} from {args.exchange} …")
        df = get_data(
            "historical",
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            exchange=args.exchange,
            use_cache=not args.no_cache,
        )
        print(f"  Loaded {len(df):,} bars "
              f"({df.index[0]} → {df.index[-1]})")
    else:
        print(f"\nGenerating {args.bars} bars of synthetic data (seed={args.seed}) …")
        df = get_data(
            "synthetic",
            n_bars=args.bars,
            start_price=args.start_price,
            seed=args.seed,
        )
    return df


def main() -> None:
    args = parse_args()

    df = load_data(args)

    print("Computing signals …")
    df_signals = generate_signals(df)
    n_long = int((df_signals["signal"] == 1).sum())
    n_short = int((df_signals["signal"] == -1).sum())
    print(f"  Long signals : {n_long}")
    print(f"  Short signals: {n_short}")

    if n_long + n_short == 0:
        print("\n  ⚠  No signals generated. Strategy conditions may be too strict "
              "for this dataset — consider loosening RSI or sentiment thresholds.")

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
