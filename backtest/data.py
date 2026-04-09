"""
Synthetic data generator for backtesting.

Produces a pandas DataFrame with columns:
    timestamp   – datetime index
    open        – open price
    high        – high price
    low         – low price
    close       – close price
    volume      – trading volume
    funding_rate – perpetual-futures funding rate
    sentiment   – crowd-sentiment score in [-1, 1]
                  (+1 = extremely bullish, -1 = extremely bearish)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_bars: int = 1000,
    start_price: float = 30_000.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic OHLCV + market-microstructure data.

    Parameters
    ----------
    n_bars:
        Number of hourly bars to generate.
    start_price:
        Starting close price.
    seed:
        Random seed for reproducibility.  Pass ``None`` for non-deterministic
        results.
    """
    rng = np.random.default_rng(seed)

    # --- Price series via geometric Brownian motion -------------------------
    drift = 0.0001          # small positive drift per bar
    vol_per_bar = 0.005     # 0.5 % hourly volatility
    log_returns = rng.normal(drift, vol_per_bar, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_returns))

    # OHLC from close
    bar_vol = rng.uniform(0.001, 0.008, size=n_bars)
    high = close * (1 + bar_vol)
    low = close * (1 - bar_vol)
    open_ = np.roll(close, 1)
    open_[0] = start_price

    volume = rng.uniform(500, 5_000, size=n_bars) * (close / start_price)

    # --- Funding rate  -------------------------------------------------------
    # Correlated loosely with recent price momentum so that extended rallies
    # produce positive funding (overcrowded longs) and sell-offs produce
    # negative funding (overcrowded shorts).
    momentum_proxy = pd.Series(close).pct_change(12).fillna(0).to_numpy()
    noise = rng.normal(0, 0.002, size=n_bars)
    funding_rate = np.clip(momentum_proxy * 0.5 + noise, -0.05, 0.05)

    # --- Sentiment score  ----------------------------------------------------
    # Similarly correlated with recent price change plus independent noise to
    # simulate social-media crowd psychology.
    sentiment_noise = rng.normal(0, 0.2, size=n_bars)
    sentiment = np.clip(momentum_proxy * 10 + sentiment_noise, -1.0, 1.0)

    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="1h")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
            "sentiment": sentiment,
        },
        index=timestamps,
    )
