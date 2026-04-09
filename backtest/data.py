"""
Data loaders for backtesting.

Two modes:
    - synthetic:  GBM price + correlated funding/sentiment (for unit tests
                  and engine sanity checks)
    - historical: real OHLCV from a crypto exchange via ccxt, real funding
                  rates from Binance, and the alternative.me Fear & Greed
                  index as a sentiment proxy.

Both return a DataFrame with the same schema:
    index        – tz-naive UTC datetime
    open, high, low, close, volume
    funding_rate – perpetual-futures funding rate (per interval, not annualized)
    sentiment    – crowd-sentiment score in [-1, 1]
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic (unchanged — keep your tests green)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_bars: int = 1000,
    start_price: float = 30_000.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Return a DataFrame of synthetic OHLCV + funding + sentiment."""
    rng = np.random.default_rng(seed)

    drift = 0.0001
    vol_per_bar = 0.005
    log_returns = rng.normal(drift, vol_per_bar, size=n_bars)
    close = start_price * np.exp(np.cumsum(log_returns))

    bar_vol = rng.uniform(0.001, 0.008, size=n_bars)
    high = close * (1 + bar_vol)
    low = close * (1 - bar_vol)
    open_ = np.roll(close, 1)
    open_[0] = start_price

    volume = rng.uniform(500, 5_000, size=n_bars) * (close / start_price)

    momentum_proxy = pd.Series(close).pct_change(12).fillna(0).to_numpy()
    noise = rng.normal(0, 0.002, size=n_bars)
    funding_rate = np.clip(momentum_proxy * 0.5 + noise, -0.05, 0.05)

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


# ---------------------------------------------------------------------------
# Historical loaders
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _fetch_ohlcv_ccxt(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    exchange_name: str = "binance",
) -> pd.DataFrame:
    """Page through ccxt fetch_ohlcv until we cover [since_ms, until_ms]."""
    import ccxt  # local import so synthetic mode has no hard dependency

    exchange = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    all_rows: list[list] = []
    cursor = since_ms
    limit = 1000

    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                     since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    return df[df.index < pd.to_datetime(until_ms, unit="ms")]


def _fetch_binance_funding(
    symbol: str,
    since_ms: int,
    until_ms: int,
) -> pd.Series:
    """Pull perpetual funding rates from Binance Futures REST API.

    Binance posts funding every 8 hours. Returns a Series indexed by timestamp.
    """
    import requests

    # Binance futures uses no slash: BTCUSDT, not BTC/USDT
    fut_symbol = symbol.replace("/", "").replace(":USDT", "")
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows: list[dict] = []
    cursor = since_ms

    while cursor < until_ms:
        resp = requests.get(url, params={
            "symbol": fut_symbol,
            "startTime": cursor,
            "endTime": until_ms,
            "limit": 1000,
        }, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)

    if not rows:
        return pd.Series(dtype=float, name="funding_rate")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_localize(None)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df.set_index("timestamp")["funding_rate"].sort_index()


def _fetch_fear_greed(since_ms: int, until_ms: int) -> pd.Series:
    """Daily Crypto Fear & Greed Index from alternative.me, rescaled to [-1, 1]."""
    import requests

    days_needed = int((until_ms - since_ms) / (1000 * 86400)) + 30
    resp = requests.get("https://api.alternative.me/fng/",
                        params={"limit": days_needed, "format": "json"},
                        timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.tz_localize(None)
    df["value"] = df["value"].astype(int)
    # Rescale 0–100 → -1..+1 (50 = neutral)
    df["sentiment"] = (df["value"] - 50) / 50.0
    return df.set_index("timestamp")["sentiment"].sort_index()


def load_historical_data(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start: str = "2023-01-01",
    end: str = "2025-01-01",
    exchange: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load real OHLCV + funding + sentiment, aligned to one DataFrame.

    Funding rates are forward-filled from their 8-hour cadence onto every bar.
    Sentiment is forward-filled from its daily cadence.
    """
    cache_file = CACHE_DIR / f"{symbol.replace('/', '')}_{timeframe}_{start}_{end}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    until_ms = int(pd.Timestamp(end).timestamp() * 1000)

    ohlcv = _fetch_ohlcv_ccxt(symbol, timeframe, since_ms, until_ms, exchange)
    if ohlcv.empty:
        raise RuntimeError(f"No OHLCV returned for {symbol} {timeframe} {start}→{end}")

    funding = _fetch_binance_funding(symbol, since_ms, until_ms)
    sentiment = _fetch_fear_greed(since_ms, until_ms)

    df = ohlcv.copy()
    df["funding_rate"] = funding.reindex(df.index, method="ffill").fillna(0.0)
    df["sentiment"] = sentiment.reindex(df.index, method="ffill").fillna(0.0)

    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    if use_cache:
        df.to_parquet(cache_file)

    return df


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def get_data(mode: str = "synthetic", **kwargs) -> pd.DataFrame:
    """Single entry point used by run_backtest.py.

    Examples
    --------
    >>> get_data("synthetic", n_bars=2000, seed=7)
    >>> get_data("historical", symbol="BTC/USDT", start="2022-01-01", end="2025-01-01")
    """
    if mode == "synthetic":
        return generate_synthetic_data(**kwargs)
    if mode == "historical":
        return load_historical_data(**kwargs)
    raise ValueError(f"Unknown data mode: {mode!r}")
