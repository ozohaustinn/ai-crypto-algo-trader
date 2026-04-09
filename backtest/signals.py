"""
Signal generation for the Sentiment-Positioning Hybrid strategy.

A signal is one of:
    1  → LONG
   -1  → SHORT
    0  → FLAT (no trade)

Entry Conditions
----------------
LONG  (all must be True):
    1. RSI < rsi_oversold             (exhaustion)
    2. funding_rate < -funding_thresh  (shorts crowded)
    3. reversal_up OR breakout_up      (price confirmation)
    4. volatility_expanding            (timing filter)
    5. sentiment < -sentiment_thresh   (crowd fear)

SHORT (all must be True):
    1. RSI > rsi_overbought            (exhaustion)
    2. funding_rate > +funding_thresh  (longs crowded)
    3. reversal_down OR breakout_down  (price confirmation)
    4. volatility_expanding            (timing filter)
    5. sentiment > +sentiment_thresh   (crowd greed)

Price confirmation uses a two-tier approach:
* *Reversal*:  the current bar's close is above/below its open (bullish /
  bearish candle), capturing the very first sign that momentum is turning.
* *Breakout*:  price exceeds the prior N-bar high/low, confirming a sustained
  directional move.
Either condition is sufficient; together they span both early reversals and
momentum continuations.
"""

from __future__ import annotations

import pandas as pd

from backtest.indicators import (
    compute_rsi,
    is_breakout_down,
    is_breakout_up,
    is_reversal_down,
    is_reversal_up,
    is_volatility_expanding,
)


def generate_signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    funding_thresh: float = 0.01,
    sentiment_thresh: float = 0.3,
    breakout_period: int = 20,
    vol_short_period: int = 7,
    vol_long_period: int = 28,
) -> pd.DataFrame:
    """Compute all indicator columns and produce a ``signal`` column.

    Parameters
    ----------
    df:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``,
        ``funding_rate``, and ``sentiment``.
    rsi_period:
        RSI look-back window.
    rsi_oversold / rsi_overbought:
        RSI thresholds for long/short entry.
    funding_thresh:
        Absolute funding-rate threshold separating crowded from neutral.
    sentiment_thresh:
        Absolute sentiment threshold separating extreme from neutral.
    breakout_period:
        Number of prior bars used for high/low breakout detection.
    vol_short_period / vol_long_period:
        ATR windows for the volatility-expansion filter.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with additional columns:

        ``rsi``, ``vol_expanding``, ``reversal_up``, ``reversal_down``,
        ``breakout_up``, ``breakout_down``, ``signal`` (1 / -1 / 0).
    """
    out = df.copy()

    # --- Indicator columns --------------------------------------------------
    out["rsi"] = compute_rsi(out["close"], rsi_period)
    out["vol_expanding"] = is_volatility_expanding(
        out["high"], out["low"], out["close"],
        vol_short_period, vol_long_period,
    )
    out["reversal_up"] = is_reversal_up(out["close"], out["open"])
    out["reversal_down"] = is_reversal_down(out["close"], out["open"])
    out["breakout_up"] = is_breakout_up(out["high"], breakout_period)
    out["breakout_down"] = is_breakout_down(out["low"], breakout_period)

    # Price confirmation: reversal (first bar of turn) OR sustained breakout
    price_confirm_up = out["reversal_up"] | out["breakout_up"]
    price_confirm_down = out["reversal_down"] | out["breakout_down"]

    # --- Long conditions ----------------------------------------------------
    long_cond = (
        (out["rsi"] < rsi_oversold)
        & (out["funding_rate"] < -funding_thresh)
        & price_confirm_up
        & out["vol_expanding"]
        & (out["sentiment"] < -sentiment_thresh)
    )

    # --- Short conditions ---------------------------------------------------
    short_cond = (
        (out["rsi"] > rsi_overbought)
        & (out["funding_rate"] > funding_thresh)
        & price_confirm_down
        & out["vol_expanding"]
        & (out["sentiment"] > sentiment_thresh)
    )

    out["signal"] = 0
    out.loc[long_cond, "signal"] = 1
    out.loc[short_cond, "signal"] = -1

    return out
