"""
Technical indicators used by the Sentiment-Positioning strategy.

All functions accept a :class:`pandas.Series` or :class:`numpy.ndarray`
and return a :class:`pandas.Series` (or scalar boolean where noted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Wilder-smoothed Relative Strength Index.

    Parameters
    ----------
    close:
        Series of closing prices.
    period:
        Look-back window (default 14).

    Returns
    -------
    pd.Series
        RSI values in [0, 100]; the first ``period`` values are ``NaN``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Average True Range (ATR)
# ---------------------------------------------------------------------------

def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute the Average True Range.

    Parameters
    ----------
    high, low, close:
        OHLC price series.
    period:
        Smoothing window.

    Returns
    -------
    pd.Series
        ATR values; the first ``period`` values are ``NaN``.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volatility filter
# ---------------------------------------------------------------------------

def is_volatility_expanding(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int = 7,
    long_period: int = 28,
) -> pd.Series:
    """Return a boolean Series indicating whether volatility is expanding.

    Volatility is deemed *expanding* when the short-term ATR is greater than
    the long-term ATR, i.e. recent price swings are larger than the baseline.

    Parameters
    ----------
    short_period:
        ATR period for the fast (recent) volatility estimate.
    long_period:
        ATR period for the slow (baseline) volatility estimate.

    Returns
    -------
    pd.Series of bool
    """
    atr_short = compute_atr(high, low, close, short_period)
    atr_long = compute_atr(high, low, close, long_period)
    return atr_short > atr_long


# ---------------------------------------------------------------------------
# Momentum / Breakout confirmation
# ---------------------------------------------------------------------------

def is_breakout_up(high: pd.Series, period: int = 20) -> pd.Series:
    """Return True where the current high exceeds the prior ``period``-bar high.

    A *bullish breakout* occurs when price pushes above the recent trading
    range, signalling upside momentum.

    Parameters
    ----------
    period:
        Number of prior bars to look back (default 20).
    """
    prior_high = high.shift(1).rolling(period).max()
    return high > prior_high


def is_breakout_down(low: pd.Series, period: int = 20) -> pd.Series:
    """Return True where the current low falls below the prior ``period``-bar low.

    A *bearish breakout* occurs when price breaks below the recent trading
    range, signalling downside momentum.

    Parameters
    ----------
    period:
        Number of prior bars to look back (default 20).
    """
    prior_low = low.shift(1).rolling(period).min()
    return low < prior_low


# ---------------------------------------------------------------------------
# Price-reversal confirmation
# ---------------------------------------------------------------------------

def is_reversal_up(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Return True where the bar closes higher than it opens (bullish bar).

    In an oversold environment this indicates that buyers stepped in and
    price may be beginning to reverse upward.  This is the *reversal*
    half of the "reversal OR breakout" price-confirmation filter.

    Parameters
    ----------
    close:
        Series of closing prices.
    open_:
        Series of opening prices.
    """
    return close > open_


def is_reversal_down(close: pd.Series, open_: pd.Series) -> pd.Series:
    """Return True where the bar closes lower than it opens (bearish bar).

    In an overbought environment this indicates that sellers are gaining
    control and price may be beginning to reverse downward.  This is the
    *reversal* half of the "reversal OR breakout" price-confirmation filter.

    Parameters
    ----------
    close:
        Series of closing prices.
    open_:
        Series of opening prices.
    """
    return close < open_
