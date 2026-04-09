"""Sentiment-Positioning Hybrid Crypto Backtest Package."""

from backtest.data import generate_synthetic_data
from backtest.indicators import (
    compute_rsi,
    compute_atr,
    is_volatility_expanding,
    is_breakout_up,
    is_breakout_down,
    is_reversal_up,
    is_reversal_down,
)
from backtest.signals import generate_signals
from backtest.engine import BacktestEngine, BacktestResult

__all__ = [
    "generate_synthetic_data",
    "compute_rsi",
    "compute_atr",
    "is_volatility_expanding",
    "is_breakout_up",
    "is_breakout_down",
    "is_reversal_up",
    "is_reversal_down",
    "generate_signals",
    "BacktestEngine",
    "BacktestResult",
]
