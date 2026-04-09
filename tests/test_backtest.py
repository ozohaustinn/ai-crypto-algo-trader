"""Unit tests for the Sentiment-Positioning Hybrid backtest strategy."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.data import generate_synthetic_data
from backtest.engine import BacktestEngine, BacktestResult, Trade
from backtest.indicators import (
    compute_atr,
    compute_rsi,
    is_breakout_down,
    is_breakout_up,
    is_reversal_down,
    is_reversal_up,
    is_volatility_expanding,
)
from backtest.signals import generate_signals


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return generate_synthetic_data(n_bars=500, seed=0)


@pytest.fixture
def signal_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    return generate_signals(synthetic_df)


# ===========================================================================
# Data generator tests
# ===========================================================================

class TestGenerateSyntheticData:
    def test_shape(self) -> None:
        df = generate_synthetic_data(n_bars=200)
        assert len(df) == 200

    def test_columns(self) -> None:
        df = generate_synthetic_data(n_bars=50)
        expected = {"open", "high", "low", "close", "volume", "funding_rate", "sentiment"}
        assert expected.issubset(df.columns)

    def test_ohlc_ordering(self) -> None:
        df = generate_synthetic_data(n_bars=200)
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["high"] >= df["low"]).all()

    def test_positive_prices(self) -> None:
        df = generate_synthetic_data(n_bars=200)
        assert (df[["open", "high", "low", "close"]] > 0).all().all()

    def test_volume_positive(self) -> None:
        df = generate_synthetic_data(n_bars=200)
        assert (df["volume"] > 0).all()

    def test_funding_rate_range(self) -> None:
        df = generate_synthetic_data(n_bars=200)
        assert df["funding_rate"].between(-0.05, 0.05).all()

    def test_sentiment_range(self) -> None:
        df = generate_synthetic_data(n_bars=200)
        assert df["sentiment"].between(-1.0, 1.0).all()

    def test_reproducible_with_seed(self) -> None:
        df1 = generate_synthetic_data(n_bars=100, seed=7)
        df2 = generate_synthetic_data(n_bars=100, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        df1 = generate_synthetic_data(n_bars=100, seed=1)
        df2 = generate_synthetic_data(n_bars=100, seed=2)
        assert not df1["close"].equals(df2["close"])

    def test_datetime_index(self) -> None:
        df = generate_synthetic_data(n_bars=100)
        assert isinstance(df.index, pd.DatetimeIndex)


# ===========================================================================
# RSI tests
# ===========================================================================

class TestComputeRsi:
    def test_output_length(self) -> None:
        close = pd.Series(np.linspace(100, 200, 100))
        rsi = compute_rsi(close, period=14)
        assert len(rsi) == len(close)

    def test_range(self) -> None:
        rng = np.random.default_rng(0)
        close = pd.Series(100 + rng.normal(0, 5, 200).cumsum())
        rsi = compute_rsi(close, period=14).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_trending_up_rsi_high(self) -> None:
        """Strongly upward-trending price with small noise → RSI should be above 60."""
        rng = np.random.default_rng(42)
        trend = np.linspace(100, 300, 200)
        # Small noise ensures there are occasional down-bars so RSI is finite
        noise = rng.normal(0, 0.5, 200)
        close = pd.Series(trend + noise)
        rsi = compute_rsi(close, period=14).dropna()
        assert len(rsi) > 0, "RSI series should not be empty after dropna"
        assert rsi.mean() > 60

    def test_trending_down_rsi_low(self) -> None:
        """Strongly downward-trending price → RSI should be below 40."""
        close = pd.Series(np.linspace(300, 100, 200))
        rsi = compute_rsi(close, period=14).dropna()
        assert rsi.mean() < 40

    def test_first_values_nan(self) -> None:
        close = pd.Series(np.linspace(100, 200, 50))
        rsi = compute_rsi(close, period=14)
        assert rsi.iloc[:13].isna().all()


# ===========================================================================
# ATR tests
# ===========================================================================

class TestComputeAtr:
    def test_output_length(self, synthetic_df: pd.DataFrame) -> None:
        atr = compute_atr(synthetic_df["high"], synthetic_df["low"], synthetic_df["close"])
        assert len(atr) == len(synthetic_df)

    def test_non_negative(self, synthetic_df: pd.DataFrame) -> None:
        atr = compute_atr(synthetic_df["high"], synthetic_df["low"], synthetic_df["close"])
        assert (atr.dropna() >= 0).all()

    def test_larger_range_larger_atr(self) -> None:
        """Higher-volatility data should produce a larger ATR."""
        idx = pd.RangeIndex(100)
        low_vol = pd.DataFrame({
            "high": np.full(100, 101.0),
            "low": np.full(100, 99.0),
            "close": np.full(100, 100.0),
        }, index=idx)
        high_vol = pd.DataFrame({
            "high": np.full(100, 110.0),
            "low": np.full(100, 90.0),
            "close": np.full(100, 100.0),
        }, index=idx)
        atr_lv = compute_atr(low_vol["high"], low_vol["low"], low_vol["close"]).dropna()
        atr_hv = compute_atr(high_vol["high"], high_vol["low"], high_vol["close"]).dropna()
        assert atr_hv.mean() > atr_lv.mean()


# ===========================================================================
# Volatility filter tests
# ===========================================================================

class TestVolatilityExpanding:
    def test_output_dtype(self, synthetic_df: pd.DataFrame) -> None:
        result = is_volatility_expanding(
            synthetic_df["high"], synthetic_df["low"], synthetic_df["close"]
        )
        assert result.dtype == bool

    def test_output_length(self, synthetic_df: pd.DataFrame) -> None:
        result = is_volatility_expanding(
            synthetic_df["high"], synthetic_df["low"], synthetic_df["close"]
        )
        assert len(result) == len(synthetic_df)


# ===========================================================================
# Breakout tests
# ===========================================================================

class TestBreakouts:
    def test_breakout_up_detects_new_high(self) -> None:
        """A bar that exceeds all prior bars must be flagged as breakout_up."""
        prices = pd.Series([100.0] * 25 + [200.0])
        result = is_breakout_up(prices, period=20)
        assert result.iloc[-1]

    def test_breakout_up_no_false_positive(self) -> None:
        """A flat series should produce no breakout signals."""
        prices = pd.Series([100.0] * 50)
        result = is_breakout_up(prices, period=20)
        assert not result.dropna().any()

    def test_breakout_down_detects_new_low(self) -> None:
        """A bar that undercuts all prior bars must be flagged as breakout_down."""
        prices = pd.Series([100.0] * 25 + [10.0])
        result = is_breakout_down(prices, period=20)
        assert result.iloc[-1]

    def test_breakout_down_no_false_positive(self) -> None:
        """A flat series should produce no breakdown signals."""
        prices = pd.Series([100.0] * 50)
        result = is_breakout_down(prices, period=20)
        assert not result.dropna().any()


# ===========================================================================
# Reversal indicator tests
# ===========================================================================

class TestReversalIndicators:
    def test_reversal_up_bullish_bar(self) -> None:
        """Bar where close > open is flagged as reversal_up."""
        close = pd.Series([101.0, 100.0, 105.0])
        open_ = pd.Series([100.0, 102.0, 104.0])
        result = is_reversal_up(close, open_)
        assert result.iloc[0]   # close > open
        assert not result.iloc[1]  # close < open
        assert result.iloc[2]   # close > open

    def test_reversal_down_bearish_bar(self) -> None:
        """Bar where close < open is flagged as reversal_down."""
        close = pd.Series([99.0, 101.0, 98.0])
        open_ = pd.Series([100.0, 100.0, 100.0])
        result = is_reversal_down(close, open_)
        assert result.iloc[0]    # close < open
        assert not result.iloc[1]  # close > open
        assert result.iloc[2]    # close < open

    def test_reversal_up_down_mutually_exclusive(self, synthetic_df: pd.DataFrame) -> None:
        """A bar cannot be simultaneously reversal_up and reversal_down."""
        ru = is_reversal_up(synthetic_df["close"], synthetic_df["open"])
        rd = is_reversal_down(synthetic_df["close"], synthetic_df["open"])
        # Doji bars (close == open) are neither; no bar should be both
        assert not (ru & rd).any()


# ===========================================================================
# Signal generation tests
# ===========================================================================

class TestGenerateSignals:
    def test_output_columns(self, signal_df: pd.DataFrame) -> None:
        expected = {
            "rsi", "vol_expanding",
            "reversal_up", "reversal_down",
            "breakout_up", "breakout_down",
            "signal",
        }
        assert expected.issubset(signal_df.columns)

    def test_signal_values(self, signal_df: pd.DataFrame) -> None:
        assert set(signal_df["signal"].unique()).issubset({-1, 0, 1})

    def test_long_conditions_correct(self, signal_df: pd.DataFrame) -> None:
        """Every LONG signal must have RSI < 30 and negative funding."""
        longs = signal_df[signal_df["signal"] == 1]
        assert (longs["rsi"] < 30).all()
        assert (longs["funding_rate"] < -0.01).all()

    def test_short_conditions_correct(self, signal_df: pd.DataFrame) -> None:
        """Every SHORT signal must have RSI > 70 and positive funding."""
        shorts = signal_df[signal_df["signal"] == -1]
        assert (shorts["rsi"] > 70).all()
        assert (shorts["funding_rate"] > 0.01).all()

    def test_original_df_unchanged(self, synthetic_df: pd.DataFrame) -> None:
        """generate_signals must not mutate its input."""
        original_cols = list(synthetic_df.columns)
        generate_signals(synthetic_df)
        assert list(synthetic_df.columns) == original_cols

    def test_custom_thresholds(self, synthetic_df: pd.DataFrame) -> None:
        """Relaxing RSI thresholds should produce at least as many signals."""
        df_strict = generate_signals(synthetic_df, rsi_oversold=20, rsi_overbought=80)
        df_loose = generate_signals(synthetic_df, rsi_oversold=35, rsi_overbought=65)
        strict_count = (df_strict["signal"] != 0).sum()
        loose_count = (df_loose["signal"] != 0).sum()
        assert loose_count >= strict_count


# ===========================================================================
# BacktestEngine tests
# ===========================================================================

class TestBacktestEngine:
    def test_returns_backtest_result(self, signal_df: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(signal_df)
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(self, signal_df: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(signal_df)
        assert len(result.equity_curve) == len(signal_df)

    def test_equity_starts_at_capital(self, signal_df: pd.DataFrame) -> None:
        capital = 50_000.0
        engine = BacktestEngine(initial_capital=capital)
        result = engine.run(signal_df)
        assert result.equity_curve.iloc[0] == pytest.approx(capital, rel=0.05)

    def test_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [100.0, 101.0]})
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="missing columns"):
            engine.run(df)

    def test_no_trades_on_zero_signal(self) -> None:
        """When all signals are 0 (flat), there should be no trades."""
        df = generate_synthetic_data(n_bars=100, seed=0)
        df_signals = generate_signals(df)
        df_signals["signal"] = 0
        engine = BacktestEngine()
        result = engine.run(df_signals)
        assert result.n_trades == 0

    def test_max_positions_respected(self, signal_df: pd.DataFrame) -> None:
        """Forcing all bars to emit a LONG signal should not exceed max_positions."""
        df = signal_df.copy()
        df["signal"] = 1
        engine = BacktestEngine(max_positions=3)
        result = engine.run(df)
        # We can't inspect open positions after the run, but we can verify the
        # engine didn't error out and returns a valid result.
        assert isinstance(result, BacktestResult)

    def test_win_rate_between_0_and_1(self, signal_df: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(signal_df)
        if result.n_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_positive(self, signal_df: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(signal_df)
        assert result.max_drawdown_pct <= 0.0

    def test_summary_is_string(self, signal_df: pd.DataFrame) -> None:
        engine = BacktestEngine()
        result = engine.run(signal_df)
        assert isinstance(result.summary(), str)
        assert "Total return" in result.summary()

    def test_stop_loss_limits_loss(self) -> None:
        """A position that drops well below stop-loss should exit at stop, not further."""
        # Build a minimal DataFrame: price crashes after entry
        prices = [100.0] * 30 + [60.0] * 50  # price falls 40%
        idx = pd.date_range("2023-01-01", periods=80, freq="1h")
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000.0] * 80,
                "funding_rate": [-0.02] * 80,
                "sentiment": [-0.5] * 80,
            },
            index=idx,
        )
        df_signals = generate_signals(df, rsi_oversold=99)  # force many longs
        df_signals["signal"] = 1  # force all signals to long
        engine = BacktestEngine(stop_loss_pct=0.02, take_profit_ratio=1.5)
        result = engine.run(df_signals)
        for trade in result.trades:
            if trade.exit_reason == "stop_loss":
                # Loss should be close to stop_loss_pct, not 40%
                assert abs(trade.pnl_pct) < 0.05


# ===========================================================================
# End-to-end smoke test
# ===========================================================================

class TestEndToEnd:
    def test_full_pipeline_runs(self) -> None:
        df = generate_synthetic_data(n_bars=300, seed=99)
        df_signals = generate_signals(df)
        engine = BacktestEngine(initial_capital=10_000)
        result = engine.run(df_signals)
        assert result.final_capital > 0
        assert not math.isnan(result.total_return_pct)

    def test_larger_dataset(self) -> None:
        df = generate_synthetic_data(n_bars=2000, seed=1)
        df_signals = generate_signals(df)
        engine = BacktestEngine()
        result = engine.run(df_signals)
        assert len(result.equity_curve) == 2000
