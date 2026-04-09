"""
Backtest engine for the Sentiment-Positioning Hybrid strategy.

Features
--------
* Event-driven bar-by-bar simulation
* 1 % risk-per-trade position sizing
* Fixed-percentage stop loss (default 2 %)
* Take-profit at 1.5× risk (default 3 %)  **or** RSI returning to 40–60
* Maximum concurrent open positions (default 5)
* Full equity-curve tracking, per-trade log, and summary statistics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class Trade:
    """Record of a single completed trade."""

    entry_bar: int
    exit_bar: int
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    size: float             # number of units (base asset)
    pnl: float              # gross P&L in quote currency
    pnl_pct: float          # P&L as a fraction of entry notional
    exit_reason: str        # "stop_loss" | "take_profit" | "rsi_neutral" | "end_of_data"


@dataclass
class BacktestResult:
    """Aggregated backtest statistics and equity curve."""

    trades: list[Trade]
    equity_curve: pd.Series
    initial_capital: float

    # --- Computed on demand -------------------------------------------------

    @property
    def final_capital(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return_pct(self) -> float:
        return (self.final_capital / self.initial_capital - 1) * 100

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def avg_pnl_pct(self) -> float:
        if not self.trades:
            return float("nan")
        return float(np.mean([t.pnl_pct for t in self.trades]))

    @property
    def max_drawdown_pct(self) -> float:
        eq = self.equity_curve
        roll_max = eq.cummax()
        drawdown = (eq - roll_max) / roll_max
        return float(drawdown.min()) * 100

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio assuming hourly bars (8 760 bars/year)."""
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return float("nan")
        return float(daily_ret.mean() / daily_ret.std() * math.sqrt(8_760))

    def summary(self) -> str:
        lines = [
            "=" * 50,
            " Sentiment-Positioning Strategy — Backtest Report",
            "=" * 50,
            f"  Initial capital     : {self.initial_capital:>12,.2f}",
            f"  Final capital       : {self.final_capital:>12,.2f}",
            f"  Total return        : {self.total_return_pct:>+11.2f} %",
            f"  Max drawdown        : {self.max_drawdown_pct:>11.2f} %",
            f"  Sharpe ratio        : {self.sharpe_ratio:>12.3f}",
            f"  Number of trades    : {self.n_trades:>12}",
            f"  Win rate            : {self.win_rate * 100:>11.1f} %"
            if self.n_trades else "  Win rate            :          N/A",
            f"  Avg trade return    : {self.avg_pnl_pct * 100:>+11.2f} %"
            if self.n_trades else "  Avg trade return    :          N/A",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class _OpenPosition:
    """Internal representation of a live position."""

    entry_bar: int
    direction: int
    entry_price: float
    size: float
    stop_price: float
    take_profit_price: float


class BacktestEngine:
    """Bar-by-bar backtest engine.

    Parameters
    ----------
    initial_capital:
        Starting equity in quote currency (e.g. USDT).
    risk_per_trade:
        Fraction of equity to risk per trade (default 0.01 = 1 %).
    stop_loss_pct:
        Stop-loss distance as a fraction of entry price (default 0.02 = 2 %).
    take_profit_ratio:
        Risk-reward ratio for take-profit (default 1.5 ×, i.e. TP = 3 % when
        SL = 2 %).
    rsi_neutral_low / rsi_neutral_high:
        RSI band for the *RSI-returns-to-neutral* exit (default 40–60).
    max_positions:
        Maximum number of concurrent open positions (default 5).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        risk_per_trade: float = 0.01,
        stop_loss_pct: float = 0.02,
        take_profit_ratio: float = 1.5,
        rsi_neutral_low: float = 40.0,
        rsi_neutral_high: float = 60.0,
        max_positions: int = 5,
    ) -> None:
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_ratio = take_profit_ratio
        self.rsi_neutral_low = rsi_neutral_low
        self.rsi_neutral_high = rsi_neutral_high
        self.max_positions = max_positions

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Execute the backtest on a signal-enriched DataFrame.

        Parameters
        ----------
        df:
            Must contain columns: ``close``, ``rsi``, ``signal``.
            Typically the output of :func:`backtest.signals.generate_signals`.

        Returns
        -------
        BacktestResult
        """
        required = {"close", "rsi", "signal"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        equity = self.initial_capital
        equity_curve: list[float] = []
        open_positions: list[_OpenPosition] = []
        completed_trades: list[Trade] = []

        close_arr = df["close"].to_numpy()
        rsi_arr = df["rsi"].to_numpy()
        signal_arr = df["signal"].to_numpy()
        n = len(df)

        for i in range(n):
            price = close_arr[i]
            rsi_val = rsi_arr[i]
            sig = signal_arr[i]

            # ---- Check exits for open positions ----------------------------
            still_open: list[_OpenPosition] = []
            for pos in open_positions:
                exit_price, exit_reason = self._check_exit(
                    pos, price, rsi_val, i == n - 1
                )
                if exit_reason is not None:
                    pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
                    pnl_pct = pos.direction * (exit_price / pos.entry_price - 1)
                    equity += pnl
                    completed_trades.append(
                        Trade(
                            entry_bar=pos.entry_bar,
                            exit_bar=i,
                            direction=pos.direction,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            size=pos.size,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            exit_reason=exit_reason,
                        )
                    )
                else:
                    still_open.append(pos)

            open_positions = still_open

            # ---- Check entries  --------------------------------------------
            if sig != 0 and len(open_positions) < self.max_positions:
                pos = self._open_position(sig, price, equity, i)
                if pos is not None:
                    open_positions.append(pos)

            equity_curve.append(equity)

        # Close any remaining positions at last bar
        # (already handled inside the loop via end_of_data flag)

        ts_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.RangeIndex(n)
        return BacktestResult(
            trades=completed_trades,
            equity_curve=pd.Series(equity_curve, index=ts_index),
            initial_capital=self.initial_capital,
        )

    # ------------------------------------------------------------------
    def _open_position(
        self,
        direction: int,
        entry_price: float,
        equity: float,
        bar: int,
    ) -> _OpenPosition | None:
        """Size and create a new position; returns None if sizing fails."""
        risk_amount = equity * self.risk_per_trade
        sl_distance = entry_price * self.stop_loss_pct

        if sl_distance <= 0:
            return None

        size = risk_amount / sl_distance

        if direction == 1:   # long
            stop_price = entry_price * (1 - self.stop_loss_pct)
            tp_price = entry_price * (1 + self.stop_loss_pct * self.take_profit_ratio)
        else:                # short
            stop_price = entry_price * (1 + self.stop_loss_pct)
            tp_price = entry_price * (1 - self.stop_loss_pct * self.take_profit_ratio)

        return _OpenPosition(
            entry_bar=bar,
            direction=direction,
            entry_price=entry_price,
            size=size,
            stop_price=stop_price,
            take_profit_price=tp_price,
        )

    # ------------------------------------------------------------------
    def _check_exit(
        self,
        pos: _OpenPosition,
        price: float,
        rsi: float,
        is_last_bar: bool,
    ) -> tuple[float, str | None]:
        """Return (exit_price, reason) or (price, None) if no exit."""
        if is_last_bar:
            return price, "end_of_data"

        if pos.direction == 1:  # long
            if price <= pos.stop_price:
                return pos.stop_price, "stop_loss"
            if price >= pos.take_profit_price:
                return pos.take_profit_price, "take_profit"
        else:                   # short
            if price >= pos.stop_price:
                return pos.stop_price, "stop_loss"
            if price <= pos.take_profit_price:
                return pos.take_profit_price, "take_profit"

        # RSI returning to neutral
        if not math.isnan(rsi) and self.rsi_neutral_low <= rsi <= self.rsi_neutral_high:
            return price, "rsi_neutral"

        return price, None
