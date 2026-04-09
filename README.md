# ai-crypto-algo-trader

Crypto bot trader

---

## Sentiment-Positioning Hybrid Strategy — Backtest

A rule-based Python backtest that implements the **Sentiment-Positioning Hybrid** crypto trading strategy.

### Strategy Logic

The strategy looks for situations where **crowd positioning is extreme, price confirms direction, and volatility supports movement**.

| Component       | Purpose                                    |
| --------------- | ------------------------------------------ |
| RSI             | Detects exhaustion (oversold / overbought) |
| Funding Rate    | Identifies crowded long or short positions |
| Sentiment Score | Captures retail crowd psychology           |
| Reversal / Breakout | Confirms directional price movement    |
| Volatility Filter | Ensures active market conditions         |

#### Long entry (all must be true)
1. RSI < 30  
2. Funding rate < −0.01 (shorts crowded)  
3. Bullish reversal or price breakout up  
4. Volatility expanding  
5. Sentiment < −0.3 (crowd fear)

#### Short entry (all must be true)
1. RSI > 70  
2. Funding rate > +0.01 (longs crowded)  
3. Bearish reversal or price breakdown  
4. Volatility expanding  
5. Sentiment > +0.3 (crowd greed)

#### Exit
* Stop loss at 2 % from entry  
* Take profit at 1.5× risk (3 %)  
* OR RSI returns to the 40–60 neutral zone

---

### Project Structure

```
backtest/
    __init__.py          package exports
    data.py              synthetic OHLCV + funding-rate + sentiment generator
    indicators.py        RSI, ATR, volatility, breakout, reversal indicators
    signals.py           combines indicators into LONG / SHORT / FLAT signals
    engine.py            bar-by-bar backtest engine with risk management
run_backtest.py          CLI entry point
tests/
    test_backtest.py     45 unit / integration tests
requirements.txt
```

---

### Installation

```bash
pip install -r requirements.txt
```

### Run the backtest

```bash
# Default run (1 000 hourly bars, $10 000 starting capital)
python run_backtest.py

# Larger simulation
python run_backtest.py --bars 5000 --seed 123

# Custom risk parameters
python run_backtest.py --bars 3000 --capital 50000 --risk 0.005 --stop-loss 0.03 --tp-ratio 2.0

# All options
python run_backtest.py --help
```

### Run the tests

```bash
python -m pytest tests/ -v
```
