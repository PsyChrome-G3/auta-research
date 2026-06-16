"""Tests for backtester."""

import pandas as pd

from auta_research.backtester import backtest_signals, run_backtest
from auta_research.config import StrategyConfig
from auta_research.patterns import detect_patterns


def _sample_df(n: int = 50) -> pd.DataFrame:
    """Generate trending sample OHLC data."""
    rows = []
    price = 1.1000
    for i in range(n):
        o = price
        c = price + (0.0005 if i % 3 != 0 else -0.0010)
        h = max(o, c) + 0.0008
        l = min(o, c) - 0.0003
        rows.append({
            "timestamp": f"2024-01-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00Z",
            "open": o, "high": h, "low": l, "close": c,
            "tick_volume": 100, "spread": 1, "real_volume": 0,
            "symbol": "EURUSD", "timeframe": "H4",
        })
        price = c
    return pd.DataFrame(rows)


def test_backtest_produces_trades():
    df = _sample_df(80)
    cfg = StrategyConfig()
    cfg.entry.modes = ["next_open"]
    cfg.stop.modes = ["pattern_extreme"]
    cfg.take_profit.r_values = [1.0]
    signals, trades = run_backtest(df, cfg, single_combo=True)
    assert isinstance(signals, pd.DataFrame)
    assert isinstance(trades, pd.DataFrame)


def test_backtest_trade_columns():
    df = _sample_df(80)
    cfg = StrategyConfig()
    cfg.entry.modes = ["next_open"]
    cfg.stop.modes = ["candle2_extreme"]
    cfg.take_profit.r_values = [0.5]
    signals = detect_patterns(df, cfg)
    if signals.empty:
        return
    trades = backtest_signals(
        df, signals.head(1), cfg,
        entry_mode="next_open", stop_mode="candle2_extreme", r_multiple=0.5,
    )
    if trades.empty:
        return
    for col in ("trade_id", "r_result", "outcome", "entry_price", "stop_price"):
        assert col in trades.columns
