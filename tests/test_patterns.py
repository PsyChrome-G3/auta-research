"""Tests for pattern detection."""

import pandas as pd

from auta_research.config import StrategyConfig
from auta_research.patterns import detect_patterns


def _make_sell_pattern_df() -> pd.DataFrame:
    """Build minimal OHLC data with a sell rejection pattern."""
    rows = []
    for i in range(30):
        rows.append({
            "timestamp": f"2024-01-{i+1:02d}T12:00:00Z",
            "open": 1.10,
            "high": 1.11,
            "low": 1.09,
            "close": 1.105,
            "tick_volume": 100,
            "spread": 1,
            "real_volume": 0,
            "symbol": "EURUSD",
            "timeframe": "H4",
        })
    # Candle 1: bullish with large upper wick
    rows[-2] = {
        "timestamp": "2024-01-29T12:00:00Z",
        "open": 1.1000, "high": 1.1100, "low": 1.0990, "close": 1.1010,
        "tick_volume": 100, "spread": 1, "real_volume": 0,
        "symbol": "EURUSD", "timeframe": "H4",
    }
    # Candle 2: bearish with upper wick, larger body
    rows[-1] = {
        "timestamp": "2024-01-30T12:00:00Z",
        "open": 1.1010, "high": 1.1030, "low": 1.0940, "close": 1.0950,
        "tick_volume": 100, "spread": 1, "real_volume": 0,
        "symbol": "EURUSD", "timeframe": "H4",
    }
    return pd.DataFrame(rows)


def test_detect_sell_pattern():
    df = _make_sell_pattern_df()
    cfg = StrategyConfig()
    signals = detect_patterns(df, cfg)
    assert len(signals) >= 1
    assert "sell" in signals["direction"].values


def test_detect_empty_on_flat_data():
    df = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z"] * 5,
        "open": [1.0] * 5,
        "high": [1.0] * 5,
        "low": [1.0] * 5,
        "close": [1.0] * 5,
        "tick_volume": [0] * 5,
        "spread": [0] * 5,
        "real_volume": [0] * 5,
        "symbol": ["EURUSD"] * 5,
        "timeframe": ["H4"] * 5,
    })
    cfg = StrategyConfig()
    signals = detect_patterns(df, cfg)
    assert len(signals) == 0


def test_butt_buddy_rejects_open_gap():
    df = _make_sell_pattern_df()
    # Gap between C1 close and C2 open
    df.iloc[-1, df.columns.get_loc("open")] = 1.1050
    cfg = StrategyConfig()
    cfg.pattern.require_butt_buddy = True
    cfg.pattern.butt_buddy_max_gap_points = 1.0
    cfg.pattern.wick_ratio_min = 1.5
    cfg.pattern.body_ratio_min = 1.2
    signals = detect_patterns(df, cfg)
    assert len(signals) == 0


def test_butt_buddy_accepts_neck_and_neck_sell():
    df = _make_sell_pattern_df()
    cfg = StrategyConfig()
    cfg.pattern.require_butt_buddy = True
    cfg.pattern.require_body_engulf = True
    cfg.pattern.wick_ratio_min = 1.5
    cfg.pattern.body_ratio_min = 1.2
    cfg.pattern.require_second_candle_wick_bias = True
    signals = detect_patterns(df, cfg)
    assert len(signals) >= 1
