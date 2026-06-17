"""Tests for optimisation grid variants."""

import pandas as pd

from auta_research.config import ResearchConfig, StrategyConfig
from auta_research.optimiser import _grid_variants, dedupe_results


def _base_research(**grid_overrides) -> ResearchConfig:
    grid = {
        "wick_ratio_min": [2.0],
        "body_ratio_min": [1.5],
        "require_body_engulf": [False],
        "candle1_allowed_colours": {
            "buy": [["bearish", "flat"]],
            "sell": [["bullish", "flat"]],
        },
        "entry_modes": ["next_open", "signal_close"],
        "stop_modes": ["pattern_extreme"],
        "tp_r_values": [0.5, 1.0],
        "atr_buffer_values": [0.0],
        "trend_filters": ["none"],
        "volatility_filters": [False],
        "session_filters": [False],
    }
    grid.update(grid_overrides)
    return ResearchConfig(optimisation={"max_variants": 100, "grid": grid})


def test_grid_variants_scalar_tp_entry_stop():
    research = _base_research()
    base = StrategyConfig()
    variants = list(_grid_variants(research, base))
    assert len(variants) == 4  # 2 entry x 2 tp
    for v in variants:
        assert isinstance(v["tp_r_value"], float)
        assert isinstance(v["entry_mode"], str)
        assert isinstance(v["stop_mode"], str)
        assert v["atr_buffer"] == 0.0

    tp_values = sorted({v["tp_r_value"] for v in variants})
    assert tp_values == [0.5, 1.0]


def test_atr_buffer_only_for_atr_stop_mode():
    research = _base_research(
        stop_modes=["pattern_extreme", "atr_buffered_pattern_extreme"],
        atr_buffer_values=[0.0, 0.1, 0.2],
        entry_modes=["next_open"],
        tp_r_values=[1.0],
    )
    variants = list(_grid_variants(research, StrategyConfig()))
    pattern = [v for v in variants if v["stop_mode"] == "pattern_extreme"]
    atr = [v for v in variants if v["stop_mode"] == ATR_BUFFERED_STOP]

    assert len(pattern) == 1
    assert pattern[0]["atr_buffer"] == 0.0
    assert sorted(v["atr_buffer"] for v in atr) == [0.0, 0.1, 0.2]


def test_dedupe_results():
    rows = [
        {
            "symbol": "EURUSD",
            "timeframe": "H4",
            "wick_ratio_min": 2.0,
            "body_ratio_min": 1.5,
            "require_body_engulf": False,
            "entry_mode": "next_open",
            "stop_mode": "pattern_extreme",
            "tp_r_value": 1.0,
            "atr_buffer": 0.0,
            "trend_filter": "none",
            "volatility_filter": False,
            "session_filter": False,
            "buy_colours": ["bearish", "flat"],
            "sell_colours": ["bullish", "flat"],
            "rank_score": 1.0,
        },
        {
            "symbol": "EURUSD",
            "timeframe": "H4",
            "wick_ratio_min": 2.0,
            "body_ratio_min": 1.5,
            "require_body_engulf": False,
            "entry_mode": "next_open",
            "stop_mode": "pattern_extreme",
            "tp_r_value": 1.0,
            "atr_buffer": 0.0,
            "trend_filter": "none",
            "volatility_filter": False,
            "session_filter": False,
            "buy_colours": ["bearish", "flat"],
            "sell_colours": ["bullish", "flat"],
            "rank_score": 2.0,
        },
    ]
    df = pd.DataFrame(rows)
    deduped, generated, skipped = dedupe_results(df)
    assert generated == 2
    assert skipped == 1
    assert len(deduped) == 1


ATR_BUFFERED_STOP = "atr_buffered_pattern_extreme"
