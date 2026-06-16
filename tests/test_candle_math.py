"""Tests for candle anatomy calculations."""

import pandas as pd

from auta_research.candle_math import (
    body_engulfs,
    body_ratio,
    candle_colour,
    enrich_candles,
    wick_ratio_buy,
    wick_ratio_sell,
)


def test_candle_colour_bullish():
    assert candle_colour(1.0, 1.1, 0.5, 0.15) == "bullish"


def test_candle_colour_bearish():
    assert candle_colour(1.1, 1.0, 0.5, 0.15) == "bearish"


def test_candle_colour_flat():
    assert candle_colour(1.0, 1.001, 0.05, 0.15) == "flat"


def test_enrich_candles():
    df = pd.DataFrame({
        "open": [1.0, 1.1],
        "high": [1.2, 1.15],
        "low": [0.9, 1.0],
        "close": [1.1, 1.05],
    })
    out = enrich_candles(df)
    assert "body" in out.columns
    assert "upper_wick" in out.columns
    assert "lower_wick" in out.columns
    assert out.iloc[0]["colour"] == "bullish"


def test_wick_ratios():
    assert wick_ratio_sell(0.2, 0.1) == 2.0
    assert wick_ratio_buy(0.1, 0.2) == 2.0


def test_body_engulf():
    assert body_engulfs(1.0, 1.05, 0.99, 1.10) is True
    assert body_engulfs(1.0, 1.10, 1.02, 1.05) is False


def test_body_ratio():
    assert body_ratio(0.05, 0.10) == 2.0
