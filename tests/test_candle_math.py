"""Tests for candle anatomy calculations."""

import pandas as pd

from auta_research.candle_math import (
    body_engulfs,
    body_ratio,
    c2_directional_wick_exceeds_c1,
    candle_colour,
    enrich_candles,
    neck_and_neck,
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


def test_neck_and_neck():
    assert neck_and_neck(1.1010, 1.1010, 0.0001) is True
    assert neck_and_neck(1.1010, 1.1020, 0.00005) is False
    assert neck_and_neck(1.1010, 1.10105, 0.0001) is True


def test_c2_directional_wick_exceeds_c1():
    # Sell: upper wick on C2 larger than C1
    assert c2_directional_wick_exceeds_c1(0.009, 0.001, 0.012, 0.002, "sell") is True
    assert c2_directional_wick_exceeds_c1(0.012, 0.001, 0.009, 0.002, "sell") is False
    # Buy: lower wick on C2 larger than C1
    assert c2_directional_wick_exceeds_c1(0.001, 0.008, 0.002, 0.012, "buy") is True
