"""Candle anatomy calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

TINY = 1e-12


def candle_colour(open_: float, close: float, body_to_range: float, flat_threshold: float) -> str:
    """Classify candle as bullish, bearish, or flat."""
    if body_to_range <= flat_threshold:
        return "flat"
    if close > open_:
        return "bullish"
    if close < open_:
        return "bearish"
    return "flat"


def enrich_candles(df: pd.DataFrame, flat_threshold: float = 0.15) -> pd.DataFrame:
    """Add candle anatomy columns to OHLC dataframe."""
    out = df.copy()
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = out["high"] - out["low"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["body_to_range"] = np.where(out["range"] > TINY, out["body"] / out["range"], 0.0)
    out["colour"] = [
        candle_colour(o, c, btr, flat_threshold)
        for o, c, btr in zip(out["open"], out["close"], out["body_to_range"])
    ]
    return out


def wick_ratio_buy(upper_wick: float, lower_wick: float) -> float:
    """Lower wick dominance ratio for buy setups."""
    return lower_wick / max(upper_wick, TINY)


def wick_ratio_sell(upper_wick: float, lower_wick: float) -> float:
    """Upper wick dominance ratio for sell setups."""
    return upper_wick / max(lower_wick, TINY)


def body_engulfs(c1_open: float, c1_close: float, c2_open: float, c2_close: float) -> bool:
    """Return True if candle2 body range fully engulfs candle1 body range."""
    c1_lo = min(c1_open, c1_close)
    c1_hi = max(c1_open, c1_close)
    c2_lo = min(c2_open, c2_close)
    c2_hi = max(c2_open, c2_close)
    return c2_lo <= c1_lo and c2_hi >= c1_hi


def body_ratio(c1_body: float, c2_body: float) -> float:
    """Ratio of candle2 body to candle1 body."""
    return c2_body / max(c1_body, TINY)
