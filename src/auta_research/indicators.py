"""Technical indicators for filters and stops."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def add_emas(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Add EMA columns to dataframe."""
    out = df.copy()
    for p in periods or [20, 50, 100, 200]:
        out[f"ema_{p}"] = ema(out["close"], p)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range series."""
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range."""
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR column."""
    out = df.copy()
    out[f"atr_{period}"] = atr(out, period)
    return out


def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 100) -> pd.Series:
    """Rolling percentile rank of ATR."""
    atr_col = atr(df, period)
    return atr_col.rolling(lookback, min_periods=lookback // 2).apply(
        lambda x: (x.iloc[-1] > x).sum() / max(len(x) - 1, 1) * 100,
        raw=False,
    )


def add_atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 100) -> pd.DataFrame:
    """Add ATR percentile column."""
    out = df.copy()
    out["atr_percentile"] = atr_percentile(out, period, lookback)
    return out


def swing_highs(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Detect swing highs using centered rolling max."""
    half = lookback // 2
    rolling_max = df["high"].rolling(lookback, center=True).max()
    return np.where(df["high"] == rolling_max, df["high"], np.nan)


def swing_lows(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Detect swing lows using centered rolling min."""
    rolling_min = df["low"].rolling(lookback, center=True).min()
    return np.where(df["low"] == rolling_min, df["low"], np.nan)


def recent_swing_high(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Forward-filled recent swing high."""
    swings = pd.Series(swing_highs(df, lookback), index=df.index)
    return swings.ffill()


def recent_swing_low(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Forward-filled recent swing low."""
    swings = pd.Series(swing_lows(df, lookback), index=df.index)
    return swings.ffill()


def enrich_indicators(df: pd.DataFrame, atr_period: int = 14, swing_lookback: int = 20) -> pd.DataFrame:
    """Add all indicator columns used by filters and scoring."""
    out = add_emas(df)
    out = add_atr(out, atr_period)
    out = add_atr_percentile(out, atr_period)
    out["recent_swing_high"] = recent_swing_high(out, swing_lookback)
    out["recent_swing_low"] = recent_swing_low(out, swing_lookback)
    return out
