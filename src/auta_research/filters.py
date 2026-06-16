"""Trade filters and signal scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from auta_research.config import StrategyConfig
from auta_research.indicators import enrich_indicators

SESSION_HOURS: dict[str, tuple[int, int]] = {
    "london": (7, 16),
    "new_york": (12, 21),
    "tokyo": (0, 9),
    "sydney": (22, 7),
}


def _parse_hour(ts: Any) -> int:
    """Extract UTC hour from timestamp."""
    if isinstance(ts, (int, float)):
        ts = pd.Timestamp(ts, unit="s", tz="UTC")
    elif isinstance(ts, str):
        ts = pd.Timestamp(ts)
    else:
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.hour


def _in_session(hour: int, session: str) -> bool:
    """Check if hour falls in session range (UTC)."""
    start, end = SESSION_HOURS.get(session, (0, 24))
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _active_trend_mode(cfg: StrategyConfig) -> str:
    """Return active trend filter mode or none."""
    if not cfg.filters.trend.enabled:
        return "none"
    modes = [m for m in cfg.filters.trend.modes if m != "none"]
    return modes[0] if modes else "none"


def check_trend_filter(row: pd.Series, direction: str, mode: str) -> bool:
    """Evaluate trend filter for a signal bar."""
    if mode == "none":
        return True
    close = row["close"]
    if mode == "ema_50":
        return close > row.get("ema_50", close) if direction == "buy" else close < row.get("ema_50", close)
    if mode == "ema_200":
        return close > row.get("ema_200", close) if direction == "buy" else close < row.get("ema_200", close)
    if mode == "price_vs_ema_200":
        ema200 = row.get("ema_200", close)
        return close > ema200 if direction == "buy" else close < ema200
    if mode == "ema_stack":
        e20, e50, e100, e200 = (
            row.get("ema_20", close),
            row.get("ema_50", close),
            row.get("ema_100", close),
            row.get("ema_200", close),
        )
        if direction == "buy":
            return e20 > e50 > e100 > e200
        return e20 < e50 < e100 < e200
    return True


def check_volatility_filter(row: pd.Series, cfg: StrategyConfig) -> bool:
    """Evaluate ATR percentile volatility filter."""
    if not cfg.filters.volatility.enabled:
        return True
    pct = row.get("atr_percentile")
    if pd.isna(pct):
        return False
    vf = cfg.filters.volatility
    return vf.min_atr_percentile <= pct <= vf.max_atr_percentile


def check_session_filter(row: pd.Series, cfg: StrategyConfig) -> bool:
    """Evaluate session time filter."""
    if not cfg.filters.session.enabled:
        return True
    hour = _parse_hour(row.get("timestamp", datetime.utcnow()))
    return any(_in_session(hour, s) for s in cfg.filters.session.allowed_sessions)


def check_location_filter(row: pd.Series, direction: str, cfg: StrategyConfig) -> bool:
    """Evaluate proximity to recent swing."""
    if not cfg.filters.location.enabled:
        return True
    atr_val = row.get(f"atr_{cfg.stop.atr_period}", row.get("atr_14", 0))
    if atr_val <= 0 or pd.isna(atr_val):
        return not cfg.filters.location.require_near_recent_swing
    close = row["close"]
    lf = cfg.filters.location
    if direction == "buy":
        swing = row.get("recent_swing_low")
        if pd.isna(swing):
            return not lf.require_near_recent_swing
        dist = abs(close - swing) / atr_val
    else:
        swing = row.get("recent_swing_high")
        if pd.isna(swing):
            return not lf.require_near_recent_swing
        dist = abs(close - swing) / atr_val
    if lf.require_near_recent_swing:
        return dist <= lf.max_distance_atr
    return True


def check_pullback_filter(row: pd.Series, direction: str, cfg: StrategyConfig) -> bool:
    """Evaluate pullback-to-EMA filter."""
    if not cfg.filters.pullback.enabled:
        return True
    atr_val = row.get(f"atr_{cfg.stop.atr_period}", row.get("atr_14", 0))
    if atr_val <= 0 or pd.isna(atr_val):
        return True
    close = row["close"]
    pf = cfg.filters.pullback
    ema_mode = _active_trend_mode(cfg)
    if direction == "buy" and ema_mode != "none":
        if not check_trend_filter(row, "buy", ema_mode if ema_mode != "none" else "price_vs_ema_200"):
            return True
        for p in pf.ema_periods:
            ema_val = row.get(f"ema_{p}")
            if pd.notna(ema_val) and abs(close - ema_val) / atr_val <= pf.max_distance_atr:
                return True
        return False
    if direction == "sell" and ema_mode != "none":
        if not check_trend_filter(row, "sell", ema_mode if ema_mode != "none" else "price_vs_ema_200"):
            return True
        for p in pf.ema_periods:
            ema_val = row.get(f"ema_{p}")
            if pd.notna(ema_val) and abs(close - ema_val) / atr_val <= pf.max_distance_atr:
                return True
        return False
    return True


def apply_filters(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    cfg: StrategyConfig,
) -> tuple[list[str], list[str]]:
    """Apply all enabled filters; return passed and failed filter names."""
    vf = cfg.filters.volatility
    lf = cfg.filters.location
    work = enrich_indicators(
        df,
        atr_period=cfg.stop.atr_period,
        swing_lookback=lf.swing_lookback,
    )
    row = work.iloc[signal_idx]
    passed: list[str] = []
    failed: list[str] = []

    trend_mode = _active_trend_mode(cfg)
    if cfg.filters.trend.enabled:
        name = f"trend_{trend_mode}"
        if check_trend_filter(row, direction, trend_mode):
            passed.append(name)
        else:
            failed.append(name)

    if cfg.filters.volatility.enabled:
        name = "volatility"
        if check_volatility_filter(row, cfg):
            passed.append(name)
        else:
            failed.append(name)

    if cfg.filters.session.enabled:
        name = "session"
        if check_session_filter(row, cfg):
            passed.append(name)
        else:
            failed.append(name)

    if cfg.filters.location.enabled:
        name = "location"
        if check_location_filter(row, direction, cfg):
            passed.append(name)
        else:
            failed.append(name)

    if cfg.filters.pullback.enabled:
        name = "pullback"
        if check_pullback_filter(row, direction, cfg):
            passed.append(name)
        else:
            failed.append(name)

    return passed, failed


def filters_pass(signal_row: pd.Series, cfg: StrategyConfig) -> bool:
    """Return True if signal passed all enabled filters (no failed filters)."""
    if not cfg.filters.trend.enabled and not cfg.filters.volatility.enabled:
        if not cfg.filters.session.enabled and not cfg.filters.location.enabled:
            if not cfg.filters.pullback.enabled:
                return True
    failed = signal_row.get("failed_filters", "")
    if isinstance(failed, str):
        return failed == "" or failed is None
    return len(failed) == 0


def score_signal(
    meta: dict[str, Any],
    row: pd.Series,
    direction: str,
    cfg: StrategyConfig,
    passed_filters: list[str],
) -> float:
    """Compute 0-100 signal quality score for ranking."""
    pat = cfg.pattern
    wick_score = min(meta["candle1_wick_ratio"] / pat.wick_ratio_min, 3.0) / 3.0 * 25
    wick2_score = min(meta["candle2_wick_ratio"] / pat.wick_ratio_min, 3.0) / 3.0 * 15
    body_score = min(meta["body_ratio"] / pat.body_ratio_min, 3.0) / 3.0 * 20
    engulf_score = 10.0 if meta.get("body_engulf") else 0.0

    trend_mode = _active_trend_mode(cfg)
    trend_score = 10.0 if check_trend_filter(row, direction, trend_mode) else 0.0
    vol_score = 10.0 if check_volatility_filter(row, cfg) else 5.0
    loc_score = 5.0 if check_location_filter(row, direction, cfg) else 0.0
    sess_score = 5.0 if check_session_filter(row, cfg) else 2.0
    filter_bonus = min(len(passed_filters) * 2, 10)

    total = wick_score + wick2_score + body_score + engulf_score + trend_score
    total += vol_score + loc_score + sess_score + filter_bonus
    return round(min(total, 100.0), 2)
