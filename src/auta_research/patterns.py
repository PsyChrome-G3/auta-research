"""Two-candle wick rejection pattern detection."""

from __future__ import annotations

from typing import Any

import pandas as pd

from auta_research.candle_math import (
    body_engulfs,
    body_ratio,
    enrich_candles,
    wick_ratio_buy,
    wick_ratio_sell,
)
from auta_research.config import StrategyConfig
from auta_research.filters import apply_filters, score_signal
from auta_research.indicators import enrich_indicators


def pattern_cache_key(cfg: StrategyConfig) -> tuple[Any, ...]:
    """Hashable key for pattern-only detection cache."""
    pat = cfg.pattern
    return (
        pat.wick_ratio_min,
        pat.body_ratio_min,
        pat.require_body_engulf,
        tuple(pat.allow_candle1_colours.get("buy", [])),
        tuple(pat.allow_candle1_colours.get("sell", [])),
        pat.require_candle2_colour.get("buy"),
        pat.require_candle2_colour.get("sell"),
        pat.require_second_candle_wick_bias,
        pat.min_body_to_range_ratio,
        pat.max_body_to_range_ratio_for_flat,
    )


def _check_direction(
    direction: str,
    c1: pd.Series,
    c2: pd.Series,
    cfg: StrategyConfig,
) -> tuple[bool, dict[str, Any]]:
    """Check if a two-candle pair matches pattern rules for direction."""
    pat = cfg.pattern
    reasons: dict[str, Any] = {}

    if direction == "buy":
        allowed_c1 = pat.allow_candle1_colours.get("buy", ["bearish", "flat"])
        required_c2 = pat.require_candle2_colour.get("buy", "bullish")
        c1_wr = wick_ratio_buy(c1["upper_wick"], c1["lower_wick"])
        c2_wr = wick_ratio_buy(c2["upper_wick"], c2["lower_wick"])
    else:
        allowed_c1 = pat.allow_candle1_colours.get("sell", ["bullish", "flat"])
        required_c2 = pat.require_candle2_colour.get("sell", "bearish")
        c1_wr = wick_ratio_sell(c1["upper_wick"], c1["lower_wick"])
        c2_wr = wick_ratio_sell(c2["upper_wick"], c2["lower_wick"])

    br = body_ratio(c1["body"], c2["body"])
    engulf = body_engulfs(c1["open"], c1["close"], c2["open"], c2["close"])

    reasons.update(
        {
            "candle1_colour": c1["colour"],
            "candle2_colour": c2["colour"],
            "candle1_wick_ratio": c1_wr,
            "candle2_wick_ratio": c2_wr,
            "body_ratio": br,
            "body_engulf": engulf,
        }
    )

    if c1["colour"] not in allowed_c1:
        reasons["fail"] = "candle1_colour"
        return False, reasons
    if c2["colour"] != required_c2:
        reasons["fail"] = "candle2_colour"
        return False, reasons
    if c1["body_to_range"] < pat.min_body_to_range_ratio and c1["colour"] != "flat":
        if c1["range"] <= 0:
            reasons["fail"] = "candle1_range"
            return False, reasons
    if c1_wr < pat.wick_ratio_min:
        reasons["fail"] = "candle1_wick_ratio"
        return False, reasons
    if pat.require_second_candle_wick_bias and c2_wr < pat.wick_ratio_min:
        reasons["fail"] = "candle2_wick_ratio"
        return False, reasons
    if br < pat.body_ratio_min:
        reasons["fail"] = "body_ratio"
        return False, reasons
    if pat.require_body_engulf and not engulf:
        reasons["fail"] = "body_engulf"
        return False, reasons

    reasons["fail"] = None
    return True, reasons


_SIGNAL_COLUMNS = [
    "signal_time", "symbol", "timeframe", "direction", "signal_bar_index",
    "candle1_time", "candle2_time",
    "candle1_open", "candle1_high", "candle1_low", "candle1_close",
    "candle2_open", "candle2_high", "candle2_low", "candle2_close",
    "candle1_body", "candle2_body",
    "candle1_upper_wick", "candle1_lower_wick",
    "candle2_upper_wick", "candle2_lower_wick",
    "candle1_wick_ratio", "candle2_wick_ratio", "body_ratio",
    "signal_score", "passed_filters", "failed_filters",
]


def detect_patterns(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    *,
    pattern_only: bool = False,
    enriched_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect all buy and sell two-candle rejection patterns."""
    flat_threshold = cfg.pattern.max_body_to_range_ratio_for_flat
    work = enrich_candles(df, flat_threshold=flat_threshold)
    if enriched_df is None:
        enriched_df = enrich_indicators(
            work,
            atr_period=cfg.stop.atr_period,
            swing_lookback=cfg.filters.location.swing_lookback,
        )
    else:
        enriched_df = enriched_df

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns and len(df) else ""
    timeframe = df["timeframe"].iloc[0] if "timeframe" in df.columns and len(df) else ""

    records: list[dict[str, Any]] = []
    n = len(work)
    for i in range(1, n):
        c1 = work.iloc[i - 1]
        c2 = work.iloc[i]

        for direction in ("buy", "sell"):
            matched, meta = _check_direction(direction, c1, c2, cfg)
            if not matched:
                continue

            if pattern_only:
                passed: list[str] = []
                failed: list[str] = []
                score = min(meta["body_ratio"] / cfg.pattern.body_ratio_min, 3.0) / 3.0 * 50
            else:
                passed, failed = apply_filters(
                    enriched_df, i, direction, cfg, enriched_df=enriched_df
                )
                score = score_signal(meta, enriched_df.iloc[i], direction, cfg, passed)

            records.append(
                {
                    "signal_time": c2.get("timestamp", c2.name),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "direction": direction,
                    "signal_bar_index": i,
                    "candle1_time": c1.get("timestamp", c1.name),
                    "candle2_time": c2.get("timestamp", c2.name),
                    "candle1_open": c1["open"],
                    "candle1_high": c1["high"],
                    "candle1_low": c1["low"],
                    "candle1_close": c1["close"],
                    "candle2_open": c2["open"],
                    "candle2_high": c2["high"],
                    "candle2_low": c2["low"],
                    "candle2_close": c2["close"],
                    "candle1_body": c1["body"],
                    "candle2_body": c2["body"],
                    "candle1_upper_wick": c1["upper_wick"],
                    "candle1_lower_wick": c1["lower_wick"],
                    "candle2_upper_wick": c2["upper_wick"],
                    "candle2_lower_wick": c2["lower_wick"],
                    "candle1_wick_ratio": meta["candle1_wick_ratio"],
                    "candle2_wick_ratio": meta["candle2_wick_ratio"],
                    "body_ratio": meta["body_ratio"],
                    "signal_score": round(score, 2),
                    "passed_filters": ",".join(passed) if passed else "",
                    "failed_filters": ",".join(failed) if failed else "",
                }
            )

    if not records:
        return pd.DataFrame(columns=_SIGNAL_COLUMNS)
    return pd.DataFrame(records)
