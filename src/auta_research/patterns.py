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


def detect_patterns(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Detect all buy and sell two-candle rejection patterns."""
    flat_threshold = cfg.pattern.max_body_to_range_ratio_for_flat
    enriched = enrich_candles(df, flat_threshold=flat_threshold)

    records: list[dict[str, Any]] = []
    for i in range(1, len(enriched)):
        c1 = enriched.iloc[i - 1]
        c2 = enriched.iloc[i]
        signal_idx = i

        for direction in ("buy", "sell"):
            matched, meta = _check_direction(direction, c1, c2, cfg)
            if not matched:
                continue

            passed, failed = apply_filters(enriched, signal_idx, direction, cfg)
            score = score_signal(meta, enriched.iloc[signal_idx], direction, cfg, passed)

            records.append(
                {
                    "signal_time": c2.get("timestamp", c2.name),
                    "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else "",
                    "timeframe": df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                    "direction": direction,
                    "signal_bar_index": signal_idx,
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
                    "signal_score": score,
                    "passed_filters": ",".join(passed) if passed else "",
                    "failed_filters": ",".join(failed) if failed else "",
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.DataFrame(records)
