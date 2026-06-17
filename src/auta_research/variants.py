"""Shared strategy variant helpers."""

from __future__ import annotations

import copy
import json
from typing import Any

from auta_research.config import StrategyConfig


def normalize_variant(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize variant dict keys and types from JSON or YAML."""
    buy = raw.get("buy_colours", ["bearish", "flat"])
    sell = raw.get("sell_colours", ["bullish", "flat"])
    return {
        "wick_ratio_min": float(raw["wick_ratio_min"]),
        "body_ratio_min": float(raw["body_ratio_min"]),
        "require_body_engulf": bool(raw.get("require_body_engulf", False)),
        "require_second_candle_wick_bias": bool(raw.get("require_second_candle_wick_bias", True)),
        "require_c2_directional_wick_larger_than_c1": bool(
            raw.get("require_c2_directional_wick_larger_than_c1", False)
        ),
        "candle2_wick_ratio_min": (
            float(raw["candle2_wick_ratio_min"])
            if raw.get("candle2_wick_ratio_min") is not None
            else None
        ),
        "c2_wick_growth_min": float(raw.get("c2_wick_growth_min", 1.0)),
        "require_butt_buddy": bool(raw.get("require_butt_buddy", False)),
        "buy_colours": list(buy),
        "sell_colours": list(sell),
        "entry_mode": str(raw["entry_mode"]),
        "stop_mode": str(raw["stop_mode"]),
        "tp_r_value": float(raw["tp_r_value"]),
        "atr_buffer": float(raw.get("atr_buffer", 0.0)),
        "trend_filter": str(raw.get("trend_filter", "none")),
        "volatility_filter": bool(raw.get("volatility_filter", False)),
        "session_filter": bool(raw.get("session_filter", False)),
    }


def parse_variant_json(text: str) -> dict[str, Any]:
    """Parse variant JSON string."""
    return normalize_variant(json.loads(text))


def apply_variant(base: StrategyConfig, variant: dict[str, Any]) -> StrategyConfig:
    """Apply a single optimisation-style variant to strategy config."""
    v = normalize_variant(variant)
    cfg = copy.deepcopy(base)
    cfg.pattern.wick_ratio_min = v["wick_ratio_min"]
    cfg.pattern.body_ratio_min = v["body_ratio_min"]
    cfg.pattern.require_body_engulf = v["require_body_engulf"]
    cfg.pattern.require_second_candle_wick_bias = v["require_second_candle_wick_bias"]
    cfg.pattern.require_c2_directional_wick_larger_than_c1 = v[
        "require_c2_directional_wick_larger_than_c1"
    ]
    cfg.pattern.candle2_wick_ratio_min = v["candle2_wick_ratio_min"]
    cfg.pattern.c2_wick_growth_min = v["c2_wick_growth_min"]
    cfg.pattern.require_butt_buddy = v["require_butt_buddy"]
    cfg.pattern.allow_candle1_colours["buy"] = v["buy_colours"]
    cfg.pattern.allow_candle1_colours["sell"] = v["sell_colours"]
    cfg.entry.modes = [v["entry_mode"]]
    cfg.stop.modes = [v["stop_mode"]]
    cfg.take_profit.r_values = [v["tp_r_value"]]
    cfg.stop.atr_buffer_values = [v["atr_buffer"]]

    trend = v["trend_filter"]
    cfg.filters.trend.enabled = trend != "none"
    cfg.filters.trend.modes = [trend] if trend != "none" else ["none"]
    cfg.filters.volatility.enabled = v["volatility_filter"]
    cfg.filters.session.enabled = v["session_filter"]
    return cfg
