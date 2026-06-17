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
