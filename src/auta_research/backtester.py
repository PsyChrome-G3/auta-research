"""Bar-by-bar backtester for two-candle rejection signals."""

from __future__ import annotations

import itertools
from typing import Any

import pandas as pd

from auta_research.config import StrategyConfig, get_point_size
from auta_research.filters import filters_pass
from auta_research.indicators import enrich_indicators
from auta_research.patterns import detect_patterns


def _spread_cost(row: pd.Series, cfg: StrategyConfig, point_size: float) -> float:
    """Return spread cost in price units."""
    if cfg.costs.spread_mode == "fixed":
        return cfg.costs.fixed_spread_points * point_size
    spread_pts = row.get("spread", 0) or 0
    return float(spread_pts) * point_size


def _slippage_cost(cfg: StrategyConfig, point_size: float) -> float:
    """Return slippage cost in price units."""
    return cfg.costs.slippage_points * point_size


def _stop_price(
    direction: str,
    c1: pd.Series,
    c2: pd.Series,
    stop_mode: str,
    atr_val: float,
    atr_buffer: float,
) -> float:
    """Calculate stop loss price."""
    pattern_high = max(c1["high"], c2["high"])
    pattern_low = min(c1["low"], c2["low"])
    if stop_mode == "candle2_extreme":
        pattern_high = c2["high"]
        pattern_low = c2["low"]
    if stop_mode == "atr_buffered_pattern_extreme":
        if direction == "buy":
            return pattern_low - atr_val * atr_buffer
        return pattern_high + atr_val * atr_buffer
    if direction == "buy":
        return pattern_low
    return pattern_high


def _resolve_entry(
    df: pd.DataFrame,
    signal_idx: int,
    direction: str,
    entry_mode: str,
    break_expiry: int,
) -> tuple[int | None, float | None]:
    """Resolve entry bar index and price."""
    c2 = df.iloc[signal_idx]
    if entry_mode == "signal_close":
        return signal_idx, float(c2["close"])
    if entry_mode == "next_open":
        if signal_idx + 1 >= len(df):
            return None, None
        return signal_idx + 1, float(df.iloc[signal_idx + 1]["open"])
    if entry_mode == "break_signal_extreme":
        extreme = c2["high"] if direction == "buy" else c2["low"]
        for j in range(signal_idx + 1, min(signal_idx + 1 + break_expiry, len(df))):
            bar = df.iloc[j]
            if direction == "buy" and bar["high"] >= extreme:
                return j, float(extreme)
            if direction == "sell" and bar["low"] <= extreme:
                return j, float(extreme)
        return None, None
    return None, None


def _simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    direction: str,
    stop_price: float,
    r_multiple: float,
    cfg: StrategyConfig,
    spread: float,
    slippage: float,
) -> dict[str, Any]:
    """Simulate a single trade forward from entry."""
    if direction == "buy":
        entry_price_adj = entry_price + spread / 2 + slippage
        risk = entry_price_adj - stop_price
        if risk <= 0:
            return {"outcome": "invalid", "r_result": 0.0}
        target = entry_price_adj + risk * r_multiple
    else:
        entry_price_adj = entry_price - spread / 2 - slippage
        risk = stop_price - entry_price_adj
        if risk <= 0:
            return {"outcome": "invalid", "r_result": 0.0}
        target = entry_price_adj - risk * r_multiple

    max_fav = 0.0
    max_adv = 0.0
    max_bars = cfg.backtest.max_bars_to_hold
    end_idx = min(entry_idx + max_bars, len(df) - 1)

    for j in range(entry_idx, end_idx + 1):
        bar = df.iloc[j]
        if direction == "buy":
            fav = (bar["high"] - entry_price_adj) / risk
            adv = (entry_price_adj - bar["low"]) / risk
            sl_hit = bar["low"] <= stop_price
            tp_hit = bar["high"] >= target
        else:
            fav = (entry_price_adj - bar["low"]) / risk
            adv = (bar["high"] - entry_price_adj) / risk
            sl_hit = bar["high"] >= stop_price
            tp_hit = bar["low"] <= target

        max_fav = max(max_fav, fav)
        max_adv = max(max_adv, adv)

        if sl_hit and tp_hit:
            handling = cfg.backtest.ambiguous_bar_handling
            if handling == "skip":
                return {"outcome": "ambiguous_skip", "r_result": 0.0, "bars_held": j - entry_idx + 1,
                        "max_favourable_excursion_r": max_fav, "max_adverse_excursion_r": max_adv}
            if handling == "optimistic":
                return {
                    "outcome": "tp", "exit_time": bar.get("timestamp"), "exit_price": target,
                    "r_result": r_multiple, "bars_held": j - entry_idx + 1,
                    "max_favourable_excursion_r": max_fav, "max_adverse_excursion_r": max_adv,
                }
            return {
                "outcome": "sl", "exit_time": bar.get("timestamp"), "exit_price": stop_price,
                "r_result": -1.0, "bars_held": j - entry_idx + 1,
                "max_favourable_excursion_r": max_fav, "max_adverse_excursion_r": max_adv,
            }
        if sl_hit:
            return {
                "outcome": "sl", "exit_time": bar.get("timestamp"), "exit_price": stop_price,
                "r_result": -1.0, "bars_held": j - entry_idx + 1,
                "max_favourable_excursion_r": max_fav, "max_adverse_excursion_r": max_adv,
            }
        if tp_hit:
            return {
                "outcome": "tp", "exit_time": bar.get("timestamp"), "exit_price": target,
                "r_result": r_multiple, "bars_held": j - entry_idx + 1,
                "max_favourable_excursion_r": max_fav, "max_adverse_excursion_r": max_adv,
            }

    final = df.iloc[end_idx]
    if direction == "buy":
        partial = (final["close"] - entry_price_adj) / risk
    else:
        partial = (entry_price_adj - final["close"]) / risk
    return {
        "outcome": "timeout",
        "exit_time": final.get("timestamp"),
        "exit_price": float(final["close"]),
        "r_result": partial,
        "bars_held": end_idx - entry_idx + 1,
        "max_favourable_excursion_r": max_fav,
        "max_adverse_excursion_r": max_adv,
    }


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: StrategyConfig,
    entry_mode: str | None = None,
    stop_mode: str | None = None,
    atr_buffer: float = 0.0,
    r_multiple: float | None = None,
    point_size: float = 0.00001,
    require_filters: bool = True,
) -> pd.DataFrame:
    """Backtest detected signals with given entry/stop/TP settings."""
    enriched = enrich_indicators(df, atr_period=cfg.stop.atr_period)
    atr_col = f"atr_{cfg.stop.atr_period}"
    entries = cfg.entry.modes if entry_mode is None else [entry_mode]
    stops = cfg.stop.modes if stop_mode is None else [stop_mode]
    r_vals = cfg.take_profit.r_values if r_multiple is None else [r_multiple]

    trade_id = 0
    trades: list[dict[str, Any]] = []

    for _, sig in signals.iterrows():
        if require_filters and not filters_pass(sig, cfg):
            continue
        signal_idx = int(sig["signal_bar_index"])
        direction = sig["direction"]
        c1_idx = signal_idx - 1
        c1 = enriched.iloc[c1_idx]
        c2 = enriched.iloc[signal_idx]
        atr_val = float(c2.get(atr_col, c2.get("atr_14", 0)) or 0)
        spread_row = c2
        spread = _spread_cost(spread_row, cfg, point_size)
        slippage = _slippage_cost(cfg, point_size)

        for em, sm, rv in itertools.product(entries, stops, r_vals):
            if sm == "atr_buffered_pattern_extreme":
                buffers = cfg.stop.atr_buffer_values or [atr_buffer]
            else:
                buffers = [0.0]

            for buf in buffers:
                if sm != "atr_buffered_pattern_extreme" and buf != 0.0:
                    continue
                entry_idx, entry_price = _resolve_entry(
                    enriched, signal_idx, direction, em, cfg.entry.break_expiry_bars
                )
                if entry_idx is None or entry_price is None:
                    continue

                stop = _stop_price(direction, c1, c2, sm, atr_val, buf)
                result = _simulate_trade(
                    enriched, entry_idx, entry_price, direction, stop, rv, cfg, spread, slippage
                )
                if result.get("outcome") in ("invalid", "ambiguous_skip"):
                    continue

                risk_dist = abs(entry_price - stop)
                reward_dist = risk_dist * rv
                trade_id += 1
                trades.append({
                    "trade_id": trade_id,
                    "symbol": sig.get("symbol", ""),
                    "timeframe": sig.get("timeframe", ""),
                    "direction": direction,
                    "signal_time": sig.get("signal_time"),
                    "entry_time": enriched.iloc[entry_idx].get("timestamp"),
                    "entry_price": entry_price,
                    "stop_price": stop,
                    "target_price": (
                        entry_price + reward_dist if direction == "buy"
                        else entry_price - reward_dist
                    ),
                    "risk_price_distance": risk_dist,
                    "reward_price_distance": reward_dist,
                    "r_multiple_target": rv,
                    "exit_time": result.get("exit_time"),
                    "exit_price": result.get("exit_price"),
                    "outcome": result.get("outcome"),
                    "r_result": result.get("r_result"),
                    "bars_held": result.get("bars_held"),
                    "max_favourable_excursion_r": result.get("max_favourable_excursion_r"),
                    "max_adverse_excursion_r": result.get("max_adverse_excursion_r"),
                    "entry_mode": em,
                    "stop_mode": sm,
                    "tp_mode": f"{rv}R",
                    "spread_used": spread,
                    "slippage_used": slippage,
                    "filters_used": sig.get("passed_filters", ""),
                    "atr_buffer": buf,
                })

    return pd.DataFrame(trades)


def run_backtest(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    point_size: float = 0.00001,
    single_combo: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect signals and run full backtest grid or single default combo."""
    signals = detect_patterns(df, cfg)
    if single_combo:
        em = cfg.entry.modes[0]
        sm = cfg.stop.modes[0]
        rv = cfg.take_profit.r_values[0]
        buf = cfg.stop.atr_buffer_values[0] if cfg.stop.atr_buffer_values else 0.0
        trades = backtest_signals(
            df, signals, cfg, entry_mode=em, stop_mode=sm,
            atr_buffer=buf, r_multiple=rv, point_size=point_size,
        )
    else:
        trades = backtest_signals(df, signals, cfg, point_size=point_size)
    return signals, trades
