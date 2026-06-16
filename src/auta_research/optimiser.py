"""Parameter grid optimisation."""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from rich.console import Console
from rich.progress import Progress

from auta_research.backtester import backtest_signals
from auta_research.patterns import detect_patterns
from auta_research.config import ResearchConfig, StrategyConfig, get_point_size, load_strategy_config
from auta_research.data_store import find_data_files, load_csv
from auta_research.metrics import compute_metrics, penalised_score

console = Console()


def _grid_variants(research: ResearchConfig, base: StrategyConfig) -> Iterator[dict[str, Any]]:
    """Yield parameter combinations from optimisation grid."""
    g = research.optimisation.grid
    count = 0
    max_v = research.optimisation.max_variants

    buy_colours = g.candle1_allowed_colours.get("buy", [["bearish", "flat"]])
    sell_colours = g.candle1_allowed_colours.get("sell", [["bullish", "flat"]])

    for wr, br, engulf, bc, sc, em, sm, tp, atr_b, trend, vol, sess in itertools.product(
        g.wick_ratio_min,
        g.body_ratio_min,
        g.require_body_engulf,
        buy_colours,
        sell_colours,
        g.entry_modes,
        g.stop_modes,
        g.tp_r_values,
        g.atr_buffer_values,
        g.trend_filters,
        g.volatility_filters,
        g.session_filters,
    ):
        if count >= max_v:
            return
        yield {
            "wick_ratio_min": wr,
            "body_ratio_min": br,
            "require_body_engulf": engulf,
            "buy_colours": bc,
            "sell_colours": sc,
            "entry_modes": em,
            "stop_modes": sm,
            "tp_r_values": tp,
            "atr_buffer": atr_b,
            "trend_filter": trend,
            "volatility_filter": vol,
            "session_filter": sess,
        }
        count += 1


def _apply_variant(base: StrategyConfig, variant: dict[str, Any]) -> StrategyConfig:
    """Apply optimisation variant to strategy config."""
    cfg = copy.deepcopy(base)
    cfg.pattern.wick_ratio_min = variant["wick_ratio_min"]
    cfg.pattern.body_ratio_min = variant["body_ratio_min"]
    cfg.pattern.require_body_engulf = variant["require_body_engulf"]
    cfg.pattern.allow_candle1_colours["buy"] = variant["buy_colours"]
    cfg.pattern.allow_candle1_colours["sell"] = variant["sell_colours"]
    cfg.entry.modes = variant["entry_modes"]
    cfg.stop.modes = variant["stop_modes"]
    cfg.take_profit.r_values = variant["tp_r_values"]
    cfg.stop.atr_buffer_values = [variant["atr_buffer"]]

    trend = variant["trend_filter"]
    cfg.filters.trend.enabled = trend != "none"
    cfg.filters.trend.modes = [trend] if trend != "none" else ["none"]
    cfg.filters.volatility.enabled = variant["volatility_filter"]
    cfg.filters.session.enabled = variant["session_filter"]
    return cfg


def optimise(research: ResearchConfig, project_root: Path) -> pd.DataFrame:
    """Run optimisation across symbols, timeframes, and parameter grid."""
    base_path = project_root / research.strategy_config
    base_cfg = load_strategy_config(base_path, project_root)
    raw_dir = project_root / research.data.raw_dir
    results_dir = project_root / research.data.results_dir / "optimisation"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    variants = list(_grid_variants(research, base_cfg))

    with Progress() as progress:
        task = progress.add_task("Optimising...", total=len(variants) * len(research.symbols) * len(research.timeframes))

        for symbol in research.symbols:
            for tf in research.timeframes:
                files = find_data_files(raw_dir, symbol, tf)
                if not files:
                    console.print(f"[yellow]No data for {symbol} {tf}, skipping[/yellow]")
                    progress.advance(task, advance=len(variants))
                    continue
                df = load_csv(files[0])
                point = get_point_size(symbol)

                for variant in variants:
                    cfg = _apply_variant(base_cfg, variant)
                    signals = detect_patterns(df, cfg)
                    all_trades: list[pd.DataFrame] = []
                    for em in variant["entry_modes"]:
                        for sm in variant["stop_modes"]:
                            for rv in variant["tp_r_values"]:
                                trades = backtest_signals(
                                    df, signals, cfg,
                                    entry_mode=em, stop_mode=sm,
                                    atr_buffer=variant["atr_buffer"],
                                    r_multiple=rv, point_size=point,
                                )
                                if not trades.empty:
                                    all_trades.append(trades)
                    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
                    metrics = compute_metrics(combined)
                    score = penalised_score(metrics, research.optimisation.min_trades_for_ranking)

                    rows.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "variant": json.dumps(variant, sort_keys=True),
                        **variant,
                        **metrics,
                        "rank_score": score,
                    })
                    progress.advance(task)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
        result_df["rank"] = range(1, len(result_df) + 1)
        out_path = results_dir / "optimisation_results.csv"
        result_df.to_csv(out_path, index=False)
        with open(results_dir / "optimisation_results.json", "w", encoding="utf-8") as f:
            json.dump(result_df.head(50).to_dict(orient="records"), f, indent=2, default=str)

        latest = project_root / research.reporting.latest_results_dir
        latest.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(latest / "optimisation_results.csv", index=False)

    return result_df
