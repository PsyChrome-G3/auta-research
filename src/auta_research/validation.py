"""Walk-forward and static train/validation/test validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console

from auta_research.backtester import backtest_signals
from auta_research.patterns import detect_patterns
from auta_research.config import ResearchConfig, StrategyConfig, get_point_size, load_strategy_config
from auta_research.data_store import find_data_files, load_csv
from auta_research.metrics import (
    aggregate_metrics_dicts,
    compute_metrics,
    degradation_pct,
    stability_score,
    win_rate_defensible,
)
from auta_research.optimiser import _apply_variant, _grid_variants

console = Console()


def _split_static(df: pd.DataFrame, train_pct: float, val_pct: float, test_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, validation, test by bar count."""
    n = len(df)
    t_end = int(n * train_pct)
    v_end = t_end + int(n * val_pct)
    return df.iloc[:t_end].copy(), df.iloc[t_end:v_end].copy(), df.iloc[v_end:].copy()


def _rolling_windows(
    df: pd.DataFrame,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Generate rolling walk-forward windows."""
    windows = []
    start = 0
    total = train_bars + val_bars + test_bars
    while start + total <= len(df):
        chunk = df.iloc[start : start + total]
        train = chunk.iloc[:train_bars]
        val = chunk.iloc[train_bars : train_bars + val_bars]
        test = chunk.iloc[train_bars + val_bars :]
        windows.append((train, val, test))
        start += step_bars
    return windows


def _best_variant_on_train(
    train_df: pd.DataFrame,
    research: ResearchConfig,
    base_cfg: StrategyConfig,
    symbol: str,
) -> tuple[dict[str, Any], StrategyConfig]:
    """Find best parameter variant on training data."""
    point = get_point_size(symbol)
    best_score = -1e9
    best_variant: dict[str, Any] = {}
    best_cfg = base_cfg

    for variant in _grid_variants(research, base_cfg):
        cfg = _apply_variant(base_cfg, variant)
        signals = detect_patterns(train_df, cfg)
        em = variant["entry_modes"][0]
        sm = variant["stop_modes"][0]
        rv = variant["tp_r_values"][0]
        trades = backtest_signals(
            train_df, signals, cfg,
            entry_mode=em, stop_mode=sm,
            atr_buffer=variant["atr_buffer"],
            r_multiple=rv, point_size=point,
        )
        metrics = compute_metrics(trades)
        score = metrics["expectancy_r"] * (metrics["trades"] ** 0.5)
        if score > best_score and metrics["trades"] >= 10:
            best_score = score
            best_variant = variant
            best_cfg = cfg

    return best_variant, best_cfg


def _eval_split(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    variant: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    """Evaluate a variant on a data split."""
    point = get_point_size(symbol)
    signals = detect_patterns(df, cfg)
    em = variant.get("entry_modes", ["next_open"])[0]
    sm = variant.get("stop_modes", ["pattern_extreme"])[0]
    rv = variant.get("tp_r_values", [1.0])[0]
    trades = backtest_signals(
        df, signals, cfg,
        entry_mode=em, stop_mode=sm,
        atr_buffer=variant.get("atr_buffer", 0.0),
        r_multiple=rv, point_size=point,
    )
    return compute_metrics(trades)


def validate(research: ResearchConfig, project_root: Path) -> dict[str, Any]:
    """Run walk-forward or static validation."""
    base_cfg = load_strategy_config(project_root / research.strategy_config, project_root)
    raw_dir = project_root / research.data.raw_dir
    out_dir = project_root / research.data.results_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    fold_test_metrics: list[dict[str, Any]] = []

    for symbol in research.symbols:
        for tf in research.timeframes:
            files = find_data_files(raw_dir, symbol, tf)
            if not files:
                console.print(f"[yellow]No data for {symbol} {tf}[/yellow]")
                continue
            df = load_csv(files[0])

            if research.validation.mode == "rolling":
                windows = _rolling_windows(
                    df,
                    research.validation.rolling.train_bars,
                    research.validation.rolling.validation_bars,
                    research.validation.rolling.test_bars,
                    research.validation.rolling.step_bars,
                )
                if not windows:
                    train, val, test = _split_static(
                        df,
                        research.validation.train_pct,
                        research.validation.validation_pct,
                        research.validation.test_pct,
                    )
                    windows = [(train, val, test)]
            else:
                train, val, test = _split_static(
                    df,
                    research.validation.train_pct,
                    research.validation.validation_pct,
                    research.validation.test_pct,
                )
                windows = [(train, val, test)]

            for fold_i, (train_df, val_df, test_df) in enumerate(windows):
                variant, cfg = _best_variant_on_train(train_df, research, base_cfg, symbol)
                train_m = _eval_split(train_df, cfg, variant, symbol)
                val_m = _eval_split(val_df, cfg, variant, symbol)
                test_m = _eval_split(test_df, cfg, variant, symbol)
                deg = degradation_pct(train_m, test_m)
                wr_check = win_rate_defensible(test_m, research.validation.win_rate_claim_threshold)

                overfit = (
                    deg > research.validation.overfit_threshold_pct
                    or wr_check["likely_overfit"]
                    or test_m.get("expectancy_r", 0) < 0
                )

                fold_test_metrics.append(test_m)
                all_results.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "fold": fold_i,
                    "variant": variant,
                    "train_metrics": train_m,
                    "validation_metrics": val_m,
                    "test_metrics": test_m,
                    "degradation_pct": deg,
                    "stability_score": 0.0,
                    "overfit_warning": overfit,
                    "win_rate_claim": wr_check,
                })

    stability = stability_score(fold_test_metrics)
    for r in all_results:
        r["stability_score"] = stability

    summary = {
        "mode": research.validation.mode,
        "folds": len(all_results),
        "stability_score": stability,
        "results": all_results,
        "aggregate_test": aggregate_metrics_dicts(
            [r["test_metrics"] for r in all_results]
        ) if all_results else {},
        "win_rate_80_supported": any(
            r["win_rate_claim"].get("defensible") for r in all_results
        ),
    }

    with open(out_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    latest = project_root / research.reporting.latest_results_dir
    latest.mkdir(parents=True, exist_ok=True)
    with open(latest / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
