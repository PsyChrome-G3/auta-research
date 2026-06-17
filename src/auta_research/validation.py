"""Walk-forward and static train/validation/test validation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from auta_research.backtester import backtest_signals
from auta_research.config import ResearchConfig, StrategyConfig, get_point_size, load_strategy_config
from auta_research.data_store import find_data_files, load_csv
from auta_research.filters import annotate_signal_filters
from auta_research.indicators import enrich_indicators
from auta_research.metrics import (
    aggregate_metrics_dicts,
    compute_metrics,
    degradation_pct,
    stability_score,
    win_rate_defensible,
)
from auta_research.optimiser import _apply_variant, _grid_variants
from auta_research.patterns import detect_patterns, pattern_cache_key

console = Console()


def _split_static(
    df: pd.DataFrame, train_pct: float, val_pct: float, test_pct: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _windows_for_df(df: pd.DataFrame, research: ResearchConfig) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Build train/val/test windows for a dataset."""
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
            return [(train, val, test)]
        return windows

    train, val, test = _split_static(
        df,
        research.validation.train_pct,
        research.validation.validation_pct,
        research.validation.test_pct,
    )
    return [(train, val, test)]


def _discover_datasets(
    research: ResearchConfig,
    raw_dir: Path,
    base_cfg: StrategyConfig,
) -> list[tuple[str, str, Path, pd.DataFrame]]:
    """Load CSVs available for validation."""
    tfs = research.optimisation.timeframes or research.timeframes
    datasets: list[tuple[str, str, Path, pd.DataFrame]] = []
    for symbol in research.symbols:
        for tf in tfs:
            files = find_data_files(raw_dir, symbol, tf)
            if not files:
                continue
            path = files[0]
            datasets.append((symbol, tf, path, load_csv(path)))
    return datasets


def _best_variant_on_train(
    train_df: pd.DataFrame,
    research: ResearchConfig,
    base_cfg: StrategyConfig,
    symbol: str,
    label: str,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> tuple[dict[str, Any], StrategyConfig]:
    """Find best parameter variant on training data."""
    point = get_point_size(symbol)
    best_score = -1e9
    best_variant: dict[str, Any] = {}
    best_cfg = base_cfg

    enriched = enrich_indicators(
        train_df,
        atr_period=base_cfg.stop.atr_period,
        swing_lookback=base_cfg.filters.location.swing_lookback,
    )
    pattern_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    for variant in _grid_variants(research, base_cfg):
        if progress is not None and task_id is not None:
            progress.update(
                task_id,
                description=(
                    f"{label} | train search | {variant['entry_mode']} | "
                    f"{variant['stop_mode']} | {variant['tp_r_value']}R"
                ),
            )

        cfg = _apply_variant(base_cfg, variant)
        pkey = pattern_cache_key(cfg)
        if pkey not in pattern_cache:
            pattern_cache[pkey] = detect_patterns(
                train_df, cfg, pattern_only=True, enriched_df=enriched
            )
        signals = annotate_signal_filters(pattern_cache[pkey], enriched, cfg)

        trades = backtest_signals(
            train_df,
            signals,
            cfg,
            entry_mode=variant["entry_mode"],
            stop_mode=variant["stop_mode"],
            atr_buffer=variant["atr_buffer"],
            r_multiple=variant["tp_r_value"],
            point_size=point,
        )
        metrics = compute_metrics(trades)
        score = metrics["expectancy_r"] * (metrics["trades"] ** 0.5)
        if score > best_score and metrics["trades"] >= 10:
            best_score = score
            best_variant = variant
            best_cfg = cfg

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    return best_variant, best_cfg


def _eval_split(
    df: pd.DataFrame,
    cfg: StrategyConfig,
    variant: dict[str, Any],
    symbol: str,
    label: str,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate a variant on a data split."""
    if progress is not None and task_id is not None:
        progress.update(task_id, description=f"{label} | evaluating split")

    point = get_point_size(symbol)
    signals = detect_patterns(df, cfg)
    trades = backtest_signals(
        df,
        signals,
        cfg,
        entry_mode=variant.get("entry_mode", "next_open"),
        stop_mode=variant.get("stop_mode", "pattern_extreme"),
        atr_buffer=variant.get("atr_buffer", 0.0),
        r_multiple=float(variant.get("tp_r_value", 1.0)),
        point_size=point,
    )
    if progress is not None and task_id is not None:
        progress.advance(task_id)
    return compute_metrics(trades)


def validate(research: ResearchConfig, project_root: Path) -> dict[str, Any]:
    """Run walk-forward or static validation."""
    base_cfg = load_strategy_config(project_root / research.strategy_config, project_root)
    raw_dir = project_root / research.data.raw_dir
    out_dir = project_root / research.data.results_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = list(_grid_variants(research, base_cfg))
    datasets = _discover_datasets(research, raw_dir, base_cfg)

    if not datasets:
        console.print("[red]No data files found for validation. Check data/raw/.[/red]")
        return {"mode": research.validation.mode, "folds": 0, "results": []}

    fold_plans: list[tuple[str, str, int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = []
    for symbol, tf, _path, df in datasets:
        windows = _windows_for_df(df, research)
        for fold_i, window in enumerate(windows):
            fold_plans.append((symbol, tf, fold_i, window))

    # Per fold: train grid search + 3 split evaluations (train, val, test).
    variant_steps = len(variants) * len(fold_plans)
    eval_steps = 3 * len(fold_plans)
    total_steps = variant_steps + eval_steps

    console.print(f"[bold]Validation plan:[/bold] mode={research.validation.mode}")
    console.print(f"  Datasets: {len(datasets)}")
    console.print(f"  Folds: {len(fold_plans)}")
    console.print(f"  Variants per fold (train search): {len(variants)}")
    console.print(f"  Total steps: [cyan]{total_steps:,}[/cyan]")

    all_results: list[dict[str, Any]] = []
    fold_test_metrics: list[dict[str, Any]] = []
    started = time.time()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task = progress.add_task("Starting validation...", total=total_steps)

        for symbol, tf, fold_i, (train_df, val_df, test_df) in fold_plans:
            label = f"{symbol}_{tf} fold {fold_i}"
            progress.update(task, description=f"{label} | selecting best on train")

            variant, cfg = _best_variant_on_train(
                train_df,
                research,
                base_cfg,
                symbol,
                label,
                progress=progress,
                task_id=task,
            )

            if not variant:
                console.print(
                    f"[yellow]{label}: no variant met min trades on train; skipping fold[/yellow]"
                )
                progress.advance(task, advance=3)
                continue

            train_m = _eval_split(train_df, cfg, variant, symbol, label, progress, task)
            val_m = _eval_split(val_df, cfg, variant, symbol, label, progress, task)
            test_m = _eval_split(test_df, cfg, variant, symbol, label, progress, task)

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

            console.print(
                f"[dim]{label} done: test expectancy {test_m.get('expectancy_r', 0):.3f}R, "
                f"win rate {test_m.get('win_rate', 0):.1%}, trades {test_m.get('trades', 0)}[/dim]"
            )

    elapsed = time.time() - started
    console.print(
        f"Validation search finished in {elapsed:.0f}s "
        f"({total_steps / max(elapsed, 1):.1f} steps/s)"
    )

    stability = stability_score(fold_test_metrics)
    for r in all_results:
        r["stability_score"] = stability

    summary = {
        "mode": research.validation.mode,
        "folds": len(all_results),
        "datasets": len(datasets),
        "variants_searched_per_fold": len(variants),
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

    console.print(f"[green]Saved:[/green] {out_dir / 'validation_results.json'}")
    return summary
