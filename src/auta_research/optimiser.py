"""Parameter grid optimisation."""

from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from auta_research.backtester import backtest_signals
from auta_research.config import ResearchConfig, StrategyConfig, get_point_size, load_strategy_config
from auta_research.data_store import find_data_files, load_csv
from auta_research.filters import annotate_signal_filters
from auta_research.indicators import enrich_indicators
from auta_research.metrics import compute_metrics, penalised_score
from auta_research.patterns import detect_patterns, pattern_cache_key
from auta_research.variants import apply_variant as _apply_variant

console = Console()

ATR_BUFFERED_STOP = "atr_buffered_pattern_extreme"

# Columns that define a unique optimisation result row.
VARIANT_DEDUP_COLUMNS = (
    "symbol",
    "timeframe",
    "wick_ratio_min",
    "body_ratio_min",
    "require_body_engulf",
    "entry_mode",
    "stop_mode",
    "tp_r_value",
    "atr_buffer",
    "trend_filter",
    "volatility_filter",
    "session_filter",
    "buy_colours",
    "sell_colours",
)


def _atr_buffers_for_stop(stop_mode: str, buffer_values: list[float]) -> list[float]:
    """Return ATR buffer grid for a stop mode (only varied for ATR-buffered stops)."""
    if stop_mode == ATR_BUFFERED_STOP:
        return buffer_values or [0.0]
    return [0.0]


def _grid_variants(research: ResearchConfig, base: StrategyConfig) -> Iterator[dict[str, Any]]:
    """Yield parameter combinations; each TP, entry, and stop is its own variant."""
    g = research.optimisation.grid
    count = 0
    max_v = research.optimisation.max_variants

    buy_colours = g.candle1_allowed_colours.get("buy", [["bearish", "flat"]])
    sell_colours = g.candle1_allowed_colours.get("sell", [["bullish", "flat"]])

    for wr, br, engulf, bc, sc, em, sm, tp, trend, vol, sess in itertools.product(
        g.wick_ratio_min,
        g.body_ratio_min,
        g.require_body_engulf,
        buy_colours,
        sell_colours,
        g.entry_modes,
        g.stop_modes,
        g.tp_r_values,
        g.trend_filters,
        g.volatility_filters,
        g.session_filters,
    ):
        for atr_b in _atr_buffers_for_stop(sm, g.atr_buffer_values):
            if count >= max_v:
                return
            yield {
                "wick_ratio_min": wr,
                "body_ratio_min": br,
                "require_body_engulf": engulf,
                "buy_colours": bc,
                "sell_colours": sc,
                "entry_mode": em,
                "stop_mode": sm,
                "tp_r_value": float(tp),
                "atr_buffer": float(atr_b),
                "trend_filter": trend,
                "volatility_filter": vol,
                "session_filter": sess,
            }
            count += 1


def _build_result_row(
    symbol: str,
    tf: str,
    path: Path,
    df: pd.DataFrame,
    variant: dict[str, Any],
    metrics: dict[str, Any],
    rank_score: float,
) -> dict[str, Any]:
    """Build one optimisation result row with scalar variant columns."""
    variant_meta = {
        k: variant[k]
        for k in (
            "wick_ratio_min",
            "body_ratio_min",
            "require_body_engulf",
            "entry_mode",
            "stop_mode",
            "tp_r_value",
            "atr_buffer",
            "trend_filter",
            "volatility_filter",
            "session_filter",
            "buy_colours",
            "sell_colours",
        )
    }
    return {
        "symbol": symbol,
        "timeframe": tf,
        "data_file": path.name,
        "bars": len(df),
        "variant": json.dumps(variant_meta, sort_keys=True),
        **variant_meta,
        **metrics,
        "rank_score": rank_score,
    }


def _discover_datasets(
    research: ResearchConfig,
    raw_dir: Path,
    base_cfg: StrategyConfig,
) -> list[tuple[str, str, Path, pd.DataFrame, pd.DataFrame]]:
    """Load all available symbol/timeframe CSVs once."""
    tfs = research.optimisation.timeframes or research.timeframes
    datasets: list[tuple[str, str, Path, pd.DataFrame, pd.DataFrame]] = []

    for symbol in research.symbols:
        for tf in tfs:
            files = find_data_files(raw_dir, symbol, tf)
            if not files:
                continue
            path = files[0]
            df = load_csv(path)
            enriched = enrich_indicators(
                df,
                atr_period=base_cfg.stop.atr_period,
                swing_lookback=base_cfg.filters.location.swing_lookback,
            )
            datasets.append((symbol, tf, path, df, enriched))

    return datasets


def _normalize_colours_column(series: pd.Series) -> pd.Series:
    """Make buy/sell colour lists comparable for deduplication."""
    def _norm(val: Any) -> str:
        if isinstance(val, list):
            return json.dumps(val, sort_keys=True)
        return str(val)

    return series.map(_norm)


def dedupe_results(result_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Remove duplicate variant rows. Returns (deduped_df, generated_count, skipped_count)."""
    if result_df.empty:
        return result_df, 0, 0

    generated = len(result_df)
    work = result_df.copy()
    for col in ("buy_colours", "sell_colours"):
        if col in work.columns:
            work[col] = _normalize_colours_column(work[col])

    deduped = work.drop_duplicates(subset=list(VARIANT_DEDUP_COLUMNS), keep="first")
    skipped = generated - len(deduped)
    return deduped.reset_index(drop=True), generated, skipped


def _save_results(result_df: pd.DataFrame, results_dir: Path, latest_dir: Path) -> tuple[int, int]:
    """Write deduplicated optimisation results. Returns (generated, skipped)."""
    if result_df.empty:
        return 0, 0

    deduped, generated, skipped = dedupe_results(result_df)
    ranked = deduped.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked.to_csv(results_dir / "optimisation_results.csv", index=False)
    with open(results_dir / "optimisation_results.json", "w", encoding="utf-8") as f:
        json.dump(ranked.head(50).to_dict(orient="records"), f, indent=2, default=str)
    latest_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(latest_dir / "optimisation_results.csv", index=False)
    return generated, skipped


def optimise(research: ResearchConfig, project_root: Path) -> pd.DataFrame:
    """Run optimisation across symbols, timeframes, and parameter grid."""
    base_path = project_root / research.strategy_config
    base_cfg = load_strategy_config(base_path, project_root)
    raw_dir = project_root / research.data.raw_dir
    results_dir = project_root / research.data.results_dir / "optimisation"
    latest_dir = project_root / research.reporting.latest_results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    variants = list(_grid_variants(research, base_cfg))
    datasets = _discover_datasets(research, raw_dir, base_cfg)
    total_jobs = len(variants) * len(datasets)

    if not datasets:
        console.print("[red]No data files found. Run pull first or check data/raw/.[/red]")
        return pd.DataFrame()

    console.print(
        f"[bold]Optimisation plan:[/bold] {len(variants)} parameter variants x "
        f"{len(datasets)} datasets = [cyan]{total_jobs:,}[/cyan] jobs"
    )
    bars_total = sum(len(d[3]) for d in datasets)
    console.print(f"Loaded {len(datasets)} CSVs ({bars_total:,} bars total)")
    if any(d[1] == "M5" for d in datasets):
        console.print(
            "[yellow]M5 data detected. This is slow. "
            "Set optimisation.timeframes: [H4, D1] in research.yaml for faster runs.[/yellow]"
        )

    rows: list[dict[str, Any]] = []
    save_every = research.optimisation.save_every
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
        task = progress.add_task("Starting...", total=total_jobs)
        done = 0

        for symbol, tf, path, df, enriched in datasets:
            point = get_point_size(symbol)
            ds_key = f"{symbol}_{tf}"
            local_pattern_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

            for variant in variants:
                progress.update(
                    task,
                    description=(
                        f"{ds_key} | {variant['entry_mode']} | "
                        f"{variant['stop_mode']} | {variant['tp_r_value']}R"
                    ),
                )

                cfg = _apply_variant(base_cfg, variant)
                pkey = pattern_cache_key(cfg)

                if pkey not in local_pattern_cache:
                    local_pattern_cache[pkey] = detect_patterns(
                        df, cfg, pattern_only=True, enriched_df=enriched
                    )
                signals = annotate_signal_filters(local_pattern_cache[pkey], enriched, cfg)

                trades = backtest_signals(
                    df,
                    signals,
                    cfg,
                    entry_mode=variant["entry_mode"],
                    stop_mode=variant["stop_mode"],
                    atr_buffer=variant["atr_buffer"],
                    r_multiple=variant["tp_r_value"],
                    point_size=point,
                )
                metrics = compute_metrics(trades)
                score = penalised_score(metrics, research.optimisation.min_trades_for_ranking)

                rows.append(
                    _variant_row(symbol, tf, path, df, variant, metrics, score)
                )

                done += 1
                progress.advance(task)

                if done % save_every == 0:
                    _save_results(pd.DataFrame(rows), results_dir, latest_dir)

    elapsed = time.time() - started
    console.print(f"Finished {done:,} jobs in {elapsed:.0f}s ({done / max(elapsed, 1):.1f} jobs/s)")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        generated, skipped = _save_results(result_df, results_dir, latest_dir)
        console.print(f"[green]Saved:[/green] {results_dir / 'optimisation_results.csv'}")
        console.print(f"Generated variants: {generated}")
        console.print(f"Duplicate variants skipped: {skipped}")
        console.print(f"Unique variants saved: {generated - skipped}")
        result_df, _, _ = dedupe_results(result_df)

    return result_df
