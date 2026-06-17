"""Fixed single-variant backtest and validate-fixed workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from auta_research.backtester import backtest_signals
from auta_research.config import (
    FixedCandidate,
    FixedCandidatesConfig,
    StrategyConfig,
    get_point_size,
    load_strategy_config,
)
from auta_research.data_store import load_csv
from auta_research.filters import annotate_signal_filters
from auta_research.indicators import enrich_indicators
from auta_research.metrics import compute_metrics, degradation_pct
from auta_research.patterns import detect_patterns
from auta_research.variants import apply_variant, normalize_variant

console = Console()


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def split_ohlc_by_date(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split OHLC data into train (before) and test (on/after split_date)."""
    split_ts = pd.Timestamp(split_date, tz="UTC")
    ts = _parse_ts(df["timestamp"])
    train = df.loc[ts < split_ts].copy()
    test = df.loc[ts >= split_ts].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run_variant_backtest(
    df: pd.DataFrame,
    variant: dict[str, Any],
    base_cfg: StrategyConfig,
) -> pd.DataFrame:
    """Backtest exactly one variant on an OHLC dataframe."""
    v = normalize_variant(variant)
    cfg = apply_variant(base_cfg, v)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) else "EURUSD"
    point = get_point_size(symbol)

    enriched = enrich_indicators(
        df,
        atr_period=cfg.stop.atr_period,
        swing_lookback=cfg.filters.location.swing_lookback,
    )
    signals = detect_patterns(df, cfg, pattern_only=True, enriched_df=enriched)
    signals = annotate_signal_filters(signals, enriched, cfg)

    trades = backtest_signals(
        df,
        signals,
        cfg,
        entry_mode=v["entry_mode"],
        stop_mode=v["stop_mode"],
        atr_buffer=v["atr_buffer"],
        r_multiple=v["tp_r_value"],
        point_size=point,
    )
    if not trades.empty:
        trades["candidate_variant"] = json.dumps(v, sort_keys=True)
        trades["r_multiple_target"] = v["tp_r_value"]
    return trades


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def backtest_fixed(
    data_path: Path,
    variant: dict[str, Any],
    output_dir: Path,
    base_cfg: StrategyConfig,
    *,
    write_latest: bool = False,
    latest_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one variant on full data; save trades.csv and summary.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_csv(data_path)
    v = normalize_variant(variant)
    trades = run_variant_backtest(df, v, base_cfg)
    metrics = compute_metrics(trades)

    trades.to_csv(output_dir / "trades.csv", index=False)
    summary = {
        "variant": v,
        "data_file": data_path.name,
        "metrics": metrics,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "variant.json", v)

    if write_latest and latest_dir is not None:
        latest_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(latest_dir / "trades.csv", index=False)
        _write_json(latest_dir / "backtest_summary.json", metrics)

    return summary


def validate_fixed(
    data_path: Path,
    variant: dict[str, Any],
    split_date: str,
    output_dir: Path,
    base_cfg: StrategyConfig,
) -> dict[str, Any]:
    """Run one variant with train/test split by signal_time on OHLC slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_csv(data_path)
    train_df, test_df = split_ohlc_by_date(df, split_date)
    v = normalize_variant(variant)

    train_trades = run_variant_backtest(train_df, v, base_cfg)
    test_trades = run_variant_backtest(test_df, v, base_cfg)

    train_m = compute_metrics(train_trades)
    test_m = compute_metrics(test_trades)
    deg = degradation_pct(train_m, test_m)
    oos_positive = test_m.get("expectancy_r", 0) > 0

    train_trades.to_csv(output_dir / "trades_train.csv", index=False)
    test_trades.to_csv(output_dir / "trades_test.csv", index=False)
    trades_full = pd.concat([train_trades, test_trades], ignore_index=True)
    trades_full.to_csv(output_dir / "trades.csv", index=False)

    _write_json(output_dir / "summary_train.json", {"variant": v, "metrics": train_m})
    _write_json(output_dir / "summary_test.json", {"variant": v, "metrics": test_m})
    _write_json(output_dir / "variant.json", v)

    summary = {
        "variant": v,
        "split_date": split_date,
        "train_metrics": train_m,
        "test_metrics": test_m,
        "degradation_pct": deg,
        "oos_positive": oos_positive,
    }
    _write_json(output_dir / "validation_summary.json", summary)

    md = [
        "# Fixed Candidate Validation",
        "",
        f"- Data: `{data_path.name}`",
        f"- Split date: **{split_date}**",
        f"- Entry: `{v['entry_mode']}` | Stop: `{v['stop_mode']}` | TP: **{v['tp_r_value']}R**",
        "",
        "## Train",
        f"- Trades: {train_m.get('trades', 0)}",
        f"- Win rate: {train_m.get('win_rate', 0):.2%}",
        f"- Expectancy: {train_m.get('expectancy_r', 0):.3f}R",
        f"- Profit factor: {train_m.get('profit_factor', 0):.3f}",
        f"- Max drawdown (R): {train_m.get('max_drawdown_r', 0):.2f}",
        "",
        "## Test (out-of-sample)",
        f"- Trades: {test_m.get('trades', 0)}",
        f"- Win rate: {test_m.get('win_rate', 0):.2%}",
        f"- Expectancy: {test_m.get('expectancy_r', 0):.3f}R",
        f"- Profit factor: {test_m.get('profit_factor', 0):.3f}",
        f"- Max drawdown (R): {test_m.get('max_drawdown_r', 0):.2f}",
        "",
        "## Comparison",
        f"- Degradation (train to test expectancy): **{deg:.1f}%**",
        f"- OOS expectancy positive: **{'yes' if oos_positive else 'no'}**",
        "",
    ]
    (output_dir / "validation_summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary


def run_candidate(
    candidate: FixedCandidate,
    output_dir: Path,
    base_cfg: StrategyConfig,
    project_root: Path,
) -> dict[str, Any]:
    """Validate one fixed candidate and save under output_dir."""
    data_path = project_root / candidate.data
    if candidate.strategy_config:
        cfg = load_strategy_config(project_root / candidate.strategy_config, project_root)
    else:
        cfg = base_cfg

    variant = candidate.variant.model_dump()
    result = validate_fixed(
        data_path,
        variant,
        candidate.split_date,
        output_dir,
        cfg,
    )
    result["name"] = candidate.name
    result["output_dir"] = str(output_dir)
    return result


def run_fixed_batch(
    batch_cfg: FixedCandidatesConfig,
    project_root: Path,
    *,
    run_prop: bool = True,
    prop_mc_runs: int | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Run validate-fixed for all configured candidates; optional prop-sim and comparison report."""
    from auta_research.config import load_prop_firm_config
    from auta_research.fixed_candidate_reports import (
        generate_fixed_candidate_comparison,
        run_prop_sim_for_candidate,
    )

    base_cfg = load_strategy_config(project_root / batch_cfg.strategy_config, project_root)
    root = project_root / batch_cfg.output_root
    results: list[dict[str, Any]] = []

    prop_cfg = None
    risk_levels = 0
    if run_prop:
        prop_cfg = load_prop_firm_config(project_root / batch_cfg.prop_firm_config, project_root)
        if prop_mc_runs is not None:
            prop_cfg.monte_carlo.runs = prop_mc_runs
        risk_levels = len(prop_cfg.risk.risk_per_trade_pct_values)

    n = len(batch_cfg.candidates)
    total_steps = n + (n * risk_levels if run_prop else 0) + 1

    if run_prop and prop_cfg and prop_cfg.monte_carlo.enabled:
        mc_runs = prop_cfg.monte_carlo.runs
        console.print(
            f"[dim]Prop-sim enabled: {mc_runs} Monte Carlo runs × {risk_levels} risk levels "
            f"× {n} candidates — expect several minutes. Use --skip-prop-sim to validate only.[/dim]"
        )

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    report_path: Path | None = None
    with progress:
        task = progress.add_task("Starting batch...", total=total_steps)

        for candidate in batch_cfg.candidates:
            out = root / candidate.name
            progress.update(task, description=f"{candidate.name} | validate-fixed")
            result = run_candidate(candidate, out, base_cfg, project_root)
            progress.advance(task)

            if prop_cfg is not None and (out / "trades_test.csv").exists():
                progress.update(task, description=f"{candidate.name} | prop-sim")
                run_prop_sim_for_candidate(
                    out,
                    prop_cfg,
                    progress=progress,
                    progress_task=task,
                    chart_samples=0,
                )
                result["prop_sim_dir"] = str(out / "prop_sim")
            results.append(result)

        progress.update(task, description="comparison report")
        report_path = generate_fixed_candidate_comparison(root, project_root / "reports")
        progress.advance(task)

    return results, report_path
