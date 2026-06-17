"""CLI entry point for auta-research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from auta_research.config import (
    find_project_root,
    get_point_size,
    load_prop_firm_config,
    load_research_config,
    load_strategy_config,
)
from auta_research.data_store import load_csv, save_csv
from auta_research.backtester import run_backtest
from auta_research.metrics import compute_metrics
from auta_research.mt5_connector import check_mt5, pull_and_save
from auta_research.optimiser import optimise
from auta_research.patterns import detect_patterns
from auta_research.reports import generate_report
from auta_research.prop_sim import discover_trade_splits, run_prop_simulation
from auta_research.prop_reports import generate_prop_sim_report
from auta_research.plotting import generate_prop_charts
from auta_research.validation import validate

console = Console()


def _project_root() -> Path:
    return find_project_root()


def cmd_pull(args: argparse.Namespace) -> int:
    """Pull OHLCV data from MT5."""
    root = _project_root()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",")]
    out_dir = root / "data" / "raw"

    ok, msg = check_mt5()
    if not ok:
        console.print(f"[red]{msg}[/red]")
        return 1

    console.print(f"Pulling {len(symbols)} symbols x {len(timeframes)} timeframes...")
    saved, errors = pull_and_save(symbols, timeframes, args.date_from, args.date_to, str(out_dir))
    for p in saved:
        console.print(f"[green]Saved:[/green] {p} ({Path(p).stat().st_size // 1024} KB)")
    if errors:
        console.print(f"\n[yellow]Failed ({len(errors)}):[/yellow]")
        for err in errors[:20]:
            console.print(f"  - {err}")
        if len(errors) > 20:
            console.print(f"  ... and {len(errors) - 20} more")
    if not saved:
        console.print("[red]No files saved. See errors above.[/red]")
        return 1
    if errors:
        console.print(f"\n[yellow]Partial success: {len(saved)} saved, {len(errors)} failed.[/yellow]")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    """Detect two-candle rejection signals."""
    root = _project_root()
    cfg = load_strategy_config(args.config, root)
    df = load_csv(args.data)
    signals = detect_patterns(df, cfg)

    out_dir = root / "data" / "results" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem
    out_path = out_dir / f"signals_{stem}.csv"
    signals.to_csv(out_path, index=False)

    console.print(f"[green]Detected {len(signals)} signals[/green] -> {out_path}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run backtest on data file."""
    root = _project_root()
    cfg = load_strategy_config(args.config, root)
    df = load_csv(args.data)
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns and len(df) else "EURUSD"
    point = get_point_size(str(symbol))

    signals, trades = run_backtest(df, cfg, point_size=point, single_combo=args.single)

    results_dir = root / "data" / "results"
    latest = results_dir / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem

    sig_path = results_dir / "signals" / f"signals_{stem}.csv"
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(sig_path, index=False)
    trades.to_csv(latest / "trades.csv", index=False)
    trades.to_csv(results_dir / f"trades_{stem}.csv", index=False)

    metrics = compute_metrics(trades)
    with open(latest / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md_lines = [
        "# Backtest Summary",
        "",
        f"- Trades: {metrics.get('trades', 0)}",
        f"- Win rate: {metrics.get('win_rate', 0):.2%}",
        f"- Expectancy: {metrics.get('expectancy_r', 0):.3f}R",
        f"- Total R: {metrics.get('total_r', 0):.2f}",
        f"- Max DD: {metrics.get('max_drawdown_r', 0):.2f}R",
        f"- Confidence: {metrics.get('confidence', 'low')}",
    ]
    (latest / "backtest_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    table = Table(title="Backtest Results")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(str(k), str(v))
    console.print(table)
    console.print(f"[green]Trades saved to {latest / 'trades.csv'}[/green]")
    return 0


def cmd_optimise(args: argparse.Namespace) -> int:
    """Run parameter optimisation."""
    root = _project_root()
    research = load_research_config(args.research_config, root)
    result = optimise(research, root)
    if result.empty:
        console.print("[yellow]No optimisation results produced. Check data files.[/yellow]")
        return 1
    console.print(f"[green]Optimised {len(result)} variants[/green]")
    console.print(result.head(10).to_string(index=False))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run walk-forward validation."""
    root = _project_root()
    research = load_research_config(args.research_config, root)
    summary = validate(research, root)
    folds = summary.get("folds", 0)
    if folds == 0:
        console.print("[yellow]No validation folds completed.[/yellow]")
        return 1
    agg = summary.get("aggregate_test", {})
    console.print(f"[green]Validation complete: {folds} folds[/green]")
    console.print(f"Stability score: {summary.get('stability_score')}")
    console.print(f"Aggregate test expectancy: {agg.get('expectancy_r', 0):.3f}R")
    console.print(f"Aggregate test win rate: {agg.get('win_rate', 0):.1%}")
    console.print(f"80% WR supported: {summary.get('win_rate_80_supported')}")
    return 0


def cmd_prop_sim(args: argparse.Namespace) -> int:
    """Run prop-firm evaluation simulator on trade log(s)."""
    root = _project_root()
    trades_path = Path(args.trades)
    if not trades_path.is_absolute():
        trades_path = root / trades_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = load_prop_firm_config(config_path, root)
    sources = discover_trade_splits(trades_path, root)
    console.print(f"[bold]Prop simulation[/bold] on {len(sources)} trade log(s):")
    for name, path in sources:
        console.print(f"  - {name}: {path}")

    out_dir = root / "data" / "results" / "prop_sim"
    assets_dir = root / "reports" / "assets"
    reports_dir = root / "reports"

    summary, mc, meta = run_prop_simulation(cfg, sources, out_dir)
    charts = generate_prop_charts(meta, assets_dir)
    report_path = generate_prop_sim_report(summary, mc, meta, cfg, reports_dir, assets_dir)

    table = Table(title="Prop Simulation Summary")
    table.add_column("Split")
    table.add_column("Risk %")
    table.add_column("Status")
    table.add_column("Pass MC")
    table.add_column("Verdict")
    for _, row in summary.head(15).iterrows():
        table.add_row(
            str(row.get("trade_split", "")),
            str(row.get("risk_per_trade_pct", "")),
            str(row.get("status", "")),
            f"{row.get('mc_pass_rate', 0):.1%}" if pd.notna(row.get("mc_pass_rate")) else "-",
            str(row.get("verdict", "")),
        )
    console.print(table)
    console.print(f"[green]Summary:[/green] {out_dir / 'prop_sim_summary.csv'}")
    if not mc.empty:
        console.print(f"[green]Monte Carlo:[/green] {out_dir / 'prop_sim_monte_carlo.csv'}")
    console.print(f"[green]Report:[/green] {report_path}")
    console.print(f"Recommended max risk: {meta.get('recommended_max_risk_pct')}%")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate Markdown report."""
    root = _project_root()
    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    report_path = generate_report(
        results_dir,
        root / "reports",
        root / "reports" / "assets",
    )
    console.print(f"[green]Report saved:[/green] {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="auta-research",
        description="AUTA 3.0 research-only backtesting tool (no live trading)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="Pull OHLCV data from MetaTrader 5")
    pull.add_argument("--symbols", required=True, help="Comma-separated symbols")
    pull.add_argument("--timeframes", required=True, help="Comma-separated timeframes")
    pull.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    pull.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    pull.set_defaults(func=cmd_pull)

    detect = sub.add_parser("detect", help="Detect two-candle rejection signals")
    detect.add_argument("--config", required=True, help="Strategy config YAML")
    detect.add_argument("--data", required=True, help="Input CSV data file")
    detect.set_defaults(func=cmd_detect)

    backtest = sub.add_parser("backtest", help="Backtest strategy on data")
    backtest.add_argument("--config", required=True, help="Strategy config YAML")
    backtest.add_argument("--data", required=True, help="Input CSV data file")
    backtest.add_argument("--single", action="store_true", help="Run single combo only")
    backtest.set_defaults(func=cmd_backtest)

    opt = sub.add_parser("optimise", help="Run parameter optimisation")
    opt.add_argument("--research-config", required=True, help="Research config YAML")
    opt.set_defaults(func=cmd_optimise)

    val = sub.add_parser("validate", help="Walk-forward validation")
    val.add_argument("--research-config", required=True, help="Research config YAML")
    val.set_defaults(func=cmd_validate)

    rep = sub.add_parser("report", help="Generate Markdown report")
    rep.add_argument("--results", default="data/results/latest", help="Results directory")
    rep.set_defaults(func=cmd_report)

    prop = sub.add_parser("prop-sim", help="Prop-firm evaluation simulator")
    prop.add_argument("--trades", required=True, help="Trade log CSV path")
    prop.add_argument("--config", default="configs/prop_firm.yaml", help="Prop firm config YAML")
    prop.set_defaults(func=cmd_prop_sim)

    return parser


def main() -> None:
    """CLI main entry."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
