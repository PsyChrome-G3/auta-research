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
    load_fixed_candidates_config,
    load_portfolio_candidates_config,
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
from auta_research.plotting import generate_prop_charts, generate_portfolio_charts
from auta_research.validation import validate
from auta_research.fixed_candidates import backtest_fixed, validate_fixed, run_fixed_batch
from auta_research.variants import parse_variant_json
from auta_research.portfolio_sim import run_portfolio_simulation
from auta_research.portfolio_reports import generate_portfolio_sim_report
from auta_research.portfolio_sensitivity import run_portfolio_sensitivity

console = Console()


def _resolve_path(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _project_root() -> Path:
    return find_project_root()


def _prop_sim_output_dir(trades_path: Path, root: Path) -> Path:
    """Use candidate prop_sim dir when trades live under a fixed-candidate folder."""
    parent = trades_path.parent
    if (parent / "variant.json").exists() or (parent / "summary_test.json").exists():
        return parent / "prop_sim"
    return root / "data" / "results" / "prop_sim"


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
    trades_path = _resolve_path(root, args.trades)
    config_path = _resolve_path(root, args.config)

    cfg = load_prop_firm_config(config_path, root)
    if args.prop_mc_runs is not None:
        cfg.monte_carlo.runs = args.prop_mc_runs

    sources = discover_trade_splits(trades_path, root)
    mode = "The5ers Bootcamp" if cfg.is_multiphase else "single-phase"
    console.print(f"[bold]Prop simulation[/bold] ({mode}) on {len(sources)} trade log(s):")
    for name, path in sources:
        console.print(f"  - {name}: {path}")

    out_dir = _prop_sim_output_dir(trades_path, root)
    assets_dir = root / "reports" / "assets"
    reports_dir = root / "reports"

    summary, mc, meta = run_prop_simulation(cfg, sources, out_dir)
    charts = generate_prop_charts(meta, assets_dir)
    report_path = generate_prop_sim_report(summary, mc, meta, cfg, reports_dir, assets_dir)

    table = Table(title="Prop Simulation Summary")
    table.add_column("Split")
    table.add_column("Risk %")
    table.add_column("Status")
    if cfg.is_multiphase:
        table.add_column("Bootcamp MC")
    else:
        table.add_column("Pass MC")
    table.add_column("Verdict")
    table.add_column("Reason")
    for _, row in summary.head(15).iterrows():
        mc_col = row.get("mc_bootcamp_pass_rate", row.get("mc_pass_rate"))
        table.add_row(
            str(row.get("trade_split", "")),
            str(row.get("risk_per_trade_pct", "")),
            str(row.get("status", "")),
            f"{mc_col:.1%}" if pd.notna(mc_col) else "-",
            str(row.get("verdict", "")),
            str(row.get("rejection_reason", "") or ""),
        )
    console.print(table)
    console.print(f"[green]Summary:[/green] {out_dir / 'prop_sim_summary.csv'}")
    if not mc.empty:
        console.print(f"[green]Monte Carlo:[/green] {out_dir / 'prop_sim_monte_carlo.csv'}")
    console.print(f"[green]Report:[/green] {report_path}")
    console.print(f"Recommended max risk: {meta.get('recommended_max_risk_pct')}%")
    return 0


def cmd_portfolio_sim(args: argparse.Namespace) -> int:
    """Run portfolio prop-firm simulation across merged candidate trade logs."""
    root = _project_root()
    portfolio_cfg = load_portfolio_candidates_config(_resolve_path(root, args.config), root)
    prop_cfg = load_prop_firm_config(_resolve_path(root, args.prop_config), root)

    if args.prop_mc_runs is not None:
        prop_cfg.monte_carlo.runs = args.prop_mc_runs

    console.print(
        f"[bold]Portfolio simulation[/bold] ({len(portfolio_cfg.portfolios)} portfolios, "
        f"MC runs={prop_cfg.monte_carlo.runs})"
    )
    for p in portfolio_cfg.portfolios:
        console.print(f"  - {p.name}: {len(p.trades)} trade log(s)")

    out_dir = root / portfolio_cfg.output_root
    assets_dir = root / "reports" / "assets"
    reports_dir = root / "reports"

    summary, mc, meta = run_portfolio_simulation(portfolio_cfg, prop_cfg, root)
    generate_portfolio_charts(meta, assets_dir)
    report_path = generate_portfolio_sim_report(summary, mc, meta, prop_cfg, reports_dir, assets_dir)

    table = Table(title="Portfolio Simulation Summary")
    table.add_column("Portfolio")
    table.add_column("Risk %")
    table.add_column("Status")
    table.add_column("Pass MC")
    table.add_column("Verdict")
    table.add_column("Reason")
    for _, row in summary.iterrows():
        table.add_row(
            str(row.get("portfolio_name", row.get("trade_split", ""))),
            str(row.get("risk_per_trade_pct", "")),
            str(row.get("status", "")),
            f"{row.get('mc_pass_rate', 0):.1%}" if pd.notna(row.get("mc_pass_rate")) else "-",
            str(row.get("verdict", "")),
            str(row.get("rejection_reason", "") or ""),
        )
    console.print(table)
    console.print(f"[green]Summary:[/green] {out_dir / 'portfolio_summary.csv'}")
    if not mc.empty:
        console.print(f"[green]Monte Carlo:[/green] {out_dir / 'portfolio_monte_carlo.csv'}")
    console.print(f"[green]Report:[/green] {report_path}")
    console.print(f"Recommended max risk: {meta.get('recommended_max_risk_pct')}%")
    return 0


def cmd_portfolio_sensitivity(args: argparse.Namespace) -> int:
    """Sweep portfolio risk grid against prop-firm rules."""
    root = _project_root()
    portfolio_cfg = load_portfolio_candidates_config(_resolve_path(root, args.config), root)
    prop_cfg = load_prop_firm_config(_resolve_path(root, args.prop_config), root)
    if args.prop_mc_runs is not None:
        prop_cfg.monte_carlo.runs = args.prop_mc_runs

    mode = "The5ers Bootcamp" if prop_cfg.is_multiphase else "single-phase"
    console.print(f"[bold]Portfolio sensitivity[/bold] ({mode}, {len(portfolio_cfg.portfolios)} portfolios)")

    df, meta = run_portfolio_sensitivity(portfolio_cfg, prop_cfg, root)
    out_dir = root / "data" / "results" / "portfolio_sensitivity"
    console.print(f"[green]Results:[/green] {out_dir / 'portfolio_sensitivity.csv'} ({len(df)} combinations)")
    if not df.empty:
        sort_col = "bootcamp_pass_rate" if "bootcamp_pass_rate" in df.columns else "pass_rate"
        if sort_col in df.columns:
            best = df.iloc[0]
            console.print(
                f"Best: {best.get('portfolio_name')} @ {best.get('risk_per_trade_pct')}% "
                f"({sort_col}={best.get(sort_col, 0):.1%})"
            )
    return 0


def cmd_backtest_fixed(args: argparse.Namespace) -> int:
    """Run a single fixed strategy variant backtest."""
    root = _project_root()
    data_path = _resolve_path(root, args.data)
    output_dir = _resolve_path(root, args.output_dir)
    cfg = load_strategy_config(args.strategy_config, root)
    variant = parse_variant_json(args.variant_json)

    summary = backtest_fixed(
        data_path,
        variant,
        output_dir,
        cfg,
        write_latest=args.write_latest,
        latest_dir=root / "data" / "results" / "latest" if args.write_latest else None,
    )
    metrics = summary.get("metrics", {})
    console.print(f"[green]Backtest fixed[/green] -> {output_dir}")
    console.print(
        f"Trades: {metrics.get('trades', 0)} | "
        f"Expectancy: {metrics.get('expectancy_r', 0):.3f}R | "
        f"Win rate: {metrics.get('win_rate', 0):.1%}"
    )
    return 0


def cmd_validate_fixed(args: argparse.Namespace) -> int:
    """Validate a single fixed strategy variant with train/test split."""
    root = _project_root()
    data_path = _resolve_path(root, args.data)
    output_dir = _resolve_path(root, args.output_dir)
    cfg = load_strategy_config(args.strategy_config, root)
    variant = parse_variant_json(args.variant_json)

    summary = validate_fixed(
        data_path,
        variant,
        args.split_date,
        output_dir,
        cfg,
    )
    train_m = summary.get("train_metrics", {})
    test_m = summary.get("test_metrics", {})
    console.print(f"[green]Validate fixed[/green] -> {output_dir}")
    console.print(
        f"Train: {train_m.get('expectancy_r', 0):.3f}R @ {train_m.get('win_rate', 0):.1%} WR | "
        f"Test: {test_m.get('expectancy_r', 0):.3f}R @ {test_m.get('win_rate', 0):.1%} WR"
    )
    console.print(
        f"Degradation: {summary.get('degradation_pct', 0):.1f}% | "
        f"OOS positive: {'yes' if summary.get('oos_positive') else 'no'}"
    )
    console.print(f"Report: {output_dir / 'validation_summary.md'}")
    return 0


def cmd_validate_fixed_batch(args: argparse.Namespace) -> int:
    """Run validate-fixed for all candidates in a batch config."""
    root = _project_root()
    batch_cfg = load_fixed_candidates_config(_resolve_path(root, args.config), root)

    console.print(f"[bold]Fixed candidate batch[/bold] ({len(batch_cfg.candidates)} candidates)")
    mc_runs = args.prop_mc_runs
    results, report_path = run_fixed_batch(
        batch_cfg,
        root,
        run_prop=not args.skip_prop_sim,
        prop_mc_runs=mc_runs,
    )

    table = Table(title="Fixed Candidate Results")
    table.add_column("Candidate")
    table.add_column("Train Exp")
    table.add_column("Test Exp")
    table.add_column("OOS+")
    for row in results:
        train_m = row.get("train_metrics", {})
        test_m = row.get("test_metrics", {})
        table.add_row(
            str(row.get("name", "")),
            f"{train_m.get('expectancy_r', 0):.3f}R",
            f"{test_m.get('expectancy_r', 0):.3f}R",
            "yes" if row.get("oos_positive") else "no",
        )
    console.print(table)
    if report_path:
        console.print(f"[green]Comparison report:[/green] {report_path}")
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
    prop.add_argument("--prop-mc-runs", type=int, default=None, help="Override Monte Carlo runs")
    prop.set_defaults(func=cmd_prop_sim)

    btf = sub.add_parser("backtest-fixed", help="Backtest one fixed strategy variant")
    btf.add_argument("--data", required=True, help="Input CSV data file")
    btf.add_argument("--variant-json", required=True, help="Variant parameters as JSON")
    btf.add_argument("--output-dir", required=True, help="Output directory for trades.csv and summary.json")
    btf.add_argument(
        "--strategy-config",
        default="configs/strategies/two_candle_rejection.yaml",
        help="Base strategy config YAML",
    )
    btf.add_argument(
        "--write-latest",
        action="store_true",
        help="Also write trades to data/results/latest (off by default)",
    )
    btf.set_defaults(func=cmd_backtest_fixed)

    vf = sub.add_parser("validate-fixed", help="Train/test validate one fixed variant")
    vf.add_argument("--data", required=True, help="Input CSV data file")
    vf.add_argument("--variant-json", required=True, help="Variant parameters as JSON")
    vf.add_argument("--split-date", required=True, help="Split date YYYY-MM-DD (test on/after)")
    vf.add_argument("--output-dir", required=True, help="Output directory for validation artifacts")
    vf.add_argument(
        "--strategy-config",
        default="configs/strategies/two_candle_rejection.yaml",
        help="Base strategy config YAML",
    )
    vf.set_defaults(func=cmd_validate_fixed)

    vfb = sub.add_parser("validate-fixed-batch", help="Validate all fixed candidates from config")
    vfb.add_argument("--config", default="configs/fixed_candidates.yaml", help="Fixed candidates YAML")
    vfb.add_argument("--skip-prop-sim", action="store_true", help="Skip prop-sim per candidate")
    vfb.add_argument(
        "--prop-mc-runs",
        type=int,
        default=None,
        help="Override Monte Carlo runs per risk level (default: prop_firm.yaml value)",
    )
    vfb.set_defaults(func=cmd_validate_fixed_batch)

    psim = sub.add_parser("portfolio-sim", help="Portfolio prop-firm simulation")
    psim.add_argument("--config", default="configs/portfolio_candidates.yaml", help="Portfolio config YAML")
    psim.add_argument("--prop-config", default="configs/prop_firm.yaml", help="Prop firm config YAML")
    psim.add_argument(
        "--prop-mc-runs",
        type=int,
        default=None,
        help="Override Monte Carlo runs per risk level",
    )
    psim.set_defaults(func=cmd_portfolio_sim)

    psens = sub.add_parser("portfolio-sensitivity", help="Portfolio risk-parameter sensitivity sweep")
    psens.add_argument("--config", default="configs/portfolio_candidates.yaml", help="Portfolio config YAML")
    psens.add_argument("--prop-config", default="configs/prop_firm.yaml", help="Prop firm config YAML")
    psens.add_argument("--prop-mc-runs", type=int, default=None, help="Override Monte Carlo runs")
    psens.set_defaults(func=cmd_portfolio_sensitivity)

    return parser


def main() -> None:
    """CLI main entry."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
