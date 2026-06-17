"""Markdown and JSON report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from auta_research.metrics import compute_metrics, win_rate_defensible
from auta_research.plotting import generate_all_charts


def _load_results(results_dir: Path) -> dict[str, Any]:
    """Load available result files from directory."""
    data: dict[str, Any] = {}
    trades_path = results_dir / "trades.csv"
    if trades_path.exists():
        data["trades"] = pd.read_csv(trades_path)
    opt_path = results_dir / "optimisation_results.csv"
    if not opt_path.exists():
        opt_path = results_dir.parent / "optimisation" / "optimisation_results.csv"
    if opt_path.exists():
        data["optimisation"] = pd.read_csv(opt_path)
    val_path = results_dir / "validation_results.json"
    if val_path.exists():
        with open(val_path, encoding="utf-8") as f:
            data["validation"] = json.load(f)
    return data


def _table_rows(df: pd.DataFrame, group_col: str, value_col: str = "r_result") -> str:
    """Build markdown table rows for grouped stats."""
    if df.empty or group_col not in df.columns:
        return "| (no data) | - | - |\n"
    rows = []
    for key, grp in df.groupby(group_col):
        wr = (grp[value_col] > 0).mean()
        exp = grp[value_col].mean()
        rows.append(f"| {key} | {wr:.2%} | {exp:.3f} |")
    return "\n".join(rows) if rows else "| (no data) | - | - |\n"


def _opt_tp_summary(opt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate optimisation rows by tp_r_value (trade-weighted)."""
    if opt.empty or "tp_r_value" not in opt.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for tp, grp in opt.groupby("tp_r_value"):
        weights = grp["trades"].fillna(0).clip(lower=0)
        total_trades = int(weights.sum())
        if total_trades > 0:
            win_rate = float((grp["win_rate"] * weights).sum() / weights.sum())
            expectancy = float((grp["expectancy_r"] * weights).sum() / weights.sum())
        else:
            win_rate = float(grp["win_rate"].mean())
            expectancy = float(grp["expectancy_r"].mean())
        rows.append({
            "tp_r_value": float(tp),
            "win_rate": win_rate,
            "expectancy_r": expectancy,
            "variant_rows": len(grp),
            "total_trades": total_trades,
        })
    return pd.DataFrame(rows).sort_values("tp_r_value").reset_index(drop=True)


def _best_variant_per_tp(opt: pd.DataFrame) -> pd.DataFrame:
    """Best rank_score row for each tp_r_value."""
    if opt.empty or "tp_r_value" not in opt.columns or "rank_score" not in opt.columns:
        return pd.DataFrame()
    idx = opt.groupby("tp_r_value")["rank_score"].idxmax()
    best = opt.loc[idx].sort_values("tp_r_value").reset_index(drop=True)
    return best


def generate_report(results_dir: Path, output_dir: Path, assets_dir: Path) -> Path:
    """Generate comprehensive Markdown research report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    data = _load_results(results_dir)
    trades = data.get("trades", pd.DataFrame())
    opt = data.get("optimisation", pd.DataFrame())
    validation = data.get("validation", {})

    metrics = compute_metrics(trades) if not trades.empty else {}
    wr_claim = win_rate_defensible(metrics) if metrics else {"verdict": "not_supported"}

    charts = {}
    if not trades.empty:
        charts = generate_all_charts(trades, assets_dir)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# AUTA 3.0 Research Report",
        "",
        f"Generated: {now}",
        "",
        "## Executive Summary",
        "",
    ]

    if metrics:
        lines.extend([
            f"- Total trades analysed: **{metrics.get('trades', 0)}**",
            f"- Win rate: **{metrics.get('win_rate', 0):.2%}**",
            f"- Expectancy (R): **{metrics.get('expectancy_r', 0):.3f}**",
            f"- Total R: **{metrics.get('total_r', 0):.2f}**",
            f"- Max drawdown (R): **{metrics.get('max_drawdown_r', 0):.2f}**",
            f"- Confidence: **{metrics.get('confidence', 'low')}**",
            f"- 80% win rate defensible: **{wr_claim.get('verdict', 'not_supported')}**",
            "",
        ])
    else:
        lines.append("No trade data found. Run backtest or optimisation first.")
        lines.append("")

    lines.extend([
        "## 1. Does the raw two-candle strategy work?",
        "",
    ])
    if metrics:
        works = metrics.get("expectancy_r", 0) > 0 and metrics.get("trades", 0) >= 30
        lines.append(
            f"Based on {metrics.get('trades', 0)} trades, expectancy is "
            f"{metrics.get('expectancy_r', 0):.3f}R. "
            f"{'Evidence suggests positive edge.' if works else 'No convincing edge detected.'}"
        )
    else:
        lines.append("Insufficient data.")
    lines.append("")

    lines.extend([
        "## 2. Best Timeframe",
        "",
        "| Timeframe | Win Rate | Expectancy |",
        "|-----------|----------|------------|",
        _table_rows(trades, "timeframe") if not trades.empty else "| - | - | - |",
        "",
        "## 3. Best Symbols",
        "",
        "| Symbol | Win Rate | Expectancy |",
        "|--------|----------|------------|",
        _table_rows(trades, "symbol") if not trades.empty else "| - | - | - |",
        "",
        "## 4. Win Rate vs R Target",
        "",
        "| R Target | Win Rate | Expectancy | Trades |",
        "|----------|----------|------------|--------|",
    ])

    if not trades.empty and "r_multiple_target" in trades.columns:
        for rv, grp in trades.groupby("r_multiple_target"):
            lines.append(
                f"| {rv}R | {(grp['r_result'] > 0).mean():.2%} | "
                f"{grp['r_result'].mean():.3f} | {len(grp)} |"
            )
    elif not opt.empty and "tp_r_value" in opt.columns:
        tp_summary = _opt_tp_summary(opt)
        for _, row in tp_summary.iterrows():
            lines.append(
                f"| {row['tp_r_value']}R | {row['win_rate']:.2%} | "
                f"{row['expectancy_r']:.3f} | {int(row['total_trades'])} |"
            )
    else:
        lines.append("| - | - | - | - |")

    lines.extend([
        "",
        "## 5. Expectancy vs R Target",
        "",
    ])
    if not trades.empty and "r_multiple_target" in trades.columns:
        best_r = trades.groupby("r_multiple_target")["r_result"].mean().idxmax()
        lines.append(f"Best TP multiple by expectancy: **{best_r}R**")
        r1 = trades[trades["r_multiple_target"] == 1.0]["r_result"].mean() if 1.0 in trades["r_multiple_target"].values else None
        if r1 is not None:
            for comp in [0.5, 0.75, 1.5, 2.0, 3.0]:
                sub = trades[trades["r_multiple_target"] == comp]["r_result"].mean() if comp in trades["r_multiple_target"].values else None
                if sub is not None:
                    lines.append(f"- 1R vs {comp}R: {r1:.3f} vs {sub:.3f}")
    elif not opt.empty and "tp_r_value" in opt.columns:
        tp_summary = _opt_tp_summary(opt)
        if not tp_summary.empty:
            best_row = tp_summary.loc[tp_summary["expectancy_r"].idxmax()]
            lines.append(
                f"Best TP multiple by optimisation expectancy: **{best_row['tp_r_value']}R** "
                f"(win rate {best_row['win_rate']:.2%}, expectancy {best_row['expectancy_r']:.3f})"
            )
            lines.append("")
            lines.append("| TP (R) | Win Rate | Expectancy | Variant Rows | Total Trades |")
            lines.append("|--------|----------|------------|--------------|--------------|")
            for _, row in tp_summary.iterrows():
                lines.append(
                    f"| {row['tp_r_value']} | {row['win_rate']:.2%} | "
                    f"{row['expectancy_r']:.3f} | {int(row['variant_rows'])} | "
                    f"{int(row['total_trades'])} |"
                )
    lines.append("")

    lines.extend([
        "## 6. Filter Comparison",
        "",
        "Compare filtered vs unfiltered by examining `filters_used` column in trade log.",
        "",
        "## 7. Optimisation Results",
        "",
    ])

    if not opt.empty:
        lines.append("### Top 5 Variants (overall rank_score)")
        lines.append("")
        lines.append(
            "| Rank | Symbol | TF | TP | Entry | Stop | Trades | Win Rate | Expectancy |"
        )
        lines.append(
            "|------|--------|----|----|-------|------|--------|----------|------------|"
        )
        for _, row in opt.head(5).iterrows():
            lines.append(
                f"| {row.get('rank', '')} | {row.get('symbol', '')} | {row.get('timeframe', '')} | "
                f"{row.get('tp_r_value', '')}R | {row.get('entry_mode', '')} | "
                f"{row.get('stop_mode', '')} | {row.get('trades', 0)} | "
                f"{row.get('win_rate', 0):.2%} | {row.get('expectancy_r', 0):.3f} |"
            )
        lines.append("")

        best_per_tp = _best_variant_per_tp(opt)
        if not best_per_tp.empty:
            lines.append("### Best Variant per TP Multiple (ranked by tp_r_value)")
            lines.append("")
            lines.append(
                "| TP | Symbol | TF | Entry | Stop | Trades | Win Rate | Expectancy | Rank Score |"
            )
            lines.append(
                "|----|--------|----|-------|------|--------|----------|------------|------------|"
            )
            for _, row in best_per_tp.iterrows():
                lines.append(
                    f"| {row.get('tp_r_value', '')}R | {row.get('symbol', '')} | "
                    f"{row.get('timeframe', '')} | {row.get('entry_mode', '')} | "
                    f"{row.get('stop_mode', '')} | {row.get('trades', 0)} | "
                    f"{row.get('win_rate', 0):.2%} | {row.get('expectancy_r', 0):.3f} | "
                    f"{row.get('rank_score', 0):.3f} |"
                )
            lines.append("")

        lines.append("### Worst 5 Variants")
        lines.append("")
        for _, row in opt.tail(5).iterrows():
            lines.append(
                f"- {row.get('symbol')} {row.get('timeframe')} "
                f"{row.get('tp_r_value', '')}R {row.get('entry_mode', '')}: "
                f"expectancy {row.get('expectancy_r', 0):.3f}, trades {row.get('trades', 0)}"
            )
    else:
        lines.append("No optimisation results found.")

    lines.extend([
        "",
        "## 8. Validation (Out-of-Sample)",
        "",
    ])
    if validation:
        lines.append(f"- Mode: {validation.get('mode', 'unknown')}")
        lines.append(f"- Folds: {validation.get('folds', 0)}")
        lines.append(f"- Stability score: {validation.get('stability_score', 0)}")
        lines.append(f"- 80% win rate supported OOS: {validation.get('win_rate_80_supported', False)}")
        agg = validation.get("aggregate_test", {})
        if agg:
            lines.append(f"- Aggregate test expectancy: {agg.get('expectancy_r', 0):.3f}")
            lines.append(f"- Aggregate test win rate: {agg.get('win_rate', 0):.2%}")
    else:
        lines.append("No validation results. Run `auta-research validate` first.")

    lines.extend([
        "",
        "## 9. Is 80% Win Rate Defensible?",
        "",
        f"Verdict: **{wr_claim.get('verdict', 'not_supported')}**",
        "",
        "Criteria: 300+ test trades, positive expectancy after costs, "
        "no catastrophic drawdown, OOS survival, multi-symbol/timeframe consistency.",
        "",
        "## 10. Highest Defensible Win Rate",
        "",
    ])
    if metrics and metrics.get("trades", 0) >= 100:
        lines.append(f"With {metrics.get('trades')} trades: {metrics.get('win_rate', 0):.2%} win rate, confidence {metrics.get('confidence')}.")
    else:
        lines.append("Insufficient trade count for high-confidence win rate claims.")

    lines.extend([
        "",
        "## 11. Recommended Next Steps",
        "",
        "1. Expand symbol and timeframe coverage if sample sizes are low.",
        "2. Run walk-forward validation before trusting optimisation winners.",
        "3. Test stricter pattern configs on higher timeframes.",
        "4. Compare session and trend filters on out-of-sample data.",
        "5. Only proceed to MQL5 EA development if OOS expectancy stays positive.",
        "",
        "## Charts",
        "",
    ])
    for name, path in charts.items():
        rel = Path(path).name
        lines.append(f"![{name}](assets/{rel})")
        lines.append("")

    lines.extend([
        "",
        "---",
        "*This report is for research purposes only. Not financial advice.*",
        "",
    ])

    report_path = output_dir / f"research_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "metrics": metrics,
        "win_rate_claim": wr_claim,
        "charts": charts,
        "report_path": str(report_path),
    }
    with open(output_dir / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return report_path
