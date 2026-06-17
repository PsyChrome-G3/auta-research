"""Markdown reports for portfolio prop-firm simulation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from auta_research.config import PropFirmConfig


def generate_portfolio_sim_report(
    summary: pd.DataFrame,
    mc: pd.DataFrame,
    meta: dict[str, Any],
    cfg: PropFirmConfig,
    output_dir: Path,
    assets_dir: Path,
) -> Path:
    """Write portfolio simulation Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# AUTA Portfolio Prop-Firm Simulation Report",
        "",
        f"Generated: {now}",
        "",
        "## Executive Summary",
        "",
        f"- Starting balance: ${cfg.account.starting_balance:,.0f}",
        f"- Profit target: {cfg.account.profit_target_pct}%",
        f"- Max total loss: {cfg.account.max_total_loss_pct}%",
        f"- Max daily loss: {cfg.account.max_daily_loss_pct}%",
        f"- Max open trades: {cfg.risk.max_open_trades}",
        f"- Recommended max risk per trade: **{meta.get('recommended_max_risk_pct')}%**",
        "",
    ]

    if meta.get("merged_trade_counts"):
        lines.append("### Merged trade counts")
        lines.append("")
        for name, count in meta["merged_trade_counts"].items():
            lines.append(f"- **{name}**: {count} trades after merge/dedupe")
        lines.append("")

    lines.extend(["## Portfolio Simulation by Risk Level", ""])
    if not summary.empty:
        cols = [
            "portfolio_name", "risk_per_trade_pct", "status", "passed",
            "final_return_pct", "max_drawdown_pct", "max_daily_loss_pct",
            "total_trades_taken", "merged_trade_count", "skipped_open_limit",
            "win_rate", "average_r", "verdict", "rejection_reason", "recommended",
        ]
        show = [c for c in cols if c in summary.columns]
        lines.append("| " + " | ".join(show) + " |")
        lines.append("| " + " | ".join(["---"] * len(show)) + " |")
        for _, row in summary.iterrows():
            cells = []
            for c in show:
                val = row[c]
                if c == "win_rate" and isinstance(val, float):
                    cells.append(f"{val:.1%}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(["## Strategy Contribution", ""])
    if not summary.empty and "strategy_contribution_pct" in summary.columns:
        for _, row in summary.iterrows():
            lines.append(f"### {row.get('portfolio_name')} @ {row.get('risk_per_trade_pct')}% risk")
            lines.append("")
            try:
                contrib = json.loads(row["strategy_contribution_pct"])
                counts = json.loads(row.get("strategy_trade_counts", "{}"))
            except (json.JSONDecodeError, TypeError):
                contrib, counts = {}, {}
            if contrib:
                lines.append("| Strategy | PnL contribution % | Trades taken |")
                lines.append("|----------|---------------------|--------------|")
                for strat, pct in sorted(contrib.items(), key=lambda x: -abs(x[1])):
                    lines.append(f"| {strat} | {pct:.1f}% | {counts.get(strat, 0)} |")
            else:
                lines.append("_No trades taken._")
            lines.append("")

    lines.extend(["## Strategy Daily Correlation", ""])
    if not summary.empty and "strategy_daily_correlation" in summary.columns:
        shown: set[str] = set()
        for _, row in summary.iterrows():
            key = f"{row.get('portfolio_name')}_{row.get('risk_per_trade_pct')}"
            if key in shown:
                continue
            shown.add(key)
            try:
                corr = json.loads(row["strategy_daily_correlation"])
            except (json.JSONDecodeError, TypeError):
                corr = {}
            if not corr:
                continue
            lines.append(f"### {row.get('portfolio_name')} (first risk row)")
            lines.append("")
            strategies = list(corr.keys())
            lines.append("| | " + " | ".join(strategies) + " |")
            lines.append("|" + "---|" * (len(strategies) + 1))
            for s in strategies:
                cells = [f"{corr[s].get(col, 0):.2f}" for col in strategies]
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")
            break

    lines.extend(["## Monte Carlo by Portfolio and Risk", ""])
    if not mc.empty:
        cols = [
            "portfolio_name", "risk_per_trade_pct", "pass_rate", "fail_rate",
            "daily_fail_rate", "total_fail_rate", "incomplete_rate",
            "median_final_return_pct", "p5_final_return_pct", "p95_final_return_pct",
            "worst_drawdown_pct", "recommended",
        ]
        show = [c for c in cols if c in mc.columns]
        lines.append("| " + " | ".join(show) + " |")
        lines.append("| " + " | ".join(["---"] * len(show)) + " |")
        for _, row in mc.iterrows():
            cells = []
            for c in show:
                val = row[c]
                if isinstance(val, float) and (
                    c.endswith("_rate") or c in ("pass_rate", "fail_rate", "incomplete_rate")
                ):
                    cells.append(f"{val:.1%}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend([
        "## Charts",
        "",
        "![Portfolio equity curve](assets/portfolio_equity_curve.png)",
        "",
        "![Portfolio Monte Carlo distribution](assets/portfolio_monte_carlo_distribution.png)",
        "",
        "## Rejection Reasons",
        "",
        "- **negative_expectancy**: average R on taken trades is not positive",
        "- **monte_carlo_pass_rate_below_threshold**: MC pass rate below configured minimum",
        "- **daily_failure_rate_too_high**: daily drawdown rule fails too often in MC",
        "- **total_failure_rate_too_high**: total drawdown rule fails too often in MC",
        "- **insufficient_trade_count**: fewer than 20 trades taken in simulation",
        "- **incomplete_too_often**: challenge rarely reaches pass/fail outcome",
        "- **no_oos_test_available**: only in-sample data, no OOS test split",
        "",
        "---",
        "*Research only. Not financial advice. No live trading.*",
        "",
    ])

    report_path = output_dir / f"portfolio_sim_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
