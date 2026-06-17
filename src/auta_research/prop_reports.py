"""Markdown reports for prop-firm simulation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from auta_research.config import PropFirmConfig


def generate_prop_sim_report(
    summary: pd.DataFrame,
    mc: pd.DataFrame,
    meta: dict[str, Any],
    cfg: PropFirmConfig,
    output_dir: Path,
    assets_dir: Path,
) -> Path:
    """Write prop-firm simulation Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# AUTA Prop-Firm Simulation Report",
        "",
        f"Generated: {now}",
        "",
        "## Executive Summary",
        "",
        f"- Starting balance: ${cfg.account.starting_balance:,.0f}",
        f"- Profit target: {cfg.account.profit_target_pct}%",
        f"- Max total loss: {cfg.account.max_total_loss_pct}%",
        f"- Max daily loss: {cfg.account.max_daily_loss_pct}%",
        f"- Recommended max risk per trade: **{meta.get('recommended_max_risk_pct')}%**",
        f"- OOS expectancy (R): **{meta.get('oos_expectancy_r')}**",
        "",
    ]

    if not summary.empty:
        verdicts = summary["verdict"].value_counts() if "verdict" in summary.columns else {}
        for label, count in verdicts.items():
            lines.append(f"- {label}: {count} row(s)")
        lines.append("")

    lines.extend(["## Simulation Results by Risk Level", ""])
    if not summary.empty:
        cols = [
            "trade_split", "risk_per_trade_pct", "status", "passed",
            "final_return_pct", "max_drawdown_pct", "max_daily_loss_pct",
            "win_rate", "average_r", "total_trades_taken", "verdict", "rejection_reason",
        ]
        show = [c for c in cols if c in summary.columns]
        lines.append("| " + " | ".join(show) + " |")
        lines.append("| " + " | ".join(["---"] * len(show)) + " |")
        for _, row in summary.iterrows():
            lines.append(
                "| " + " | ".join(
                    f"{row[c]:.2%}" if c == "win_rate" and isinstance(row[c], float) else str(row[c])
                    for c in show
                ) + " |"
            )
        lines.append("")

    lines.extend(["## Monte Carlo by Risk Level", ""])
    if not mc.empty:
        cols = [
            "trade_split", "risk_per_trade_pct", "pass_rate", "fail_rate",
            "daily_fail_rate", "total_fail_rate", "incomplete_rate",
            "median_final_return_pct",
            "p5_final_return_pct", "p95_final_return_pct", "worst_drawdown_pct",
            "recommended",
        ]
        show = [c for c in cols if c in mc.columns]
        lines.append("| " + " | ".join(show) + " |")
        lines.append("| " + " | ".join(["---"] * len(show)) + " |")
        for _, row in mc.iterrows():
            cells = []
            for c in show:
                val = row[c]
                if isinstance(val, float) and c.endswith("_rate") or c in (
                    "pass_rate", "fail_rate", "daily_fail_rate", "total_fail_rate",
                ):
                    cells.append(f"{val:.1%}")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend([
        "## Win Rate and Expectancy by TP (trade log)",
        "",
        "See main research report for TP breakdown if trade log includes r_multiple_target.",
        "",
        "## Charts",
        "",
        "![Prop equity curve](assets/prop_equity_curve.png)",
        "",
        "![Monte Carlo distribution](assets/prop_monte_carlo_distribution.png)",
        "",
        "## Verdict Definitions",
        "",
        "- **rejected**: negative expectancy or unacceptable failure profile",
        "- **promising but too risky**: edge exists but drawdown rules fail too often",
        "- **passes in-sample only**: passes on train/full but not validated OOS",
        "- **passes OOS candidate**: positive OOS with acceptable MC pass rate",
        "- **demo forward-test candidate**: OOS edge, low failure rates, not cluster-dependent",
        "",
        "---",
        "*Research only. Not financial advice. No live trading.*",
        "",
    ])

    report_path = output_dir / f"prop_sim_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
