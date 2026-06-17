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
    is_multiphase = meta.get("simulation_mode") == "multi_phase_bootcamp" or cfg.is_multiphase

    lines = [
        "# AUTA Prop-Firm Simulation Report",
        "",
        f"Generated: {now}",
        "",
        f"- Simulation mode: **{'The5ers Bootcamp (multi-phase)' if is_multiphase else 'Single-phase challenge'}**",
        f"- Program: {meta.get('program_name', cfg.name)}",
        "",
    ]

    if is_multiphase:
        lines.extend(_multiphase_executive_summary(cfg, meta, summary))
        lines.extend(_multiphase_results_table(summary))
        lines.extend(_multiphase_mc_table(mc))
    else:
        lines.extend([
            "## Executive Summary (Single-Phase Challenge)",
            "",
            f"- Starting balance: ${cfg.account.starting_balance:,.0f}",
            f"- Profit target: {cfg.account.profit_target_pct}%",
            f"- Max total loss: {cfg.account.max_total_loss_pct}%",
            f"- Max daily loss: {cfg.account.max_daily_loss_pct}%",
            f"- Recommended max risk per trade: **{meta.get('recommended_max_risk_pct')}%**",
            f"- OOS expectancy (R): **{meta.get('oos_expectancy_r')}**",
            "",
        ])
        if not summary.empty and "verdict" in summary.columns:
            for label, count in summary["verdict"].value_counts().items():
                lines.append(f"- {label}: {count} row(s)")
            lines.append("")
        lines.extend(_single_phase_results_table(summary))
        lines.extend(_single_phase_mc_table(mc))

    lines.extend([
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


def _multiphase_executive_summary(cfg: PropFirmConfig, meta: dict, summary: pd.DataFrame) -> list[str]:
    lines = [
        "## Executive Summary (The5ers Bootcamp Multi-Phase)",
        "",
        f"- Currency: {cfg.program.currency if cfg.program else 'GBP'}",
        f"- Leverage: {cfg.program.leverage if cfg.program else 'n/a'}",
        f"- Bootcamp steps: {len(cfg.bootcamp_phases())}",
        f"- Recommended risk per trade: **{meta.get('recommended_risk_per_trade')}%**",
        f"- OOS expectancy (R): **{meta.get('oos_expectancy_r')}**",
        "",
    ]
    for phase in cfg.phases:
        lines.append(
            f"- **{phase.name}**: start {phase.starting_balance:,.0f}, "
            f"target {phase.profit_target_pct}%, max loss {phase.max_total_loss_pct}%"
            + (f", daily pause {phase.daily_pause_pct}%" if phase.daily_pause_pct else ", no daily pause")
        )
    lines.append("")
    if not summary.empty and "bootcamp_evaluation_passed" in summary.columns:
        passed = summary["bootcamp_evaluation_passed"].sum()
        lines.append(f"- Deterministic bootcamp passes (rows): {passed}/{len(summary)}")
        lines.append("")
    return lines


def _multiphase_results_table(summary: pd.DataFrame) -> list[str]:
    lines = ["## Bootcamp Phase Results (Deterministic Run)", ""]
    if summary.empty:
        return lines + ["_No results._", ""]
    cols = [
        "trade_split", "risk_per_trade_pct", "max_open_trades", "max_trades_per_day",
        "phase_passed_step_1", "phase_passed_step_2", "phase_passed_step_3",
        "bootcamp_evaluation_passed", "funded_phase_passed_or_survived",
        "phase_failed_name", "phase_failure_reason",
        "total_trades_taken", "verdict", "rejection_reason",
    ]
    show = [c for c in cols if c in summary.columns]
    lines.append("| " + " | ".join(show) + " |")
    lines.append("| " + " | ".join(["---"] * len(show)) + " |")
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in show) + " |")
    lines.append("")
    return lines


def _multiphase_mc_table(mc: pd.DataFrame) -> list[str]:
    lines = ["## Monte Carlo (Bootcamp Pass Rates by Step)", ""]
    if mc.empty:
        return lines + ["_No Monte Carlo results._", ""]
    cols = [
        "trade_split", "risk_per_trade_pct", "bootcamp_pass_rate",
        "step_1_pass_rate", "step_2_pass_rate", "step_3_pass_rate",
        "funded_survival_rate", "funded_pass_rate",
        "fail_rate_by_phase", "median_trades_to_pass_by_phase",
        "incomplete_rate", "recommended",
    ]
    show = [c for c in cols if c in mc.columns]
    lines.append("| " + " | ".join(show) + " |")
    lines.append("| " + " | ".join(["---"] * len(show)) + " |")
    for _, row in mc.iterrows():
        cells = []
        for c in show:
            val = row[c]
            if isinstance(val, float) and c.endswith("_rate"):
                cells.append(f"{val:.1%}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _single_phase_results_table(summary: pd.DataFrame) -> list[str]:
    lines = ["## Simulation Results by Risk Level", ""]
    if summary.empty:
        return lines + ["_No results._", ""]
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
    return lines


def _single_phase_mc_table(mc: pd.DataFrame) -> list[str]:
    lines = ["## Monte Carlo by Risk Level", ""]
    if mc.empty:
        return lines + ["_No Monte Carlo results._", ""]
    cols = [
        "trade_split", "risk_per_trade_pct", "pass_rate", "fail_rate",
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
            if isinstance(val, float) and (c.endswith("_rate") or c in ("pass_rate", "fail_rate")):
                cells.append(f"{val:.1%}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines
