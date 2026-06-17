"""Comparison reports for fixed strategy candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from auta_research.config import PropFirmConfig
from auta_research.prop_sim import discover_trade_splits, run_prop_simulation


def run_prop_sim_for_candidate(
    candidate_dir: Path,
    prop_cfg: PropFirmConfig,
    trades_file: str = "trades_test.csv",
    *,
    progress: Any | None = None,
    progress_task: int | None = None,
    chart_samples: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run prop-sim on a candidate trade log and save under candidate_dir/prop_sim."""
    trades_path = candidate_dir / trades_file
    if not trades_path.exists():
        trades_path = candidate_dir / "trades.csv"
    sources = discover_trade_splits(trades_path, candidate_dir)
    out_dir = candidate_dir / "prop_sim"
    return run_prop_simulation(
        prop_cfg,
        sources,
        out_dir,
        progress=progress,
        progress_task=progress_task,
        chart_samples=chart_samples,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prop_metrics_from_candidate(candidate_dir: Path) -> dict[str, Any]:
    """Read prop-sim outputs saved under candidate_dir/prop_sim."""
    prop_dir = candidate_dir / "prop_sim"
    meta = _load_json(prop_dir / "prop_sim_meta.json")
    out: dict[str, Any] = {
        "prop_pass_rate": None,
        "prop_fail_rate": None,
        "prop_recommended_risk_pct": meta.get("recommended_max_risk_pct"),
        "prop_verdict": None,
    }

    mc_path = prop_dir / "prop_sim_monte_carlo.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        test_mc = mc[mc["trade_split"] == "test"] if "trade_split" in mc.columns else mc
        if test_mc.empty:
            test_mc = mc
        if not test_mc.empty:
            best = test_mc.loc[test_mc["pass_rate"].idxmax()]
            out["prop_pass_rate"] = float(best["pass_rate"])
            out["prop_fail_rate"] = float(best["fail_rate"])
            if out["prop_recommended_risk_pct"] is None and "recommended" in test_mc.columns:
                rec = test_mc[test_mc["recommended"] == True]  # noqa: E712
                if not rec.empty:
                    out["prop_recommended_risk_pct"] = float(rec.iloc[0]["risk_per_trade_pct"])

    summary_path = prop_dir / "prop_sim_summary.csv"
    if summary_path.exists():
        ps = pd.read_csv(summary_path)
        test_ps = ps[ps["trade_split"] == "test"] if "trade_split" in ps.columns else ps
        if not test_ps.empty and "verdict" in test_ps.columns:
            out["prop_verdict"] = str(test_ps.iloc[0]["verdict"])

    return out


def collect_candidate_row(candidate_dir: Path) -> dict[str, Any]:
    """Collect comparison metrics for one candidate directory."""
    name = candidate_dir.name
    test_summary = _load_json(candidate_dir / "summary_test.json").get("metrics", {})
    validation = _load_json(candidate_dir / "validation_summary.json")
    prop = _prop_metrics_from_candidate(candidate_dir)

    return {
        "candidate": name,
        "test_trades": test_summary.get("trades", 0),
        "test_expectancy_r": test_summary.get("expectancy_r", 0),
        "test_win_rate": test_summary.get("win_rate", 0),
        "test_profit_factor": test_summary.get("profit_factor", 0),
        "test_max_drawdown_r": test_summary.get("max_drawdown_r", 0),
        "train_expectancy_r": validation.get("train_metrics", {}).get("expectancy_r", 0),
        "degradation_pct": validation.get("degradation_pct", 0),
        "oos_positive": validation.get("oos_positive", False),
        **prop,
    }


def _rank_table(df: pd.DataFrame, sort_col: str, ascending: bool, title: str) -> list[str]:
    if df.empty or sort_col not in df.columns:
        return [f"## {title}", "", "_No data._", ""]
    ranked = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    lines = [
        f"## {title}",
        "",
        "| Rank | Candidate | Value |",
        "|------|-----------|-------|",
    ]
    for i, row in ranked.iterrows():
        val = row[sort_col]
        if isinstance(val, float):
            if "rate" in sort_col or sort_col.endswith("_win_rate"):
                val_str = f"{val:.1%}"
            else:
                val_str = f"{val:.3f}"
        elif isinstance(val, bool):
            val_str = "yes" if val else "no"
        else:
            val_str = str(val)
        lines.append(f"| {i + 1} | {row['candidate']} | {val_str} |")
    lines.append("")
    return lines


def generate_fixed_candidate_comparison(
    candidates_root: Path,
    reports_dir: Path,
) -> Path:
    """Build ranked comparison report across fixed candidate directories."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if candidates_root.is_dir():
        for child in sorted(candidates_root.iterdir()):
            if child.is_dir() and (child / "summary_test.json").exists():
                rows.append(collect_candidate_row(child))

    df = pd.DataFrame(rows)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Fixed Candidate Comparison",
        "",
        f"Generated: {now}",
        "",
    ]

    if df.empty:
        lines.append("No fixed candidate results found.")
    else:
        primary = df.sort_values(
            by=["oos_positive", "test_expectancy_r", "prop_pass_rate"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        primary.insert(0, "rank", range(1, len(primary) + 1))

        lines.extend([
            "## Overview (OOS positive, then test expectancy, then prop pass rate)",
            "",
            "| Rank | Candidate | Test Exp | Test WR | Test PF | Test DD | "
            "OOS+ | Prop Pass | Prop Fail | Rec Risk % | Verdict |",
            "|------|-----------|----------|---------|---------|---------|"
            "------|-----------|-----------|------------|---------|",
        ])
        for _, r in primary.iterrows():
            lines.append(
                f"| {int(r['rank'])} | {r['candidate']} | "
                f"{r.get('test_expectancy_r', 0):.3f} | "
                f"{r.get('test_win_rate', 0):.1%} | "
                f"{r.get('test_profit_factor', 0):.2f} | "
                f"{r.get('test_max_drawdown_r', 0):.2f} | "
                f"{'yes' if r.get('oos_positive') else 'no'} | "
                f"{(r.get('prop_pass_rate') or 0):.1%} | "
                f"{(r.get('prop_fail_rate') or 0):.1%} | "
                f"{r.get('prop_recommended_risk_pct', '')} | "
                f"{r.get('prop_verdict', '')} |"
            )
        lines.append("")

        lines.extend(_rank_table(df, "test_expectancy_r", False, "Ranked by test expectancy"))
        lines.extend(_rank_table(df, "test_profit_factor", False, "Ranked by test profit factor"))
        lines.extend(_rank_table(df, "test_max_drawdown_r", True, "Ranked by test drawdown (lower is better)"))
        lines.extend(_rank_table(df, "prop_pass_rate", False, "Ranked by prop-sim pass rate"))
        lines.extend(_rank_table(df, "prop_fail_rate", True, "Ranked by prop-sim failure rate (lower is better)"))
        lines.extend(_rank_table(df, "prop_recommended_risk_pct", False, "Ranked by recommended risk per trade"))

    report_path = reports_dir / f"fixed_candidate_comparison_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
