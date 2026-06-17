"""Portfolio risk-parameter sensitivity analysis for prop-firm simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from auta_research.config import PortfolioCandidatesConfig, PropFirmConfig, resolve_path
from auta_research.portfolio_sim import load_portfolio_trades
from auta_research.prop_multiphase import (
    RiskSettings,
    multiphase_mc_to_dict,
    multiphase_sim_to_dict,
    run_multiphase_monte_carlo,
    simulate_multiphase_program,
)
from auta_research.prop_sim import (
    MonteCarloResult,
    assign_verdict,
    run_monte_carlo,
    simulate_account,
    sim_to_dict,
    verdict_to_dict,
)
from auta_research.prop_multiphase import _multiphase_to_sim_result


def run_portfolio_sensitivity(
    portfolio_cfg: PortfolioCandidatesConfig,
    prop_cfg: PropFirmConfig,
    project_root: Path,
    *,
    output_root: str = "data/results/portfolio_sensitivity",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sweep risk grid across portfolios; rank by pass/bootcamp rate."""
    out_dir = project_root / output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    is_multiphase = prop_cfg.is_multiphase

    for portfolio in portfolio_cfg.portfolios:
        merged = load_portfolio_trades(
            portfolio.trades,
            project_root,
            dedupe_same_signal=portfolio_cfg.dedupe_same_signal,
        )
        oos_exp = float(merged["r_result"].mean()) if len(merged) > 0 else 0.0

        for risk_dict in prop_cfg.risk.iter_risk_settings(full_grid=True):
            risk_pct = float(risk_dict["risk_per_trade_pct"])
            row: dict[str, Any] = {
                "portfolio_name": portfolio.name,
                "simulation_mode": "multi_phase_bootcamp" if is_multiphase else "single_phase_challenge",
                **{k: v for k, v in risk_dict.items()},
                "merged_trades": len(merged),
            }

            if is_multiphase:
                risk = RiskSettings.from_dict(risk_dict)
                sim = simulate_multiphase_program(merged, prop_cfg, risk, portfolio.name)
                row.update(multiphase_sim_to_dict(sim))
                if prop_cfg.monte_carlo.enabled:
                    mc = run_multiphase_monte_carlo(merged, prop_cfg, risk, portfolio.name)
                    row.update(multiphase_mc_to_dict(mc))
                    mc_adapter = MonteCarloResult(
                        risk_per_trade_pct=risk_pct,
                        trade_split=portfolio.name,
                        runs=mc.runs,
                        pass_rate=mc.bootcamp_pass_rate,
                        fail_rate=mc.fail_rate,
                        daily_fail_rate=mc.daily_fail_rate,
                        total_fail_rate=mc.total_fail_rate,
                        incomplete_rate=mc.incomplete_rate,
                        median_final_return_pct=mc.median_final_return_pct,
                        p5_final_return_pct=mc.p5_final_return_pct,
                        p95_final_return_pct=mc.p95_final_return_pct,
                        worst_drawdown_pct=mc.worst_drawdown_pct,
                    )
                else:
                    mc_adapter = None
                sim_adapter = _multiphase_to_sim_result(sim, portfolio.name, risk_pct)
            else:
                risk_settings = RiskSettings.from_dict(risk_dict)
                sim_adapter = simulate_account(merged, prop_cfg, risk_pct, portfolio.name)
                row.update(sim_to_dict(sim_adapter))
                row["max_open_trades"] = risk_settings.max_open_trades
                row["max_trades_per_day"] = risk_settings.max_trades_per_day
                row["simulation_mode"] = "single_phase_challenge"
                mc_adapter = None
                if prop_cfg.monte_carlo.enabled:
                    mc_adapter = run_monte_carlo(merged, prop_cfg, risk_pct, portfolio.name)
                    row.update(mc_adapter.__dict__)

            verdict = assign_verdict(sim_adapter, mc_adapter, prop_cfg, oos_exp)
            row.update(verdict_to_dict(verdict))
            rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "bootcamp_pass_rate" if is_multiphase and "bootcamp_pass_rate" in df.columns else "pass_rate"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    df.to_csv(out_dir / "portfolio_sensitivity.csv", index=False)
    meta = {
        "simulation_mode": "multi_phase_bootcamp" if is_multiphase else "single_phase_challenge",
        "program_name": prop_cfg.name,
        "portfolios": [p.name for p in portfolio_cfg.portfolios],
        "risk_combinations": len(prop_cfg.risk.iter_risk_settings(full_grid=True)),
    }
    with open(out_dir / "portfolio_sensitivity_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df, meta
