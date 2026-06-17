"""Portfolio-level prop-firm simulation across multiple strategy candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auta_research.config import PortfolioCandidatesConfig, PropFirmConfig, resolve_path
from auta_research.prop_sim import (
    MonteCarloResult,
    OutcomeStatus,
    SimResult,
    _prepare_trades,
    _risk_base,
    _trade_r,
    assign_verdict,
    load_trade_log,
    recommend_max_risk,
    sim_to_dict,
    verdict_to_dict,
)


def strategy_name_from_path(path: Path) -> str:
    """Derive strategy label from candidate directory name."""
    if path.parent.name in ("fixed_candidates", "results", "data"):
        return path.stem
    return path.parent.name


def _to_timestamp(val: Any) -> pd.Timestamp:
    if pd.isna(val):
        return pd.Timestamp("1970-01-01", tz="UTC")
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class PortfolioSimResult(SimResult):
    """Portfolio simulation with per-strategy attribution."""

    portfolio_name: str = ""
    strategy_contribution_pct: dict[str, float] = field(default_factory=dict)
    strategy_trade_counts: dict[str, int] = field(default_factory=dict)
    strategy_daily_correlation: dict[str, dict[str, float]] = field(default_factory=dict)
    merged_trade_count: int = 0
    taken_trade_count: int = 0
    skipped_open_limit: int = 0


def load_portfolio_trades(
    trade_paths: list[str | Path],
    project_root: Path,
    *,
    dedupe_same_signal: bool = True,
) -> pd.DataFrame:
    """Load, tag, merge, and optionally dedupe portfolio trade logs."""
    frames: list[pd.DataFrame] = []
    for raw in trade_paths:
        path = resolve_path(raw, project_root)
        df = load_trade_log(path)
        df = df.copy()
        df["strategy_name"] = strategy_name_from_path(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    sort_col = "signal_time"
    merged["_signal_ts"] = merged["signal_time"].map(_to_timestamp)
    merged = merged.sort_values("_signal_ts").reset_index(drop=True)

    if dedupe_same_signal:
        sym = merged["symbol"].astype(str) if "symbol" in merged.columns else ""
        merged["_dedupe_key"] = sym + "|" + merged["signal_time"].astype(str)
        merged = merged.drop_duplicates(subset=["_dedupe_key"], keep="first")
        merged = merged.drop(columns=["_dedupe_key"], errors="ignore")

    merged = merged.drop(columns=["_signal_ts"], errors="ignore")
    return _prepare_trades(merged.sort_values("signal_time").reset_index(drop=True))


def _strategy_daily_correlation(taken: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pearson correlation of daily R sums across strategies."""
    if taken.empty or "strategy_name" not in taken.columns:
        return {}
    daily = (
        taken.groupby(["_trade_day", "strategy_name"], as_index=False)["r_result"]
        .sum()
        .pivot(index="_trade_day", columns="strategy_name", values="r_result")
        .fillna(0.0)
    )
    if daily.shape[1] < 2:
        return {}
    corr = daily.corr()
    return {
        row: {col: round(float(corr.loc[row, col]), 4) for col in corr.columns}
        for row in corr.index
    }


def simulate_portfolio_account(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_per_trade_pct: float,
    portfolio_name: str = "portfolio",
) -> PortfolioSimResult:
    """Chronological portfolio simulation with max_open_trades and account rules."""
    trades = _prepare_trades(trades)
    acct = cfg.account
    risk_cfg = cfg.risk
    starting = acct.starting_balance
    balance = starting
    profit_target = starting * (1 + acct.profit_target_pct / 100)
    floor_balance = starting * (1 - acct.max_total_loss_pct / 100)
    max_daily_loss_amt = starting * acct.max_daily_loss_pct / 100
    stop_daily_loss_amt = starting * risk_cfg.stop_after_daily_loss_pct / 100
    max_open = risk_cfg.max_open_trades

    status: OutcomeStatus = "incomplete"
    passed = False
    failed_daily = False
    failed_total = False
    incomplete = True

    equity_curve = [balance]
    daily_pnl: dict[str, float] = {}
    trading_days_set: set[str] = set()
    strategy_pnl: dict[str, float] = {}
    strategy_counts: dict[str, int] = {}
    taken_rows: list[pd.Series] = []

    current_day = ""
    day_pnl = 0.0
    trades_today = 0
    day_stopped = False
    consecutive_losses = 0
    days_to_pass: int | None = None
    skipped_open_limit = 0

    open_exits: list[pd.Timestamp] = []

    taken_r: list[float] = []
    day_pnls_list: list[float] = []
    peak_balance = balance
    max_drawdown_pct = 0.0
    max_daily_loss_pct = 0.0

    sort_col = "entry_time" if "entry_time" in trades.columns else "signal_time"

    for _, row in trades.iterrows():
        if passed or failed_daily or failed_total:
            break

        entry_ts = _to_timestamp(row.get("entry_time", row.get("signal_time")))
        exit_raw = row.get("exit_time")
        exit_ts = _to_timestamp(exit_raw) if pd.notna(exit_raw) else entry_ts

        open_exits = [e for e in open_exits if e > entry_ts]
        if len(open_exits) >= max_open:
            skipped_open_limit += 1
            continue

        day = str(row["_trade_day"])
        if day != current_day:
            if current_day:
                day_pnls_list.append(day_pnl)
                if day_pnl < 0:
                    max_daily_loss_pct = max(max_daily_loss_pct, -day_pnl / starting * 100)
            current_day = day
            day_pnl = daily_pnl.get(day, 0.0)
            trades_today = 0
            day_stopped = False

        if acct.max_trading_days is not None and len(trading_days_set) > acct.max_trading_days:
            break

        if trades_today >= risk_cfg.max_trades_per_day:
            continue
        if day_stopped:
            continue
        if consecutive_losses >= risk_cfg.stop_after_consecutive_losses:
            continue

        r = _trade_r(row, cfg)
        risk_amt = _risk_base(balance, starting, risk_cfg.compound) * risk_per_trade_pct / 100
        pnl = r * risk_amt

        balance += pnl
        day_pnl += pnl
        daily_pnl[day] = day_pnl
        trades_today += 1
        trading_days_set.add(day)
        taken_r.append(r)
        equity_curve.append(balance)
        open_exits.append(exit_ts)

        strat = str(row.get("strategy_name", "unknown"))
        strategy_pnl[strat] = strategy_pnl.get(strat, 0.0) + pnl
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        taken_rows.append(row)

        if pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        peak_balance = max(peak_balance, balance)
        dd_pct = (peak_balance - balance) / starting * 100
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)

        if day_pnl <= -stop_daily_loss_amt:
            day_stopped = True

        if day_pnl <= -max_daily_loss_amt:
            status = "failed_daily_drawdown"
            failed_daily = True
            incomplete = False
            break

        if balance <= floor_balance:
            status = "failed_total_drawdown"
            failed_total = True
            incomplete = False
            break

        if balance >= profit_target and len(trading_days_set) >= acct.min_trading_days:
            status = "passed"
            passed = True
            incomplete = False
            if days_to_pass is None:
                days_to_pass = len(trading_days_set)
            break

    if current_day:
        day_pnls_list.append(day_pnl)
        if day_pnl < 0:
            max_daily_loss_pct = max(max_daily_loss_pct, -day_pnl / starting * 100)

    if not passed and not failed_daily and not failed_total:
        status = "incomplete"
        incomplete = True

    wins = [x for x in taken_r if x > 0]
    losses = [x for x in taken_r if x < 0]
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    streak = max_streak = 0
    for val in taken_r:
        if val < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    total_pnl = sum(strategy_pnl.values())
    contrib_pct = {
        k: round(v / total_pnl * 100, 2) if total_pnl != 0 else 0.0
        for k, v in strategy_pnl.items()
    }

    taken_df = pd.DataFrame(taken_rows) if taken_rows else pd.DataFrame()
    corr = _strategy_daily_correlation(taken_df)

    largest_win_day = max(day_pnls_list) if day_pnls_list else 0.0
    largest_loss_day = min(day_pnls_list) if day_pnls_list else 0.0

    return PortfolioSimResult(
        risk_per_trade_pct=risk_per_trade_pct,
        trade_split=portfolio_name,
        status=status,
        passed=passed,
        failed_daily_drawdown=failed_daily,
        failed_total_drawdown=failed_total,
        incomplete=incomplete,
        final_balance=round(balance, 2),
        final_return_pct=round((balance - starting) / starting * 100, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        max_daily_loss_pct=round(max_daily_loss_pct, 4),
        days_to_pass=days_to_pass,
        trading_days=len(trading_days_set),
        total_trades_taken=len(taken_r),
        win_rate=round(len(wins) / len(taken_r), 4) if taken_r else 0.0,
        average_r=round(float(np.mean(taken_r)), 4) if taken_r else 0.0,
        profit_factor=round(min(pf, 999.0), 4),
        longest_losing_streak=max_streak,
        largest_winning_day=round(largest_win_day, 2),
        largest_losing_day=round(largest_loss_day, 2),
        equity_curve=equity_curve,
        daily_pnl=daily_pnl,
        portfolio_name=portfolio_name,
        strategy_contribution_pct=contrib_pct,
        strategy_trade_counts=strategy_counts,
        strategy_daily_correlation=corr,
        merged_trade_count=len(trades),
        taken_trade_count=len(taken_r),
        skipped_open_limit=skipped_open_limit,
    )


def run_portfolio_monte_carlo(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_per_trade_pct: float,
    portfolio_name: str,
) -> MonteCarloResult:
    """Monte Carlo on merged portfolio trade sequence."""
    mc = cfg.monte_carlo
    runs = mc.runs
    rng = np.random.default_rng(42)

    if trades.empty:
        return MonteCarloResult(
            risk_per_trade_pct=risk_per_trade_pct,
            trade_split=portfolio_name,
            runs=0,
            pass_rate=0.0,
            fail_rate=1.0,
            daily_fail_rate=0.0,
            total_fail_rate=0.0,
            incomplete_rate=1.0,
            median_final_return_pct=0.0,
            p5_final_return_pct=0.0,
            p95_final_return_pct=0.0,
            worst_drawdown_pct=0.0,
        )

    statuses: list[str] = []
    returns: list[float] = []
    drawdowns: list[float] = []

    for _ in range(runs):
        if mc.bootstrap_with_replacement:
            idx = rng.integers(0, len(trades), size=len(trades))
            sample = trades.iloc[idx].copy()
        elif mc.shuffle_trades:
            sample = trades.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        else:
            sample = trades

        sample = sample.reset_index(drop=True)
        sim = simulate_portfolio_account(sample, cfg, risk_per_trade_pct, portfolio_name)
        statuses.append(sim.status)
        returns.append(sim.final_return_pct)
        drawdowns.append(sim.max_drawdown_pct)

    statuses_arr = np.array(statuses)
    pass_rate = float((statuses_arr == "passed").mean())
    daily_fail = float((statuses_arr == "failed_daily_drawdown").mean())
    total_fail = float((statuses_arr == "failed_total_drawdown").mean())
    incomplete = float((statuses_arr == "incomplete").mean())

    return MonteCarloResult(
        risk_per_trade_pct=risk_per_trade_pct,
        trade_split=portfolio_name,
        runs=runs,
        pass_rate=round(pass_rate, 4),
        fail_rate=round(daily_fail + total_fail, 4),
        daily_fail_rate=round(daily_fail, 4),
        total_fail_rate=round(total_fail, 4),
        incomplete_rate=round(incomplete, 4),
        median_final_return_pct=round(float(np.median(returns)), 4),
        p5_final_return_pct=round(float(np.percentile(returns, 5)), 4),
        p95_final_return_pct=round(float(np.percentile(returns, 95)), 4),
        worst_drawdown_pct=round(float(np.max(drawdowns)), 4),
    )


def portfolio_sim_to_dict(sim: PortfolioSimResult) -> dict[str, Any]:
    """Serialize portfolio result for CSV output."""
    d = sim_to_dict(sim)
    d["portfolio_name"] = sim.portfolio_name
    d["merged_trade_count"] = sim.merged_trade_count
    d["taken_trade_count"] = sim.taken_trade_count
    d["skipped_open_limit"] = sim.skipped_open_limit
    d["strategy_contribution_pct"] = json.dumps(sim.strategy_contribution_pct)
    d["strategy_trade_counts"] = json.dumps(sim.strategy_trade_counts)
    d["strategy_daily_correlation"] = json.dumps(sim.strategy_daily_correlation)
    return d


def run_portfolio_simulation(
    portfolio_cfg: PortfolioCandidatesConfig,
    prop_cfg: PropFirmConfig,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run portfolio prop simulation for all configured portfolios."""
    if prop_cfg.is_multiphase:
        return _run_multiphase_portfolio_simulation(portfolio_cfg, prop_cfg, project_root)
    return _run_single_phase_portfolio_simulation(portfolio_cfg, prop_cfg, project_root)


def _run_single_phase_portfolio_simulation(
    portfolio_cfg: PortfolioCandidatesConfig,
    prop_cfg: PropFirmConfig,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out_dir = project_root / portfolio_cfg.output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    mc_objects: list[MonteCarloResult] = []
    equity_by_key: dict[str, list[float]] = {}
    mc_returns_by_key: dict[str, list[float]] = {}
    merged_trades_by_portfolio: dict[str, pd.DataFrame] = {}

    for portfolio in portfolio_cfg.portfolios:
        merged = load_portfolio_trades(
            portfolio.trades,
            project_root,
            dedupe_same_signal=portfolio_cfg.dedupe_same_signal,
        )
        merged_trades_by_portfolio[portfolio.name] = merged
        oos_expectancy = float(merged["r_result"].mean()) if len(merged) > 0 else 0.0

        for risk_pct in prop_cfg.risk.risk_per_trade_pct_values:
            sim = simulate_portfolio_account(merged, prop_cfg, risk_pct, portfolio.name)
            row = portfolio_sim_to_dict(sim)
            mc_result: MonteCarloResult | None = None

            if prop_cfg.monte_carlo.enabled:
                mc_result = run_portfolio_monte_carlo(merged, prop_cfg, risk_pct, portfolio.name)
                mc_objects.append(mc_result)
                mc_row = mc_result.__dict__.copy()
                mc_row["portfolio_name"] = portfolio.name
                mc_rows.append(mc_row)

                if prop_cfg.monte_carlo.runs > 0:
                    key = f"{portfolio.name}_{risk_pct}"
                    mc_returns_by_key[key] = _portfolio_mc_samples(
                        merged, prop_cfg, risk_pct, portfolio.name, min(200, prop_cfg.monte_carlo.runs)
                    )

            verdict_result = assign_verdict(
                sim,
                mc_result,
                prop_cfg,
                oos_expectancy,
            )
            row.update(verdict_to_dict(verdict_result))
            if mc_result:
                row["mc_pass_rate"] = mc_result.pass_rate
                row["mc_fail_rate"] = mc_result.fail_rate
                row["mc_total_fail_rate"] = mc_result.total_fail_rate
                row["mc_daily_fail_rate"] = mc_result.daily_fail_rate
                row["mc_incomplete_rate"] = mc_result.incomplete_rate

            summary_rows.append(row)
            equity_by_key[f"{portfolio.name}_{risk_pct}"] = sim.equity_curve

    summary_df = pd.DataFrame(summary_rows)
    mc_df = pd.DataFrame(mc_rows)

    recommended = recommend_max_risk(mc_objects, prop_cfg)
    if recommended is not None and not mc_df.empty:
        mc_df.loc[mc_df["risk_per_trade_pct"] == recommended, "recommended"] = True
        if not summary_df.empty:
            summary_df.loc[summary_df["risk_per_trade_pct"] == recommended, "recommended"] = True

    meta = {
        "recommended_max_risk_pct": recommended,
        "portfolios": [p.name for p in portfolio_cfg.portfolios],
        "equity_curves": equity_by_key,
        "mc_return_samples": mc_returns_by_key,
        "merged_trade_counts": {k: len(v) for k, v in merged_trades_by_portfolio.items()},
    }

    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)
    if not mc_df.empty:
        mc_df.to_csv(out_dir / "portfolio_monte_carlo.csv", index=False)

    with open(out_dir / "portfolio_meta.json", "w", encoding="utf-8") as f:
        slim = {k: v for k, v in meta.items() if k not in ("equity_curves", "mc_return_samples")}
        json.dump(slim, f, indent=2, default=str)

    return summary_df, mc_df, meta


def _run_multiphase_portfolio_simulation(
    portfolio_cfg: PortfolioCandidatesConfig,
    prop_cfg: PropFirmConfig,
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Portfolio simulation using multi-phase bootcamp programme rules."""
    from auta_research.prop_multiphase import (
        RiskSettings,
        multiphase_mc_to_dict,
        multiphase_sim_to_dict,
        recommend_multiphase_risk,
        run_multiphase_monte_carlo,
        simulate_multiphase_program,
    )
    from auta_research.prop_sim import assign_verdict, verdict_to_dict
    from auta_research.prop_multiphase import _multiphase_to_sim_result

    out_dir = project_root / portfolio_cfg.output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    mc_objects = []
    equity_by_key: dict[str, list[float]] = {}
    merged_trades_by_portfolio: dict[str, pd.DataFrame] = {}

    for portfolio in portfolio_cfg.portfolios:
        merged = load_portfolio_trades(
            portfolio.trades,
            project_root,
            dedupe_same_signal=portfolio_cfg.dedupe_same_signal,
        )
        merged_trades_by_portfolio[portfolio.name] = merged
        oos_expectancy = float(merged["r_result"].mean()) if len(merged) > 0 else 0.0

        for risk_dict in prop_cfg.risk.iter_risk_settings(full_grid=False):
            risk = RiskSettings.from_dict(risk_dict)
            sim = simulate_multiphase_program(merged, prop_cfg, risk, portfolio.name)
            row = multiphase_sim_to_dict(sim)
            row["portfolio_name"] = portfolio.name
            mc_result = None

            if prop_cfg.monte_carlo.enabled:
                mc_result = run_multiphase_monte_carlo(merged, prop_cfg, risk, portfolio.name)
                mc_objects.append(mc_result)
                mc_row = multiphase_mc_to_dict(mc_result)
                mc_row["portfolio_name"] = portfolio.name
                mc_rows.append(mc_row)

            mc_adapter = None
            if mc_result:
                from auta_research.prop_sim import MonteCarloResult

                mc_adapter = MonteCarloResult(
                    risk_per_trade_pct=risk.risk_per_trade_pct,
                    trade_split=portfolio.name,
                    runs=mc_result.runs,
                    pass_rate=mc_result.bootcamp_pass_rate,
                    fail_rate=mc_result.fail_rate,
                    daily_fail_rate=mc_result.daily_fail_rate,
                    total_fail_rate=mc_result.total_fail_rate,
                    incomplete_rate=mc_result.incomplete_rate,
                    median_final_return_pct=mc_result.median_final_return_pct,
                    p5_final_return_pct=mc_result.p5_final_return_pct,
                    p95_final_return_pct=mc_result.p95_final_return_pct,
                    worst_drawdown_pct=mc_result.worst_drawdown_pct,
                )
                row["mc_bootcamp_pass_rate"] = mc_result.bootcamp_pass_rate
                row["mc_step_1_pass_rate"] = mc_result.step_1_pass_rate
                row["mc_step_2_pass_rate"] = mc_result.step_2_pass_rate
                row["mc_step_3_pass_rate"] = mc_result.step_3_pass_rate
                row["mc_funded_survival_rate"] = mc_result.funded_survival_rate

            sim_adapter = _multiphase_to_sim_result(sim, portfolio.name, risk.risk_per_trade_pct)
            verdict_result = assign_verdict(sim_adapter, mc_adapter, prop_cfg, oos_expectancy)
            row.update(verdict_to_dict(verdict_result))
            summary_rows.append(row)
            equity_by_key[f"{portfolio.name}_{risk.risk_per_trade_pct}"] = sim.equity_curve

    summary_df = pd.DataFrame(summary_rows)
    mc_df = pd.DataFrame(mc_rows)
    recommended = recommend_multiphase_risk(mc_objects, prop_cfg)
    if recommended is not None and not summary_df.empty:
        summary_df.loc[summary_df["risk_per_trade_pct"] == recommended, "recommended_risk_per_trade"] = True

    meta = {
        "simulation_mode": "multi_phase_bootcamp",
        "program_type": prop_cfg.program_type,
        "program_name": prop_cfg.name,
        "recommended_risk_per_trade": recommended,
        "portfolios": [p.name for p in portfolio_cfg.portfolios],
        "equity_curves": equity_by_key,
        "merged_trade_counts": {k: len(v) for k, v in merged_trades_by_portfolio.items()},
    }

    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)
    if not mc_df.empty:
        mc_df.to_csv(out_dir / "portfolio_monte_carlo.csv", index=False)
    with open(out_dir / "portfolio_meta.json", "w", encoding="utf-8") as f:
        slim = {k: v for k, v in meta.items() if k != "equity_curves"}
        json.dump(slim, f, indent=2, default=str)

    return summary_df, mc_df, meta


def _portfolio_mc_samples(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_pct: float,
    portfolio_name: str,
    samples: int,
) -> list[float]:
    rng = np.random.default_rng(99)
    out: list[float] = []
    for _ in range(samples):
        idx = rng.integers(0, len(trades), size=len(trades))
        sample = trades.iloc[idx].reset_index(drop=True)
        sim = simulate_portfolio_account(sample, cfg, risk_pct, portfolio_name)
        out.append(sim.final_return_pct)
    return out
