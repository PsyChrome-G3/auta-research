"""Multi-phase prop-firm programme simulation (e.g. The5ers Bootcamp)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from auta_research.config import PropFirmConfig, PropPhaseConfig
from auta_research.prop_sim import (
    OutcomeStatus,
    _prepare_trades,
    _risk_base,
    _to_timestamp,
    _trade_r,
)

PhaseFailureReason = Literal[
    "failed_total_drawdown",
    "failed_daily_drawdown",
    "incomplete_trades_exhausted",
    "incomplete_target_not_reached",
    "none",
]


@dataclass
class RiskSettings:
    risk_per_trade_pct: float
    max_open_trades: int = 1
    max_trades_per_day: int = 3
    stop_after_daily_loss_pct: float = 2.0
    stop_after_consecutive_losses: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, float | int]) -> RiskSettings:
        return cls(
            risk_per_trade_pct=float(d["risk_per_trade_pct"]),
            max_open_trades=int(d.get("max_open_trades", 1)),
            max_trades_per_day=int(d.get("max_trades_per_day", 3)),
            stop_after_daily_loss_pct=float(d.get("stop_after_daily_loss_pct", 2.0)),
            stop_after_consecutive_losses=int(d.get("stop_after_consecutive_losses", 3)),
        )


@dataclass
class PhaseSimResult:
    phase_name: str
    passed: bool
    failed: bool
    incomplete: bool
    status: OutcomeStatus
    failure_reason: PhaseFailureReason
    starting_balance: float
    final_balance: float
    final_return_pct: float
    max_drawdown_pct: float
    trading_days: int
    trades_taken: int
    trades_to_pass: int | None
    days_to_pass: int | None
    end_trade_idx: int
    taken_r: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class MultiPhaseSimResult:
    trade_split: str
    risk_settings: RiskSettings
    phase_results: dict[str, PhaseSimResult] = field(default_factory=dict)
    phase_passed_step_1: bool = False
    phase_passed_step_2: bool = False
    phase_passed_step_3: bool = False
    bootcamp_evaluation_passed: bool = False
    funded_phase_passed_or_survived: bool | None = None
    phase_failed_name: str | None = None
    phase_failure_reason: PhaseFailureReason | None = None
    total_trades_taken: int = 0
    average_r: float = 0.0
    win_rate: float = 0.0
    equity_curve: list[float] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.bootcamp_evaluation_passed

    @property
    def status(self) -> OutcomeStatus:
        if self.bootcamp_evaluation_passed:
            return "passed"
        if self.phase_failure_reason == "failed_daily_drawdown":
            return "failed_daily_drawdown"
        if self.phase_failure_reason == "failed_total_drawdown":
            return "failed_total_drawdown"
        return "incomplete"


@dataclass
class MultiPhaseMonteCarloResult:
    trade_split: str
    risk_settings: RiskSettings
    runs: int
    bootcamp_pass_rate: float
    step_1_pass_rate: float
    step_2_pass_rate: float
    step_3_pass_rate: float
    funded_survival_rate: float
    funded_pass_rate: float
    fail_rate_by_phase: dict[str, float]
    median_trades_to_pass_by_phase: dict[str, float]
    median_days_to_pass_by_phase: dict[str, float]
    pass_rate: float = 0.0
    fail_rate: float = 0.0
    daily_fail_rate: float = 0.0
    total_fail_rate: float = 0.0
    incomplete_rate: float = 0.0
    median_final_return_pct: float = 0.0
    p5_final_return_pct: float = 0.0
    p95_final_return_pct: float = 0.0
    worst_drawdown_pct: float = 0.0
    recommended: bool = False

    @property
    def risk_per_trade_pct(self) -> float:
        return self.risk_settings.risk_per_trade_pct


def _phase_passed_flags(phase_results: dict[str, PhaseSimResult]) -> tuple[bool, bool, bool]:
    def _passed(name: str) -> bool:
        pr = phase_results.get(name)
        return bool(pr and pr.passed)

    return _passed("step_1"), _passed("step_2"), _passed("step_3")


def simulate_phase(
    trades: pd.DataFrame,
    phase: PropPhaseConfig,
    cfg: PropFirmConfig,
    risk: RiskSettings,
    start_idx: int = 0,
) -> tuple[PhaseSimResult, int]:
    """Simulate one programme phase from start_idx until pass, fail, or trades exhausted."""
    trades = _prepare_trades(trades)
    starting = phase.starting_balance
    balance = starting
    profit_target = starting * (1 + phase.profit_target_pct / 100)
    floor_balance = starting * (1 - phase.max_total_loss_pct / 100)

    enforce_daily_fail = phase.max_daily_loss_pct is not None
    max_daily_loss_amt = starting * phase.max_daily_loss_pct / 100 if enforce_daily_fail else 0.0
    enforce_daily_pause = phase.daily_pause_pct is not None
    pause_daily_amt = starting * phase.daily_pause_pct / 100 if enforce_daily_pause else 0.0
    stop_daily_amt = starting * risk.stop_after_daily_loss_pct / 100

    status: OutcomeStatus = "incomplete"
    passed = False
    failed = False
    failure_reason: PhaseFailureReason = "none"

    equity_curve = [balance]
    trading_days: set[str] = set()
    taken_r: list[float] = []
    open_exits: list[pd.Timestamp] = []

    current_day = ""
    day_pnl = 0.0
    trades_today = 0
    day_stopped = False
    consecutive_losses = 0
    days_to_pass: int | None = None
    trades_to_pass: int | None = None

    peak_balance = balance
    max_drawdown_pct = 0.0
    idx = start_idx

    while idx < len(trades):
        if passed or failed:
            break

        row = trades.iloc[idx]
        idx += 1

        entry_ts = _to_timestamp(row.get("entry_time", row.get("signal_time")))
        exit_raw = row.get("exit_time")
        exit_ts = _to_timestamp(exit_raw) if pd.notna(exit_raw) else entry_ts
        open_exits = [e for e in open_exits if e > entry_ts]
        if len(open_exits) >= risk.max_open_trades:
            continue

        day = str(row["_trade_day"])
        if day != current_day:
            current_day = day
            day_pnl = 0.0
            trades_today = 0
            day_stopped = False

        if phase.time_limit_days is not None and len(trading_days) >= phase.time_limit_days:
            break

        if trades_today >= risk.max_trades_per_day:
            continue
        if day_stopped:
            continue
        if consecutive_losses >= risk.stop_after_consecutive_losses:
            continue

        r = _trade_r(row, cfg)
        risk_amt = _risk_base(balance, starting, cfg.risk.compound) * risk.risk_per_trade_pct / 100
        pnl = r * risk_amt

        balance += pnl
        day_pnl += pnl
        trades_today += 1
        trading_days.add(day)
        taken_r.append(r)
        equity_curve.append(balance)
        open_exits.append(exit_ts)

        if pnl < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0

        peak_balance = max(peak_balance, balance)
        dd_pct = (peak_balance - balance) / starting * 100
        max_drawdown_pct = max(max_drawdown_pct, dd_pct)

        if enforce_daily_pause and day_pnl <= -pause_daily_amt:
            day_stopped = True

        if enforce_daily_pause and pause_daily_amt > 0 and day_pnl <= -stop_daily_amt:
            day_stopped = True

        if enforce_daily_fail and day_pnl <= -max_daily_loss_amt:
            status = "failed_daily_drawdown"
            failed = True
            failure_reason = "failed_daily_drawdown"
            break

        if balance <= floor_balance:
            status = "failed_total_drawdown"
            failed = True
            failure_reason = "failed_total_drawdown"
            break

        if balance >= profit_target and len(trading_days) >= phase.min_trading_days:
            status = "passed"
            passed = True
            failure_reason = "none"
            days_to_pass = len(trading_days)
            trades_to_pass = len(taken_r)
            break

    if not passed and not failed:
        if idx >= len(trades) and len(taken_r) == 0:
            failure_reason = "incomplete_trades_exhausted"
        else:
            failure_reason = "incomplete_target_not_reached"
        status = "incomplete"

    return PhaseSimResult(
        phase_name=phase.name,
        passed=passed,
        failed=failed,
        incomplete=not passed and not failed,
        status=status,
        failure_reason=failure_reason,
        starting_balance=starting,
        final_balance=round(balance, 2),
        final_return_pct=round((balance - starting) / starting * 100, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        trading_days=len(trading_days),
        trades_taken=len(taken_r),
        trades_to_pass=trades_to_pass,
        days_to_pass=days_to_pass,
        end_trade_idx=idx,
        taken_r=taken_r,
        equity_curve=equity_curve,
    ), idx


def simulate_multiphase_program(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk: RiskSettings,
    trade_split: str = "full",
    *,
    include_funded: bool | None = None,
) -> MultiPhaseSimResult:
    """Run bootcamp steps sequentially, then optional funded trader phase."""
    trades = _prepare_trades(trades)
    if include_funded is None:
        include_funded = cfg.program.include_funded_phase if cfg.program else True

    result = MultiPhaseSimResult(trade_split=trade_split, risk_settings=risk)
    idx = 0
    all_taken_r: list[float] = []
    combined_equity: list[float] = []

    for phase in cfg.bootcamp_phases():
        if idx >= len(trades):
            result.phase_failed_name = phase.name
            result.phase_failure_reason = "incomplete_trades_exhausted"
            phase_res = PhaseSimResult(
                phase_name=phase.name,
                passed=False,
                failed=False,
                incomplete=True,
                status="incomplete",
                failure_reason="incomplete_trades_exhausted",
                starting_balance=phase.starting_balance,
                final_balance=phase.starting_balance,
                final_return_pct=0.0,
                max_drawdown_pct=0.0,
                trading_days=0,
                trades_taken=0,
                trades_to_pass=None,
                days_to_pass=None,
                end_trade_idx=idx,
                taken_r=[],
            )
            result.phase_results[phase.name] = phase_res
            break

        phase_res, idx = simulate_phase(trades, phase, cfg, risk, idx)
        result.phase_results[phase.name] = phase_res
        all_taken_r.extend(phase_res.taken_r)
        combined_equity.extend(phase_res.equity_curve[1:] if combined_equity else phase_res.equity_curve)

        if not phase_res.passed:
            result.phase_failed_name = phase.name
            result.phase_failure_reason = phase_res.failure_reason
            break

    s1, s2, s3 = _phase_passed_flags(result.phase_results)
    result.phase_passed_step_1 = s1
    result.phase_passed_step_2 = s2
    result.phase_passed_step_3 = s3
    result.bootcamp_evaluation_passed = s1 and s2 and s3

    funded = cfg.funded_phase()
    if include_funded and funded and result.bootcamp_evaluation_passed and idx < len(trades):
        funded_res, idx = simulate_phase(trades, funded, cfg, risk, idx)
        result.phase_results[funded.name] = funded_res
        all_taken_r.extend(funded_res.taken_r)
        combined_equity.extend(funded_res.equity_curve[1:])
        result.funded_phase_passed_or_survived = funded_res.passed or (
            not funded_res.failed and funded_res.trades_taken > 0
        )
    elif include_funded and funded and result.bootcamp_evaluation_passed:
        result.funded_phase_passed_or_survived = False

    result.total_trades_taken = len(all_taken_r)
    result.equity_curve = combined_equity or [0.0]
    if all_taken_r:
        result.average_r = round(float(np.mean(all_taken_r)), 4)
        result.win_rate = round(float(np.mean([1 if x > 0 else 0 for x in all_taken_r])), 4)

    return result


def run_multiphase_monte_carlo(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk: RiskSettings,
    trade_split: str = "full",
    *,
    progress: Any | None = None,
    progress_task: int | None = None,
) -> MultiPhaseMonteCarloResult:
    """Monte Carlo over trade sequences for a multi-phase programme."""
    mc = cfg.monte_carlo
    runs = mc.runs
    rng = np.random.default_rng(42)
    trades = _prepare_trades(trades)

    if trades.empty or runs == 0:
        return _empty_multiphase_mc(trade_split, risk, runs)

    bootcamp_passed: list[bool] = []
    step_flags: dict[str, list[bool]] = {p.name: [] for p in cfg.bootcamp_phases()}
    funded_survived: list[bool] = []
    funded_passed: list[bool] = []
    fail_phase_counts: dict[str, int] = {p.name: 0 for p in cfg.phases}
    trades_to_pass: dict[str, list[int]] = {p.name: [] for p in cfg.phases}
    days_to_pass: dict[str, list[int]] = {p.name: [] for p in cfg.phases}
    returns: list[float] = []
    drawdowns: list[float] = []
    statuses: list[str] = []

    report_every = max(1, runs // 20)
    for i in range(runs):
        if mc.bootstrap_with_replacement:
            idx = rng.integers(0, len(trades), size=len(trades))
            sample = trades.iloc[idx].copy()
        elif mc.shuffle_trades:
            sample = trades.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        else:
            sample = trades

        sample = sample.reset_index(drop=True)
        sim = simulate_multiphase_program(sample, cfg, risk, trade_split)
        bootcamp_passed.append(sim.bootcamp_evaluation_passed)
        statuses.append(sim.status)

        for phase in cfg.bootcamp_phases():
            pr = sim.phase_results.get(phase.name)
            step_flags[phase.name].append(bool(pr and pr.passed))
            if pr and pr.failed:
                fail_phase_counts[phase.name] = fail_phase_counts.get(phase.name, 0) + 1
            if pr and pr.passed and pr.trades_to_pass is not None:
                trades_to_pass[phase.name].append(pr.trades_to_pass)
            if pr and pr.passed and pr.days_to_pass is not None:
                days_to_pass[phase.name].append(pr.days_to_pass)

        if sim.phase_failed_name and sim.phase_failure_reason in (
            "failed_total_drawdown",
            "failed_daily_drawdown",
        ):
            fail_phase_counts[sim.phase_failed_name] = fail_phase_counts.get(sim.phase_failed_name, 0) + 1

        if sim.funded_phase_passed_or_survived is not None:
            funded_survived.append(sim.funded_phase_passed_or_survived)
            funded = sim.phase_results.get("funded_trader")
            funded_passed.append(bool(funded and funded.passed))

        last_phase = list(sim.phase_results.values())[-1] if sim.phase_results else None
        if last_phase:
            returns.append(last_phase.final_return_pct)
            drawdowns.append(last_phase.max_drawdown_pct)

        if progress is not None and progress_task is not None and (i + 1) % report_every == 0:
            progress.update(
                progress_task,
                description=f"MC {trade_split} @ {risk.risk_per_trade_pct}% ({i + 1}/{runs})",
            )

    statuses_arr = np.array(statuses)
    daily_fail = float((statuses_arr == "failed_daily_drawdown").mean())
    total_fail = float((statuses_arr == "failed_total_drawdown").mean())
    incomplete = float((statuses_arr == "incomplete").mean())

    fail_by_phase = {k: round(v / runs, 4) for k, v in fail_phase_counts.items()}

    def _median_map(src: dict[str, list[int]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, vals in src.items():
            if vals:
                out[name] = round(float(np.median(vals)), 2)
        return out

    boot_rate = float(np.mean(bootcamp_passed)) if bootcamp_passed else 0.0

    return MultiPhaseMonteCarloResult(
        trade_split=trade_split,
        risk_settings=risk,
        runs=runs,
        bootcamp_pass_rate=round(boot_rate, 4),
        step_1_pass_rate=round(float(np.mean(step_flags.get("step_1", [False]))), 4),
        step_2_pass_rate=round(float(np.mean(step_flags.get("step_2", [False]))), 4),
        step_3_pass_rate=round(float(np.mean(step_flags.get("step_3", [False]))), 4),
        funded_survival_rate=round(float(np.mean(funded_survived)), 4) if funded_survived else 0.0,
        funded_pass_rate=round(float(np.mean(funded_passed)), 4) if funded_passed else 0.0,
        fail_rate_by_phase=fail_by_phase,
        median_trades_to_pass_by_phase=_median_map(trades_to_pass),
        median_days_to_pass_by_phase=_median_map(days_to_pass),
        pass_rate=round(boot_rate, 4),
        fail_rate=round(daily_fail + total_fail, 4),
        daily_fail_rate=round(daily_fail, 4),
        total_fail_rate=round(total_fail, 4),
        incomplete_rate=round(incomplete, 4),
        median_final_return_pct=round(float(np.median(returns)), 4) if returns else 0.0,
        p5_final_return_pct=round(float(np.percentile(returns, 5)), 4) if returns else 0.0,
        p95_final_return_pct=round(float(np.percentile(returns, 95)), 4) if returns else 0.0,
        worst_drawdown_pct=round(float(np.max(drawdowns)), 4) if drawdowns else 0.0,
    )


def _empty_multiphase_mc(trade_split: str, risk: RiskSettings, runs: int) -> MultiPhaseMonteCarloResult:
    return MultiPhaseMonteCarloResult(
        trade_split=trade_split,
        risk_settings=risk,
        runs=runs,
        bootcamp_pass_rate=0.0,
        step_1_pass_rate=0.0,
        step_2_pass_rate=0.0,
        step_3_pass_rate=0.0,
        funded_survival_rate=0.0,
        funded_pass_rate=0.0,
        fail_rate_by_phase={},
        median_trades_to_pass_by_phase={},
        median_days_to_pass_by_phase={},
        pass_rate=0.0,
        fail_rate=1.0,
        incomplete_rate=1.0,
    )


def multiphase_sim_to_dict(sim: MultiPhaseSimResult) -> dict[str, Any]:
    """Flatten multi-phase deterministic result for CSV."""
    last = list(sim.phase_results.values())[-1] if sim.phase_results else None
    return {
        "simulation_mode": "multi_phase_bootcamp",
        "program_type": "multi_step_bootcamp",
        "trade_split": sim.trade_split,
        "risk_per_trade_pct": sim.risk_settings.risk_per_trade_pct,
        "max_open_trades": sim.risk_settings.max_open_trades,
        "max_trades_per_day": sim.risk_settings.max_trades_per_day,
        "status": sim.status,
        "passed": sim.bootcamp_evaluation_passed,
        "phase_passed_step_1": sim.phase_passed_step_1,
        "phase_passed_step_2": sim.phase_passed_step_2,
        "phase_passed_step_3": sim.phase_passed_step_3,
        "bootcamp_evaluation_passed": sim.bootcamp_evaluation_passed,
        "funded_phase_passed_or_survived": sim.funded_phase_passed_or_survived,
        "phase_failed_name": sim.phase_failed_name,
        "phase_failure_reason": sim.phase_failure_reason,
        "final_balance": last.final_balance if last else 0.0,
        "final_return_pct": last.final_return_pct if last else 0.0,
        "max_drawdown_pct": max((p.max_drawdown_pct for p in sim.phase_results.values()), default=0.0),
        "total_trades_taken": sim.total_trades_taken,
        "win_rate": sim.win_rate,
        "average_r": sim.average_r,
    }


def multiphase_mc_to_dict(mc: MultiPhaseMonteCarloResult) -> dict[str, Any]:
    """Flatten multi-phase MC result for CSV."""
    return {
        "simulation_mode": "multi_phase_bootcamp",
        "program_type": "multi_step_bootcamp",
        "trade_split": mc.trade_split,
        "risk_per_trade_pct": mc.risk_settings.risk_per_trade_pct,
        "max_open_trades": mc.risk_settings.max_open_trades,
        "max_trades_per_day": mc.risk_settings.max_trades_per_day,
        "runs": mc.runs,
        "bootcamp_pass_rate": mc.bootcamp_pass_rate,
        "step_1_pass_rate": mc.step_1_pass_rate,
        "step_2_pass_rate": mc.step_2_pass_rate,
        "step_3_pass_rate": mc.step_3_pass_rate,
        "funded_survival_rate": mc.funded_survival_rate,
        "funded_pass_rate": mc.funded_pass_rate,
        "fail_rate_by_phase": json.dumps(mc.fail_rate_by_phase),
        "median_trades_to_pass_by_phase": json.dumps(mc.median_trades_to_pass_by_phase),
        "median_days_to_pass_by_phase": json.dumps(mc.median_days_to_pass_by_phase),
        "pass_rate": mc.pass_rate,
        "fail_rate": mc.fail_rate,
        "daily_fail_rate": mc.daily_fail_rate,
        "total_fail_rate": mc.total_fail_rate,
        "incomplete_rate": mc.incomplete_rate,
        "median_final_return_pct": mc.median_final_return_pct,
        "p5_final_return_pct": mc.p5_final_return_pct,
        "p95_final_return_pct": mc.p95_final_return_pct,
        "worst_drawdown_pct": mc.worst_drawdown_pct,
        "recommended": mc.recommended,
    }


def recommend_multiphase_risk(
    mc_rows: list[MultiPhaseMonteCarloResult],
    cfg: PropFirmConfig,
) -> float | None:
    """Pick highest risk with acceptable bootcamp MC pass rate."""
    vcfg = cfg.verdict
    best: float | None = None
    for row in sorted(mc_rows, key=lambda r: r.risk_settings.risk_per_trade_pct):
        if (
            row.bootcamp_pass_rate >= vcfg.min_mc_pass_rate
            and row.total_fail_rate <= vcfg.max_total_fail_rate
            and row.daily_fail_rate <= vcfg.max_daily_fail_rate
        ):
            best = row.risk_settings.risk_per_trade_pct
    return best


def run_multiphase_prop_simulation(
    cfg: PropFirmConfig,
    trade_sources: list[tuple[str, Any]],
    output_dir: Any,
    *,
    progress: Any | None = None,
    progress_task: int | None = None,
    chart_samples: int = 0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run multi-phase prop simulation for all trade splits and risk combinations."""
    from pathlib import Path

    import pandas as pd

    from auta_research.prop_sim import assign_verdict, load_trade_log, verdict_to_dict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    mc_objects: list[MultiPhaseMonteCarloResult] = []
    equity_by_key: dict[str, list[float]] = {}
    oos_expectancy: float | None = None

    risk_combos = cfg.risk.iter_risk_settings(full_grid=False)

    for split_name, path in trade_sources:
        trades = load_trade_log(path)
        if split_name == "test" and len(trades) > 0:
            oos_expectancy = float(trades["r_result"].mean())

        for risk_dict in risk_combos:
            risk = RiskSettings.from_dict(risk_dict)
            if progress is not None and progress_task is not None:
                progress.update(
                    progress_task,
                    description=f"bootcamp {split_name} @ {risk.risk_per_trade_pct}%",
                )

            sim = simulate_multiphase_program(trades, cfg, risk, split_name)
            row = multiphase_sim_to_dict(sim)
            mc_result: MultiPhaseMonteCarloResult | None = None

            if cfg.monte_carlo.enabled:
                mc_result = run_multiphase_monte_carlo(
                    trades,
                    cfg,
                    risk,
                    split_name,
                    progress=progress,
                    progress_task=progress_task,
                )
                mc_objects.append(mc_result)
                mc_row = multiphase_mc_to_dict(mc_result)
                mc_rows.append(mc_row)

            mc_adapter = None
            if mc_result:
                from auta_research.prop_sim import MonteCarloResult

                mc_adapter = MonteCarloResult(
                    risk_per_trade_pct=risk.risk_per_trade_pct,
                    trade_split=split_name,
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
                row["mc_pass_rate"] = mc_result.bootcamp_pass_rate
                row["mc_step_1_pass_rate"] = mc_result.step_1_pass_rate
                row["mc_step_2_pass_rate"] = mc_result.step_2_pass_rate
                row["mc_step_3_pass_rate"] = mc_result.step_3_pass_rate
                row["mc_funded_survival_rate"] = mc_result.funded_survival_rate
                row["mc_fail_rate_by_phase"] = json.dumps(mc_result.fail_rate_by_phase)
                row["mc_median_trades_to_pass"] = json.dumps(mc_result.median_trades_to_pass_by_phase)
                row["mc_median_days_to_pass"] = json.dumps(mc_result.median_days_to_pass_by_phase)

            sim_adapter = _multiphase_to_sim_result(sim, split_name, risk.risk_per_trade_pct)
            verdict_result = assign_verdict(sim_adapter, mc_adapter, cfg, oos_expectancy if split_name == "test" else None)
            row.update(verdict_to_dict(verdict_result))

            summary_rows.append(row)
            equity_by_key[f"{split_name}_{risk.risk_per_trade_pct}"] = sim.equity_curve
            if progress is not None and progress_task is not None:
                progress.advance(progress_task)

    summary_df = pd.DataFrame(summary_rows)
    mc_df = pd.DataFrame(mc_rows)

    recommended = recommend_multiphase_risk(mc_objects, cfg)
    if recommended is not None and not summary_df.empty:
        summary_df.loc[summary_df["risk_per_trade_pct"] == recommended, "recommended_risk_per_trade"] = True
    if recommended is not None and not mc_df.empty:
        mc_df.loc[mc_df["risk_per_trade_pct"] == recommended, "recommended"] = True

    meta = {
        "simulation_mode": "multi_phase_bootcamp",
        "program_type": cfg.program_type,
        "program_name": cfg.name,
        "recommended_max_risk_pct": recommended,
        "recommended_risk_per_trade": recommended,
        "trade_splits": [s for s, _ in trade_sources],
        "equity_curves": equity_by_key,
        "oos_expectancy_r": oos_expectancy,
    }

    summary_df.to_csv(output_dir / "prop_sim_summary.csv", index=False)
    if not mc_df.empty:
        mc_df.to_csv(output_dir / "prop_sim_monte_carlo.csv", index=False)

    meta_path = output_dir / "prop_sim_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        slim = {k: v for k, v in meta.items() if k not in ("equity_curves",)}
        json.dump(slim, f, indent=2, default=str)

    return summary_df, mc_df, meta


def _multiphase_to_sim_result(sim: MultiPhaseSimResult, trade_split: str, risk_pct: float):
    """Adapt multi-phase result for single-phase verdict helpers."""
    from auta_research.prop_sim import SimResult

    return SimResult(
        risk_per_trade_pct=risk_pct,
        trade_split=trade_split,
        status=sim.status,
        passed=sim.bootcamp_evaluation_passed,
        failed_daily_drawdown=sim.phase_failure_reason == "failed_daily_drawdown",
        failed_total_drawdown=sim.phase_failure_reason == "failed_total_drawdown",
        incomplete=not sim.bootcamp_evaluation_passed and sim.phase_failure_reason not in (
            "failed_daily_drawdown",
            "failed_total_drawdown",
        ),
        final_balance=list(sim.phase_results.values())[-1].final_balance if sim.phase_results else 0.0,
        final_return_pct=list(sim.phase_results.values())[-1].final_return_pct if sim.phase_results else 0.0,
        max_drawdown_pct=max((p.max_drawdown_pct for p in sim.phase_results.values()), default=0.0),
        max_daily_loss_pct=0.0,
        days_to_pass=None,
        trading_days=sum(p.trading_days for p in sim.phase_results.values()),
        total_trades_taken=sim.total_trades_taken,
        win_rate=sim.win_rate,
        average_r=sim.average_r,
        profit_factor=0.0,
        longest_losing_streak=0,
        largest_winning_day=0.0,
        largest_losing_day=0.0,
    )
