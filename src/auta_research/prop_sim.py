"""Prop-firm style funded account evaluation simulator (research only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from auta_research.config import PropFirmConfig

OutcomeStatus = Literal[
    "passed",
    "failed_daily_drawdown",
    "failed_total_drawdown",
    "incomplete",
]

VERDICT_LABELS = (
    "rejected",
    "promising but too risky",
    "passes in-sample only",
    "passes OOS candidate",
    "demo forward-test candidate",
)

REJECTION_REASONS = (
    "negative_expectancy",
    "monte_carlo_pass_rate_below_threshold",
    "daily_failure_rate_too_high",
    "total_failure_rate_too_high",
    "insufficient_trade_count",
    "incomplete_too_often",
    "no_oos_test_available",
)

MIN_TRADES_FOR_VERDICT = 20


@dataclass
class VerdictResult:
    """Prop-firm research verdict with optional rejection reason."""

    verdict: str
    rejection_reason: str | None = None


@dataclass
class SimResult:
    """Result of a single chronological account simulation."""

    risk_per_trade_pct: float
    trade_split: str
    status: OutcomeStatus
    passed: bool
    failed_daily_drawdown: bool
    failed_total_drawdown: bool
    incomplete: bool
    final_balance: float
    final_return_pct: float
    max_drawdown_pct: float
    max_daily_loss_pct: float
    days_to_pass: int | None
    trading_days: int
    total_trades_taken: int
    win_rate: float
    average_r: float
    profit_factor: float
    longest_losing_streak: int
    largest_winning_day: float
    largest_losing_day: float
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: dict[str, float] = field(default_factory=dict)
    cluster_dependent: bool = False


@dataclass
class MonteCarloResult:
    """Monte Carlo summary for one risk level and trade split."""

    risk_per_trade_pct: float
    trade_split: str
    runs: int
    pass_rate: float
    fail_rate: float
    daily_fail_rate: float
    total_fail_rate: float
    incomplete_rate: float
    median_final_return_pct: float
    p5_final_return_pct: float
    p95_final_return_pct: float
    worst_drawdown_pct: float
    recommended: bool = False


def _to_timestamp(val: Any) -> pd.Timestamp:
    if pd.isna(val):
        return pd.Timestamp("1970-01-01", tz="UTC")
    ts = pd.Timestamp(val)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _parse_trade_day(ts: Any) -> str:
    """Extract UTC calendar day from a trade timestamp."""
    if pd.isna(ts):
        return "unknown"
    if isinstance(ts, (int, float)):
        dt = pd.Timestamp(ts, unit="s", tz="UTC")
    else:
        dt = pd.Timestamp(ts)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt.strftime("%Y-%m-%d")


def load_trade_log(path: str | Path) -> pd.DataFrame:
    """Load and validate a trade log CSV."""
    df = pd.read_csv(path)
    required = {"r_result", "signal_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trade log missing columns: {sorted(missing)}")

    df = df.copy()
    df["r_result"] = pd.to_numeric(df["r_result"], errors="coerce").fillna(0.0)
    return _prepare_trades(df)


def discover_trade_splits(primary: Path, project_root: Path) -> list[tuple[str, Path]]:
    """Return trade logs to simulate based on the requested path."""
    if not primary.exists():
        raise FileNotFoundError(f"No trade log found at {primary}")

    primary = primary.resolve()
    parent = primary.parent
    name = primary.name

    if name == "trades_test.csv":
        return [("test", primary)]
    if name == "trades_train.csv":
        return [("train", primary)]
    if name == "trades.csv":
        train = parent / "trades_train.csv"
        test = parent / "trades_test.csv"
        if train.exists() and test.exists():
            return [("train", train), ("test", test)]
        return [("full", primary)]

    return [("full", primary)]


def _trade_r(row: pd.Series, cfg: PropFirmConfig) -> float:
    """Return R multiple used for PnL."""
    if not cfg.execution.use_trade_log_r_results:
        return float(row.get("r_result", 0.0))
    r = float(row["r_result"])
    if not cfg.execution.include_spread_slippage:
        # Trades already net of costs; no further adjustment unless disabled explicitly.
        return r
    return r


def _risk_base(balance: float, starting: float, compound: bool) -> float:
    return balance if compound else starting


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Ensure trade log has sort order and _trade_day column."""
    df = trades.copy()
    if "_trade_day" not in df.columns:
        time_col = "entry_time" if "entry_time" in df.columns else "signal_time"
        df["_trade_day"] = df[time_col].map(_parse_trade_day)
    if "r_result" not in df.columns:
        raise ValueError("Trade log must include r_result column")
    df["r_result"] = pd.to_numeric(df["r_result"], errors="coerce").fillna(0.0)
    sort_col = "entry_time" if "entry_time" in df.columns else "signal_time"
    return df.sort_values(sort_col).reset_index(drop=True)


def simulate_account(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_per_trade_pct: float,
    trade_split: str = "full",
) -> SimResult:
    """Run one chronological prop-firm evaluation on a trade log."""
    trades = _prepare_trades(trades)
    acct = cfg.account
    risk_cfg = cfg.risk
    starting = acct.starting_balance
    balance = starting
    profit_target = starting * (1 + acct.profit_target_pct / 100)
    floor_balance = starting * (1 - acct.max_total_loss_pct / 100)
    max_daily_loss_amt = starting * acct.max_daily_loss_pct / 100
    stop_daily_loss_amt = starting * risk_cfg.stop_after_daily_loss_pct / 100

    status: OutcomeStatus = "incomplete"
    passed = False
    failed_daily = False
    failed_total = False
    incomplete = True

    equity_curve = [balance]
    daily_pnl: dict[str, float] = {}
    trading_days_set: set[str] = set()

    current_day = ""
    day_pnl = 0.0
    trades_today = 0
    day_stopped = False
    consecutive_losses = 0
    days_to_pass: int | None = None

    taken_r: list[float] = []
    day_pnls_list: list[float] = []

    peak_balance = balance
    max_drawdown_pct = 0.0
    max_daily_loss_pct = 0.0

    for _, row in trades.iterrows():
        if passed or failed_daily or failed_total:
            break

        day = str(row["_trade_day"])
        if day != current_day:
            if current_day:
                day_pnls_list.append(day_pnl)
                day_loss_pct = (-day_pnl / starting * 100) if day_pnl < 0 else 0.0
                max_daily_loss_pct = max(max_daily_loss_pct, day_loss_pct)
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

    largest_win_day = max(day_pnls_list) if day_pnls_list else 0.0
    largest_loss_day = min(day_pnls_list) if day_pnls_list else 0.0

    result = SimResult(
        risk_per_trade_pct=risk_per_trade_pct,
        trade_split=trade_split,
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
    )
    result.cluster_dependent = _is_cluster_dependent(trades, cfg, risk_per_trade_pct, result)
    return result


def _is_cluster_dependent(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_pct: float,
    baseline: SimResult,
) -> bool:
    """True if pass outcome likely depends on a small cluster of top trades."""
    if baseline.passed or len(trades) < 20:
        return False
    trimmed = trades.sort_values("r_result", ascending=False).iloc[int(len(trades) * 0.1) :]
    if trimmed.empty:
        return False
    trimmed_result = simulate_account(trimmed, cfg, risk_pct, baseline.trade_split)
    if baseline.passed and not trimmed_result.passed:
        return True
    if baseline.average_r <= 0:
        return True
    half = len(trades) // 2
    first = simulate_account(trades.iloc[:half], cfg, risk_pct, baseline.trade_split)
    second = simulate_account(trades.iloc[half:], cfg, risk_pct, baseline.trade_split)
    if first.average_r > 0 and second.average_r < 0:
        return True
    return False


def run_monte_carlo(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_per_trade_pct: float,
    trade_split: str = "full",
    *,
    progress: Any | None = None,
    progress_task: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo resampling of trade order/outcomes."""
    mc = cfg.monte_carlo
    runs = mc.runs
    rng = np.random.default_rng(42)

    r_vals = trades["r_result"].astype(float).values
    if len(r_vals) == 0:
        return MonteCarloResult(
            risk_per_trade_pct=risk_per_trade_pct,
            trade_split=trade_split,
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

    report_every = max(1, runs // 20)
    for i in range(runs):
        if mc.bootstrap_with_replacement:
            idx = rng.integers(0, len(r_vals), size=len(r_vals))
            sample = trades.iloc[idx].copy()
        elif mc.shuffle_trades:
            sample = trades.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 1_000_000)))
        else:
            sample = trades

        sample = sample.reset_index(drop=True)
        sim = simulate_account(sample, cfg, risk_per_trade_pct, trade_split)
        statuses.append(sim.status)
        returns.append(sim.final_return_pct)
        drawdowns.append(sim.max_drawdown_pct)

        if progress is not None and progress_task is not None and (i + 1) % report_every == 0:
            progress.update(
                progress_task,
                description=f"prop-sim MC {trade_split} @ {risk_per_trade_pct}% ({i + 1}/{runs})",
            )

    statuses_arr = np.array(statuses)
    pass_rate = float((statuses_arr == "passed").mean())
    daily_fail = float((statuses_arr == "failed_daily_drawdown").mean())
    total_fail = float((statuses_arr == "failed_total_drawdown").mean())
    incomplete = float((statuses_arr == "incomplete").mean())
    fail_rate = daily_fail + total_fail

    return MonteCarloResult(
        risk_per_trade_pct=risk_per_trade_pct,
        trade_split=trade_split,
        runs=runs,
        pass_rate=round(pass_rate, 4),
        fail_rate=round(fail_rate, 4),
        daily_fail_rate=round(daily_fail, 4),
        total_fail_rate=round(total_fail, 4),
        incomplete_rate=round(incomplete, 4),
        median_final_return_pct=round(float(np.median(returns)), 4),
        p5_final_return_pct=round(float(np.percentile(returns, 5)), 4),
        p95_final_return_pct=round(float(np.percentile(returns, 95)), 4),
        worst_drawdown_pct=round(float(np.max(drawdowns)), 4),
    )


def recommend_max_risk(mc_rows: list[MonteCarloResult], cfg: PropFirmConfig) -> float | None:
    """Pick highest risk with acceptable Monte Carlo pass and fail rates."""
    verdict_cfg = cfg.verdict
    best: float | None = None
    for row in sorted(mc_rows, key=lambda r: r.risk_per_trade_pct):
        if row.trade_split == "train":
            continue
        if (
            row.pass_rate >= verdict_cfg.min_mc_pass_rate
            and row.total_fail_rate <= verdict_cfg.max_total_fail_rate
            and row.daily_fail_rate <= verdict_cfg.max_daily_fail_rate
        ):
            best = row.risk_per_trade_pct
    return best


def assign_verdict(
    sim: SimResult,
    mc: MonteCarloResult | None,
    cfg: PropFirmConfig,
    oos_expectancy: float | None = None,
) -> VerdictResult:
    """Assign research verdict label and rejection reason for prop-firm survivability."""
    vcfg = cfg.verdict
    is_oos = sim.trade_split in ("test", "oos") or sim.trade_split not in ("train", "full")
    is_train = sim.trade_split in ("train", "full")

    if sim.total_trades_taken < MIN_TRADES_FOR_VERDICT:
        return VerdictResult("rejected", "insufficient_trade_count")

    if sim.average_r <= 0:
        return VerdictResult("rejected", "negative_expectancy")

    if mc and mc.incomplete_rate > 0.5 and not sim.passed:
        return VerdictResult("rejected", "incomplete_too_often")

    if mc and mc.pass_rate < 0.3:
        return VerdictResult("rejected", "monte_carlo_pass_rate_below_threshold")

    if mc and mc.total_fail_rate > vcfg.max_total_fail_rate:
        reason = "total_failure_rate_too_high"
        if sim.average_r > 0 and mc.pass_rate >= 0.3:
            return VerdictResult("promising but too risky", reason)
        return VerdictResult("rejected", reason)

    if mc and mc.daily_fail_rate > vcfg.max_daily_fail_rate:
        reason = "daily_failure_rate_too_high"
        if sim.average_r > 0 and mc.pass_rate >= 0.3:
            return VerdictResult("promising but too risky", reason)
        return VerdictResult("rejected", reason)

    if sim.cluster_dependent:
        return VerdictResult("promising but too risky", "total_failure_rate_too_high")

    oos_ok = oos_expectancy is not None and oos_expectancy >= vcfg.min_oos_expectancy_r

    if is_oos and oos_ok and mc and mc.pass_rate >= vcfg.min_mc_pass_rate:
        if not sim.cluster_dependent and mc.total_fail_rate <= vcfg.max_total_fail_rate:
            return VerdictResult("demo forward-test candidate")
        return VerdictResult("passes OOS candidate")

    if is_train and sim.passed and mc and mc.pass_rate >= vcfg.min_mc_pass_rate:
        return VerdictResult("passes in-sample only")

    if is_train and not is_oos and oos_expectancy is None and sim.trade_split == "full":
        if mc and mc.pass_rate < vcfg.min_mc_pass_rate:
            return VerdictResult("rejected", "no_oos_test_available")

    if mc and mc.pass_rate >= vcfg.min_mc_pass_rate * 0.8:
        return VerdictResult("promising but too risky", "monte_carlo_pass_rate_below_threshold")

    if mc and mc.pass_rate < vcfg.min_mc_pass_rate:
        return VerdictResult("rejected", "monte_carlo_pass_rate_below_threshold")

    return VerdictResult("rejected", "monte_carlo_pass_rate_below_threshold")


def verdict_to_dict(result: VerdictResult) -> dict[str, str | None]:
    """Serialize verdict for CSV/report output."""
    return {
        "verdict": result.verdict,
        "rejection_reason": result.rejection_reason,
    }


def sim_to_dict(sim: SimResult) -> dict[str, Any]:
    """Serialize SimResult without large lists."""
    d = {k: v for k, v in sim.__dict__.items() if k not in ("equity_curve", "daily_pnl")}
    return d


def run_prop_simulation(
    cfg: PropFirmConfig,
    trade_sources: list[tuple[str, Path]],
    output_dir: Path,
    *,
    progress: Any | None = None,
    progress_task: int | None = None,
    chart_samples: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run full prop simulation for all splits and risk levels."""
    if cfg.is_multiphase:
        from auta_research.prop_multiphase import run_multiphase_prop_simulation

        return run_multiphase_prop_simulation(
            cfg,
            trade_sources,
            output_dir,
            progress=progress,
            progress_task=progress_task,
            chart_samples=chart_samples,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    mc_objects: list[MonteCarloResult] = []
    equity_by_key: dict[str, list[float]] = {}
    mc_returns_by_key: dict[str, list[float]] = {}

    oos_expectancy: float | None = None

    meta_mode = {"simulation_mode": "single_phase_challenge", "program_type": "single_challenge"}

    for split_name, path in trade_sources:
        trades = load_trade_log(path)
        if split_name == "test" and len(trades) > 0:
            oos_expectancy = float(trades["r_result"].mean())

        for risk_pct in cfg.risk.risk_per_trade_pct_values:
            if progress is not None and progress_task is not None:
                progress.update(
                    progress_task,
                    description=f"prop-sim {split_name} @ {risk_pct}% risk",
                )
            sim = simulate_account(trades, cfg, risk_pct, split_name)
            row = sim_to_dict(sim)
            row["simulation_mode"] = "single_phase_challenge"
            row["program_type"] = "single_challenge"
            mc_result: MonteCarloResult | None = None

            if cfg.monte_carlo.enabled:
                mc_result = run_monte_carlo(
                    trades,
                    cfg,
                    risk_pct,
                    split_name,
                    progress=progress,
                    progress_task=progress_task,
                )
                mc_objects.append(mc_result)
                mc_row = mc_result.__dict__.copy()
                mc_row["simulation_mode"] = "single_phase_challenge"
                mc_row["program_type"] = "single_challenge"
                mc_rows.append(mc_row)

                if chart_samples > 0:
                    key = f"{split_name}_{risk_pct}"
                    mc_returns_by_key[key] = _mc_return_samples(trades, cfg, risk_pct, chart_samples)

            verdict_result = assign_verdict(sim, mc_result, cfg, oos_expectancy if split_name == "test" else None)
            row.update(verdict_to_dict(verdict_result))
            if mc_result:
                row["mc_pass_rate"] = mc_result.pass_rate
                row["mc_fail_rate"] = mc_result.fail_rate
                row["mc_total_fail_rate"] = mc_result.total_fail_rate
                row["mc_daily_fail_rate"] = mc_result.daily_fail_rate
                row["mc_incomplete_rate"] = mc_result.incomplete_rate

            summary_rows.append(row)
            equity_by_key[f"{split_name}_{risk_pct}"] = sim.equity_curve
            if progress is not None and progress_task is not None:
                progress.advance(progress_task)

    summary_df = pd.DataFrame(summary_rows)
    mc_df = pd.DataFrame(mc_rows)

    recommended = recommend_max_risk(mc_objects, cfg)
    if recommended is not None and not mc_df.empty:
        mc_df.loc[mc_df["risk_per_trade_pct"] == recommended, "recommended"] = True

    meta = {
        **meta_mode,
        "recommended_max_risk_pct": recommended,
        "recommended_risk_per_trade": recommended,
        "trade_splits": [s for s, _ in trade_sources],
        "equity_curves": equity_by_key,
        "mc_return_samples": mc_returns_by_key,
        "oos_expectancy_r": oos_expectancy,
    }

    summary_df.to_csv(output_dir / "prop_sim_summary.csv", index=False)
    if not mc_df.empty:
        mc_df.to_csv(output_dir / "prop_sim_monte_carlo.csv", index=False)

    meta_path = output_dir / "prop_sim_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        slim = {k: v for k, v in meta.items() if k not in ("equity_curves", "mc_return_samples")}
        json.dump(slim, f, indent=2, default=str)

    return summary_df, mc_df, meta


def _mc_return_samples(
    trades: pd.DataFrame,
    cfg: PropFirmConfig,
    risk_pct: float,
    samples: int,
) -> list[float]:
    """Lightweight return samples for plotting."""
    rng = np.random.default_rng(99)
    r_vals = trades["r_result"].astype(float).values
    out: list[float] = []
    for _ in range(samples):
        idx = rng.integers(0, len(r_vals), size=len(r_vals))
        sample = trades.iloc[idx].reset_index(drop=True)
        sim = simulate_account(sample, cfg, risk_pct, "mc")
        out.append(sim.final_return_pct)
    return out
