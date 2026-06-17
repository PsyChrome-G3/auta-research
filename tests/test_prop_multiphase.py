"""Tests for multi-phase prop-firm simulation (The5ers Bootcamp)."""

from __future__ import annotations

import pandas as pd
import pytest

from auta_research.config import PropFirmConfig, PropPhaseConfig, load_prop_firm_config
from auta_research.prop_multiphase import (
    RiskSettings,
    run_multiphase_monte_carlo,
    simulate_multiphase_program,
    simulate_phase,
)


def _bootcamp_cfg() -> PropFirmConfig:
    return PropFirmConfig(
        program={"name": "test_bootcamp", "type": "multi_step_bootcamp", "currency": "GBP"},
        phases=[
            PropPhaseConfig(name="step_1", starting_balance=3724, profit_target_pct=6, max_total_loss_pct=5),
            PropPhaseConfig(name="step_2", starting_balance=7449, profit_target_pct=6, max_total_loss_pct=5),
            PropPhaseConfig(name="step_3", starting_balance=11173, profit_target_pct=6, max_total_loss_pct=5),
            PropPhaseConfig(
                name="funded_trader",
                starting_balance=14898,
                profit_target_pct=5,
                max_total_loss_pct=4,
                max_daily_loss_pct=3,
                daily_pause_pct=3,
            ),
        ],
        risk={"risk_per_trade_pct_values": [0.5], "max_open_trades": 1, "max_trades_per_day": 10},
        monte_carlo={"enabled": True, "runs": 30},
    )


def _trades(n: int, win_r: float = 1.0, lose_r: float = -1.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        r = win_r if i % 4 != 0 else lose_r
        rows.append({
            "signal_time": f"2024-01-{(i % 28) + 1:02d}T12:00:00Z",
            "entry_time": f"2024-01-{(i % 28) + 1:02d}T13:00:00Z",
            "exit_time": f"2024-01-{(i % 28) + 1:02d}T14:00:00Z",
            "symbol": "EURUSD",
            "r_result": r,
        })
    return pd.DataFrame(rows)


def test_load_the5ers_config():
    cfg = load_prop_firm_config("configs/prop_firms/the5ers_bootcamp_20k_gbp.yaml")
    assert cfg.is_multiphase
    assert cfg.program_type == "multi_step_bootcamp"
    assert len(cfg.bootcamp_phases()) == 3
    assert cfg.funded_phase() is not None
    assert cfg.funded_phase().daily_pause_pct == 3.0
    assert cfg.bootcamp_phases()[0].max_daily_loss_pct is None


def test_step_phase_no_daily_pause_enforced():
    cfg = _bootcamp_cfg()
    phase = cfg.bootcamp_phases()[0]
    trades = _trades(5, win_r=-1.0, lose_r=-1.0)
    risk = RiskSettings(risk_per_trade_pct=1.0, max_trades_per_day=5)
    result, _ = simulate_phase(trades, phase, cfg, risk)
    assert not result.failed_daily_drawdown if hasattr(result, "failed_daily_drawdown") else result.failure_reason != "failed_daily_drawdown"


def test_bootcamp_passes_with_strong_edge():
    cfg = _bootcamp_cfg()
    trades = _trades(120, win_r=1.5, lose_r=-0.5)
    risk = RiskSettings(risk_per_trade_pct=0.5, max_trades_per_day=10)
    result = simulate_multiphase_program(trades, cfg, risk)
    assert result.phase_passed_step_1
    assert result.bootcamp_evaluation_passed or result.phase_passed_step_2


def test_bootcamp_fails_on_total_drawdown():
    cfg = _bootcamp_cfg()
    trades = _trades(30, win_r=-1.0, lose_r=-1.0)
    risk = RiskSettings(risk_per_trade_pct=2.0, max_trades_per_day=10)
    result = simulate_multiphase_program(trades, cfg, risk)
    assert not result.bootcamp_evaluation_passed
    assert result.phase_failure_reason == "failed_total_drawdown"


def test_multiphase_monte_carlo_runs():
    cfg = _bootcamp_cfg()
    trades = _trades(80, win_r=1.2, lose_r=-0.8)
    risk = RiskSettings(risk_per_trade_pct=0.5)
    mc = run_multiphase_monte_carlo(trades, cfg, risk, "test")
    assert mc.runs == 30
    assert 0.0 <= mc.bootcamp_pass_rate <= 1.0
    assert "step_1" in mc.fail_rate_by_phase or mc.step_1_pass_rate >= 0


def test_single_phase_config_still_works():
    cfg = load_prop_firm_config("configs/prop_firm.yaml")
    assert not cfg.is_multiphase
