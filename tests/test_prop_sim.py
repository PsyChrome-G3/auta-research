"""Tests for prop-firm simulator."""

from pathlib import Path

import pandas as pd

from auta_research.config import PropFirmConfig
from auta_research.prop_sim import (
    assign_verdict,
    discover_trade_splits,
    run_monte_carlo,
    simulate_account,
    SimResult,
)


def _sample_trades(n: int = 40, win_r: float = 1.0, lose_r: float = -1.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        r = win_r if i % 3 != 0 else lose_r
        rows.append({
            "signal_time": f"2024-01-{(i % 28) + 1:02d}T12:00:00Z",
            "entry_time": f"2024-01-{(i % 28) + 1:02d}T13:00:00Z",
            "symbol": "EURUSD",
            "timeframe": "H4",
            "direction": "buy",
            "r_result": r,
        })
    return pd.DataFrame(rows)


def test_simulate_passes_with_strong_edge():
    cfg = PropFirmConfig(
        account={"starting_balance": 100000, "profit_target_pct": 2.0, "max_total_loss_pct": 10.0},
        risk={"risk_per_trade_pct_values": [0.5], "max_trades_per_day": 10, "compound": False},
        monte_carlo={"enabled": False},
    )
    trades = _sample_trades(50, win_r=1.5, lose_r=-1.0)
    result = simulate_account(trades, cfg, 0.5)
    assert result.total_trades_taken > 0
    assert result.final_balance != 100000


def test_simulate_fails_total_drawdown():
    cfg = PropFirmConfig(
        account={"starting_balance": 10000, "profit_target_pct": 50.0, "max_total_loss_pct": 2.0},
        risk={"risk_per_trade_pct_values": [1.0], "max_trades_per_day": 20},
        monte_carlo={"enabled": False},
    )
    trades = _sample_trades(20, win_r=-1.0, lose_r=-1.0)
    result = simulate_account(trades, cfg, 1.0)
    assert result.failed_total_drawdown or result.final_balance < 10000


def test_monte_carlo_runs():
    cfg = PropFirmConfig(monte_carlo={"enabled": True, "runs": 50})
    trades = _sample_trades(30)
    mc = run_monte_carlo(trades, cfg, 0.25)
    assert mc.runs == 50
    assert 0.0 <= mc.pass_rate <= 1.0


def test_assign_verdict_rejected_on_negative_edge():
    cfg = PropFirmConfig()
    sim = SimResult(
        risk_per_trade_pct=0.5,
        trade_split="full",
        status="incomplete",
        passed=False,
        failed_daily_drawdown=False,
        failed_total_drawdown=False,
        incomplete=True,
        final_balance=99000,
        final_return_pct=-1.0,
        max_drawdown_pct=1.0,
        max_daily_loss_pct=0.5,
        days_to_pass=None,
        trading_days=5,
        total_trades_taken=25,
        win_rate=0.3,
        average_r=-0.2,
        profit_factor=0.5,
        longest_losing_streak=4,
        largest_winning_day=100,
        largest_losing_day=-200,
    )
    result = assign_verdict(sim, None, cfg)
    assert result.verdict == "rejected"
    assert result.rejection_reason == "negative_expectancy"


def test_discover_trade_splits_full_only(tmp_path: Path):
    primary = tmp_path / "trades.csv"
    primary.write_text("signal_time,r_result\n2024-01-01T00:00:00Z,1.0\n", encoding="utf-8")
    sources = discover_trade_splits(primary, tmp_path)
    assert sources == [("full", primary.resolve())]
