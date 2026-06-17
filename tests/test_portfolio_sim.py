"""Tests for portfolio prop-firm simulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auta_research.config import PropFirmConfig
from auta_research.portfolio_sim import (
    load_portfolio_trades,
    simulate_portfolio_account,
    strategy_name_from_path,
)


def _trade_row(
    day: int,
    symbol: str,
    strategy: str,
    r: float,
    hour: int = 12,
) -> dict:
    ts = f"2024-01-{day:02d}T{hour:02d}:00:00Z"
    return {
        "signal_time": ts,
        "entry_time": ts,
        "exit_time": f"2024-01-{day:02d}T{hour + 1:02d}:00:00Z",
        "symbol": symbol,
        "timeframe": "H4",
        "direction": "buy",
        "r_result": r,
        "strategy_name": strategy,
    }


def test_strategy_name_from_path():
    path = Path("data/results/fixed_candidates/EURUSD_H4_2R_candle2/trades_test.csv")
    assert strategy_name_from_path(path) == "EURUSD_H4_2R_candle2"


def test_load_portfolio_merges_and_dedupes(tmp_path: Path):
    a = tmp_path / "EURUSD_A" / "trades_test.csv"
    b = tmp_path / "EURUSD_B" / "trades_test.csv"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    rows_a = [_trade_row(i, "EURUSD", "A", 1.0) for i in range(1, 25)]
    rows_b = [_trade_row(i, "EURUSD", "B", 0.5) for i in range(1, 25)]
    rows_b[0]["signal_time"] = rows_a[0]["signal_time"]
    pd.DataFrame(rows_a).to_csv(a, index=False)
    pd.DataFrame(rows_b).to_csv(b, index=False)

    merged = load_portfolio_trades([a, b], tmp_path, dedupe_same_signal=True)
    assert "strategy_name" in merged.columns
    assert len(merged) < len(rows_a) + len(rows_b)
    assert merged["signal_time"].is_monotonic_increasing or len(merged) <= 2


def test_portfolio_max_open_trades_limits_entries():
    cfg = PropFirmConfig(
        account={"starting_balance": 100000, "profit_target_pct": 50.0, "max_total_loss_pct": 20.0},
        risk={"risk_per_trade_pct_values": [0.5], "max_open_trades": 1, "max_trades_per_day": 10},
        monte_carlo={"enabled": False},
    )
    rows = [
        _trade_row(1, "EURUSD", "s1", 1.0, hour=10),
        _trade_row(1, "GBPUSD", "s2", 1.0, hour=11),
        _trade_row(1, "USDJPY", "s3", 1.0, hour=12),
    ]
    rows[0]["exit_time"] = "2024-01-01T20:00:00Z"
    rows[1]["exit_time"] = "2024-01-01T20:00:00Z"
    rows[2]["exit_time"] = "2024-01-01T20:00:00Z"
    trades = pd.DataFrame(rows)
    trades = trades.sort_values("signal_time").reset_index(drop=True)

    result = simulate_portfolio_account(trades, cfg, 0.5, "test_portfolio")
    assert result.total_trades_taken == 1
    assert result.skipped_open_limit >= 2


def test_portfolio_contribution_tracking():
    cfg = PropFirmConfig(
        account={"starting_balance": 100000, "profit_target_pct": 50.0, "max_total_loss_pct": 20.0},
        risk={"risk_per_trade_pct_values": [0.5], "max_open_trades": 5, "max_trades_per_day": 10},
        monte_carlo={"enabled": False},
    )
    rows = [
        _trade_row(i, "EURUSD", "strat_a", 1.0) for i in range(1, 25)
    ] + [
        _trade_row(i, "GBPJPY", "strat_b", 0.5) for i in range(1, 25)
    ]
    trades = pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)
    trades["strategy_name"] = trades["strategy_name"]

    result = simulate_portfolio_account(trades, cfg, 0.5, "diversified")
    assert result.total_trades_taken >= 20
    assert "strat_a" in result.strategy_contribution_pct
    assert "strat_b" in result.strategy_contribution_pct
