"""Tests for fixed-candidate backtest and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auta_research.config import StrategyConfig
from auta_research.fixed_candidates import backtest_fixed, split_ohlc_by_date, validate_fixed
from auta_research.prop_sim import discover_trade_splits
from auta_research.variants import normalize_variant


def _sample_ohlc(n: int = 120) -> pd.DataFrame:
    rows = []
    price = 1.1000
    for i in range(n):
        month = 1 + (i // 30) % 12
        year = 2024 + (i // 365)
        day = (i % 28) + 1
        o = price
        c = price + (0.0005 if i % 4 != 0 else -0.0012)
        h = max(o, c) + 0.0010
        l = min(o, c) - 0.0005
        rows.append({
            "timestamp": f"{year}-{month:02d}-{day:02d}T12:00:00Z",
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "tick_volume": 100,
            "spread": 1,
            "real_volume": 0,
            "symbol": "EURUSD",
            "timeframe": "H4",
        })
        price = c
    return pd.DataFrame(rows)


VARIANT = {
    "wick_ratio_min": 1.5,
    "body_ratio_min": 1.2,
    "require_body_engulf": False,
    "entry_mode": "next_open",
    "stop_mode": "candle2_extreme",
    "tp_r_value": 0.5,
    "atr_buffer": 0.0,
    "trend_filter": "none",
    "volatility_filter": False,
    "session_filter": False,
    "buy_colours": ["bearish", "flat"],
    "sell_colours": ["bullish", "flat"],
}


def test_split_ohlc_by_date():
    df = _sample_ohlc(100)
    df.loc[df.index < 50, "timestamp"] = "2024-06-01T12:00:00Z"
    df.loc[df.index >= 50, "timestamp"] = "2025-07-01T12:00:00Z"
    train, test = split_ohlc_by_date(df, "2025-06-01")
    assert len(train) + len(test) == len(df)
    assert len(test) > 0
    assert pd.to_datetime(train["timestamp"], utc=True).max() < pd.Timestamp("2025-06-01", tz="UTC")
    assert pd.to_datetime(test["timestamp"], utc=True).min() >= pd.Timestamp("2025-06-01", tz="UTC")


def test_backtest_fixed_writes_outputs(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    data_path.write_text(_sample_ohlc(80).to_csv(index=False), encoding="utf-8")
    out = tmp_path / "out"
    cfg = StrategyConfig()

    summary = backtest_fixed(data_path, VARIANT, out, cfg)
    assert (out / "trades.csv").exists()
    assert (out / "summary.json").exists()
    assert (out / "variant.json").exists()
    assert "metrics" in summary
    assert not (tmp_path / "latest").exists()


def test_backtest_fixed_optional_latest(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    data_path.write_text(_sample_ohlc(80).to_csv(index=False), encoding="utf-8")
    out = tmp_path / "out"
    latest = tmp_path / "latest"
    cfg = StrategyConfig()

    backtest_fixed(data_path, VARIANT, out, cfg, write_latest=True, latest_dir=latest)
    assert (latest / "trades.csv").exists()


def test_validate_fixed_writes_split_outputs(tmp_path: Path):
    data_path = tmp_path / "data.csv"
    df = _sample_ohlc(200)
    df.loc[df.index < 100, "timestamp"] = "2024-06-01T12:00:00Z"
    df.loc[df.index >= 100, "timestamp"] = "2025-07-01T12:00:00Z"
    data_path.write_text(df.to_csv(index=False), encoding="utf-8")
    out = tmp_path / "candidate"
    cfg = StrategyConfig()

    summary = validate_fixed(data_path, VARIANT, "2025-06-01", out, cfg)
    for name in (
        "trades_train.csv",
        "trades_test.csv",
        "summary_train.json",
        "summary_test.json",
        "validation_summary.md",
        "validation_summary.json",
    ):
        assert (out / name).exists(), name

    assert "train_metrics" in summary
    assert "test_metrics" in summary
    assert "degradation_pct" in summary
    assert "oos_positive" in summary
    md = (out / "validation_summary.md").read_text(encoding="utf-8")
    assert "Degradation" in md
    assert "OOS expectancy positive" in md


def test_discover_trade_splits_test_only(tmp_path: Path):
    candidate = tmp_path / "EURUSD_H4_1p5R_candle2"
    candidate.mkdir()
    test = candidate / "trades_test.csv"
    test.write_text("signal_time,r_result\n2025-07-01T00:00:00Z,0.5\n", encoding="utf-8")
    sources = discover_trade_splits(test, tmp_path)
    assert sources == [("test", test.resolve())]


def test_discover_trade_splits_train_and_test_siblings(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    train = candidate / "trades_train.csv"
    test = candidate / "trades_test.csv"
    trades = candidate / "trades.csv"
    train.write_text("signal_time,r_result\n2024-01-01T00:00:00Z,1.0\n", encoding="utf-8")
    test.write_text("signal_time,r_result\n2025-07-01T00:00:00Z,0.5\n", encoding="utf-8")
    trades.write_text("signal_time,r_result\n2024-01-01T00:00:00Z,1.0\n", encoding="utf-8")
    sources = discover_trade_splits(trades, tmp_path)
    labels = [s[0] for s in sources]
    assert labels == ["train", "test"]


def test_normalize_variant_from_yaml_style():
    v = normalize_variant(VARIANT)
    assert v["tp_r_value"] == 0.5
    assert v["stop_mode"] == "candle2_extreme"
