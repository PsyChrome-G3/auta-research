"""MetaTrader 5 data connector (research only, no order execution)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from auta_research.data_store import STANDARD_COLUMNS, build_filename, save_csv

MT5_AVAILABLE = False
_mt5: Any = None

try:
    import MetaTrader5 as _mt5_module

    _mt5 = _mt5_module
    MT5_AVAILABLE = True
except ImportError:
    pass

TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def check_mt5() -> tuple[bool, str]:
    """Check if MT5 package and terminal are available."""
    if not MT5_AVAILABLE:
        return False, (
            "MetaTrader5 Python package is not installed. "
            "Install with: uv pip install MetaTrader5. "
            "You can also load pre-exported CSV files instead."
        )
    if not _mt5.initialize():
        err = _mt5.last_error()
        _mt5.shutdown()
        return False, (
            f"Could not connect to MetaTrader 5 terminal. Error: {err}. "
            "Ensure MT5 is running and logged in, or use pre-exported CSV files."
        )
    _mt5.shutdown()
    return True, "OK"


def _mt5_timeframe(tf: str) -> int:
    """Map timeframe string to MT5 constant."""
    tf_upper = tf.upper()
    attr = f"TIMEFRAME_{tf_upper}"
    if hasattr(_mt5, attr):
        return getattr(_mt5, attr)
    raise ValueError(f"Unsupported timeframe: {tf}")


def _parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD to UTC datetime."""
    return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def pull_rates(
    symbol: str,
    timeframe: str,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    """Pull OHLCV rates from MT5 for a symbol and timeframe."""
    if not MT5_AVAILABLE:
        raise RuntimeError(check_mt5()[1])

    if not _mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {_mt5.last_error()}")

    try:
        if not _mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select symbol {symbol}: {_mt5.last_error()}")

        tf = _mt5_timeframe(timeframe)
        start = _parse_date(date_from)
        end = _parse_date(date_to)
        rates = _mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No data returned for {symbol} {timeframe}: {_mt5.last_error()}")

        df = pd.DataFrame(rates)
        ts = pd.to_datetime(df["time"], unit="s", utc=True)
        df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.rename(columns={"tick_volume": "tick_volume"})
        if "real_volume" not in df.columns:
            df["real_volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = 0
        df["symbol"] = symbol.upper()
        df["timeframe"] = timeframe.upper()
        return df[STANDARD_COLUMNS]
    finally:
        _mt5.shutdown()


def pull_and_save(
    symbols: list[str],
    timeframes: list[str],
    date_from: str,
    date_to: str,
    output_dir: str,
) -> list[str]:
    """Pull data for multiple symbols/timeframes and save CSV files."""
    ok, msg = check_mt5()
    if not ok:
        raise RuntimeError(msg)

    saved: list[str] = []
    for symbol in symbols:
        for tf in timeframes:
            df = pull_rates(symbol.strip().upper(), tf.strip().upper(), date_from, date_to)
            fname = build_filename(symbol, tf, date_from, date_to)
            path = save_csv(df, f"{output_dir}/{fname}")
            saved.append(str(path))
    return saved
