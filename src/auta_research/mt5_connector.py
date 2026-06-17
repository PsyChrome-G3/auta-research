"""MetaTrader 5 data connector (research only, no order execution)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

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

UTC = ZoneInfo("UTC")

# Max calendar days per request (MT5 history limits vary by timeframe).
CHUNK_DAYS: dict[str, int] = {
    "M1": 7,
    "M5": 30,
    "M15": 60,
    "M30": 90,
    "H1": 180,
    "H4": 365,
    "D1": 1825,
    "W1": 3650,
    "MN1": 7300,
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


def _parse_date_start(s: str) -> datetime:
    """Parse YYYY-MM-DD to UTC start of day."""
    dt = datetime.strptime(s[:10], "%Y-%m-%d")
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)


def _parse_date_end(s: str) -> datetime:
    """Parse YYYY-MM-DD to UTC end of day."""
    dt = datetime.strptime(s[:10], "%Y-%m-%d")
    return dt.replace(hour=23, minute=59, second=59, microsecond=0, tzinfo=UTC)


def _chunk_ranges(
    date_from: str, date_to: str, timeframe: str
) -> list[tuple[datetime, datetime]]:
    """Split a date range into MT5-friendly chunks."""
    start = _parse_date_start(date_from)
    end = _parse_date_end(date_to)
    if start > end:
        raise ValueError(f"date_from ({date_from}) must be before date_to ({date_to})")

    chunk_days = CHUNK_DAYS.get(timeframe.upper(), 90)
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days) - timedelta(seconds=1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(seconds=1)
    return chunks


def _copy_rates_chunk(
    resolved: str, tf: int, chunk_start: datetime, chunk_end: datetime
) -> Any:
    """Request rates for one chunk, trying datetime then unix timestamps."""
    rates = _mt5.copy_rates_range(resolved, tf, chunk_start, chunk_end)
    if rates is not None and len(rates) > 0:
        return rates
    # Some MT5 builds reject tz-aware datetime; retry with unix seconds.
    rates = _mt5.copy_rates_range(
        resolved, tf, int(chunk_start.timestamp()), int(chunk_end.timestamp())
    )
    return rates


def _resolve_symbol(symbol: str) -> str | None:
    """Resolve broker-specific symbol name."""
    symbol = symbol.upper().strip()
    info = _mt5.symbol_info(symbol)
    if info is not None:
        return info.name

    for suffix in ("m", ".a", ".r", ".pro", "-ecn", ".i", ".e"):
        candidate = f"{symbol}{suffix}"
        info = _mt5.symbol_info(candidate)
        if info is not None:
            return info.name

    matches: list[str] = []
    for item in _mt5.symbols_get() or []:
        name = item.name.upper()
        if name == symbol or name.startswith(symbol):
            matches.append(item.name)

    if not matches:
        return None
    if symbol in [m.upper() for m in matches]:
        for m in matches:
            if m.upper() == symbol:
                return m
    return sorted(matches, key=len)[0]


def _rates_to_dataframe(rates: Any, symbol: str, timeframe: str) -> pd.DataFrame:
    """Convert MT5 rates array to standard dataframe."""
    df = pd.DataFrame(rates)
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "real_volume" not in df.columns:
        df["real_volume"] = 0
    if "spread" not in df.columns:
        df["spread"] = 0
    df["symbol"] = symbol.upper()
    df["timeframe"] = timeframe.upper()
    return df[STANDARD_COLUMNS]


def pull_rates(
    symbol: str,
    timeframe: str,
    date_from: str,
    date_to: str,
    *,
    _initialized: bool = False,
) -> pd.DataFrame:
    """Pull OHLCV rates from MT5 for a symbol and timeframe."""
    if not MT5_AVAILABLE:
        raise RuntimeError(check_mt5()[1])

    own_session = not _initialized
    if own_session and not _mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {_mt5.last_error()}")

    try:
        resolved = _resolve_symbol(symbol)
        if resolved is None:
            raise RuntimeError(
                f"Symbol {symbol} not found in MT5 terminal. "
                "Check Market Watch or broker symbol names."
            )

        if not _mt5.symbol_select(resolved, True):
            raise RuntimeError(
                f"Could not select symbol {resolved}: {_mt5.last_error()}"
            )

        tf = _mt5_timeframe(timeframe)
        chunks = _chunk_ranges(date_from, date_to, timeframe)
        parts: list[pd.DataFrame] = []

        for chunk_start, chunk_end in chunks:
            rates = _copy_rates_chunk(resolved, tf, chunk_start, chunk_end)
            if rates is None or len(rates) == 0:
                continue
            parts.append(_rates_to_dataframe(rates, symbol, timeframe))

        if not parts:
            err = _mt5.last_error()
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe} ({date_from} to {date_to}): {err}. "
                "Try opening the symbol chart in MT5, increase 'Max bars in chart' "
                "(Tools > Options > Charts), and scroll the chart to load history."
            )

        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df
    finally:
        if own_session:
            _mt5.shutdown()


def pull_and_save(
    symbols: list[str],
    timeframes: list[str],
    date_from: str,
    date_to: str,
    output_dir: str,
) -> tuple[list[str], list[str]]:
    """Pull data for multiple symbols/timeframes and save CSV files.

    Returns (saved_paths, error_messages).
    """
    ok, msg = check_mt5()
    if not ok:
        raise RuntimeError(msg)

    if not _mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {_mt5.last_error()}")

    saved: list[str] = []
    errors: list[str] = []

    try:
        for symbol in symbols:
            for tf in timeframes:
                sym = symbol.strip().upper()
                tf_clean = tf.strip().upper()
                try:
                    df = pull_rates(
                        sym, tf_clean, date_from, date_to, _initialized=True
                    )
                    fname = build_filename(sym, tf_clean, date_from, date_to)
                    path = save_csv(df, f"{output_dir}/{fname}")
                    saved.append(str(path))
                except Exception as exc:
                    errors.append(f"{sym} {tf_clean}: {exc}")
    finally:
        _mt5.shutdown()

    return saved, errors
