"""CSV data loading and saving."""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
    "symbol",
    "timeframe",
]


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp column to ISO UTC strings."""
    out = df.copy()
    if "timestamp" not in out.columns:
        return out
    if not pd.api.types.is_numeric_dtype(out["timestamp"]):
        with suppress(Exception):
            ts = pd.to_datetime(out["timestamp"], utc=True)
            out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        ts = pd.to_datetime(out["timestamp"], unit="s", utc=True)
        out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def parse_filename(path: Path) -> dict[str, str]:
    """Parse SYMBOL_TIMEFRAME_YYYYMMDD_YYYYMMDD.csv filename."""
    m = re.match(r"^([A-Z0-9]+)_([A-Z0-9]+)_(\d{8})_(\d{8})\.csv$", path.name, re.IGNORECASE)
    if not m:
        return {}
    return {
        "symbol": m.group(1).upper(),
        "timeframe": m.group(2).upper(),
        "date_from": m.group(3),
        "date_to": m.group(4),
    }


def build_filename(symbol: str, timeframe: str, date_from: str, date_to: str) -> str:
    """Build standard raw data filename."""
    d_from = date_from.replace("-", "")[:8]
    d_to = date_to.replace("-", "")[:8]
    return f"{symbol.upper()}_{timeframe.upper()}_{d_from}_{d_to}.csv"


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV CSV with standard columns."""
    p = Path(path)
    df = pd.read_csv(p)
    meta = parse_filename(p)
    if "symbol" not in df.columns and meta.get("symbol"):
        df["symbol"] = meta["symbol"]
    if "timeframe" not in df.columns and meta.get("timeframe"):
        df["timeframe"] = meta["timeframe"]
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("tick_volume", "spread", "real_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    df = _normalize_timestamp(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save dataframe to CSV ensuring standard columns exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out = _normalize_timestamp(out)
    for col in STANDARD_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[STANDARD_COLUMNS]
    out.to_csv(p, index=False)
    return p


def find_data_files(raw_dir: str | Path, symbol: str | None = None, timeframe: str | None = None) -> list[Path]:
    """Find CSV files in raw data directory."""
    raw = Path(raw_dir)
    if not raw.exists():
        return []
    files = sorted(raw.glob("*.csv"))
    result = []
    for f in files:
        meta = parse_filename(f)
        if symbol and meta.get("symbol", "").upper() != symbol.upper():
            continue
        if timeframe and meta.get("timeframe", "").upper() != timeframe.upper():
            continue
        result.append(f)
    return result
