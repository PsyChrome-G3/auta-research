"""Generate sample OHLCV CSV for offline testing."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

random.seed(42)


def generate_bars(n: int = 500, start_price: float = 1.1000) -> pd.DataFrame:
    rows = []
    price = start_price
    for i in range(n):
        direction = random.choice([-1, -1, 1, 1, 0])
        body = random.uniform(0.0003, 0.0015)
        if direction < 0:
            o = price
            c = price - body
        elif direction > 0:
            o = price
            c = price + body
        else:
            o = price
            c = price + random.uniform(-0.0001, 0.0001)
        upper = max(o, c) + random.uniform(0.0002, 0.0010)
        lower = min(o, c) - random.uniform(0.0001, 0.0005)
        day = (i % 28) + 1
        hour = (i % 24)
        rows.append({
            "timestamp": f"2024-{(i // 28) % 12 + 1:02d}-{day:02d}T{hour:02d}:00:00Z",
            "open": round(o, 5),
            "high": round(upper, 5),
            "low": round(lower, 5),
            "close": round(c, 5),
            "tick_volume": random.randint(50, 500),
            "spread": random.randint(1, 3),
            "real_volume": 0,
            "symbol": "EURUSD",
            "timeframe": "H4",
        })
        price = c

    # Inject a sell pattern near the end
    idx = n - 2
    rows[idx] = {
        **rows[idx],
        "open": 1.1200, "high": 1.1300, "low": 1.1190, "close": 1.1210,
    }
    rows[idx + 1] = {
        **rows[idx + 1],
        "open": 1.1210, "high": 1.1230, "low": 1.1140, "close": 1.1150,
    }
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "raw" / "EURUSD_H4_20240101_20260616.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_bars(500)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} bars)")
