"""Tests for MT5 connector helpers (no live terminal required)."""

from auta_research.mt5_connector import _chunk_ranges, _parse_date_end, _parse_date_start


def test_parse_dates():
    start = _parse_date_start("2024-01-01")
    end = _parse_date_end("2024-01-31")
    assert start < end
    assert start.hour == 0
    assert end.hour == 23


def test_chunk_ranges_m5():
    chunks = _chunk_ranges("2024-01-01", "2024-03-01", "M5")
    assert len(chunks) >= 2
    assert chunks[0][0] < chunks[0][1]
    assert chunks[-1][1] >= _parse_date_start("2024-03-01")
