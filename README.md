# AUTA Research

Research-only backtesting toolkit for the AUTA 3.0 two-candle wick rejection trading strategy.

**This tool does not place live trades.** It connects to MetaTrader 5 only to pull historical OHLCV data, or you can load pre-exported CSV files. No order execution code is included.

## Purpose

Systematically investigate a two-candle rejection pattern:

- **Sell setup:** Candle 1 (ideally bullish/flat) with strong upper wick, followed by bearish Candle 2 with upper-wick rejection and a materially larger body.
- **Buy setup:** Candle 1 (ideally bearish/flat) with strong lower wick, followed by bullish Candle 2 with lower-wick rejection and a materially larger body.

The engine detects signals, backtests entry/stop/TP variants, runs optimisation grids, performs walk-forward validation, and produces CSV/JSON/Markdown reports with charts.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended)
- MetaTrader 5 terminal (optional, for `pull` command only)

## Install

```bash
cd auta-research
uv venv
uv pip install -e ".[dev]"
```

Verify installation:

```bash
auta-research --help
```

## MetaTrader 5 Setup

1. Install MetaTrader 5 and log in to your broker account.
2. Enable algorithmic trading in MT5 (Tools > Options > Expert Advisors).
3. Keep the MT5 terminal running while pulling data.
4. Install the Python package: `uv pip install MetaTrader5`

If MT5 is unavailable, export OHLCV CSVs manually and place them in `data/raw/` using the naming format:

```
SYMBOL_TIMEFRAME_YYYYMMDD_YYYYMMDD.csv
```

Required columns: `timestamp`, `open`, `high`, `low`, `close`, `tick_volume`, `spread`, `real_volume`, `symbol`, `timeframe`

## Example Commands

### Pull data from MT5

```bash
auta-research pull --symbols EURUSD,GBPUSD,USDJPY,XAGUSD --timeframes M5,M15,H1,H4,D1 --from 2024-01-01 --to 2026-06-16
```

### Detect signals

```bash
auta-research detect --config configs/strategies/two_candle_rejection.yaml --data data/raw/EURUSD_H4_20240101_20260616.csv
```

### Backtest

```bash
auta-research backtest --config configs/strategies/two_candle_rejection.yaml --data data/raw/EURUSD_H4_20240101_20260616.csv
```

Use `--single` to run one entry/stop/TP combo instead of the full grid.

### Optimise

```bash
auta-research optimise --research-config configs/research.yaml
```

### Walk-forward validation

```bash
auta-research validate --research-config configs/research.yaml
```

### Generate report

```bash
auta-research report --results data/results/latest
```

## Strategy Configs

| Config | Description |
|--------|-------------|
| `configs/strategies/two_candle_rejection.yaml` | Default pattern rules |
| `configs/strategies/two_candle_rejection_strict.yaml` | Stricter wick/body/engulf rules |
| `configs/strategies/two_candle_rejection_trend_filtered.yaml` | With EMA trend filter |

## Backtest Assumptions

- **Entry modes:** next candle open, signal candle close, or break of signal extreme.
- **Stop modes:** pattern extreme, candle 2 extreme, or ATR-buffered pattern extreme.
- **TP:** tested at 0.5R through 3R.
- **Ambiguous bars** (both SL and TP inside same candle): defaults to **conservative** (SL first).
- **Spread/slippage:** applied from data or fixed config values.
- **Timeout:** trades exit at close after `max_bars_to_hold` with partial R.
- OHLC data cannot reveal true intrabar sequence; conservative handling is the default.

## Output Locations

| Output | Path |
|--------|------|
| Raw data | `data/raw/` |
| Signals | `data/results/signals/` |
| Trade logs | `data/results/latest/trades.csv` |
| Optimisation | `data/results/optimisation/` |
| Validation | `data/results/validation/` |
| Reports | `reports/` |
| Charts | `reports/assets/` |

## Tests

```bash
uv run pytest tests/ -v
```

## Confidence Rules

- Fewer than 100 trades: **low** confidence
- 100-299 trades: **medium** confidence at best
- 300+ trades: **high** confidence possible
- 80% win rate claims require strong OOS evidence across symbols/timeframes

## Roadmap: MQL5 EA

Once a variant shows positive out-of-sample expectancy with adequate sample size:

1. Port pattern detection logic to MQL5.
2. Implement the proven entry/stop/TP rules as an Expert Advisor.
3. Forward-test on demo account before any live deployment.
4. Keep this Python tool for ongoing research and parameter updates.

## Disclaimer

This software is for **research and education only**. It is not financial advice. Past backtest performance does not guarantee future results. Always validate strategies out-of-sample before risking capital.
