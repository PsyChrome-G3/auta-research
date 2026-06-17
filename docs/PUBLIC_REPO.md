# Public repository guidelines

This repo is a **generic research framework** for two-candle wick-rejection backtesting. It is safe to share; your **live strategy parameters, watchlists, and results should stay local**.

## Do not commit

| Path | Why |
|------|-----|
| `configs/fixed_candidates.yaml` | Your tuned candidates |
| `configs/portfolio_candidates.yaml` | Your portfolio composition |
| `configs/locked_butt_buddy_strategy.yaml` | Locked symbol/timeframe watchlist |
| `configs/strategies/two_candle_rejection_butt_buddy.yaml` | Proprietary strict variant |
| `configs/private/` | All local strategy configs |
| `scripts/` | Local screening / sim scripts with baked-in lists |
| `data/raw/*.csv` | Broker data |
| `data/results/**` | Backtest and prop-sim outputs |
| `reports/**` | Generated reports and charts |

## Safe to commit

- Python source under `src/auta_research/`
- Generic strategy YAML under `configs/strategies/two_candle_rejection*.yaml` (not `*_butt_buddy.yaml`)
- Example configs under `configs/examples/`
- `configs/prop_firm.yaml` and `configs/prop_firms/*` (public prop-firm rule templates)
- Tests and `tools/generate_sample_data.py`

## Local workflow

```bash
mkdir -p configs/private
cp configs/examples/fixed_candidates.example.yaml configs/private/fixed_candidates.yaml
# edit, then:
auta-research validate-fixed-batch --config configs/private/fixed_candidates.yaml
```

## If proprietary configs were already pushed

Files like `configs/fixed_candidates.yaml` may exist in **git history** even after removal. To purge from GitHub:

```bash
# Install git-filter-repo, then from repo root:
git filter-repo --path configs/fixed_candidates.yaml --invert-paths
git filter-repo --path configs/portfolio_candidates.yaml --invert-paths
git push --force origin main
```

Coordinate with anyone else using the repo before force-pushing. After purge, rotate any parameters you consider compromised.
