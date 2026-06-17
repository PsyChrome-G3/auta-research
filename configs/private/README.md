# Private configs (not in git)

Copy example files from `configs/examples/` here and customise for your own research.

Recommended local files:

| File | Purpose |
|------|---------|
| `fixed_candidates.yaml` | Your tuned symbol/variant candidates |
| `portfolio_candidates.yaml` | Portfolio trade-log groupings for prop-sim |
| `locked_butt_buddy_strategy.yaml` | Locked watchlist after screening |
| `research.local.yaml` | Full symbol universe / optimisation overrides |

Run commands with explicit paths, e.g.:

```bash
auta-research validate-fixed-batch --config configs/private/fixed_candidates.yaml
```

Or copy into `configs/` root (those paths are gitignored).
