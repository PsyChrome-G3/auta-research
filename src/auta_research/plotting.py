"""Chart generation for research reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_equity_curve(trades: pd.DataFrame, out_path: Path) -> None:
    """Plot cumulative R equity curve."""
    if trades.empty or "r_result" not in trades.columns:
        return
    _ensure_dir(out_path.parent)
    equity = trades.sort_values("entry_time")["r_result"].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(equity)), equity.values)
    ax.set_title("Equity Curve (R)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_drawdown(trades: pd.DataFrame, out_path: Path) -> None:
    """Plot drawdown curve in R."""
    if trades.empty:
        return
    _ensure_dir(out_path.parent)
    equity = trades.sort_values("entry_time")["r_result"].cumsum()
    dd = equity - equity.cummax()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(range(len(dd)), dd.values, 0, alpha=0.5)
    ax.set_title("Drawdown (R)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown R")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_win_rate_by_timeframe(trades: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of win rate by timeframe."""
    if trades.empty or "timeframe" not in trades.columns:
        return
    _ensure_dir(out_path.parent)
    grouped = trades.groupby("timeframe").apply(
        lambda x: (x["r_result"] > 0).mean(), include_groups=False
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Win Rate by Timeframe")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_expectancy_by_tp(trades: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of expectancy by TP R multiple."""
    if trades.empty or "r_multiple_target" not in trades.columns:
        return
    _ensure_dir(out_path.parent)
    grouped = trades.groupby("r_multiple_target")["r_result"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=ax)
    ax.set_title("Expectancy by TP Multiple")
    ax.set_ylabel("Average R")
    ax.axhline(0, color="gray", linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trade_count_by_symbol(trades: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of trade count by symbol."""
    if trades.empty or "symbol" not in trades.columns:
        return
    _ensure_dir(out_path.parent)
    counts = trades.groupby("symbol").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Trade Count by Symbol")
    ax.set_ylabel("Trades")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_all_charts(trades: pd.DataFrame, assets_dir: Path) -> dict[str, str]:
    """Generate all report charts."""
    charts = {
        "equity_curve": assets_dir / "equity_curve.png",
        "drawdown": assets_dir / "drawdown.png",
        "win_rate_by_timeframe": assets_dir / "win_rate_by_timeframe.png",
        "expectancy_by_tp": assets_dir / "expectancy_by_tp.png",
        "trade_count_by_symbol": assets_dir / "trade_count_by_symbol.png",
    }
    plot_equity_curve(trades, charts["equity_curve"])
    plot_drawdown(trades, charts["drawdown"])
    plot_win_rate_by_timeframe(trades, charts["win_rate_by_timeframe"])
    plot_expectancy_by_tp(trades, charts["expectancy_by_tp"])
    plot_trade_count_by_symbol(trades, charts["trade_count_by_symbol"])
    return {k: str(v) for k, v in charts.items()}
