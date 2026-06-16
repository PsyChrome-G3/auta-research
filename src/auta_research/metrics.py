"""Trade performance metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def aggregate_metrics_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of metric dicts (weighted by trade count)."""
    if not metric_dicts:
        return {}
    total_trades = sum(m.get("trades", 0) for m in metric_dicts)
    if total_trades == 0:
        return compute_metrics(pd.DataFrame())
    weights = [m.get("trades", 0) / total_trades for m in metric_dicts]
    agg: dict[str, Any] = {
        "trades": total_trades,
        "win_rate": round(sum(m.get("win_rate", 0) * w for m, w in zip(metric_dicts, weights)), 4),
        "average_r": round(sum(m.get("average_r", 0) * w for m, w in zip(metric_dicts, weights)), 4),
        "expectancy_r": round(sum(m.get("expectancy_r", 0) * w for m, w in zip(metric_dicts, weights)), 4),
        "total_r": round(sum(m.get("total_r", 0) for m in metric_dicts), 4),
        "confidence": confidence_label(total_trades),
    }
    return agg


def confidence_label(trade_count: int) -> str:
    """Assign confidence based on sample size."""
    if trade_count < 100:
        return "low"
    if trade_count < 300:
        return "medium"
    return "high"


def compute_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    """Compute summary metrics from trade log."""
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "average_r": 0.0,
            "median_r": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "expectancy_r": 0.0,
            "sharpe_like_score": 0.0,
            "longest_losing_streak": 0,
            "average_bars_held": 0.0,
            "exposure_estimate": 0.0,
            "confidence": "low",
            "overfit_warning": True,
        }

    r = trades["r_result"].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = (r > 0).mean()
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    equity = r.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()

    streak = 0
    max_streak = 0
    for val in r:
        if val < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    std = r.std()
    sharpe_like = (r.mean() / std * np.sqrt(len(r))) if std > 0 else 0.0
    n = len(trades)
    conf = confidence_label(n)

    overfit = False
    if n < 100 and win_rate > 0.75:
        overfit = True
    if n < 30:
        overfit = True

    return {
        "trades": int(n),
        "win_rate": round(float(win_rate), 4),
        "average_r": round(float(r.mean()), 4),
        "median_r": round(float(r.median()), 4),
        "total_r": round(float(r.sum()), 4),
        "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else 999.0,
        "max_drawdown_r": round(float(max_dd), 4),
        "expectancy_r": round(float(r.mean()), 4),
        "sharpe_like_score": round(float(sharpe_like), 4),
        "longest_losing_streak": int(max_streak),
        "average_bars_held": round(float(trades["bars_held"].mean()), 2),
        "exposure_estimate": round(float(trades["bars_held"].sum()) / max(n, 1), 2),
        "confidence": conf,
        "overfit_warning": overfit,
    }


def penalised_score(metrics: dict[str, Any], min_trades: int = 30) -> float:
    """Rank variants with sample-size penalty."""
    n = metrics.get("trades", 0)
    if n < min_trades:
        return -999.0 + n
    base = metrics.get("expectancy_r", 0.0) * np.sqrt(n)
    if metrics.get("overfit_warning"):
        base *= 0.5
    if metrics.get("confidence") == "low":
        base *= 0.7
    elif metrics.get("confidence") == "medium":
        base *= 0.9
    pf = metrics.get("profit_factor", 0)
    if pf < 1.0:
        base *= 0.5
    return round(float(base), 4)


def degradation_pct(train_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> float:
    """Calculate performance degradation from train to test."""
    train_exp = train_metrics.get("expectancy_r", 0)
    test_exp = test_metrics.get("expectancy_r", 0)
    if train_exp <= 0:
        return 100.0 if test_exp <= 0 else 0.0
    deg = (train_exp - test_exp) / abs(train_exp) * 100
    return round(max(float(deg), 0.0), 2)


def stability_score(fold_metrics: list[dict[str, Any]]) -> float:
    """Score stability across walk-forward folds."""
    if not fold_metrics:
        return 0.0
    expectancies = [m.get("expectancy_r", 0) for m in fold_metrics]
    if not expectancies:
        return 0.0
    mean_exp = np.mean(expectancies)
    std_exp = np.std(expectancies)
    if std_exp == 0:
        return 1.0 if mean_exp > 0 else 0.0
    return round(float(mean_exp / std_exp), 4)


def win_rate_defensible(metrics: dict[str, Any], threshold: float = 0.80) -> dict[str, Any]:
    """Assess whether a high win rate claim is defensible."""
    wr = metrics.get("win_rate", 0)
    n = metrics.get("trades", 0)
    exp = metrics.get("expectancy_r", 0)
    dd = metrics.get("max_drawdown_r", 0)
    conf = metrics.get("confidence", "low")

    defensible = (
        wr >= threshold
        and n >= 300
        and exp > 0
        and dd > -50
        and conf in ("medium", "high")
    )
    likely_overfit = wr >= threshold and n < 100

    return {
        "claimed_win_rate": wr,
        "defensible": defensible,
        "likely_overfit": likely_overfit,
        "confidence": conf,
        "trade_count": n,
        "verdict": (
            "supported" if defensible
            else "likely_overfit" if likely_overfit
            else "not_supported"
        ),
    }
