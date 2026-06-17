"""Configuration loading and validation for AUTA research."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class PatternConfig(BaseModel):
    wick_ratio_min: float = 2.0
    candle2_wick_ratio_min: float | None = None
    body_ratio_min: float = 1.5
    require_body_engulf: bool = False
    require_second_candle_wick_bias: bool = True
    require_c2_directional_wick_larger_than_c1: bool = False
    c2_wick_growth_min: float = 1.0
    require_butt_buddy: bool = False
    butt_buddy_max_gap_points: float = 5.0
    min_body_to_range_ratio: float = 0.05
    max_body_to_range_ratio_for_flat: float = 0.15
    allow_candle1_colours: dict[str, list[str]] = Field(
        default_factory=lambda: {"buy": ["bearish", "flat"], "sell": ["bullish", "flat"]}
    )
    require_candle2_colour: dict[str, str] = Field(
        default_factory=lambda: {"buy": "bullish", "sell": "bearish"}
    )


class EntryConfig(BaseModel):
    modes: list[str] = Field(default_factory=lambda: ["next_open", "signal_close", "break_signal_extreme"])
    break_expiry_bars: int = 5


class StopConfig(BaseModel):
    modes: list[str] = Field(
        default_factory=lambda: [
            "pattern_extreme",
            "candle2_extreme",
            "atr_buffered_pattern_extreme",
        ]
    )
    atr_period: int = 14
    atr_buffer_values: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])


class TakeProfitConfig(BaseModel):
    r_values: list[float] = Field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])


class CostsConfig(BaseModel):
    spread_mode: Literal["from_data", "fixed"] = "from_data"
    fixed_spread_points: float = 0.0
    slippage_points: float = 0.0
    commission_per_lot: float = 0.0


class TrendFilterConfig(BaseModel):
    enabled: bool = False
    modes: list[str] = Field(default_factory=lambda: ["none", "ema_50", "ema_200", "ema_stack", "price_vs_ema_200"])


class VolatilityFilterConfig(BaseModel):
    enabled: bool = False
    min_atr_percentile: float = 20.0
    max_atr_percentile: float = 90.0
    atr_period: int = 14
    percentile_lookback: int = 100


class SessionFilterConfig(BaseModel):
    enabled: bool = False
    allowed_sessions: list[str] = Field(default_factory=lambda: ["london", "new_york"])


class LocationFilterConfig(BaseModel):
    enabled: bool = False
    require_near_recent_swing: bool = False
    swing_lookback: int = 20
    max_distance_atr: float = 0.5


class PullbackFilterConfig(BaseModel):
    enabled: bool = False
    ema_periods: list[int] = Field(default_factory=lambda: [20, 50])
    max_distance_atr: float = 0.5


class FiltersConfig(BaseModel):
    trend: TrendFilterConfig = Field(default_factory=TrendFilterConfig)
    volatility: VolatilityFilterConfig = Field(default_factory=VolatilityFilterConfig)
    session: SessionFilterConfig = Field(default_factory=SessionFilterConfig)
    location: LocationFilterConfig = Field(default_factory=LocationFilterConfig)
    pullback: PullbackFilterConfig = Field(default_factory=PullbackFilterConfig)


class BacktestConfig(BaseModel):
    max_bars_to_hold: int = 100
    ambiguous_bar_handling: Literal["conservative", "optimistic", "skip"] = "conservative"


class StrategyConfig(BaseModel):
    name: str = "two_candle_rejection"
    description: str = ""
    pattern: PatternConfig = Field(default_factory=PatternConfig)
    entry: EntryConfig = Field(default_factory=EntryConfig)
    stop: StopConfig = Field(default_factory=StopConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


class OptimisationGridConfig(BaseModel):
    wick_ratio_min: list[float] = Field(default_factory=lambda: [2.0])
    body_ratio_min: list[float] = Field(default_factory=lambda: [1.5])
    require_body_engulf: list[bool] = Field(default_factory=lambda: [False])
    candle1_allowed_colours: dict[str, list[list[str]]] = Field(
        default_factory=lambda: {
            "buy": [["bearish", "flat"]],
            "sell": [["bullish", "flat"]],
        }
    )
    entry_modes: list[str] = Field(default_factory=lambda: ["next_open"])
    stop_modes: list[str] = Field(default_factory=lambda: ["pattern_extreme"])
    tp_r_values: list[float] = Field(default_factory=lambda: [1.0])
    atr_buffer_values: list[float] = Field(default_factory=lambda: [0.0])
    trend_filters: list[str] = Field(default_factory=lambda: ["none"])
    volatility_filters: list[bool] = Field(default_factory=lambda: [False])
    session_filters: list[bool] = Field(default_factory=lambda: [False])


class OptimisationConfig(BaseModel):
    max_variants: int = 500
    min_trades_for_ranking: int = 30
    timeframes: list[str] | None = None
    save_every: int = 25
    grid: OptimisationGridConfig = Field(default_factory=OptimisationGridConfig)


class RollingValidationConfig(BaseModel):
    train_bars: int = 2000
    validation_bars: int = 500
    test_bars: int = 500
    step_bars: int = 500


class ValidationConfig(BaseModel):
    mode: Literal["static", "rolling"] = "static"
    train_pct: float = 0.5
    validation_pct: float = 0.25
    test_pct: float = 0.25
    rolling: RollingValidationConfig = Field(default_factory=RollingValidationConfig)
    overfit_threshold_pct: float = 30.0
    win_rate_claim_threshold: float = 0.80


class ReportingConfig(BaseModel):
    output_dir: str = "reports"
    assets_dir: str = "reports/assets"
    latest_results_dir: str = "data/results/latest"


class ResearchDataConfig(BaseModel):
    raw_dir: str = "data/raw"
    results_dir: str = "data/results"
    date_from: str = "2024-01-01"
    date_to: str = "2026-06-16"


class ResearchConfig(BaseModel):
    name: str = "auta_research_default"
    description: str = ""
    strategy_config: str = "configs/strategies/two_candle_rejection.yaml"
    data: ResearchDataConfig = Field(default_factory=ResearchDataConfig)
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    timeframes: list[str] = Field(default_factory=lambda: ["H4"])
    optimisation: OptimisationConfig = Field(default_factory=OptimisationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)


class SymbolsConfig(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    point_values: dict[str, float] = Field(default_factory=dict)


class PropAccountConfig(BaseModel):
    starting_balance: float = 100_000.0
    profit_target_pct: float = 8.0
    max_total_loss_pct: float = 6.0
    max_daily_loss_pct: float = 3.0
    min_trading_days: int = 0
    max_trading_days: int | None = None


class PropPhaseConfig(BaseModel):
    """One phase in a multi-step prop programme (e.g. The5ers Bootcamp step)."""

    name: str
    starting_balance: float
    profit_target_pct: float
    max_total_loss_pct: float
    max_daily_loss_pct: float | None = None
    daily_pause_pct: float | None = None
    time_limit_days: int | None = None
    min_trading_days: int = 0


class PropProgramConfig(BaseModel):
    name: str = "prop_firm_default"
    type: str = "single_challenge"
    currency: str = "USD"
    leverage: str | None = None
    include_funded_phase: bool = True


class PropRiskConfig(BaseModel):
    risk_per_trade_pct_values: list[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0]
    )
    max_open_trades: int = 1
    max_open_trades_values: list[int] | None = None
    max_trades_per_day: int = 3
    max_trades_per_day_values: list[int] | None = None
    stop_after_daily_loss_pct: float = 2.0
    stop_after_daily_loss_pct_values: list[float] | None = None
    stop_after_consecutive_losses: int = 3
    stop_after_consecutive_losses_values: list[int] | None = None
    compound: bool = False

    def iter_risk_settings(self, *, full_grid: bool = False) -> list[dict[str, float | int]]:
        """Risk parameter combinations. full_grid=True expands all _values lists (sensitivity)."""
        if full_grid and (
            self.max_open_trades_values
            or self.max_trades_per_day_values
            or self.stop_after_daily_loss_pct_values
            or self.stop_after_consecutive_losses_values
        ):
            return self._full_risk_grid()
        return [
            {
                "risk_per_trade_pct": float(rp),
                "max_open_trades": int(self.max_open_trades),
                "max_trades_per_day": int(self.max_trades_per_day),
                "stop_after_daily_loss_pct": float(self.stop_after_daily_loss_pct),
                "stop_after_consecutive_losses": int(self.stop_after_consecutive_losses),
            }
            for rp in self.risk_per_trade_pct_values
        ]

    def _full_risk_grid(self) -> list[dict[str, float | int]]:
        import itertools

        open_vals = self.max_open_trades_values or [self.max_open_trades]
        day_vals = self.max_trades_per_day_values or [self.max_trades_per_day]
        stop_daily_vals = self.stop_after_daily_loss_pct_values or [self.stop_after_daily_loss_pct]
        stop_loss_vals = self.stop_after_consecutive_losses_values or [self.stop_after_consecutive_losses]
        combos: list[dict[str, float | int]] = []
        for risk_pct, open_t, day_t, stop_d, stop_c in itertools.product(
            self.risk_per_trade_pct_values,
            open_vals,
            day_vals,
            stop_daily_vals,
            stop_loss_vals,
        ):
            combos.append({
                "risk_per_trade_pct": float(risk_pct),
                "max_open_trades": int(open_t),
                "max_trades_per_day": int(day_t),
                "stop_after_daily_loss_pct": float(stop_d),
                "stop_after_consecutive_losses": int(stop_c),
            })
        return combos


class PropExecutionConfig(BaseModel):
    use_trade_log_r_results: bool = True
    include_spread_slippage: bool = True


class PropMonteCarloConfig(BaseModel):
    enabled: bool = True
    runs: int = 1000
    shuffle_trades: bool = True
    bootstrap_with_replacement: bool = True


class PropVerdictConfig(BaseModel):
    min_mc_pass_rate: float = 0.55
    max_total_fail_rate: float = 0.25
    max_daily_fail_rate: float = 0.35
    min_oos_expectancy_r: float = 0.0


class PropFirmConfig(BaseModel):
    name: str = "prop_firm_default"
    description: str = ""
    program: PropProgramConfig | None = None
    phases: list[PropPhaseConfig] = Field(default_factory=list)
    account: PropAccountConfig = Field(default_factory=PropAccountConfig)
    risk: PropRiskConfig = Field(default_factory=PropRiskConfig)
    execution: PropExecutionConfig = Field(default_factory=PropExecutionConfig)
    monte_carlo: PropMonteCarloConfig = Field(default_factory=PropMonteCarloConfig)
    verdict: PropVerdictConfig = Field(default_factory=PropVerdictConfig)

    @property
    def is_multiphase(self) -> bool:
        return len(self.phases) > 0

    @property
    def program_type(self) -> str:
        if self.program is not None:
            return self.program.type
        return "single_challenge" if not self.is_multiphase else "multi_step_bootcamp"

    def bootcamp_phases(self) -> list[PropPhaseConfig]:
        return [p for p in self.phases if p.name != "funded_trader"]

    def funded_phase(self) -> PropPhaseConfig | None:
        for phase in self.phases:
            if phase.name == "funded_trader":
                return phase
        return None


class FixedCandidateVariant(BaseModel):
    wick_ratio_min: float = 1.5
    body_ratio_min: float = 1.2
    require_body_engulf: bool = False
    require_second_candle_wick_bias: bool = True
    require_c2_directional_wick_larger_than_c1: bool = False
    candle2_wick_ratio_min: float | None = None
    c2_wick_growth_min: float = 1.0
    require_butt_buddy: bool = False
    entry_mode: str = "next_open"
    stop_mode: str = "pattern_extreme"
    tp_r_value: float = 1.0
    atr_buffer: float = 0.0
    trend_filter: str = "none"
    volatility_filter: bool = False
    session_filter: bool = False
    buy_colours: list[str] = Field(default_factory=lambda: ["bearish", "flat"])
    sell_colours: list[str] = Field(default_factory=lambda: ["bullish", "flat"])


class FixedCandidate(BaseModel):
    name: str
    data: str
    split_date: str = "2025-06-01"
    strategy_config: str = "configs/strategies/two_candle_rejection.yaml"
    variant: FixedCandidateVariant = Field(default_factory=FixedCandidateVariant)


class FixedCandidatesConfig(BaseModel):
    output_root: str = "data/results/fixed_candidates"
    strategy_config: str = "configs/strategies/two_candle_rejection.yaml"
    prop_firm_config: str = "configs/prop_firm.yaml"
    candidates: list[FixedCandidate] = Field(default_factory=list)


class PortfolioDefinition(BaseModel):
    name: str
    trades: list[str] = Field(default_factory=list)


class PortfolioCandidatesConfig(BaseModel):
    dedupe_same_signal: bool = True
    output_root: str = "data/results/portfolio_sim"
    portfolios: list[PortfolioDefinition] = Field(default_factory=list)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to project root or given base."""
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None:
        return (base / p).resolve()
    return p.resolve()


def find_project_root(start: Path | None = None) -> Path:
    """Find project root by looking for pyproject.toml."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "auta_research").exists():
            return parent
    return current


def load_strategy_config(path: str | Path, base: Path | None = None) -> StrategyConfig:
    """Load and validate a strategy YAML config."""
    resolved = resolve_path(path, base)
    data = load_yaml(resolved)
    return StrategyConfig.model_validate(data)


def load_research_config(path: str | Path, base: Path | None = None) -> ResearchConfig:
    """Load and validate a research YAML config."""
    resolved = resolve_path(path, base)
    data = load_yaml(resolved)
    return ResearchConfig.model_validate(data)


def load_fixed_candidates_config(path: str | Path, base: Path | None = None) -> FixedCandidatesConfig:
    """Load fixed candidate batch config."""
    resolved = resolve_path(path, base)
    data = load_yaml(resolved)
    return FixedCandidatesConfig.model_validate(data)


def load_prop_firm_config(path: str | Path, base: Path | None = None) -> PropFirmConfig:
    """Load and validate prop firm simulator YAML config."""
    resolved = resolve_path(path, base)
    data = load_yaml(resolved)
    cfg = PropFirmConfig.model_validate(data)
    if cfg.program is not None and cfg.name == "prop_firm_default":
        cfg.name = cfg.program.name
    return cfg


def load_portfolio_candidates_config(path: str | Path, base: Path | None = None) -> PortfolioCandidatesConfig:
    """Load portfolio simulation config."""
    resolved = resolve_path(path, base)
    data = load_yaml(resolved)
    return PortfolioCandidatesConfig.model_validate(data)


def load_symbols_config(path: str | Path, base: Path | None = None) -> SymbolsConfig:
    """Load symbols metadata config."""
    resolved = resolve_path(path, base)
    data = load_yaml(resolved)
    return SymbolsConfig.model_validate(data)


def get_point_size(symbol: str, symbols_config: SymbolsConfig | None = None) -> float:
    """Return point size for a symbol."""
    if symbols_config is None:
        symbols_config = SymbolsConfig()
    pv = symbols_config.point_values
    if symbol in pv:
        return pv[symbol]
    if "JPY" in symbol:
        return pv.get("JPY", 0.001)
    if symbol in ("XAUUSD", "XPDUSD"):
        return pv.get(symbol, 0.01)
    if symbol == "XAGUSD":
        return pv.get("XAGUSD", 0.001)
    return pv.get("default", 0.00001)
