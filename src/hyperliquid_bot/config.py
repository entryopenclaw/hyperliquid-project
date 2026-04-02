from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import deep_merge, expand_env


@dataclass(slots=True)
class HyperliquidConfig:
    network: str = "testnet"
    account_address: str = ""
    secret_key_env: str = "HL_SECRET_KEY"
    vault_address: str = ""
    api_url: str = ""


@dataclass(slots=True)
class MarketConfig:
    symbol: str = "BTC"
    candle_interval: str = "1m"
    depth_levels: int = 5
    feature_window: int = 200
    startup_candle_lookback: int = 30
    min_warm_price_points: int = 20


@dataclass(slots=True)
class ExecutionConfig:
    mode: str = "paper"
    paper_starting_balance_usd: float = 10_000.0
    paper_latency_bps: float = 0.5
    paper_max_latency_bps: float = 3.0
    paper_fill_tolerance_bps: float = 2.5
    paper_partial_fill_min_fraction: float = 0.25
    resting_order_max_age_s: int = 20
    limit_offset_bps: float = 1.0
    ioc_slippage_bps: float = 8.0
    reconcile_interval_s: int = 15
    schedule_cancel_after_s: int = 20
    max_open_orders: int = 3


@dataclass(slots=True)
class RiskConfig:
    max_leverage: float = 2.0
    base_order_notional_usd: float = 50.0
    max_position_notional_usd: float = 250.0
    max_daily_loss_usd: float = 25.0
    max_drawdown_usd: float = 40.0
    max_slippage_bps: float = 12.0
    max_spread_bps: float = 8.0
    max_data_age_s: float = 5.0
    max_reject_streak: int = 3
    cooldown_after_stop_s: int = 300
    min_account_value_usd: float = 50.0


@dataclass(slots=True)
class StrategyConfig:
    long_entry_bps: float = 2.0
    short_entry_bps: float = -2.0
    exit_threshold_bps: float = 0.5
    min_confidence: float = 0.55
    position_scale: float = 1.0
    urgency_confidence: float = 0.8


@dataclass(slots=True)
class TrainingConfig:
    enabled: bool = True
    retrain_interval_hours: int = 24
    min_training_rows: int = 300
    train_window_rows: int = 5000
    validation_rows: int = 1000
    lookahead_bars: int = 3
    promotion_min_expectancy_bps: float = 0.5
    promotion_min_win_rate: float = 0.52
    promotion_max_drawdown_usd: float = 20.0
    promotion_max_turnover: float = 40.0


@dataclass(slots=True)
class BacktestConfig:
    assumed_notional_usd: float = 100.0
    fee_bps: float = 0.7
    slippage_bps: float = 0.8
    min_signal_bps: float = 0.1


@dataclass(slots=True)
class StorageConfig:
    root_dir: str = "data"
    sqlite_path: str = "data/runtime.db"
    raw_events_path: str = "data/raw_events.jsonl"
    features_path: str = "data/features.jsonl"
    models_dir: str = "data/models"
    reports_dir: str = "data/reports"
    status_path: str = "data/status.json"


@dataclass(slots=True)
class MonitoringConfig:
    enable_http: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"


@dataclass(slots=True)
class BotConfig:
    hyperliquid: HyperliquidConfig
    market: MarketConfig
    execution: ExecutionConfig
    risk: RiskConfig
    strategy: StrategyConfig
    training: TrainingConfig
    backtest: BacktestConfig
    storage: StorageConfig
    monitoring: MonitoringConfig
    source_path: Path

    @property
    def secret_key(self) -> str:
        return os.getenv(self.hyperliquid.secret_key_env, "")

    def validate(self) -> None:
        if self.hyperliquid.network not in {"testnet", "mainnet"}:
            raise ValueError("hyperliquid.network must be 'testnet' or 'mainnet'")
        if self.execution.mode not in {"paper", "shadow", "live"}:
            raise ValueError("execution.mode must be paper, shadow, or live")
        if self.execution.paper_starting_balance_usd <= 0:
            raise ValueError("execution.paper_starting_balance_usd must be > 0")
        if self.market.startup_candle_lookback <= 0:
            raise ValueError("market.startup_candle_lookback must be > 0")
        if self.market.min_warm_price_points <= 0:
            raise ValueError("market.min_warm_price_points must be > 0")
        if self.market.startup_candle_lookback < self.market.min_warm_price_points:
            raise ValueError("market.startup_candle_lookback must be >= market.min_warm_price_points")
        if self.execution.paper_latency_bps < 0:
            raise ValueError("execution.paper_latency_bps must be >= 0")
        if self.execution.paper_max_latency_bps <= 0:
            raise ValueError("execution.paper_max_latency_bps must be > 0")
        if self.execution.paper_latency_bps > self.execution.paper_max_latency_bps:
            raise ValueError("execution.paper_latency_bps must be <= execution.paper_max_latency_bps")
        if self.execution.paper_fill_tolerance_bps <= 0:
            raise ValueError("execution.paper_fill_tolerance_bps must be > 0")
        if not 0 < self.execution.paper_partial_fill_min_fraction <= 1:
            raise ValueError("execution.paper_partial_fill_min_fraction must be in (0, 1]")
        if self.execution.resting_order_max_age_s <= 0:
            raise ValueError("execution.resting_order_max_age_s must be > 0")
        if self.risk.max_position_notional_usd < self.risk.base_order_notional_usd:
            raise ValueError("risk.max_position_notional_usd must be >= base_order_notional_usd")
        if self.training.validation_rows <= 0:
            raise ValueError("training.validation_rows must be > 0")
        if self.training.retrain_interval_hours <= 0:
            raise ValueError("training.retrain_interval_hours must be > 0")
        if self.backtest.assumed_notional_usd <= 0:
            raise ValueError("backtest.assumed_notional_usd must be > 0")
        if self.backtest.min_signal_bps < 0:
            raise ValueError("backtest.min_signal_bps must be >= 0")


def _defaults() -> dict[str, Any]:
    return {
        "hyperliquid": asdict(HyperliquidConfig()),
        "market": asdict(MarketConfig()),
        "execution": asdict(ExecutionConfig()),
        "risk": asdict(RiskConfig()),
        "strategy": asdict(StrategyConfig()),
        "training": asdict(TrainingConfig()),
        "backtest": asdict(BacktestConfig()),
        "storage": asdict(StorageConfig()),
        "monitoring": asdict(MonitoringConfig()),
    }


def _build(config: dict[str, Any], source_path: Path) -> BotConfig:
    built = BotConfig(
        hyperliquid=HyperliquidConfig(**config["hyperliquid"]),
        market=MarketConfig(**config["market"]),
        execution=ExecutionConfig(**config["execution"]),
        risk=RiskConfig(**config["risk"]),
        strategy=StrategyConfig(**config["strategy"]),
        training=TrainingConfig(**config["training"]),
        backtest=BacktestConfig(**config["backtest"]),
        storage=StorageConfig(**config["storage"]),
        monitoring=MonitoringConfig(**config["monitoring"]),
        source_path=source_path,
    )
    built.validate()
    return built


def load_config(path: str | Path) -> BotConfig:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    merged = deep_merge(_defaults(), expand_env(raw))
    return _build(merged, source_path=config_path)
