from __future__ import annotations

from pathlib import Path

from hyperliquid_bot.config import (
    BotConfig,
    ExecutionConfig,
    HyperliquidConfig,
    MarketConfig,
    MonitoringConfig,
    RiskConfig,
    StorageConfig,
    StrategyConfig,
    TrainingConfig,
)
from hyperliquid_bot.orchestrator import AutonomousBot


def _bot_config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        hyperliquid=HyperliquidConfig(),
        market=MarketConfig(),
        execution=ExecutionConfig(),
        risk=RiskConfig(),
        strategy=StrategyConfig(),
        training=TrainingConfig(
            min_training_rows=20,
            train_window_rows=100,
            validation_rows=10,
            lookahead_bars=1,
            promotion_min_expectancy_bps=0.1,
            promotion_min_win_rate=0.5,
            promotion_max_drawdown_usd=1000.0,
            promotion_max_turnover=1000.0,
        ),
        storage=StorageConfig(
            root_dir=str(tmp_path / "data"),
            sqlite_path=str(tmp_path / "data" / "runtime.db"),
            raw_events_path=str(tmp_path / "data" / "raw_events.jsonl"),
            features_path=str(tmp_path / "data" / "features.jsonl"),
            models_dir=str(tmp_path / "data" / "models"),
            reports_dir=str(tmp_path / "data" / "reports"),
            status_path=str(tmp_path / "data" / "status.json"),
        ),
        monitoring=MonitoringConfig(),
        source_path=tmp_path / "bot.toml",
    )


def test_train_cycle_promotes_and_loads_model(tmp_path: Path) -> None:
    bot = AutonomousBot(_bot_config(tmp_path))
    mid = 100.0
    for idx in range(40):
        bot.storage.record_feature_row(
            {
                "timestamp": f"2026-01-01T00:{idx:02d}:00+00:00",
                "symbol": "BTC",
                "mid_price": mid,
                "spread_bps": 1.0,
                "features": {
                    "depth_imbalance": 1.0,
                    "trade_imbalance": 0.8,
                    "momentum_5": 0.01,
                },
            }
        )
        mid += 0.2

    outcome = bot.train()

    assert outcome["accepted"]
    assert outcome["model_version"].startswith("model-")
    assert bot.signal.model.version == outcome["model_version"]
    assert bot.monitoring.metrics["active_model_version"] == outcome["model_version"]
